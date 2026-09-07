from collections import Counter
from pathlib import Path
import sqlite3

import pytest

from config import load_config
from data.exp3 import (
    CLIMATE_BANDS,
    build_exp3_aggregation_records,
    build_exp3_attribute_test_records,
    build_exp3_continent_rows,
    build_exp3_sft_records,
    generate_exp3_qa,
    materialize_exp3_dataset,
    validate_exp3_fact_count,
    verify_exp3_dataset,
    verify_exp3_qa,
)
from evaluation.inference import load_verified_qa_split
from evaluation.metrics import score_prediction, unordered_normalized_exact_match
from scripts import run_exp03
from training.checkpoint_retention import retain_best_exp3_checkpoint
from training.cpt import verify_cpt_artifacts
from training.target_sft import load_target_sft_dataset
from utils.io import read_json, read_jsonl, write_json


@pytest.fixture
def exp3_config() -> dict:
    return load_config(Path("configs/exp03_continent_inverse.yaml"))


def test_exp3_fact_count_and_generation_invariants() -> None:
    with pytest.raises(ValueError, match="divisible by 2"):
        validate_exp3_fact_count(3)
    with pytest.raises(ValueError, match="at most 34 rows"):
        validate_exp3_fact_count(70)

    rows = build_exp3_continent_rows(2025, 68)
    assert len(rows) == 34
    assert len({row["continent_id"] for row in rows}) == len(rows)
    assert len({row["continent_name"] for row in rows}) == len(rows)
    counts = Counter(row["climate_band"] for row in rows)
    assert set(counts) == set(CLIMATE_BANDS)
    assert set(counts.values()) == {2}
    assert rows == build_exp3_continent_rows(2025, 68)


def test_exp3_database_is_standalone_and_reuses_cpt_serialization(
    tmp_path: Path, exp3_config: dict
) -> None:
    dataset_dir = tmp_path / "dataset"
    materialize_exp3_dataset(exp3_config, dataset_dir, fact_count=10)
    dataset = verify_exp3_dataset(dataset_dir, fact_count=10, seed=2025)
    with sqlite3.connect(dataset["database"]) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(continent)").fetchall()
    assert tables == [("continent",)]
    assert [column[1] for column in columns] == [
        "continent_id",
        "continent_name",
        "climate_band",
    ]
    assert len(dataset["rows"]) == 5
    cpt_manifest = read_json(dataset["cpt_manifest"])
    assert cpt_manifest["cpt_example_count"] == 5
    assert cpt_manifest["cpt_example_logical_fact_count"] == 10
    assert cpt_manifest["cpt_examples_independently_tokenized"] is True
    assert not (dataset["cpt_dir"] / "train.txt").exists()
    provenance = verify_cpt_artifacts(
        exp3_config,
        table_count=1,
        fact_count=10,
        database_path=dataset["database"],
        database_manifest_path=dataset["manifest_path"],
        readable_book_path=dataset["cpt_dir"] / "book_readable.txt",
        train_text_path=None,
        cpt_manifest_path=dataset["cpt_manifest"],
    )
    assert provenance["cpt_example_count"] == 5


def test_exp3_sft_and_inverse_aggregation_qa(
    tmp_path: Path, exp3_config: dict
) -> None:
    dataset_dir = tmp_path / "dataset"
    qa_dir = tmp_path / "qa"
    materialize_exp3_dataset(exp3_config, dataset_dir, fact_count=10)
    generate_exp3_qa(dataset_dir, qa_dir, fact_count=10, seed=2025)
    verified = verify_exp3_qa(
        qa_dir, dataset_dir=dataset_dir, fact_count=10, seed=2025
    )
    rows = verify_exp3_dataset(
        dataset_dir, fact_count=10, seed=2025
    )["rows"]

    sft = read_jsonl(verified["sft_data_dir"] / "train" / "H0.jsonl")
    assert sft == build_exp3_sft_records(rows)
    assert len(sft) == len(rows) == 5
    assert all(
        record["question"]
        == f"What climate band does {row['continent_name']} belong to?"
        and record["gold_answer"] == row["climate_band"]
        and record["target_field"] == "climate_band"
        for record, row in zip(sft, rows, strict=True)
    )
    assert all(
        read_jsonl(verified["sft_data_dir"] / "train" / f"H{hop}.jsonl") == []
        for hop in (1, 2, 3)
    )

    attribute_records = read_jsonl(qa_dir / "attribute_test" / "H0.jsonl")
    assert attribute_records == build_exp3_attribute_test_records(rows)
    assert len(attribute_records) == len(rows)
    assert [record["question"] for record in attribute_records] == [
        record["question"] for record in sft
    ]
    assert [record["gold_answer"] for record in attribute_records] == [
        record["gold_answer"] for record in sft
    ]

    test_records = read_jsonl(qa_dir / "aggregation_test" / "H0.jsonl")
    assert test_records == build_exp3_aggregation_records(rows, split="test")
    assert len(test_records) == len({row["climate_band"] for row in rows})
    by_id = {row["continent_name"]: row["continent_id"] for row in rows}
    for record in test_records:
        names = record["gold_answer"].split(", ")
        assert 1 <= len(names) <= 2
        assert names == sorted(names, key=by_id.__getitem__)

    loaded_sft, dev, _ = load_target_sft_dataset(
        qa_dir,
        dataset_dir="target_sft",
        training_split="train",
        dev_split=None,
        table_count=1,
        fact_count=10,
    )
    loaded_test, _ = load_verified_qa_split(
        qa_dir / "aggregation_test",
        split="test",
        expected_table_count=1,
        expected_fact_count=10,
        manifest_hash_key="aggregation_test_manifest_sha256",
    )
    assert loaded_sft == sft
    assert dev == []
    assert loaded_test == test_records
    assert not (qa_dir / "validation").exists()
    assert not (qa_dir / "test").exists()
    cpt_test = verified["cpt_test"]
    assert cpt_test["manifest"]["natural_language_questions"] is False
    assert cpt_test["manifest"]["source_cpt_logical_fact_count"] == 10
    assert len(cpt_test["probes"]) == 10
    assert all("?" not in probe["prompt"] for probe in cpt_test["probes"])


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("Beloria, Arvania", True),
        ("Arvania", False),
        ("Arvania, Beloria, Caldor", False),
        ("Arvania, Arvania", False),
    ],
)
def test_exp3_unordered_em_uses_multiset_semantics(
    prediction: str, expected: bool
) -> None:
    assert (
        unordered_normalized_exact_match(prediction, "Arvania, Beloria")
        is expected
    )


def test_exp3_cpt_test_uses_raw_declarative_completion_prefixes(
    tmp_path: Path, exp3_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    qa_dir = tmp_path / "qa"
    checkpoint = tmp_path / "checkpoint"
    output_dir = tmp_path / "result"
    materialize_exp3_dataset(exp3_config, dataset_dir, fact_count=2)
    generate_exp3_qa(dataset_dir, qa_dir, fact_count=2, seed=2025)
    qa = verify_exp3_qa(
        qa_dir, dataset_dir=dataset_dir, fact_count=2, seed=2025
    )
    write_json(checkpoint / "config.json", {})
    write_json(checkpoint / "training_metadata.json", {})
    seen_prompts: list[str] = []

    def fake_evaluate(records, *, prompt_formatter, **_kwargs):
        seen_prompts.extend(prompt_formatter(record["question"]) for record in records)
        return [
            score_prediction(
                record,
                f" {record['gold_answer']} climate band.",
                f"{record['gold_answer']} climate band.",
            )
            for record in records
        ], {"model_identity": "fake", "tokenizer_identity": "fake"}

    monkeypatch.setattr(run_exp03, "evaluate_with_local_checkpoint", fake_evaluate)
    run_exp03._evaluate_cpt_test(
        exp3_config,
        checkpoint=checkpoint,
        qa=qa,
        output_dir=output_dir,
        fact_count=2,
        layers=12,
    )
    assert seen_prompts == [probe["prompt"] for probe in qa["cpt_test"]["probes"]]
    assert all("Question:" not in prompt and "?" not in prompt for prompt in seen_prompts)
    metrics = read_json(output_dir / "metrics.json")
    assert metrics["completion_prefix_match"]["accuracy"] == 1.0


def test_exp3_best_checkpoint_replaces_only_on_strict_improvement(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_json(first / "config.json", {"marker": "first"})
    write_json(second / "config.json", {"marker": "second"})
    destination = tmp_path / "best" / "checkpoint"

    def metadata(score: float, source: Path, epochs: int) -> dict:
        return {
            "experiment": "exp03_continent_inverse",
            "N": 10,
            "epochs": epochs,
            "seed": 2025,
            "EM": score,
            "best_by_test_em": score,
            "attribute_test_em": 1.0,
            "aggregation_test_em": score,
            "aggregation_unordered_em": score,
            "checkpoint_source": str(source),
            "run": f"run-{epochs}",
        }

    assert retain_best_exp3_checkpoint(
        first, destination, metadata(0.62, first, 10)
    )
    assert not retain_best_exp3_checkpoint(
        second, destination, metadata(0.62, second, 25)
    )
    assert read_json(destination / "config.json")["marker"] == "first"
    assert retain_best_exp3_checkpoint(
        second, destination, metadata(0.81, second, 25)
    )
    assert read_json(destination / "config.json")["marker"] == "second"
    retained = read_json(destination.parent / "best_metadata.json")
    assert retained["epochs"] == 25
    assert retained["EM"] == 0.81

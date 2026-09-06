import argparse
from copy import deepcopy
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.evaluate as evaluate_script
import scripts.run_exp02 as exp2_runner
import scripts.train as train_script

from config import load_config
from data.cpt_test import generate_exp2_cpt_test, verify_exp2_cpt_test
from data.materialize import (
    build_exp2_database_manifest,
    materialize_selected_tables_database,
)
from data.qa import (
    assign_exp2_target_sft_splits,
    generate_condition_qa,
    generate_qa_candidates,
    generate_target_sft_qa,
    load_verified_semantic_chains,
)
from data.serialize import (
    EXP2_CPT_EXAMPLE_METHOD_VERSION,
    build_selected_cpt_records,
    database_schema_sha256,
    serialize_database_cpt,
)
from data.world import (
    NATURAL_IDENTIFIER_FIELDS,
    SEMANTIC_ENTITY_SPECS,
    build_world_for_chain_count,
    facts_per_selected_chain,
    validate_exp2_fact_count,
    validate_selected_tables,
)
from experiment import read_checkpoint_model_architecture
from experiment import resolve_model_checkpoint, verify_checkpoint_layers
from training.cpt import (
    _create_exp2_cpt_progress_bar,
    _seeded_dataloader_generator,
    build_cpt_training_plan,
    collate_independent_cpt_examples,
    tokenize_independent_cpt_records,
    verify_cpt_artifacts,
)
from training.target_sft import build_target_sft_training_plan, load_target_sft_dataset
from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, read_text, write_json
from utils.paths import (
    EXP02_RESULTS_DIR,
    exp2_condition_label,
    exp2_evaluation_result_dir,
)


@pytest.fixture(scope="module")
def exp2_config() -> dict:
    return load_config(Path("configs/exp02_capacity_boundary.yaml"))


@pytest.mark.parametrize(
    ("tables", "expected"),
    [
        (["continent"], 2),
        (["country"], 3),
        (["continent", "country"], 5),
        ([spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS], 40),
        (["course", "student"], 8),
    ],
)
def test_exp2_selected_table_fact_semantics(tables: list[str], expected: int) -> None:
    assert facts_per_selected_chain(tables) == expected
    assert validate_exp2_fact_count(expected * 7, tables) == 7


def test_exp2_table_selection_validation_and_t() -> None:
    assert validate_selected_tables(["course", "student"]) == ("course", "student")
    assert len(validate_selected_tables(["continent", "country"])) == 2
    with pytest.raises(ValueError, match="at least one"):
        validate_selected_tables([])
    with pytest.raises(ValueError, match="duplicate"):
        validate_selected_tables(["continent", "continent"])
    with pytest.raises(ValueError, match="unknown canonical"):
        validate_selected_tables(["planet"])
    with pytest.raises(ValueError, match="Nearest valid values are 1000 and 1002"):
        validate_exp2_fact_count(1001, ["continent"])


def _bundle(tmp_path: Path, config: dict, tables: list[str], chains: int) -> Path:
    selected = validate_selected_tables(tables)
    bundle = tmp_path / "bundle"
    database = bundle / "database.sqlite"
    world = build_world_for_chain_count(config, chains)
    materialization = materialize_selected_tables_database(
        world, selected, chains, database
    )
    schema_hash = database_schema_sha256(database)
    manifest = build_exp2_database_manifest(
        config,
        {**materialization, "artifact_path": str(bundle.resolve())},
        generation_timestamp="20260905_120000_000000",
        canonical_database_sha256="a" * 64,
        canonical_database_manifest_sha256="b" * 64,
        canonical_schema_sha256=schema_hash,
        generated_schema_sha256=schema_hash,
        configuration_sha256="c" * 64,
        database_sha256=hash_file(database),
    )
    write_json(bundle / "manifest.json", manifest)
    cpt = bundle / "cpt"
    cpt_manifest = serialize_database_cpt(
        config,
        database,
        bundle / "manifest.json",
        None,
        readable_book_path=cpt / "book_readable.txt",
        expected_table_count=len(selected),
        expected_logical_fact_count=chains * facts_per_selected_chain(selected),
    )
    write_json(cpt / "manifest.json", cpt_manifest)
    return bundle


def test_exp2_hidden_fk_support_preserves_schema_but_not_exposure(
    tmp_path: Path, exp2_config: dict
) -> None:
    bundle = _bundle(tmp_path, exp2_config, ["student"], 5)
    with sqlite3.connect(bundle / "database.sqlite") as connection:
        table_count = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        assert table_count == 12
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute('SELECT COUNT(*) FROM "enrollment"').fetchone() == (5,)
    manifest = __import__("json").loads((bundle / "manifest.json").read_text())
    assert manifest["T"] == 1
    assert manifest["N"] == 20
    assert manifest["attribute_fact_count"] == 15
    assert manifest["relation_fact_count"] == 5
    book = read_text(bundle / "cpt" / "book_readable.txt")
    assert "Student Records" in book
    assert "Enrollment Records" not in book
    assert "Course Records" not in book
    assert not (bundle / "cpt" / "train.txt").exists()
    cpt_manifest = read_json(bundle / "cpt" / "manifest.json")
    assert cpt_manifest["readable_book_artifact"] == "book_readable.txt"
    assert cpt_manifest["record_passes_per_cpt_epoch"] == 1
    assert cpt_manifest["cpt_examples_independently_tokenized"] is True
    assert cpt_manifest["cpt_examples_include_book_context"] is False
    assert "fact_exposure" not in cpt_manifest
    assert "train_text_sha256" not in cpt_manifest
    assert not any(key.endswith("_per_exposure") for key in cpt_manifest)


def _exp2_qa_bundle(
    tmp_path: Path, config: dict, *, chains: int = 5
) -> tuple[Path, Path, dict]:
    dataset = _bundle(tmp_path / "dataset", config, ["continent"], chains)
    qa_root = tmp_path / "qa"
    generate_condition_qa(
        config,
        dataset / "database.sqlite",
        dataset / "manifest.json",
        qa_root,
        expected_table_count=1,
        expected_logical_fact_count=chains * 2,
        source_training_data_dir=dataset,
        generation_timestamp="20260905_120000_000000",
    )
    result = generate_target_sft_qa(
        config,
        dataset / "database.sqlite",
        dataset / "manifest.json",
        qa_root,
        expected_table_count=1,
        expected_logical_fact_count=chains * 2,
        source_training_data_dir=dataset,
        generation_timestamp="20260905_120000_000000",
    )
    return dataset, qa_root, result


def test_exp2_target_sft_uses_all_reserved_chains_and_creates_no_dev(
    tmp_path: Path, exp2_config: dict
) -> None:
    _, qa_root, result = _exp2_qa_bundle(tmp_path, exp2_config)
    evaluation_manifest = read_json(qa_root / "split_manifest.json")
    sft_manifest = result["split_manifest"]
    assert sft_manifest["train_chain_indices"] == evaluation_manifest[
        "reserved_chain_indices"
    ]
    assert sft_manifest["train_chain_count"] == evaluation_manifest[
        "reserved_chain_count"
    ]
    assert assign_exp2_target_sft_splits(
        evaluation_manifest["reserved_chain_indices"]
    ) == {"train": evaluation_manifest["reserved_chain_indices"]}
    assert not (qa_root / "target_sft" / "dev").exists()
    assert "dev_manifest" not in result
    assert not any(key.startswith("dev_") for key in sft_manifest)


def test_exp2_target_sft_generation_does_not_modify_validation_or_test(
    tmp_path: Path, exp2_config: dict
) -> None:
    dataset = _bundle(tmp_path / "dataset", exp2_config, ["continent"], 5)
    qa_root = tmp_path / "qa"
    generate_condition_qa(
        exp2_config,
        dataset / "database.sqlite",
        dataset / "manifest.json",
        qa_root,
        expected_table_count=1,
        expected_logical_fact_count=10,
        source_training_data_dir=dataset,
        generation_timestamp="20260905_120000_000000",
    )
    held_out_paths = sorted(
        path
        for split in ("validation", "test")
        for path in (qa_root / split).iterdir()
        if path.is_file()
    )
    before = {path: hash_file(path) for path in held_out_paths}
    generate_target_sft_qa(
        exp2_config,
        dataset / "database.sqlite",
        dataset / "manifest.json",
        qa_root,
        expected_table_count=1,
        expected_logical_fact_count=10,
        source_training_data_dir=dataset,
        generation_timestamp="20260905_120000_000000",
    )
    assert {path: hash_file(path) for path in held_out_paths} == before
    manifest = read_json(qa_root / "target_sft" / "split_manifest.json")
    assert not (
        set(manifest["train_chain_indices"])
        & set(manifest["validation_chain_indices"])
    )
    assert not set(manifest["train_chain_indices"]) & set(
        manifest["test_chain_indices"]
    )
    assert not set(manifest["validation_chain_indices"]) & set(
        manifest["test_chain_indices"]
    )


def test_exp2_sft_loader_reads_train_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp2_config: dict,
) -> None:
    _, qa_root, _ = _exp2_qa_bundle(tmp_path, exp2_config)
    import training.target_sft as target_sft_module

    original_read_jsonl = target_sft_module.read_jsonl

    def train_only_read_jsonl(path: Path):
        if "validation" in Path(path).parts or "test" in Path(path).parts:
            pytest.fail("Exp02 SFT loaded held-out QA records")
        return original_read_jsonl(path)

    monkeypatch.setattr(target_sft_module, "read_jsonl", train_only_read_jsonl)
    train_records, dev_records, provenance = load_target_sft_dataset(
        qa_root,
        dataset_dir="target_sft",
        training_split="train",
        dev_split=None,
        table_count=1,
        fact_count=10,
    )
    assert train_records
    assert dev_records == []
    assert provenance["dev_split_used"] is False
    assert provenance["validation_split_used"] is False
    assert provenance["test_split_used"] is False


def test_exp2_cpt_authenticates_independent_records_and_keeps_book_as_artifact(
    tmp_path: Path, exp2_config: dict
) -> None:
    bundle = _bundle(tmp_path, exp2_config, ["continent"], 5)
    provenance = verify_cpt_artifacts(
        exp2_config,
        table_count=1,
        fact_count=10,
        database_path=bundle / "database.sqlite",
        database_manifest_path=bundle / "manifest.json",
        readable_book_path=bundle / "cpt" / "book_readable.txt",
        train_text_path=bundle / "cpt" / "train.txt",
        cpt_manifest_path=bundle / "cpt" / "manifest.json",
    )
    assert provenance["readable_book_artifact"] == "book_readable.txt"
    assert provenance["readable_book_artifact_path"] == str(
        (bundle / "cpt" / "book_readable.txt").resolve()
    )
    assert provenance["record_passes_per_cpt_epoch"] == 1
    assert provenance["cpt_example_count"] == 5
    assert not (bundle / "cpt" / "train.txt").exists()


class _Exp2CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        return [ord(character) + 2 for character in text]


def test_exp2_cpt_records_are_isolated_complete_and_independently_tokenized(
    tmp_path: Path, exp2_config: dict
) -> None:
    bundle = _bundle(tmp_path, exp2_config, ["continent", "country"], 5)
    records, metadata = build_selected_cpt_records(
        bundle / "database.sqlite", read_json(bundle / "manifest.json")
    )
    book = read_text(bundle / "cpt" / "book_readable.txt")
    assert metadata["method_version"] == EXP2_CPT_EXAMPLE_METHOD_VERSION
    assert metadata["record_count"] == 15
    assert metadata["logical_fact_count"] == 25
    assert sum(len(record["covered_fact_sha256"]) for record in records) == 25
    assert len({record["id"] for record in records}) == len(records)
    for record in records:
        assert "\n" not in record["text"]
        assert record["text"] in book.splitlines()
        assert "The Academic Database Book" not in record["text"]
        assert not record["text"].endswith(" Records")

    tokenizer = _Exp2CharacterTokenizer()
    examples, statistics = tokenize_independent_cpt_records(
        records, tokenizer, context_length=512
    )
    assert tokenizer.encoded_texts == [record["text"] for record in records]
    assert len(examples) == statistics["sequence_count"] == len(records)
    assert statistics["eos_token_count"] == len(records)
    assert statistics["cross_record_attention"] is False
    assert statistics["global_stream_chunking"] is False
    plan = build_cpt_training_plan(
        exp2_config,
        table_count=2,
        fact_count=25,
        sequence_count=len(examples),
    )
    assert plan["trained_sequence_count_per_epoch"] == len(records)
    assert plan["dropped_sequences_per_epoch"] == 0
    for record, example in zip(records, examples, strict=True):
        assert example["record_id"] == record["id"]
        assert example["input_ids"][-1] == tokenizer.eos_token_id
        assert example["attention_mask"] == [1] * len(example["input_ids"])
        assert example["labels"] == example["input_ids"]

    batch = collate_independent_cpt_examples(
        [examples[0], examples[-1]], pad_token_id=tokenizer.pad_token_id
    )
    for input_ids, attention_mask, labels in zip(
        batch["input_ids"].tolist(),
        batch["attention_mask"].tolist(),
        batch["labels"].tolist(),
        strict=True,
    ):
        for token_id, attends, label in zip(
            input_ids, attention_mask, labels, strict=True
        ):
            assert label == token_id if attends else label == -100


def test_exp2_cpt_shuffle_is_deterministic_but_changes_record_order_each_epoch(
    tmp_path: Path, exp2_config: dict
) -> None:
    torch = pytest.importorskip("torch")
    bundle = _bundle(tmp_path, exp2_config, ["continent"], 10)
    records, _ = build_selected_cpt_records(
        bundle / "database.sqlite", read_json(bundle / "manifest.json")
    )

    def two_epochs() -> tuple[list[int], list[int]]:
        generator = _seeded_dataloader_generator(
            torch, exp2_config["experiment"]["seed"]
        )
        sampler = torch.utils.data.RandomSampler(records, generator=generator)
        return list(iter(sampler)), list(iter(sampler))

    first_epoch, second_epoch = two_epochs()
    repeated_first, repeated_second = two_epochs()
    assert first_epoch == repeated_first
    assert second_epoch == repeated_second
    assert first_epoch != second_epoch
    assert sorted(first_epoch) == sorted(second_epoch) == list(range(len(records)))


def test_exp2_cpt_progress_bar_is_epoch_scoped(capsys: pytest.CaptureFixture[str]) -> None:
    progress = _create_exp2_cpt_progress_bar(
        {"epochs": 3, "optimizer_steps": 9, "learning_rate": 3e-5}
    )
    try:
        assert progress.total == 3
        assert progress.unit == "epoch"
        assert progress.desc == "Exp02 CPT epoch 0/3"
        assert "optimizer_step=0/9" in progress.postfix
        assert "loss=n/a" in progress.postfix
        assert "lr=3.000e-05" in progress.postfix
        progress.set_description("Exp02 CPT epoch 1/3")
        progress.set_postfix(
            optimizer_step="1/9", loss="2.500000", lr="1.000e-05"
        )
        progress.update(1)
        assert progress.n == 1
        assert progress.desc.startswith("Exp02 CPT epoch 1/3")
        assert "optimizer_step=1/9" in progress.postfix
        assert "loss=2.500000" in progress.postfix
        assert "lr=1.000e-05" in progress.postfix
    finally:
        progress.close()
    capsys.readouterr()


def test_exp2_cpt_test_uses_only_seen_facts_and_is_deterministic(
    tmp_path: Path, exp2_config: dict
) -> None:
    all_tables = [spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS]
    bundle = _bundle(tmp_path / "dataset", exp2_config, all_tables, 40)
    first = generate_exp2_cpt_test(
        exp2_config,
        training_data_dir=bundle,
        output_dir=tmp_path / "first" / "cpt_test",
    )
    second = generate_exp2_cpt_test(
        exp2_config,
        training_data_dir=bundle,
        output_dir=tmp_path / "second" / "cpt_test",
    )
    assert first["probes"] == second["probes"]
    assert first["manifest"] == second["manifest"]
    verified = verify_exp2_cpt_test(
        training_data_dir=bundle, cpt_test_dir=first["output_dir"]
    )
    cpt_records, _ = build_selected_cpt_records(
        bundle / "database.sqlite", read_json(bundle / "manifest.json")
    )
    source_by_id = {record["id"]: record for record in cpt_records}
    assert len(verified["probes"]) == len(cpt_records) * 2
    assert {probe["source_cpt_record_id"] for probe in verified["probes"]} == set(
        source_by_id
    )
    for probe in verified["probes"]:
        source = source_by_id[probe["source_cpt_record_id"]]
        assert probe["source_fact"] == source["text"]
        assert "\n" not in probe["prompt"]
        assert "?" not in probe["prompt"]
        assert "The Academic Database Book" not in probe["prompt"]
        assert " Records" not in probe["prompt"]
        if probe["probe_type"] == "canonical_completion":
            assert source["text"].startswith(
                probe["prompt"] + probe["gold_answer"]
            )
        else:
            assert probe["prompt"] + probe["gold_answer"] + "." != source["text"]

    canonical = next(
        probe
        for probe in verified["probes"]
        if probe["probe_type"] == "canonical_completion"
    )
    assert canonical["prompt"].endswith("is a continent with a ")
    assert canonical["attribute"] == "climate_band"


def test_exp2_exact_epoch_plans_have_no_hidden_exposure_or_dev_selection(
    exp2_config: dict,
) -> None:
    config = deepcopy(exp2_config)
    config["training"]["cpt_epochs"] = 200
    config["target_sft"]["epochs"] = 200
    cpt_plan = build_cpt_training_plan(
        config, table_count=1, fact_count=10, sequence_count=3
    )
    sft_plan = build_target_sft_training_plan(
        config, table_count=1, fact_count=10, example_count=5
    )
    assert cpt_plan["requested_cpt_epochs"] == 200
    assert cpt_plan["requested_record_passes"] == 200
    assert cpt_plan["total_optimizer_steps"] == 200
    assert "passes_over_serialized_corpus" not in cpt_plan
    assert "effective_fact_exposure" not in cpt_plan
    assert sft_plan["requested_sft_epochs"] == 200
    assert sft_plan["total_optimizer_steps"] == 200
    assert sft_plan["checkpoint_selection"] == "final_requested_epoch"
    assert sft_plan["early_stopping_enabled"] is False
    assert sft_plan["dev_split_used"] is False
    assert "early_stopping_patience" not in sft_plan


def test_exp2_sft_checkpoint_provenance_requires_the_final_requested_epoch(
    tmp_path: Path,
) -> None:
    qa_root = tmp_path / "qa"
    write_json(qa_root / "target_sft" / "split_manifest.json", {"version": 1})
    cpt_checkpoint = tmp_path / "cpt"
    cpt_checkpoint.mkdir()
    checkpoint = tmp_path / "sft"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 12})
    metadata = {
        "experiment": "exp02_capacity_boundary",
        "stage": "target-sft",
        "model": "gpt2",
        "T": 1,
        "N": 10,
        "L": 12,
        "source_checkpoint": str(cpt_checkpoint.resolve()),
        "requested_sft_epochs": 200,
        "completed_sft_epochs": 200,
        "final_checkpoint_epoch": 200,
        "checkpoint_selection": "final_requested_epoch",
        "early_stopping_enabled": False,
        "dev_split_used": False,
        "target_sft_split_manifest_sha256": hash_file(
            qa_root / "target_sft" / "split_manifest.json"
        ),
        "provenance": {"selected_tables": ["continent"]},
    }
    write_json(checkpoint / "training_metadata.json", metadata)
    qa = {"root": qa_root}
    exp2_runner._verify_sft_checkpoint(
        checkpoint,
        qa=qa,
        cpt_checkpoint=cpt_checkpoint.resolve(),
        selected_tables=("continent",),
        model_name="gpt2",
        layers=12,
        table_count=1,
        fact_count=10,
        requested_epochs=200,
    )
    metadata["completed_sft_epochs"] = 199
    metadata["final_checkpoint_epoch"] = 199
    write_json(checkpoint / "training_metadata.json", metadata)
    with pytest.raises(ValueError, match="completed_sft_epochs mismatch"):
        exp2_runner._verify_sft_checkpoint(
            checkpoint,
            qa=qa,
            cpt_checkpoint=cpt_checkpoint.resolve(),
            selected_tables=("continent",),
            model_name="gpt2",
            layers=12,
            table_count=1,
            fact_count=10,
            requested_epochs=200,
        )


def test_exp2_determinism_and_nested_prefixes(exp2_config: dict) -> None:
    small = build_world_for_chain_count(exp2_config, 7)
    repeated = build_world_for_chain_count(exp2_config, 7)
    large = build_world_for_chain_count(exp2_config, 12)
    assert small == repeated
    assert large["chains"][:7] == small["chains"]


def test_exp2_world_is_stable_across_the_old_1000_chain_boundary(
    exp2_config: dict,
) -> None:
    world_999 = build_world_for_chain_count(exp2_config, 999)
    world_1000 = build_world_for_chain_count(exp2_config, 1000)
    world_1002 = build_world_for_chain_count(exp2_config, 1002)
    repeated = build_world_for_chain_count(exp2_config, 1002)
    assert world_1000["chains"][:999] == world_999["chains"]
    assert world_1002["chains"][:1000] == world_1000["chains"]
    assert repeated == world_1002

    identifiers = [
        entity["entity_id"]
        for chain in world_1002["chains"]
        for entity in chain["entities"]
    ]
    assert len(identifiers) == len(set(identifiers))
    assert all(
        len(entity["entity_id"][3:]) == 6
        for chain in world_1002["chains"][998:]
        for entity in chain["entities"]
    )
    for position, spec in enumerate(SEMANTIC_ENTITY_SPECS):
        natural_name = NATURAL_IDENTIFIER_FIELDS.get(spec["entity_type"])
        if natural_name is None:
            continue
        values = {
            next(
                attribute["value"]
                for attribute in chain["entities"][position]["attributes"]
                if attribute["name"] == natural_name
            )
            for chain in world_1002["chains"]
        }
        assert len(values) == 1002


def test_exp2_world_scales_only_at_genuine_natural_namespace_capacity(
    exp2_config: dict,
) -> None:
    world = build_world_for_chain_count(exp2_config, 2600)
    city_position = next(
        index
        for index, spec in enumerate(SEMANTIC_ENTITY_SPECS)
        if spec["entity_type"] == "city"
    )
    city_names = [
        next(
            attribute["value"]
            for attribute in chain["entities"][city_position]["attributes"]
            if attribute["name"] == "city_name"
        )
        for chain in world["chains"]
    ]
    assert len(city_names) == len(set(city_names)) == 2600
    assert all(" Record " not in name for name in city_names[:2560])
    assert all(" Record " in name for name in city_names[2560:])


def test_exp2_world_preserves_existing_canonical_prefix(exp2_config: dict) -> None:
    canonical_world = read_json(
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "generated_databases"
        / "exp01_first_feasibility"
        / "master_world"
        / "world.json"
    )
    generated = build_world_for_chain_count(
        exp2_config, canonical_world["construction"]["total_chains"]
    )
    assert generated["chains"] == canonical_world["chains"]


def test_exp2_qa_uses_only_exposed_paths(tmp_path: Path, exp2_config: dict) -> None:
    bundle = _bundle(tmp_path, exp2_config, ["continent", "country"], 5)
    chains, manifest = load_verified_semantic_chains(
        bundle / "database.sqlite",
        bundle / "manifest.json",
        expected_table_count=2,
        expected_logical_fact_count=25,
    )
    candidates = generate_qa_candidates(
        chains, [0], "train", exposed_positions=set(manifest["selected_positions"])
    )
    assert candidates["H0"]
    assert candidates["H1"]
    assert candidates["H2"] == candidates["H3"] == []
    assert {record["source_entity_type"] for record in candidates["H0"]} <= {
        "continent", "country"
    }
    assert all(
        record["source_entity_type"] == "country"
        and record["target_entity_type"] == "continent"
        for record in candidates["H1"]
    )


def test_exp2_paths_and_model_defaults() -> None:
    first = exp2_condition_label(["continent"], 500, "20260905_120000_000001")
    second = exp2_condition_label(["continent"], 500, "20260905_120000_000002")
    assert first != second
    assert first.startswith("T01_N500_continent_")
    checkpoint, layers = resolve_model_checkpoint("gpt2")
    assert checkpoint.name == "gpt2"
    assert layers == 12


def test_exp2_qwen3_model_registry_reads_local_checkpoint_config() -> None:
    checkpoint, layers = resolve_model_checkpoint("qwen3-0.6b-base")
    assert checkpoint.name == "qwen3-0.6b-base"
    assert layers == 28
    assert read_checkpoint_model_architecture(checkpoint) == {
        "native_layers": 28,
        "hidden_size": 1024,
        "attention_heads": 16,
        "context_length": 32768,
    }


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (
            "gpt2",
            {
                "native_layers": 12,
                "hidden_size": 768,
                "attention_heads": 12,
                "context_length": 1024,
            },
        ),
        (
            "qwen3-0.6b-base",
            {
                "native_layers": 28,
                "hidden_size": 1024,
                "attention_heads": 16,
                "context_length": 32768,
            },
        ),
    ],
)
def test_exp2_wrapper_resolved_config_uses_cli_model_architecture(
    tmp_path: Path, model_name: str, expected: dict[str, int]
) -> None:
    checkpoint, native_layers = resolve_model_checkpoint(model_name)
    config = exp2_runner._write_resolved_config(
        base_config_path=Path("configs/exp02_capacity_boundary.yaml"),
        output_path=tmp_path / "resolved_config.yaml",
        model_name=model_name,
        native_layers=native_layers,
        model_architecture=read_checkpoint_model_architecture(checkpoint),
        overrides={},
    )
    assert config["model"]["name"] == model_name
    for key, value in expected.items():
        assert config["model"][key] == value
    assert config["training"]["context_length"] == 512
    assert config["target_sft"]["context_length"] == 128
    assert config["evaluation"]["context_length"] == 256


def test_exp2_train_cpt_uses_cli_selected_qwen_model_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    captured: dict[str, object] = {}
    condition = {
        "T": 1,
        "N": 100,
        "cpt_dir": tmp_path / "cpt",
        "database": tmp_path / "database.sqlite",
        "manifest_path": tmp_path / "manifest.json",
        "cpt_manifest": tmp_path / "cpt" / "manifest.json",
    }
    monkeypatch.setattr(
        train_script, "load_exp2_dataset_condition", lambda _: condition
    )
    monkeypatch.setattr(train_script, "verify_checkpoint_layers", lambda *_, **__: {})

    def run_cpt(config, **kwargs):
        captured["config"] = deepcopy(config)
        captured["kwargs"] = kwargs
        return {"optimizer_steps": 0}

    monkeypatch.setattr(train_script, "run_cpt_training", run_cpt)
    train_script._run_exp2(
        argparse.Namespace(
            stage="cpt",
            table_count=None,
            fact_count=None,
            layers=None,
            source_checkpoint=None,
            training_data_dir=tmp_path,
            sft_data_dir=None,
            model="qwen3-0.6b-base",
        ),
        deepcopy(exp2_config),
    )
    config = captured["config"]
    assert config["model"]["name"] == "qwen3-0.6b-base"
    assert config["model"]["native_layers"] == 28
    assert config["model"]["hidden_size"] == 1024
    assert config["model"]["attention_heads"] == 16
    assert captured["kwargs"]["layers"] == 28


def test_exp2_train_target_sft_uses_cli_selected_qwen_model_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    captured: dict[str, object] = {}
    source_checkpoint = tmp_path / "qwen-cpt"
    source_checkpoint.mkdir()
    write_json(
        source_checkpoint / "config.json",
        {
            "model_type": "qwen3",
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "max_position_embeddings": 32768,
        },
    )
    write_json(source_checkpoint / "training_metadata.json", {})
    sft_dir = tmp_path / "qa" / "target_sft"
    sft_dir.mkdir(parents=True)
    write_json(
        sft_dir / "split_manifest.json",
        {
            "experiment_name": "exp02_capacity_boundary",
            "T": 1,
            "requested_N": 100,
        },
    )
    monkeypatch.setattr(train_script, "verify_checkpoint_layers", lambda *_, **__: {})

    def run_sft(config, **kwargs):
        captured["config"] = deepcopy(config)
        captured["kwargs"] = kwargs
        return {"optimizer_steps": 0}

    monkeypatch.setattr(train_script, "run_target_sft_training", run_sft)
    train_script._run_exp2(
        argparse.Namespace(
            stage="target-sft",
            table_count=None,
            fact_count=None,
            layers=None,
            source_checkpoint=source_checkpoint,
            training_data_dir=None,
            sft_data_dir=sft_dir,
            model="qwen3-0.6b-base",
        ),
        deepcopy(exp2_config),
    )
    config = captured["config"]
    assert config["model"]["name"] == "qwen3-0.6b-base"
    assert config["model"]["native_layers"] == 28
    assert config["model"]["context_length"] == 32768
    assert captured["kwargs"]["layers"] == 28


def test_exp2_evaluation_result_path_uses_exact_n_and_timestamp() -> None:
    assert exp2_evaluation_result_dir(
        table_count=2,
        fact_count=1000,
        split="validation",
        stage="eval_cpt",
        timestamp="12-34-56_05-09-2026",
    ) == (
        EXP02_RESULTS_DIR
        / "t_sweep"
        / "T02"
        / "n_sweep"
        / "N1000"
        / "validation"
        / "eval_cpt"
        / "12-34-56_05-09-2026"
    )
    with pytest.raises(ValueError, match="HH-MM-SS_DD-MM-YYYY"):
        exp2_evaluation_result_dir(
            table_count=2,
            fact_count=1000,
            split="validation",
            stage="eval_cpt",
            timestamp="20260905_123456",
        )


def test_exp2_evaluation_stage_comes_from_checkpoint_metadata() -> None:
    assert evaluate_script._exp2_evaluation_stage(
        {"experiment": "exp02_capacity_boundary", "stage": "cpt"}
    ) == "eval_cpt"
    assert evaluate_script._exp2_evaluation_stage(
        {"experiment": "exp02_capacity_boundary", "stage": "target-sft"}
    ) == "eval_sft"
    with pytest.raises(ValueError, match="stage must be"):
        evaluate_script._exp2_evaluation_stage(
            {"experiment": "exp02_capacity_boundary"}
        )


def _exp2_qa_manifest() -> dict:
    return {
        "experiment_name": "exp02_capacity_boundary",
        "T": 2,
        "requested_N": 1000,
        "selected_tables": ["continent", "country"],
        "source_database_sha256": "a" * 64,
        "source_database_manifest_sha256": "b" * 64,
        "source_dataset_manifest_sha256": "b" * 64,
    }


def _exp2_checkpoint_metadata(
    stage: str, *, model: str = "gpt2", layers: int = 12
) -> dict:
    metadata = {
        "experiment": "exp02_capacity_boundary",
        "stage": stage,
        "model": model,
        "T": 2,
        "N": 1000,
        "L": layers,
        "checkpoint_layer_verification": {
            "requested_layers": layers,
            "actual_layers": layers,
        },
    }
    if stage == "cpt":
        metadata.update(
            {
                "experiment_condition": {
                    "table_count": 2,
                    "fact_count": 1000,
                    "layers": layers,
                    "selected_tables": ["continent", "country"],
                },
                "provenance": {
                    "experiment_name": "exp02_capacity_boundary",
                    "T": 2,
                    "N": 1000,
                    "selected_tables": ["continent", "country"],
                    "source_database_sha256": "a" * 64,
                    "database_manifest_sha256": "b" * 64,
                },
            }
        )
    else:
        metadata.update(
            {
                "current_database_condition": {
                    "T": 2,
                    "N": 1000,
                    "layers": layers,
                    "selected_tables": ["continent", "country"],
                    "source_database_sha256": "a" * 64,
                    "source_database_manifest_sha256": "b" * 64,
                },
                "provenance": {
                    "selected_tables": ["continent", "country"],
                    "source_database_sha256": "a" * 64,
                    "source_database_manifest_sha256": "b" * 64,
                },
            }
        )
    return metadata


def test_exp2_result_reservation_never_reuses_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluate_script,
        "exp2_evaluation_result_dir",
        lambda **kwargs: tmp_path
        / f"T{kwargs['table_count']:02d}"
        / f"N{kwargs['fact_count']}"
        / kwargs["split"]
        / kwargs["stage"]
        / kwargs["timestamp"],
    )
    started_at = datetime(2026, 9, 5, 12, 34, 56, tzinfo=timezone.utc)
    first, first_timestamp = evaluate_script._reserve_exp2_result_directory(
        table_count=2,
        fact_count=1000,
        split="validation",
        stage="eval_cpt",
        started_at=started_at,
    )
    second, second_timestamp = evaluate_script._reserve_exp2_result_directory(
        table_count=2,
        fact_count=1000,
        split="validation",
        stage="eval_cpt",
        started_at=started_at,
    )
    assert first_timestamp == "12-34-56_05-09-2026"
    assert second_timestamp == "12-34-57_05-09-2026"
    assert first != second


@pytest.mark.parametrize(
    ("checkpoint_stage", "evaluation_stage"),
    [("cpt", "eval_cpt"), ("target-sft", "eval_sft")],
)
@pytest.mark.parametrize(
    ("model_name", "layers", "checkpoint_config"),
    [
        ("gpt2", 12, {"model_type": "gpt2", "n_layer": 12}),
        ("qwen3-0.6b-base", 28, {"model_type": "qwen3", "num_hidden_layers": 28}),
    ],
)
def test_exp2_evaluator_writes_standard_files_to_authenticated_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp2_config: dict,
    checkpoint_stage: str,
    evaluation_stage: str,
    model_name: str,
    layers: int,
    checkpoint_config: dict,
) -> None:
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    write_json(
        qa_root / "split_manifest.json",
        _exp2_qa_manifest(),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", checkpoint_config)
    write_json(
        checkpoint / "training_metadata.json",
        _exp2_checkpoint_metadata(
            checkpoint_stage, model=model_name, layers=layers
        ),
    )
    output_dir = (
        tmp_path
        / "results"
        / "exp02_capacity_boundary"
        / "t_sweep"
        / "T02"
        / "n_sweep"
        / "N1000"
        / "validation"
        / evaluation_stage
        / "12-34-56_05-09-2026"
    )
    captured: dict[str, object] = {}

    def reserve(**kwargs):
        captured.update(kwargs)
        output_dir.mkdir(parents=True)
        return output_dir, "12-34-56_05-09-2026"

    monkeypatch.setattr(evaluate_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_script, "_reserve_exp2_result_directory", reserve)
    monkeypatch.setattr(
        evaluate_script,
        "load_verified_qa_split",
        lambda *_, **__: (
            [{"id": "example"}],
            {
                "qa_split_manifest_sha256": "c" * 64,
                "qa_manifest_sha256": "d" * 64,
                "input_hashes": {},
            },
        ),
    )
    monkeypatch.setattr(
        evaluate_script,
        "evaluate_with_local_checkpoint",
        lambda *_, **__: ([{"id": "example"}], {"model_identity": model_name}),
    )
    monkeypatch.setattr(
        evaluate_script,
        "compute_evaluation_metrics",
        lambda _: {"overall": {"normalized_exact_match_accuracy": 1.0}},
    )
    evaluate_script._evaluate_exp2(
        argparse.Namespace(
            qa_data_dir=qa_root,
            table_count=None,
            fact_count=None,
            layers=None,
            checkpoint=checkpoint,
            split="validation",
            batch_size=None,
            model=model_name,
            run_name=None,
        ),
        exp2_config,
    )
    assert captured == {
        "table_count": 2,
        "fact_count": 1000,
        "split": "validation",
        "stage": evaluation_stage,
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "evaluation_config.json",
        "metrics.json",
        "predictions.jsonl",
    }
    evaluation_config = __import__("json").loads(
        (output_dir / "evaluation_config.json").read_text(encoding="utf-8")
    )
    assert evaluation_config["checkpoint_stage"] == checkpoint_stage
    assert evaluation_config["evaluation_stage"] == evaluation_stage
    assert evaluation_config["M"] == model_name


def test_exp2_evaluation_rejects_wrong_cli_model_before_inference() -> None:
    checkpoint_metadata = _exp2_checkpoint_metadata(
        "cpt", model="qwen3-0.6b-base", layers=28
    )
    with pytest.raises(ValueError, match="model identity mismatch"):
        evaluate_script._authenticate_exp2_checkpoint_condition(
            checkpoint_metadata=checkpoint_metadata,
            qa_manifest=_exp2_qa_manifest(),
            actual_layers=28,
            expected_model="gpt2",
        )


def _mutate_checkpoint_condition(metadata: dict, field: str) -> None:
    condition = (
        metadata["experiment_condition"]
        if metadata["stage"] == "cpt"
        else metadata["current_database_condition"]
    )
    provenance = metadata["provenance"]
    if field == "selected_tables":
        condition[field] = ["continent", "region"]
        provenance[field] = ["continent", "region"]
    elif field == "T":
        metadata["T"] = 3
        condition["table_count" if metadata["stage"] == "cpt" else "T"] = 3
        if "T" in provenance:
            provenance["T"] = 3
    elif field == "N":
        metadata["N"] = 1005
        condition["fact_count" if metadata["stage"] == "cpt" else "N"] = 1005
        if "N" in provenance:
            provenance["N"] = 1005
    else:
        if metadata["stage"] == "cpt":
            provenance["database_manifest_sha256"] = "e" * 64
        else:
            condition["source_database_manifest_sha256"] = "e" * 64
            provenance["source_database_manifest_sha256"] = "e" * 64


@pytest.mark.parametrize("checkpoint_stage", ["cpt", "target-sft"])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("selected_tables", "selected_tables mismatch"),
        ("T", "checkpoint T mismatch"),
        ("N", "checkpoint N mismatch"),
        ("manifest", "source database manifest hash mismatch"),
    ],
)
def test_exp2_evaluation_rejects_checkpoint_qa_condition_mismatch_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp2_config: dict,
    checkpoint_stage: str,
    field: str,
    message: str,
) -> None:
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    write_json(qa_root / "split_manifest.json", _exp2_qa_manifest())
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 12})
    checkpoint_metadata = deepcopy(_exp2_checkpoint_metadata(checkpoint_stage))
    _mutate_checkpoint_condition(checkpoint_metadata, field)
    write_json(checkpoint / "training_metadata.json", checkpoint_metadata)

    monkeypatch.setattr(
        evaluate_script,
        "load_verified_qa_split",
        lambda *_, **__: pytest.fail("QA records loaded before provenance rejection"),
    )
    monkeypatch.setattr(
        evaluate_script,
        "evaluate_with_local_checkpoint",
        lambda *_, **__: pytest.fail("model inference ran before provenance rejection"),
    )
    with pytest.raises(ValueError, match=message):
        evaluate_script._evaluate_exp2(
            argparse.Namespace(
                qa_data_dir=qa_root,
                table_count=None,
                fact_count=None,
                layers=None,
                checkpoint=checkpoint,
                split="validation",
                batch_size=None,
                model=None,
                run_name=None,
            ),
            exp2_config,
        )


def test_exp2_explicit_architecture_depth_override(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gpt2-l6"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 6})
    resolved, layers = resolve_model_checkpoint("gpt2", source_checkpoint=checkpoint)
    assert resolved == checkpoint.resolve()
    assert layers == 6
    assert verify_checkpoint_layers(resolved, 6)["actual_layers"] == 6
    with pytest.raises(ValueError, match="actually has L6"):
        verify_checkpoint_layers(resolved, 12)


def test_exp2_config_has_no_fixed_t_or_n_sweeps(exp2_config: dict) -> None:
    assert "t_sweep" not in exp2_config["data"]
    assert "n_sweep" not in exp2_config["data"]
    assert "fact_exposure" not in exp2_config["training"]
    assert "dev_split" not in exp2_config["target_sft"]
    assert "early_stopping_patience" not in exp2_config["target_sft"]
    cpt_plan = build_cpt_training_plan(
        exp2_config, table_count=1, fact_count=500, sequence_count=2
    )
    assert cpt_plan["L"] == 12
    assert cpt_plan["independent_record_sequences"] is True
    assert cpt_plan["cross_record_attention"] is False
    assert cpt_plan["record_passes_per_epoch"] == 1
    assert cpt_plan["requested_record_passes"] == cpt_plan["epochs"]
    assert "fact_exposure" not in cpt_plan
    sft_plan = build_target_sft_training_plan(
        exp2_config, table_count=1, fact_count=500, example_count=2
    )
    assert sft_plan["L"] == 12
    assert sft_plan["requested_sft_epochs"] == sft_plan["epochs"]
    assert sft_plan["early_stopping_enabled"] is False
    assert sft_plan["checkpoint_selection"] == "final_requested_epoch"
    assert sft_plan["dev_split_used"] is False
    assert "early_stopping_patience" not in sft_plan


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (
            ["scripts/train.py", "--stage", "cpt"],
            "--training-data-dir is required for Experiment-2 CPT training",
        ),
        (
            ["scripts/train.py", "--stage", "target-sft"],
            "--sft-data-dir is required for target-SFT training",
        ),
        (
            ["scripts/generate_target_sft_qa.py"],
            "--training-data-dir is required for Experiment-2 target-SFT QA generation",
        ),
        (
            [
                "scripts/evaluate.py",
                "--checkpoint",
                "models/base_models/gpt2",
                "--split",
                "validation",
            ],
            "--qa-data-dir is required for Experiment-2 evaluation",
        ),
    ],
)
def test_exp2_explicit_path_guards(command: list[str], message: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            *command,
            "--config",
            "configs/exp02_capacity_boundary.yaml",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr


def _cached_dataset_bundle(
    root: Path,
    name: str,
    *,
    tables: list[str],
    fact_count: int,
    seed: int = 2025,
) -> Path:
    bundle = root / name
    cpt_dir = bundle / "cpt"
    cpt_dir.mkdir(parents=True)
    database = bundle / "database.sqlite"
    database.write_bytes(f"database:{tables}:{fact_count}:{seed}".encode())
    per_chain = facts_per_selected_chain(tables)
    manifest = {
        "experiment_name": "exp02_capacity_boundary",
        "T": len(tables),
        "requested_N": fact_count,
        "selected_tables": tables,
        "facts_per_selected_chain": per_chain,
        "selected_chain_count": fact_count // per_chain,
        "database_sha256": hash_file(database),
        "seed": seed,
    }
    manifest_path = bundle / "manifest.json"
    write_json(manifest_path, manifest)
    (cpt_dir / "book_readable.txt").write_text("book\n", encoding="utf-8")
    write_json(
        cpt_dir / "manifest.json",
        {
            "experiment_name": "exp02_capacity_boundary",
            "T": len(tables),
            "requested_N": fact_count,
            "selected_tables": tables,
            "source_database_sha256": hash_file(database),
            "source_database_manifest_sha256": hash_file(manifest_path),
            "readable_book_sha256": hash_file(cpt_dir / "book_readable.txt"),
            "readable_book_artifact": "book_readable.txt",
            "record_passes_per_cpt_epoch": 1,
            "cpt_examples_independently_tokenized": True,
            "cpt_examples_include_book_context": False,
            "logical_facts_in_book": fact_count,
        },
    )
    return bundle


def _cached_qa_bundle(root: Path, name: str, *, dataset: Path) -> Path:
    condition = exp2_runner._verify_dataset_bundle(
        dataset,
        selected_tables=tuple(
            read_json(dataset / "manifest.json")["selected_tables"]
        ),
        requested_n=read_json(dataset / "manifest.json")["requested_N"],
    )
    qa_root = root / name
    for relative in (
        "validation/manifest.json",
        "test/manifest.json",
        "target_sft/train/manifest.json",
    ):
        path = qa_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"placeholder": True})
    shared = {
        "experiment_name": "exp02_capacity_boundary",
        "T": condition["T"],
        "requested_N": condition["N"],
        "selected_tables": condition["selected_tables"],
        "train_chain_indices": [0],
    }
    write_json(
        qa_root / "split_manifest.json",
        {
            **shared,
            "source_training_data_dir": str(dataset.resolve()),
            "source_database_sha256": condition["manifest"]["database_sha256"],
            "reserved_chain_indices": [0],
            "validation_chain_indices": [1],
            "test_chain_indices": [2],
        },
    )
    train_hash = hash_json_object([0])
    write_json(
        qa_root / "target_sft" / "split_manifest.json",
        {
            **shared,
            "sft_split_method_version": "all_reserved_train_v1",
            "train_chain_count": 1,
            "train_chain_indices_sha256": train_hash,
            "chain_assignment_hashes": {"train": train_hash},
            "target_sft_chain_assignments_sha256": hash_json_object(
                {"train": [0]}
            ),
            "source_evaluation_split_manifest_sha256": hash_file(
                qa_root / "split_manifest.json"
            ),
            "train_manifest_sha256": hash_file(
                qa_root / "target_sft" / "train" / "manifest.json"
            ),
        },
    )
    return qa_root


def test_exp2_runner_finds_oldest_authenticated_dataset_without_running_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "datasets"
    oldest = _cached_dataset_bundle(
        root,
        "T01_N500_continent_20260101_000000_000001",
        tables=["continent"],
        fact_count=500,
    )
    _cached_dataset_bundle(
        root,
        "T01_N500_continent_20260102_000000_000001",
        tables=["continent"],
        fact_count=500,
    )
    monkeypatch.setattr(
        exp2_runner,
        "_run_command",
        lambda *_: pytest.fail("generate_databases.py was called"),
    )
    matches = exp2_runner._find_existing_dataset_bundles(
        selected_tables=("continent",), requested_n=500, seed=2025, root=root
    )
    assert [match["bundle"] for match in matches][0] == oldest.resolve()


def test_exp2_runner_dataset_cache_identity_includes_n_tables_and_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _cached_dataset_bundle(
        root,
        "fixture",
        tables=["continent"],
        fact_count=500,
    )
    assert not exp2_runner._find_existing_dataset_bundles(
        selected_tables=("continent",), requested_n=1000, seed=2025, root=root
    )
    assert not exp2_runner._find_existing_dataset_bundles(
        selected_tables=("continent", "country"),
        requested_n=500,
        seed=2025,
        root=root,
    )
    assert not exp2_runner._find_existing_dataset_bundles(
        selected_tables=("continent",), requested_n=500, seed=7, root=root
    )


def test_exp2_runner_derives_baseline_n_without_training_settings(
    exp2_config: dict,
) -> None:
    changed_training = deepcopy(exp2_config)
    changed_training["training"]["cpt_epochs"] = 999
    changed_training["target_sft"]["epochs"] = 777
    assert exp2_runner._automatic_dataset_n(
        changed_training,
        selected_tables=("continent",),
        requested_n=None,
    ) == 500


def test_exp2_runner_finds_only_qa_bound_to_selected_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "datasets"
    qa_root = tmp_path / "qa"
    selected = _cached_dataset_bundle(
        dataset_root,
        "dataset-a",
        tables=["continent"],
        fact_count=500,
    )
    duplicate = _cached_dataset_bundle(
        dataset_root,
        "dataset-b",
        tables=["continent"],
        fact_count=500,
    )
    _cached_qa_bundle(qa_root, "qa-older-incompatible", dataset=duplicate)
    compatible = _cached_qa_bundle(qa_root, "qa-newer-compatible", dataset=selected)
    condition = exp2_runner._verify_dataset_bundle(
        selected, selected_tables=("continent",), requested_n=500
    )
    monkeypatch.setattr(
        exp2_runner,
        "_run_command",
        lambda *_: pytest.fail("generate_target_sft_qa.py was called"),
    )
    found = exp2_runner._find_existing_qa_bundle(
        dataset_condition=condition,
        selected_tables=("continent",),
        root=qa_root,
    )
    assert found is not None
    assert found["root"] == compatible.resolve()


class _StopExp2Wrapper(Exception):
    pass


def _runner_args(**updates: object) -> argparse.Namespace:
    values = {
        "tables": ["continent"],
        "fact_count": 500,
        "model": "gpt2",
        "layers": 12,
        "base_model": None,
        "config": Path("configs/exp02_capacity_boundary.yaml"),
        "cpt_epochs": None,
        "sft_epochs": None,
        "cpt_batch_size": None,
        "cpt_gradient_accumulation": None,
        "sft_batch_size": None,
        "sft_gradient_accumulation": None,
        "cpt_learning_rate": None,
        "sft_learning_rate": None,
        "dataset_path": None,
        "qa_path": None,
        "cpt_checkpoint": None,
        "sft_checkpoint": None,
        "evaluate_test": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _patch_runner_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exp2_config: dict,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    write_json(
        base_model / "config.json",
        {
            "model_type": "gpt2",
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "n_positions": 1024,
        },
    )
    reused: list[str] = []
    monkeypatch.setattr(exp2_runner, "_parse_args", lambda: args)
    monkeypatch.setattr(exp2_runner, "_infer_resume_inputs", lambda _: None)
    monkeypatch.setattr(
        exp2_runner,
        "resolve_model_checkpoint",
        lambda *_, **__: (base_model, 12),
    )
    monkeypatch.setattr(exp2_runner, "verify_checkpoint_layers", lambda *_, **__: {})
    monkeypatch.setattr(exp2_runner, "_create_run_dir", lambda: run_dir)
    monkeypatch.setattr(
        exp2_runner,
        "_write_resolved_config",
        lambda **_: deepcopy(exp2_config),
    )
    monkeypatch.setattr(exp2_runner, "_write_json_atomic", lambda *_, **__: None)
    monkeypatch.setattr(
        exp2_runner,
        "_reuse_stage",
        lambda **kwargs: reused.append(kwargs["stage"]),
    )
    return reused, base_model


def test_exp2_runner_reuses_compatible_pair_and_skips_both_generators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    args = _runner_args()
    reused, _ = _patch_runner_before_training(
        monkeypatch, tmp_path, exp2_config, args
    )
    older_without_qa = {"bundle": tmp_path / "old", "T": 1, "N": 500}
    paired_dataset = {"bundle": tmp_path / "paired", "T": 1, "N": 500}
    qa = {
        "root": tmp_path / "qa",
        "sft_data_dir": tmp_path / "qa" / "target_sft",
    }
    monkeypatch.setattr(
        exp2_runner,
        "_find_existing_dataset_bundles",
        lambda **_: [older_without_qa, paired_dataset],
    )
    monkeypatch.setattr(
        exp2_runner,
        "_find_existing_qa_bundle",
        lambda **kwargs: qa
        if kwargs["dataset_condition"] is paired_dataset
        else None,
    )
    monkeypatch.setattr(
        exp2_runner,
        "_verify_dataset_bundle",
        lambda path, **_: paired_dataset
        if path == paired_dataset["bundle"]
        else pytest.fail("wrapper did not select the compatible DB+QA pair"),
    )
    monkeypatch.setattr(exp2_runner, "_verify_qa_bundle", lambda *_, **__: qa)

    executed: list[str] = []

    def execute(**kwargs):
        executed.append(kwargs["stage"])
        if kwargs["stage"] == "cpt":
            return tmp_path / "cpt"
        raise _StopExp2Wrapper

    monkeypatch.setattr(exp2_runner, "_execute_stage", execute)
    monkeypatch.setattr(
        exp2_runner,
        "_run_command",
        lambda *_: pytest.fail("a generation command was called"),
    )
    with pytest.raises(_StopExp2Wrapper):
        exp2_runner.main()
    assert args.dataset_path == paired_dataset["bundle"]
    assert args.qa_path == qa["root"]
    assert reused[:2] == ["generate_dataset", "generate_qa"]
    assert "generate_dataset" not in executed
    assert "generate_qa" not in executed


def test_exp2_runner_reuses_db_and_generates_only_missing_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    args = _runner_args()
    reused, _ = _patch_runner_before_training(
        monkeypatch, tmp_path, exp2_config, args
    )
    dataset = {"bundle": tmp_path / "dataset", "T": 1, "N": 500}
    monkeypatch.setattr(
        exp2_runner, "_find_existing_dataset_bundles", lambda **_: [dataset]
    )
    monkeypatch.setattr(exp2_runner, "_find_existing_qa_bundle", lambda **_: None)
    monkeypatch.setattr(exp2_runner, "_verify_dataset_bundle", lambda *_, **__: dataset)
    executed: list[str] = []

    def execute(**kwargs):
        executed.append(kwargs["stage"])
        if kwargs["stage"] == "cpt":
            return tmp_path / "cpt"
        if kwargs["stage"] == "generate_qa":
            return {
                "root": tmp_path / "qa",
                "sft_data_dir": tmp_path / "qa" / "target_sft",
            }
        raise _StopExp2Wrapper

    monkeypatch.setattr(exp2_runner, "_execute_stage", execute)
    with pytest.raises(_StopExp2Wrapper):
        exp2_runner.main()
    assert reused == ["generate_dataset"]
    assert "generate_dataset" not in executed
    assert "generate_qa" in executed


def test_exp2_runner_explicit_dataset_and_qa_still_bypass_cache_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    dataset_path = tmp_path / "explicit-dataset"
    qa_path = tmp_path / "explicit-qa"
    args = _runner_args(dataset_path=dataset_path, qa_path=qa_path)
    reused, _ = _patch_runner_before_training(
        monkeypatch, tmp_path, exp2_config, args
    )
    condition = {"bundle": dataset_path.resolve(), "T": 1, "N": 500}
    qa = {
        "root": qa_path.resolve(),
        "sft_data_dir": qa_path.resolve() / "target_sft",
    }
    monkeypatch.setattr(exp2_runner, "_verify_dataset_bundle", lambda *_, **__: condition)
    monkeypatch.setattr(exp2_runner, "_verify_qa_bundle", lambda *_, **__: qa)
    monkeypatch.setattr(
        exp2_runner,
        "_find_existing_dataset_bundles",
        lambda **_: pytest.fail("dataset cache search replaced explicit path"),
    )
    monkeypatch.setattr(
        exp2_runner,
        "_find_existing_qa_bundle",
        lambda **_: pytest.fail("QA cache search replaced explicit path"),
    )

    def execute(**kwargs):
        if kwargs["stage"] == "cpt":
            return tmp_path / "cpt"
        raise _StopExp2Wrapper

    monkeypatch.setattr(exp2_runner, "_execute_stage", execute)
    with pytest.raises(_StopExp2Wrapper):
        exp2_runner.main()
    assert reused[:2] == ["generate_dataset", "generate_qa"]

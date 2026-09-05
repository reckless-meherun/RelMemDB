import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.evaluate as evaluate_script

from config import load_config
from data.materialize import (
    build_exp2_database_manifest,
    materialize_selected_tables_database,
)
from data.qa import generate_qa_candidates, load_verified_semantic_chains
from data.serialize import database_schema_sha256, serialize_database_cpt
from data.world import (
    SEMANTIC_ENTITY_SPECS,
    build_world_for_chain_count,
    facts_per_selected_chain,
    validate_exp2_fact_count,
    validate_selected_tables,
)
from experiment import resolve_model_checkpoint, verify_checkpoint_layers
from training.cpt import build_cpt_training_plan
from training.target_sft import build_target_sft_training_plan
from utils.hashing import hash_file
from utils.io import read_text, write_json
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
        cpt / "train.txt",
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


def test_exp2_determinism_and_nested_prefixes(exp2_config: dict) -> None:
    small = build_world_for_chain_count(exp2_config, 7)
    repeated = build_world_for_chain_count(exp2_config, 7)
    large = build_world_for_chain_count(exp2_config, 12)
    assert small == repeated
    assert large["chains"][:7] == small["chains"]


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


def test_exp2_evaluator_writes_standard_files_to_authenticated_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp2_config: dict
) -> None:
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    write_json(
        qa_root / "split_manifest.json",
        {
            "experiment_name": "exp02_capacity_boundary",
            "T": 2,
            "requested_N": 1000,
            "selected_tables": ["continent", "country"],
            "source_database_sha256": "a" * 64,
            "source_database_manifest_sha256": "b" * 64,
        },
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 12})
    write_json(
        checkpoint / "training_metadata.json",
        {
            "experiment": "exp02_capacity_boundary",
            "stage": "cpt",
            "model": "gpt2",
            "provenance": {"source_database_sha256": "a" * 64},
        },
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
        / "eval_cpt"
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
        lambda *_, **__: ([{"id": "example"}], {"model_identity": "gpt2"}),
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
            run_name=None,
        ),
        exp2_config,
    )
    assert captured == {
        "table_count": 2,
        "fact_count": 1000,
        "split": "validation",
        "stage": "eval_cpt",
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "evaluation_config.json",
        "metrics.json",
        "predictions.jsonl",
    }
    evaluation_config = __import__("json").loads(
        (output_dir / "evaluation_config.json").read_text(encoding="utf-8")
    )
    assert evaluation_config["checkpoint_stage"] == "cpt"
    assert evaluation_config["evaluation_stage"] == "eval_cpt"


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
    assert build_cpt_training_plan(
        exp2_config, table_count=1, fact_count=500, sequence_count=2
    )["L"] == 12
    assert build_target_sft_training_plan(
        exp2_config, table_count=1, fact_count=500, example_count=2
    )["L"] == 12


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

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

from config import load_config
from data.materialize import (
    build_exp2_database_manifest,
    materialize_selected_tables_database,
)
from data.qa import generate_qa_candidates, load_verified_semantic_chains
from data.serialize import database_schema_sha256, serialize_database_cpt
from data.world import (
    NATURAL_IDENTIFIER_FIELDS,
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


def _exp2_checkpoint_metadata(stage: str) -> dict:
    metadata = {
        "experiment": "exp02_capacity_boundary",
        "stage": stage,
        "model": "gpt2",
        "T": 2,
        "N": 1000,
        "L": 12,
        "checkpoint_layer_verification": {
            "requested_layers": 12,
            "actual_layers": 12,
        },
    }
    if stage == "cpt":
        metadata.update(
            {
                "experiment_condition": {
                    "table_count": 2,
                    "fact_count": 1000,
                    "layers": 12,
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
                    "layers": 12,
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
def test_exp2_evaluator_writes_standard_files_to_authenticated_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp2_config: dict,
    checkpoint_stage: str,
    evaluation_stage: str,
) -> None:
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    write_json(
        qa_root / "split_manifest.json",
        _exp2_qa_manifest(),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 12})
    write_json(
        checkpoint / "training_metadata.json",
        _exp2_checkpoint_metadata(checkpoint_stage),
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
    (cpt_dir / "train.txt").write_text("train\n", encoding="utf-8")
    write_json(
        cpt_dir / "manifest.json",
        {
            "experiment_name": "exp02_capacity_boundary",
            "T": len(tables),
            "requested_N": fact_count,
            "selected_tables": tables,
            "source_database_sha256": hash_file(database),
            "source_database_manifest_sha256": hash_file(manifest_path),
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
        "target_sft/dev/manifest.json",
    ):
        path = qa_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"placeholder": True})
    shared = {
        "experiment_name": "exp02_capacity_boundary",
        "T": condition["T"],
        "requested_N": condition["N"],
        "selected_tables": condition["selected_tables"],
    }
    write_json(
        qa_root / "split_manifest.json",
        {
            **shared,
            "source_training_data_dir": str(dataset.resolve()),
            "source_database_sha256": condition["manifest"]["database_sha256"],
        },
    )
    write_json(qa_root / "target_sft" / "split_manifest.json", shared)
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

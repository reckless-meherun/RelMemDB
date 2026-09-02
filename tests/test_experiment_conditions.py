from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.evaluate as evaluate_script
import scripts.generate_databases as database_script
import scripts.run_experiment as experiment_driver
import scripts.train as train_script
from config import load_config
from data.materialize import build_database_manifest, materialize_database
from data.qa_reference import (
    QAReferenceCompatibilityError,
    verify_qa_reference_compatibility,
)
from experiment import ExperimentCondition, verify_checkpoint_layers
from utils.hashing import hash_file
from utils.io import read_json, write_json
from utils.paths import (
    EXP01_GENERATED_DATABASES_DIR,
    EXP01_QA_DIR,
    database_condition_dir,
    evaluation_result_dir,
    qa_reference_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_QA = EXP01_QA_DIR / "t_sweep_N10K" / "T12"
MASTER_WORLD = EXP01_GENERATED_DATABASES_DIR / "master_world" / "world.json"
PROTECTED_QA_HASHES = {
    "split_manifest.json": "08b2dbeca48e2d42cae466062699d87ccec9bb45218bf3eea280ccf5423db7df",
    "target_sft/split_manifest.json": "cb961da2f481ba537df748ca7bf48bc3cb778943715e0fcc0733d3ec241a23ab",
    "target_sft/train/H0.jsonl": "1e2984935192ddd46a34c8cdef943aa4c9dc049dde252c90db6f1904c62181a9",
    "target_sft/train/H1.jsonl": "b84651889378a355de5ee416190f4d059671f292549f29008c27e69828d0b4d5",
    "target_sft/train/H2.jsonl": "00cc3d2eb8f8e4b6747c53bd7b28bbc2e9e63db94c09a0c565e547411ad47af7",
    "target_sft/train/H3.jsonl": "4d1df742d711f977fb9bcbc37ecb4982a8939ee24068c7eaba8a10435525debc",
    "target_sft/train/manifest.json": "cf14bd5c3815136c967f6869402d45085b417a2aad23a9e0d40f158767ab3ebc",
    "target_sft/dev/H0.jsonl": "95f3163dfb185f0a1f54ba326bca612a6f1ad390c2d9d7f76d8940098e44d5b4",
    "target_sft/dev/H1.jsonl": "9a43aa497f0a1bab0cf5a509b547159812849d5d8af0881229a10786bd4a27b7",
    "target_sft/dev/H2.jsonl": "00de4420968ae6901256c590057c001eda7d845fa3a089937fe3a6078445a7b7",
    "target_sft/dev/H3.jsonl": "61b17d306f5906a7d853f13b3b1e0db9824c79bee0bc930be49249be63e05936",
    "target_sft/dev/manifest.json": "bbb79a0c22967382f2548d9fc1c15d7142c1bf8664eeb12d0d7006a1dfe26aa7",
    "validation/H0.jsonl": "890b072911dfdfd8dd36f0a67b72691ace493b913e7fa9cd4ee360135ba863e3",
    "validation/H1.jsonl": "73d6002ad5bea18c332806f4c82d8820c968b7d85b90e00f9556341b9af1cc83",
    "validation/H2.jsonl": "ed07caaf46d260c4e2084763a73bf00e03eb62180583bcb3105234719ec62550",
    "validation/H3.jsonl": "fe403f194797af969e4b39a92dd23dd00e82e92895b97cd9d34826b2665a6a85",
    "validation/manifest.json": "dea93834ffa9bc3d87330e75d0c2dc2a7fdc92ef2d2794947c72bc3ffe883ad3",
    "test/H0.jsonl": "a1208001da2fcddd718de5f0e66b3d563516f4a23ecb47c35c772871079d7d94",
    "test/H1.jsonl": "5bf88dd5768faa354cd52a8c95049044a0dcf9ac7ef433112116ef418b1465b1",
    "test/H2.jsonl": "bd3b762a08789cc156b1dcbda480675963a5cab70db1efffbc892e3fa22bbd62",
    "test/H3.jsonl": "c690c9d990ff6b33c76c6dc95806e8e71f223f66c89ca52646ce6fbf7240aef3",
    "test/manifest.json": "182ada4d4ea0ca4ffc146dbe6d8138df3e3fc1f426fa20960adb7aa7eb36c878",
}


def _materialize_fixture(
    root: Path, world: dict, *, table_count: int, fact_count: int, name: str
) -> tuple[Path, Path]:
    output_dir = root / name
    database_path = output_dir / "database.sqlite"
    manifest_path = output_dir / "manifest.json"
    materialization = materialize_database(
        world,
        table_count=table_count,
        logical_fact_count=fact_count,
        output_path=database_path,
    )
    manifest = build_database_manifest(
        load_config(),
        materialization,
        sweep="synthetic_test_fixture",
        master_world_sha256="fixture",
        configuration_sha256="fixture",
        database_sha256=hash_file(database_path),
    )
    write_json(manifest_path, manifest)
    return database_path, manifest_path


@pytest.fixture(scope="module")
def semantic_databases(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("semantic_conditions")
    world = read_json(MASTER_WORLD)
    artifacts = {
        (4, 10_000): _materialize_fixture(
            root, world, table_count=4, fact_count=10_000, name="t4_n10k"
        ),
        (8, 10_000): _materialize_fixture(
            root, world, table_count=8, fact_count=10_000, name="t8_n10k"
        ),
        (12, 10_000): _materialize_fixture(
            root, world, table_count=12, fact_count=10_000, name="t12_n10k"
        ),
        (8, 20_000): _materialize_fixture(
            root, world, table_count=8, fact_count=20_000, name="t8_n20k"
        ),
        (8, 5_000): _materialize_fixture(
            root, world, table_count=8, fact_count=5_000, name="t8_n5k"
        ),
    }
    mismatched_world = deepcopy(world)
    mismatched_world["chains"][0]["entities"][0]["attributes"][0]["value"] = (
        "Deliberately Mismatched Continent"
    )
    artifacts["mismatch"] = _materialize_fixture(
        root,
        mismatched_world,
        table_count=4,
        fact_count=10_000,
        name="semantic_mismatch",
    )
    return artifacts


def test_canonical_qa_reference_is_invariant_across_conditions() -> None:
    config = load_config()
    paths = {
        (table_count, fact_count, layers): qa_reference_dir(config)
        for table_count, fact_count, layers in (
            (4, 10_000, 12),
            (8, 10_000, 12),
            (12, 10_000, 6),
            (12, 10_000, 9),
            (8, 20_000, 12),
        )
    }
    assert set(paths.values()) == {CANONICAL_QA}
    for split in ("target_sft/train", "target_sft/dev", "validation", "test"):
        expected = {
            hop: hash_file(CANONICAL_QA / split / f"H{hop}.jsonl") for hop in range(4)
        }
        assert all(
            {hop: hash_file(path / split / f"H{hop}.jsonl") for hop in range(4)}
            == expected
            for path in paths.values()
        )


@pytest.mark.parametrize("table_count", [4, 8, 12])
def test_t_sweep_semantically_matches_fixed_qa(
    semantic_databases, table_count: int
) -> None:
    database, manifest = semantic_databases[(table_count, 10_000)]
    result = verify_qa_reference_compatibility(
        load_config(),
        table_count,
        10_000,
        current_database_path=database,
        current_database_manifest_path=manifest,
    )
    assert result["compatible"] is True
    assert result["required_chain_count"] == 250


def test_nested_n20k_semantically_matches_fixed_qa(semantic_databases) -> None:
    database, manifest = semantic_databases[(8, 20_000)]
    result = verify_qa_reference_compatibility(
        load_config(),
        8,
        20_000,
        current_database_path=database,
        current_database_manifest_path=manifest,
    )
    assert result["compatible"] is True
    assert result["current_database_condition"]["chain_count"] == 500
    assert (
        result["current_database_condition"]["database_sha256"]
        != result["qa_reference"]["source_database_sha256"]
    )
    assert result["physical_database_sha_equality_required"] is False


def test_compatible_layouts_share_one_semantic_fingerprint(semantic_databases) -> None:
    fingerprints = set()
    for table_count, fact_count in (
        (4, 10_000),
        (8, 10_000),
        (12, 10_000),
        (8, 20_000),
    ):
        database, manifest = semantic_databases[(table_count, fact_count)]
        result = verify_qa_reference_compatibility(
            load_config(),
            table_count,
            fact_count,
            current_database_path=database,
            current_database_manifest_path=manifest,
        )
        fingerprints.add(result["semantic_compatibility_fingerprint"])
    assert len(fingerprints) == 1


def test_n5k_is_rejected_for_missing_required_chains(semantic_databases) -> None:
    database, manifest = semantic_databases[(8, 5_000)]
    with pytest.raises(
        QAReferenceCompatibilityError,
        match="does not contain all chains required by the canonical QA benchmark",
    ):
        verify_qa_reference_compatibility(
            load_config(),
            8,
            5_000,
            current_database_path=database,
            current_database_manifest_path=manifest,
        )


def test_semantic_mismatch_is_rejected(semantic_databases) -> None:
    database, manifest = semantic_databases["mismatch"]
    with pytest.raises(QAReferenceCompatibilityError, match="semantic content"):
        verify_qa_reference_compatibility(
            load_config(),
            4,
            10_000,
            current_database_path=database,
            current_database_manifest_path=manifest,
        )


def test_condition_validation_and_generic_paths() -> None:
    config = load_config()
    condition = ExperimentCondition.from_config(
        config, table_count=4, fact_count=20_000, layers=9
    )
    assert condition.label == "T4_N20K_L9"
    assert database_condition_dir(4, 20_000) == (
        EXP01_GENERATED_DATABASES_DIR / "conditions" / "T4_N20K"
    )
    assert database_script._condition_destination(config, 4, 20_000) == (
        "condition",
        EXP01_GENERATED_DATABASES_DIR / "conditions" / "T4_N20K",
    )
    with pytest.raises(ValueError, match="between 1 and 12"):
        ExperimentCondition.from_config(
            config, table_count=13, fact_count=10_000, layers=12
        )
    with pytest.raises(ValueError, match="divisible"):
        ExperimentCondition.from_config(
            config, table_count=4, fact_count=10_001, layers=12
        )
    with pytest.raises(ValueError, match="master world"):
        ExperimentCondition.from_config(
            config, table_count=4, fact_count=20_040, layers=12
        )


def test_checkpoint_layer_count_must_match(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "gpt2", "n_layer": 6})
    assert verify_checkpoint_layers(checkpoint, 6)["actual_layers"] == 6
    with pytest.raises(ValueError, match="requested L9.*actually has L6"):
        verify_checkpoint_layers(checkpoint, 9)


def test_condition_aware_artifact_names_do_not_collide() -> None:
    config = load_config()
    conditions = [
        ExperimentCondition.from_config(config, table_count=t, fact_count=n, layers=l)
        for t, n, l in (
            (4, 10_000, 12),
            (8, 10_000, 12),
            (4, 20_000, 12),
            (4, 10_000, 6),
        )
    ]
    checkpoint_names = {
        train_script.build_cpt_paths(
            config,
            table_count=condition.table_count,
            fact_count=condition.fact_count,
            layers=condition.layers,
        )["output_checkpoint"]
        for condition in conditions
    }
    result_paths = {
        evaluation_result_dir(
            condition.table_count,
            condition.fact_count,
            condition.layers,
            "validation",
            "post_cpt",
        )
        for condition in conditions
    }
    assert len(checkpoint_names) == len(conditions)
    assert len(result_paths) == len(conditions)


def test_through_validation_never_contains_test() -> None:
    config = load_config()
    condition = ExperimentCondition.from_config(
        config, table_count=4, fact_count=10_000, layers=12
    )
    commands = experiment_driver.build_stage_commands(
        config_path=PROJECT_ROOT / "configs/exp01_first_feasibility.yaml",
        config=config,
        condition=condition,
        stage="through-validation",
        source_checkpoint=Path("/real/matching/l12/source"),
    )
    assert experiment_driver.stage_sequence("through-validation") == (
        "prepare",
        "cpt",
        "eval-cpt",
        "sft",
        "eval-sft",
    )
    assert all("test" not in command for command in commands)
    assert [
        command[command.index("--split") + 1]
        for command in commands
        if "--split" in command
    ] == ["validation", "validation"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_loads_only_the_canonical_qa_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"n_layer": 12})
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        evaluate_script,
        "_parse_args",
        lambda: argparse.Namespace(
            table_count=4,
            fact_count=10_000,
            layers=12,
            split=split,
            checkpoint=checkpoint,
            run_name="fixture",
            batch_size=None,
            config=PROJECT_ROOT / "configs/exp01_first_feasibility.yaml",
        ),
    )

    def fake_compatibility(config, table_count, fact_count, **kwargs):
        captured["compatibility"] = (table_count, fact_count, kwargs["reference_dir"])
        return {
            "current_database_condition": {
                "table_count": table_count,
                "fact_count": fact_count,
            },
            "qa_reference": {"table_count": 12, "fact_count": 10_000},
            "semantic_compatibility_fingerprint": "fixture",
        }

    def fake_loader(path, **kwargs):
        captured["qa_load"] = (path, kwargs)
        return (
            [{"id": "fixture"}],
            {
                "qa_manifest_sha256": "manifest",
                "qa_split_manifest_sha256": "split",
                "input_hashes": {},
            },
        )

    monkeypatch.setattr(
        evaluate_script, "verify_qa_reference_compatibility", fake_compatibility
    )
    monkeypatch.setattr(evaluate_script, "load_verified_qa_split", fake_loader)

    def fake_result_dir(*args):
        captured["result_args"] = args
        return tmp_path / "result"

    monkeypatch.setattr(evaluate_script, "evaluation_result_dir", fake_result_dir)
    monkeypatch.setattr(evaluate_script, "prepare_result_directory", lambda _: None)
    monkeypatch.setattr(
        evaluate_script,
        "evaluate_with_local_checkpoint",
        lambda *_, **__: ([{"id": "fixture"}], {"model_identity": "fixture"}),
    )
    monkeypatch.setattr(
        evaluate_script,
        "compute_evaluation_metrics",
        lambda _: {"overall": {"normalized_exact_match_accuracy": 1.0}},
    )
    monkeypatch.setattr(evaluate_script, "write_json", lambda *_, **__: None)
    monkeypatch.setattr(evaluate_script, "write_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(evaluate_script, "PROJECT_ROOT", tmp_path)
    evaluate_script.main()
    loaded_path, loader_kwargs = captured["qa_load"]
    assert loaded_path == CANONICAL_QA / split
    assert loader_kwargs == {
        "split": split,
        "expected_table_count": 12,
        "expected_fact_count": 10_000,
    }
    assert captured["compatibility"] == (4, 10_000, CANONICAL_QA)
    assert captured["result_args"] == (4, 10_000, 12, split, "fixture")


def test_canonical_qa_bytes_are_unchanged() -> None:
    assert {
        relative: hash_file(CANONICAL_QA / relative) for relative in PROTECTED_QA_HASHES
    } == PROTECTED_QA_HASHES

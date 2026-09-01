from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from config import load_config, validate_config
from data.materialize import build_database_manifest, materialize_database
from data.qa import (
    generate_condition_qa,
    generate_qa_records,
    load_verified_logical_chains,
)
from data.serialize import serialize_database_cpt
from data.world import build_master_world
from utils.hashing import hash_file
from utils.io import write_json
from utils.paths import (
    EXP01_QA_DIR,
    PROJECT_ROOT,
    n_sweep_database_dir,
    n_sweep_qa_dir,
    t_sweep_database_dir,
    t_sweep_qa_dir,
)


@pytest.fixture(scope="module")
def default_config() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="module")
def small_config(default_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(default_config)
    data = config["data"]
    data["reuse_t8_n10k"] = False
    data["t_sweep"]["fact_count"] = 400
    data["n_sweep"]["fact_counts"] = [200, 400]
    data["optional_n40k"]["fact_count"] = 800
    validate_config(config)
    return config


@pytest.fixture(scope="module")
def small_world(small_config: dict[str, Any]) -> dict[str, Any]:
    return build_master_world(small_config)


def _build_serialized_test_condition(
    tmp_path: Path,
    config: dict[str, Any],
    world: dict[str, Any],
    *,
    table_count: int = 4,
    logical_fact_count: int = 200,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    database_path = tmp_path / "database.sqlite"
    database_manifest_path = tmp_path / "database_manifest.json"
    train_text_path = tmp_path / "train.txt"
    cpt_manifest_path = tmp_path / "cpt_manifest.json"
    materialization = materialize_database(
        world,
        table_count=table_count,
        logical_fact_count=logical_fact_count,
        output_path=database_path,
    )
    database_manifest = build_database_manifest(
        config,
        materialization,
        sweep="test_sweep",
        master_world_sha256="master-hash",
        configuration_sha256="configuration-hash",
        database_sha256=hash_file(database_path),
    )
    write_json(database_manifest_path, database_manifest)
    cpt_manifest = serialize_database_cpt(
        config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        train_text_path=train_text_path,
    )
    write_json(cpt_manifest_path, cpt_manifest)
    return (
        database_path,
        database_manifest_path,
        train_text_path,
        cpt_manifest_path,
        cpt_manifest,
    )


def _load_condition_chains(
    condition_dir: Path, table_count: int
) -> list[dict[str, Any]]:
    chains, _ = load_verified_logical_chains(
        condition_dir / "database.sqlite",
        condition_dir / "manifest.json",
        expected_table_count=table_count,
    )
    return chains


def _qa_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"{json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        for record in records
    ).encode("utf-8")


def test_physical_database_reconstruction_across_t() -> None:
    reconstructed = [
        _load_condition_chains(t_sweep_database_dir(table_count), table_count)
        for table_count in (4, 8, 12)
    ]
    assert reconstructed[0] == reconstructed[1] == reconstructed[2]

    for chain in reconstructed[0]:
        assert len(chain["entities"]) == 12
        assert sum(
            len(entity["attributes"]) for entity in chain["entities"]
        ) == 17
        assert len(chain["previous_entity_ids"][1:]) == 11
        assert all(chain["previous_entity_ids"][1:])


def test_qa_counts_per_chain() -> None:
    chain = _load_condition_chains(t_sweep_database_dir(4), 4)[0]
    records = generate_qa_records([chain])

    assert sum(record["type"] == "attribute" for record in records["H0"]) == 17
    assert sum(record["type"] == "relation" for record in records["H0"]) == 11
    assert len(records["H0"]) == 28
    assert len(records["H1"]) == 11
    assert len(records["H2"]) == 10
    assert len(records["H3"]) == 9


def test_qa_gold_answers_and_support_paths() -> None:
    chains = _load_condition_chains(t_sweep_database_dir(4), 4)[:2]
    records = generate_qa_records(chains)
    entity_locations = {
        entity["entity_id"]: (chain, position)
        for chain in chains
        for position, entity in enumerate(chain["entities"])
    }
    h0_by_id = {record["id"]: record for record in records["H0"]}
    attribute_ids: dict[tuple[str, str], str] = {}
    relation_ids: dict[str, str] = {}

    for chain in chains:
        for position, entity in enumerate(chain["entities"]):
            entity_id = entity["entity_id"]
            for slot, expected_answer in entity["attributes"].items():
                question = f"What is {slot} of entity {entity_id}?"
                matches = [
                    record
                    for record in records["H0"]
                    if record["question"] == question
                ]
                assert len(matches) == 1
                assert matches[0]["answer"] == expected_answer
                attribute_ids[(entity_id, slot)] = matches[0]["id"]
            if position > 0:
                question = (
                    "Which entity is immediately previous to entity "
                    f"{entity_id}?"
                )
                matches = [
                    record
                    for record in records["H0"]
                    if record["question"] == question
                ]
                assert len(matches) == 1
                assert matches[0]["answer"] == chain["previous_entity_ids"][position]
                relation_ids[entity_id] = matches[0]["id"]

    for hop in (1, 2, 3):
        for record in records[f"H{hop}"]:
            prefix = "Starting from entity "
            source_entity_id = record["question"][len(prefix) :].split(",", 1)[0]
            chain, source_position = entity_locations[source_entity_id]
            target_entity = chain["entities"][source_position - hop]
            expected_supports = [
                relation_ids[chain["entities"][position]["entity_id"]]
                for position in range(source_position, source_position - hop, -1)
            ]
            expected_supports.append(
                attribute_ids[(target_entity["entity_id"], "attribute_0")]
            )
            assert record["answer"] == target_entity["attributes"]["attribute_0"]
            assert record["support_fact_ids"] == expected_supports
            assert len(record["support_fact_ids"]) == hop + 1
            assert all(support_id in h0_by_id for support_id in expected_supports)


def test_qa_ids_are_unique_stable_and_t_independent() -> None:
    record_sets = []
    for table_count in (4, 8, 12):
        chains = _load_condition_chains(
            t_sweep_database_dir(table_count), table_count
        )[:3]
        records = generate_qa_records(chains)
        all_ids = [
            record["id"]
            for hop_records in records.values()
            for record in hop_records
        ]
        assert len(all_ids) == len(set(all_ids))
        record_sets.append(records)
    assert record_sets[0] == record_sets[1] == record_sets[2]


def test_qa_t_sweep_jsonl_bytes_are_identical() -> None:
    records_by_t = [
        generate_qa_records(
            _load_condition_chains(t_sweep_database_dir(table_count), table_count)
        )
        for table_count in (4, 8, 12)
    ]
    for hop in ("H0", "H1", "H2", "H3"):
        serialized = [
            _qa_jsonl_bytes(records[hop]) for records in records_by_t
        ]
        assert serialized[0] == serialized[1] == serialized[2]


def test_qa_n_sweep_prefix_nesting() -> None:
    condition_records = [
        generate_qa_records(
            _load_condition_chains(condition_dir, 8)
        )
        for condition_dir in (
            n_sweep_database_dir(5_000),
            t_sweep_database_dir(8),
            n_sweep_database_dir(20_000),
        )
    ]
    for hop in ("H0", "H1", "H2", "H3"):
        n5k, n10k, n20k = [records[hop] for records in condition_records]
        assert len(n5k) < len(n10k) < len(n20k)
        assert n5k == n10k[: len(n5k)]
        assert n10k == n20k[: len(n10k)]


def test_qa_record_schema_and_no_physical_leakage() -> None:
    chain = _load_condition_chains(t_sweep_database_dir(4), 4)[0]
    records = generate_qa_records([chain])
    expected_h0_fields = {"id", "hop", "type", "question", "answer"}
    expected_relational_fields = expected_h0_fields | {"support_fact_ids"}

    for record in records["H0"]:
        assert set(record) == expected_h0_fields
    for hop in (1, 2, 3):
        for record in records[f"H{hop}"]:
            assert set(record) == expected_relational_fields

    for hop_records in records.values():
        for record in hop_records:
            lowercase_question = record["question"].lower()
            for forbidden in (
                "table_",
                "p00_",
                "rowid",
                "sqlite",
                " sql ",
                "t4",
                "t8",
                "t12",
                "n5k",
                "n10k",
                "n20k",
                str(PROJECT_ROOT).lower(),
            ):
                assert forbidden not in lowercase_question


def test_qa_condition_generation_is_deterministic(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    database_path, database_manifest_path, _, _, _ = (
        _build_serialized_test_condition(tmp_path, default_config, small_world)
    )
    output_paths = {
        hop: tmp_path / f"{hop}.jsonl" for hop in ("H0", "H1", "H2", "H3")
    }
    qa_manifest_path = tmp_path / "qa_manifest.json"
    first_manifest = generate_condition_qa(
        default_config,
        database_path,
        database_manifest_path,
        output_paths,
        expected_table_count=4,
        expected_logical_fact_count=200,
    )
    write_json(qa_manifest_path, first_manifest)
    first_bytes = {
        path: path.read_bytes()
        for path in (*output_paths.values(), qa_manifest_path)
    }

    second_manifest = generate_condition_qa(
        default_config,
        database_path,
        database_manifest_path,
        output_paths,
        expected_table_count=4,
        expected_logical_fact_count=200,
    )
    write_json(qa_manifest_path, second_manifest)
    assert second_manifest == first_manifest
    assert all(path.read_bytes() == content for path, content in first_bytes.items())
    assert first_manifest["h0_definition"] == "attribute_and_direct_relation_facts"
    assert first_manifest["h0_attribute_count"] == 85
    assert first_manifest["h0_relation_count"] == 55
    assert first_manifest["H0_count"] == 140
    assert first_manifest["H1_count"] == 55
    assert first_manifest["H2_count"] == 50
    assert first_manifest["H3_count"] == 45
    assert first_manifest["total_QA_count"] == 290
    for hop in ("H0", "H1", "H2", "H3"):
        assert first_manifest[f"{hop}_sha256"] == hash_file(output_paths[hop])


def test_no_duplicate_n10k_qa_directory() -> None:
    assert n_sweep_qa_dir(10_000) == t_sweep_qa_dir(8)
    assert not (n_sweep_qa_dir(5_000).parent / "N10K").exists()


def test_optional_n40k_qa_placeholders_remain_empty() -> None:
    qa_dir = n_sweep_qa_dir(40_000)
    for filename in ("H0.jsonl", "H1.jsonl", "H2.jsonl", "H3.jsonl", "manifest.json"):
        assert (qa_dir / filename).stat().st_size == 0


def test_preflight_qa_baseline_and_skill_training_populated() -> None:
    preflight_dir = EXP01_QA_DIR / "preflight_relational_qa"
    baseline_dir = preflight_dir / "baseline"
    for filename in ("H1.jsonl", "H2.jsonl", "H3.jsonl", "manifest.json"):
        assert (baseline_dir / filename).stat().st_size > 0

    skill_training_dir = preflight_dir / "skill_training"
    for filename in ("train.jsonl", "val.jsonl", "manifest.json"):
        assert (skill_training_dir / filename).stat().st_size > 0

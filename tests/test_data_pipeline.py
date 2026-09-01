from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from config import load_config
from data.materialize import build_database_manifest, materialize_database
from data.qa import (
    HOP_NAMES,
    RAW_ENTITY_IDENTIFIER,
    answer_is_in_question,
    assign_chain_splits,
    filter_qa_candidates,
    generate_condition_qa,
    generate_qa_candidates,
    load_verified_semantic_chains,
)
from data.world import (
    NATURAL_IDENTIFIER_FIELDS,
    SEMANTIC_ENTITY_SPECS,
    build_master_world,
)
from utils.hashing import hash_file
from utils.io import read_json, read_jsonl, write_json
from utils.paths import EXP01_QA_DIR, PROJECT_ROOT


@pytest.fixture(scope="module")
def default_config() -> dict[str, Any]:
    return load_config()


def _alphabetic_index(index: int) -> str:
    value = index + 1
    letters = []
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _world_with_nonleaky_test_anchors(config: dict[str, Any]) -> dict[str, Any]:
    world = deepcopy(build_master_world(config))
    for chain_index, chain in enumerate(world["chains"]):
        label = _alphabetic_index(chain_index)
        for entity in chain["entities"]:
            field = NATURAL_IDENTIFIER_FIELDS.get(entity["entity_type"])
            if field is None:
                continue
            for attribute in entity["attributes"]:
                if attribute["name"] == field:
                    entity_label = entity["entity_type"].replace("_", " ").title()
                    attribute["value"] = f"{entity_label} Meridian {label}"
                    break
    return world


@pytest.fixture(scope="module")
def qa_condition(
    tmp_path_factory: pytest.TempPathFactory,
    default_config: dict[str, Any],
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("semantic_qa")
    database_path = root / "database.sqlite"
    database_manifest_path = root / "manifest.json"
    world = _world_with_nonleaky_test_anchors(default_config)
    materialization = materialize_database(world, 12, 10_000, database_path)
    database_manifest = build_database_manifest(
        default_config,
        materialization,
        sweep="temporary_test",
        master_world_sha256="temporary-master-world",
        configuration_sha256="temporary-config",
        database_sha256=hash_file(database_path),
    )
    write_json(database_manifest_path, database_manifest)
    chains, _ = load_verified_semantic_chains(
        database_path,
        database_manifest_path,
        expected_table_count=12,
        expected_logical_fact_count=10_000,
    )
    output_dir = root / "qa"
    source_hash_before = hash_file(database_path)
    result = generate_condition_qa(
        default_config,
        database_path,
        database_manifest_path,
        output_dir,
        expected_table_count=12,
        expected_logical_fact_count=10_000,
    )
    assert hash_file(database_path) == source_hash_before
    return {
        "root": root,
        "database_path": database_path,
        "database_manifest_path": database_manifest_path,
        "chains": chains,
        "output_dir": output_dir,
        "result": result,
    }


def _records_for_split(output_dir: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    return {
        hop_name: read_jsonl(output_dir / split / f"{hop_name}.jsonl")
        for hop_name in HOP_NAMES
    }


def test_semantic_academic_schema_is_reconstructed_from_temporary_sqlite(
    qa_condition: dict[str, Any],
) -> None:
    chains = qa_condition["chains"]
    assert len(chains) == 250
    for chain_index, chain in enumerate(chains):
        assert chain["chain_index"] == chain_index
        assert [entity["entity_type"] for entity in chain["entities"]] == [
            spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS
        ]
        assert all(entity["natural_anchor"] for entity in chain["entities"])
        assert all(
            chain["entities"][position]["relation_target_id"]
            == chain["entities"][position - 1]["entity_id"]
            for position in range(1, 12)
        )


def test_deterministic_150_50_50_chain_split_is_disjoint_and_nested_friendly() -> None:
    assignments = assign_chain_splits(250)
    assert {name: len(indices) for name, indices in assignments.items()} == {
        "reserved": 150,
        "validation": 50,
        "test": 50,
    }
    assert assignments["reserved"][:6] == [0, 1, 2, 5, 6, 7]
    assert assignments["validation"][:3] == [3, 8, 13]
    assert assignments["test"][:3] == [4, 9, 14]
    assert not set(assignments["validation"]) & set(assignments["test"])
    assert assign_chain_splits(125) == {
        split: [index for index in indices if index < 125]
        for split, indices in assignments.items()
    }


@pytest.mark.parametrize("split", ["validation", "test"])
def test_candidate_counts_and_retained_record_schema(
    qa_condition: dict[str, Any], split: str
) -> None:
    manifest = qa_condition["result"][f"{split}_manifest"]
    assert {
        hop: manifest["counts"][hop]["candidate_count"] for hop in HOP_NAMES
    } == {"H0": 1300, "H1": 450, "H2": 350, "H3": 300}
    assert manifest["candidate_total"] == 2400
    assert manifest["zero_context"] is True
    records = _records_for_split(qa_condition["output_dir"], split)
    expected_h0_fields = {
        "id",
        "split",
        "hop",
        "question",
        "gold_answer",
        "fact_type",
        "source_entity_type",
        "target_entity_type",
        "target_field",
    }
    expected_relational_fields = expected_h0_fields - {"fact_type"} | {
        "support_fact_ids"
    }
    assert all(set(record) == expected_h0_fields for record in records["H0"])
    for hop in (1, 2, 3):
        assert all(set(record) == expected_relational_fields for record in records[f"H{hop}"])
        assert all(record["hop"] == hop for record in records[f"H{hop}"])


def test_h0_candidate_fact_selection_and_non_tautological_exclusions(
    qa_condition: dict[str, Any],
) -> None:
    assignments = assign_chain_splits(250)
    candidates = generate_qa_candidates(
        qa_condition["chains"], assignments["validation"], "validation"
    )
    attribute_records = [
        record for record in candidates["H0"] if record["fact_type"] == "attribute"
    ]
    relation_records = [
        record for record in candidates["H0"] if record["fact_type"] == "relation"
    ]
    assert len(attribute_records) == 17 * 50
    assert len(relation_records) == 9 * 50
    excluded_anchor_fields = {
        *NATURAL_IDENTIFIER_FIELDS.values(),
        "section_label",
        "academic_term",
    }
    assert not {record["target_field"] for record in attribute_records} & excluded_anchor_fields
    assert not any(
        record["source_entity_type"] in {"course_offering", "enrollment"}
        for record in relation_records
    )


def test_relational_questions_have_exact_declared_hop_depths(
    qa_condition: dict[str, Any],
) -> None:
    records = _records_for_split(qa_condition["output_dir"], "validation")
    expected_pairs = {
        "H1": {
            "country": "continent",
            "region": "country",
            "city": "region",
            "campus": "city",
            "school": "campus",
            "department": "school",
            "subject": "department",
            "course": "subject",
            "student": "enrollment",
        },
        "H2": {
            "region": "continent",
            "city": "country",
            "campus": "region",
            "school": "city",
            "department": "campus",
            "subject": "school",
            "course": "department",
        },
        "H3": {
            "city": "continent",
            "campus": "country",
            "school": "region",
            "department": "city",
            "subject": "campus",
            "course": "school",
        },
    }
    for hop_name, pairs in expected_pairs.items():
        assert {
            record["source_entity_type"] for record in records[hop_name]
        } == set(pairs)
        assert all(
            record["target_entity_type"] == pairs[record["source_entity_type"]]
            for record in records[hop_name]
        )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_support_ids_resolve_to_retained_h0_and_no_model_facing_leakage(
    qa_condition: dict[str, Any], split: str
) -> None:
    records = _records_for_split(qa_condition["output_dir"], split)
    h0_ids = {record["id"] for record in records["H0"]}
    raw_ids = {
        entity["entity_id"]
        for chain in qa_condition["chains"]
        for entity in chain["entities"]
    }
    for hop_name, hop_records in records.items():
        for record in hop_records:
            assert record["split"] == split
            assert "context" not in record
            assert not answer_is_in_question(record["question"], record["gold_answer"])
            assert RAW_ENTITY_IDENTIFIER.search(record["question"]) is None
            assert RAW_ENTITY_IDENTIFIER.search(record["gold_answer"]) is None
            assert not any(raw_id in record["question"] for raw_id in raw_ids)
            assert not any(raw_id in record["gold_answer"] for raw_id in raw_ids)
            assert "previous-entity" not in record["question"]
            assert "attribute_" not in record["question"]
            if hop_name != "H0":
                assert len(record["support_fact_ids"]) == record["hop"] + 1
                assert set(record["support_fact_ids"]) <= h0_ids


def test_natural_anchors_and_exact_database_answers_are_used(
    qa_condition: dict[str, Any],
) -> None:
    validation_chain = qa_condition["chains"][3]
    continent, country = validation_chain["entities"][:2]
    records = _records_for_split(qa_condition["output_dir"], "validation")
    country_questions = [
        record for record in records["H0"] if country["natural_anchor"] in record["question"]
    ]
    assert country_questions
    relation = next(
        record
        for record in country_questions
        if record["fact_type"] == "relation"
    )
    assert relation["question"] == (
        f"Which continent does {country['natural_anchor']} belong to?"
    )
    assert relation["gold_answer"] == continent["natural_anchor"]


def test_strict_leakage_filter_and_support_closure(
    qa_condition: dict[str, Any],
) -> None:
    candidates = generate_qa_candidates(
        qa_condition["chains"], [3], "validation"
    )
    supported_relational = candidates["H1"][0]
    support_to_remove = supported_relational["support_fact_ids"][-1]
    leaking_h0 = next(
        record for record in candidates["H0"] if record["id"] == support_to_remove
    )
    leaking_h0["question"] += f" {leaking_h0['gold_answer']}"
    directly_leaking_relational = candidates["H1"][1]
    directly_leaking_relational["question"] += (
        f" {directly_leaking_relational['gold_answer']}"
    )

    retained, audit = filter_qa_candidates(candidates)
    retained_ids = {
        record["id"] for hop_records in retained.values() for record in hop_records
    }
    assert leaking_h0["id"] not in retained_ids
    assert directly_leaking_relational["id"] not in retained_ids
    assert all(
        support_to_remove not in record["support_fact_ids"]
        for hop_name in HOP_NAMES[1:]
        for record in retained[hop_name]
    )
    exclusions = {item["id"]: item for item in audit["excluded_items"]}
    assert exclusions[leaking_h0["id"]]["reason"] == (
        "gold_answer_contained_in_normalized_question"
    )
    assert exclusions[directly_leaking_relational["id"]]["reason"] == (
        "gold_answer_contained_in_normalized_question"
    )
    closure_exclusions = [
        item
        for item in audit["excluded_items"]
        if item["reason"] == "required_h0_support_not_retained"
    ]
    assert closure_exclusions
    assert all(
        support_to_remove in item["missing_support_fact_ids"]
        for item in closure_exclusions
    )


def test_split_manifests_record_filtering_provenance_and_file_hashes(
    qa_condition: dict[str, Any],
) -> None:
    output_dir = qa_condition["output_dir"]
    split_manifest = read_json(output_dir / "split_manifest.json")
    assert split_manifest["reserved_chain_count"] == 150
    assert split_manifest["validation_chain_count"] == 50
    assert split_manifest["test_chain_count"] == 50
    assert split_manifest["target_qa_training_generated"] is False
    assert split_manifest["zero_context"] is True
    assert not (output_dir / "train").exists()
    for split in ("validation", "test"):
        manifest_path = output_dir / split / "manifest.json"
        manifest = read_json(manifest_path)
        assert manifest["candidate_total"] == 2400
        assert isinstance(manifest["excluded_items"], list)
        for hop_name in HOP_NAMES:
            counts = manifest["counts"][hop_name]
            assert counts["candidate_count"] == {
                "H0": 1300,
                "H1": 450,
                "H2": 350,
                "H3": 300,
            }[hop_name]
            assert counts["final_retained_count"] == (
                counts["candidate_count"]
                - counts["leakage_filtered_count"]
                - counts["support_closure_filtered_count"]
            )
            path = output_dir / split / f"{hop_name}.jsonl"
            assert manifest["output_file_hashes"][path.name] == hash_file(path)


def test_condition_generation_is_byte_deterministic(
    qa_condition: dict[str, Any], default_config: dict[str, Any]
) -> None:
    second_output = qa_condition["root"] / "qa_second"
    second_result = generate_condition_qa(
        default_config,
        qa_condition["database_path"],
        qa_condition["database_manifest_path"],
        second_output,
        expected_table_count=12,
        expected_logical_fact_count=10_000,
    )
    assert second_result == qa_condition["result"]
    first_output = qa_condition["output_dir"]
    relative_files = [
        Path("split_manifest.json"),
        *(
            Path(split) / filename
            for split in ("validation", "test")
            for filename in (*[f"{hop}.jsonl" for hop in HOP_NAMES], "manifest.json")
        ),
    ]
    assert all(
        (first_output / relative).read_bytes() == (second_output / relative).read_bytes()
        for relative in relative_files
    )


def test_database_manifest_tampering_fails_before_output(
    qa_condition: dict[str, Any], default_config: dict[str, Any]
) -> None:
    tampered_path = qa_condition["root"] / "tampered_manifest.json"
    tampered = read_json(qa_condition["database_manifest_path"])
    tampered["database_sha256"] = "0" * 64
    write_json(tampered_path, tampered)
    output_dir = qa_condition["root"] / "tampered_output"
    with pytest.raises(ValueError, match="manifest hash"):
        generate_condition_qa(
            default_config,
            qa_condition["database_path"],
            tampered_path,
            output_dir,
            expected_table_count=12,
            expected_logical_fact_count=10_000,
        )
    assert not output_dir.exists()


def test_obsolete_opaque_qa_assumptions_are_removed() -> None:
    source = (PROJECT_ROOT / "src" / "data" / "qa.py").read_text(encoding="utf-8")
    for obsolete in ("pXX", "attribute_0", "previous_entity", "table_00"):
        assert obsolete not in source


def test_preflight_qa_artifacts_are_preserved() -> None:
    preflight_dir = EXP01_QA_DIR / "preflight_relational_qa"
    for relative_path in (
        "baseline/H1.jsonl",
        "baseline/H2.jsonl",
        "baseline/H3.jsonl",
        "baseline/manifest.json",
        "skill_training/train.jsonl",
        "skill_training/val.jsonl",
        "skill_training/manifest.json",
    ):
        assert (preflight_dir / relative_path).stat().st_size > 0

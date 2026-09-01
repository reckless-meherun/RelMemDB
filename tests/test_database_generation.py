from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import sqlite3
from typing import Any

import pytest

from config import ConfigError, load_config, validate_config
from data.materialize import materialize_database, partition_latent_positions
from data.serialize import build_database_serialization_block, inspect_database_schema
from data.world import (
    IDENTIFIER_MODULUS,
    NATURAL_IDENTIFIER_FIELDS,
    SEMANTIC_ENTITY_SPECS,
    build_master_world,
    derive_master_world_counts,
)
from utils.hashing import hash_file, hash_json_object


CANONICAL_SCHEMA = {
    "continent": ["continent_id", "continent_name", "climate_band"],
    "country": ["country_id", "country_name", "currency_name", "continent_id"],
    "region": ["region_id", "region_name", "administrative_type", "country_id"],
    "city": ["city_id", "city_name", "population_band", "region_id"],
    "campus": ["campus_id", "campus_name", "campus_type", "city_id"],
    "school": ["school_id", "school_name", "founding_period", "campus_id"],
    "department": ["department_id", "department_name", "focus_area", "school_id"],
    "subject": ["subject_id", "subject_name", "subject_level", "discipline_group", "department_id"],
    "course": ["course_id", "course_title", "credit_hours", "delivery_mode", "subject_id"],
    "course_offering": ["offering_id", "section_label", "meeting_period", "room_label", "course_id"],
    "enrollment": ["enrollment_id", "academic_term", "final_grade", "enrollment_status", "offering_id"],
    "student": ["student_id", "full_name", "study_year", "scholarship_status", "primary_enrollment_id"],
}


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="module")
def world(config: dict[str, Any]) -> dict[str, Any]:
    return build_master_world(config)


def _user_tables(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _stored_values(path: Path) -> Counter[str | int]:
    values: Counter[str | int] = Counter()
    with sqlite3.connect(path) as connection:
        for table in _user_tables(connection):
            for row in connection.execute(f'SELECT * FROM "{table}"'):
                values.update(row)
    return values


def test_academic_world_is_deterministic_and_seeded(config: dict[str, Any]) -> None:
    first = build_master_world(config)
    second = build_master_world(config)
    assert first == second
    assert hash_json_object(first) == hash_json_object(second)

    changed = deepcopy(config)
    changed["experiment"]["seed"] += 1
    changed_world = build_master_world(changed)
    assert changed_world != first
    assert [entity["entity_type"] for entity in changed_world["chains"][0]["entities"]] == [
        spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS
    ]


def test_master_world_counts_and_n_definition(
    config: dict[str, Any], world: dict[str, Any]
) -> None:
    counts = derive_master_world_counts(config)
    assert config["experiment"]["seed"] == 2025
    assert counts == {
        "latent_positions": 12,
        "identifier_fields_per_chain": 12,
        "attribute_facts_per_chain": 29,
        "relation_facts_per_chain": 11,
        "experimental_facts_per_chain": 40,
        "total_chains": 500,
        "total_experimental_facts": 20_000,
        "maximum_supported_configured_n": 20_000,
    }
    assert len(world["chains"][:250]) == 250
    assert 250 * (29 + 11) == 10_000
    assert 250 * 12 == 3_000


def test_each_chain_has_exact_semantic_facts(world: dict[str, Any]) -> None:
    for chain in world["chains"]:
        entities = chain["entities"]
        assert [entity["position"] for entity in entities] == list(range(12))
        assert [entity["entity_type"] for entity in entities] == [
            spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS
        ]
        assert sum(len(entity["attributes"]) for entity in entities) == 29
        assert len(chain["relations"]) == 11
        for relation in chain["relations"]:
            source = relation["source_position"]
            target = relation["target_position"]
            assert source == target + 1
            assert relation["source_entity_id"] == entities[source]["entity_id"]
            assert relation["target_entity_id"] == entities[target]["entity_id"]


def test_values_are_human_readable_and_identifiers_are_neutral(world: dict[str, Any]) -> None:
    opaque = re.compile(r"^(?:e|v)_[0-9a-f]{24,}$")
    identifier = re.compile(r"^[A-Z]{3}[0-9]{6}$")
    first_chain = world["chains"][0]
    suffixes = []
    for entity in first_chain["entities"]:
        assert identifier.fullmatch(entity["entity_id"])
        suffixes.append(entity["entity_id"][3:])
        for attribute in entity["attributes"]:
            value = attribute["value"]
            assert isinstance(value, (str, int)) and not isinstance(value, bool)
            assert not opaque.fullmatch(str(value))
            if isinstance(value, str):
                assert value.strip()
    assert len(set(suffixes)) == 12
    assert first_chain["entities"][-1]["attributes"][0]["name"] == "full_name"
    assert " " in first_chain["entities"][-1]["attributes"][0]["value"]


def test_natural_identifying_fields_are_unique_across_full_world(
    world: dict[str, Any],
) -> None:
    values_by_entity_type = {
        entity_type: [] for entity_type in NATURAL_IDENTIFIER_FIELDS
    }
    for chain in world["chains"]:
        for entity in chain["entities"]:
            field = NATURAL_IDENTIFIER_FIELDS.get(entity["entity_type"])
            if field is None:
                continue
            attributes = {
                attribute["name"]: attribute["value"]
                for attribute in entity["attributes"]
            }
            values_by_entity_type[entity["entity_type"]].append(attributes[field])

    for entity_type, values in values_by_entity_type.items():
        assert len(values) == len(world["chains"])
        assert len(values) == len(set(values)), entity_type
        assert all(isinstance(value, str) and " " in value for value in values)


def test_ids_are_globally_unique_deterministic_and_append_stable(
    config: dict[str, Any], world: dict[str, Any]
) -> None:
    identifiers = [
        entity["entity_id"]
        for chain in world["chains"]
        for entity in chain["entities"]
    ]
    numeric_suffixes = [identifier[3:] for identifier in identifiers]
    assert len(identifiers) == len(set(identifiers)) == 6_000
    assert len(numeric_suffixes) == len(set(numeric_suffixes)) == 6_000
    assert identifiers == [
        entity["entity_id"]
        for chain in build_master_world(config)["chains"]
        for entity in chain["entities"]
    ]

    extended_config = deepcopy(config)
    extended_config["data"]["n_sweep"]["fact_counts"].append(40_000)
    extended_world = build_master_world(extended_config)
    assert len(extended_world["chains"]) == 1_000
    assert extended_world["chains"][: len(world["chains"])] == world["chains"]


def test_related_ids_have_no_sequential_or_affine_chain_index_shortcut(
    world: dict[str, Any],
) -> None:
    suffixes_by_position = [
        [int(chain["entities"][position]["entity_id"][3:]) for chain in world["chains"]]
        for position in range(len(SEMANTIC_ENTITY_SPECS))
    ]
    for suffixes in suffixes_by_position:
        successive_differences = {
            (right - left) % IDENTIFIER_MODULUS
            for left, right in zip(suffixes, suffixes[1:])
        }
        assert len(successive_differences) > 450

    for parent_position in range(len(SEMANTIC_ENTITY_SPECS) - 1):
        related_differences = {
            (child - parent) % IDENTIFIER_MODULUS
            for parent, child in zip(
                suffixes_by_position[parent_position],
                suffixes_by_position[parent_position + 1],
                strict=True,
            )
        }
        assert len(related_differences) > 450

    for chain in world["chains"]:
        related_suffixes = {
            entity["entity_id"][3:] for entity in chain["entities"]
        }
        assert len(related_suffixes) == len(SEMANTIC_ENTITY_SPECS)


def test_invalid_fact_accounting_is_rejected(config: dict[str, Any]) -> None:
    changed = deepcopy(config)
    changed["data"]["master_world"]["descriptive_facts_per_chain"] = 28
    with pytest.raises(ConfigError, match="must equal the sum"):
        validate_config(changed)


def test_canonical_t12_schema_fks_integrity_and_n(
    tmp_path: Path, world: dict[str, Any]
) -> None:
    path = tmp_path / "canonical.sqlite"
    metadata = materialize_database(world, 12, 10_000, path)
    with sqlite3.connect(path) as connection:
        schema = inspect_database_schema(connection)
        actual_schema = {table["name"]: table["columns"] for table in schema["tables"]}
        assert actual_schema == CANONICAL_SCHEMA
        assert schema["table_count"] == 12
        assert schema["total_schema_column_count"] == 52
        assert schema["identifier_schema_column_count"] == 12
        assert schema["experimental_schema_column_count"] == 40
        assert schema["schema_foreign_key_count"] == 11
        assert schema["schema_cross_table_foreign_key_count"] == 11
        assert {
            (
                relationship["source_table"],
                relationship["source_column"],
                relationship["target_table"],
                relationship["target_column"],
            )
            for relationship in schema["relationships"]
        } == {
            ("country", "continent_id", "continent", "continent_id"),
            ("region", "country_id", "country", "country_id"),
            ("city", "region_id", "region", "region_id"),
            ("campus", "city_id", "city", "city_id"),
            ("school", "campus_id", "campus", "campus_id"),
            ("department", "school_id", "school", "school_id"),
            ("subject", "department_id", "department", "department_id"),
            ("course", "subject_id", "subject", "subject_id"),
            ("course_offering", "course_id", "course", "course_id"),
            ("enrollment", "offering_id", "course_offering", "offering_id"),
            ("student", "primary_enrollment_id", "enrollment", "enrollment_id"),
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in CANONICAL_SCHEMA
        } == {table: 250 for table in CANONICAL_SCHEMA}
        course_types = {
            row[1]: row[2].upper()
            for row in connection.execute('PRAGMA table_info("course")')
        }
        assert course_types["credit_hours"] == "INTEGER"
    assert metadata["selected_chain_count"] == 250
    assert metadata["attribute_fact_count"] == 7_250
    assert metadata["relation_fact_count"] == 2_750
    assert metadata["identifier_field_count"] == 3_000
    assert metadata["actual_logical_fact_count"] == 10_000
    assert metadata["stored_value_count"] == 13_000
    assert metadata["identifiers_counted_as_experimental_facts"] is False


def test_canonical_sqlite_output_is_deterministic(
    tmp_path: Path, world: dict[str, Any]
) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    first_metadata = materialize_database(world, 12, 10_000, first)
    second_metadata = materialize_database(world, 12, 10_000, second)
    assert first_metadata == second_metadata
    assert hash_file(first) == hash_file(second)


def test_future_t_materializations_preserve_logical_content(
    tmp_path: Path, world: dict[str, Any]
) -> None:
    hashes = set()
    stored_values = []
    for table_count in (4, 8, 12):
        path = tmp_path / f"t{table_count}.sqlite"
        metadata = materialize_database(world, table_count, 200, path)
        hashes.add(metadata["logical_content_sha256"])
        stored_values.append(_stored_values(path))
        with sqlite3.connect(path) as connection:
            schema = inspect_database_schema(connection)
            assert schema["table_count"] == table_count
            assert schema["schema_foreign_key_count"] == 11
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert len(hashes) == 1
    assert stored_values[0] == stored_values[1] == stored_values[2]
    assert partition_latent_positions(12, 4) == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    ]


def test_future_n_subsets_are_prefix_nested(
    tmp_path: Path, world: dict[str, Any]
) -> None:
    values = []
    for fact_count in (5_000, 10_000, 20_000):
        path = tmp_path / f"n{fact_count}.sqlite"
        metadata = materialize_database(world, 8, fact_count, path)
        assert metadata["selected_chain_count"] == fact_count // 40
        assert metadata["actual_logical_fact_count"] == fact_count
        values.append(_stored_values(path))
    n5k, n10k, n20k = values
    assert n5k < n10k < n20k
    assert world["chains"][:125] == world["chains"][:250][:125]
    assert world["chains"][:250] == world["chains"][:500][:250]


def test_serialization_schema_assumptions_exclude_identifiers(
    tmp_path: Path, world: dict[str, Any]
) -> None:
    path = tmp_path / "semantic.sqlite"
    materialize_database(world, 12, 200, path)
    block, metadata = build_database_serialization_block(path)
    assert "TABLE student" in block
    assert "TABLE course_offering" in block
    assert "credit_hours=" in block
    assert metadata["logical_fact_occurrences"] == 200
    assert metadata["identifier_occurrences"] == 60
    assert metadata["stored_value_occurrences"] == 260
    assert metadata["schema_relationship_count"] == 11

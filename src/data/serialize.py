import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from data.world import NATURAL_IDENTIFIER_FIELDS, SEMANTIC_ENTITY_SPECS
from utils.hashing import hash_file, hash_json_object, hash_text
from utils.io import read_json, write_text

SERIALIZATION_FORMAT_VERSION = 3
SERIALIZATION_STYLE = "natural_language_db_book_v1"

SCHEMA_ENTITY_DESCRIPTIONS = (
    "A continent represents a geographic area and has a name and climate band.",
    "A country represents a country with a name and currency.",
    "A region represents an administrative region.",
    "A city represents a city and its population band.",
    "A campus represents a university campus.",
    "A school represents an academic school.",
    "A department represents an academic department.",
    "A subject represents an academic subject.",
    "A course represents a university course.",
    "A course offering represents a scheduled offering of a course.",
    "An enrollment represents a student's enrollment in a course offering.",
    "A student represents a student and their study information.",
)

SCHEMA_RELATION_DESCRIPTIONS = (
    "A country belongs to a continent.",
    "A region belongs to a country.",
    "A city belongs to a region.",
    "A campus is located in a city.",
    "A school is located at a campus.",
    "A department belongs to a school.",
    "A subject belongs to a department.",
    "A course belongs to a subject.",
    "A course offering is an offering of a course.",
    "An enrollment is for a course offering.",
    "A student has a primary enrollment.",
)

RAW_ENTITY_IDENTIFIER = re.compile(
    r"\b(?:CTN|CTR|REG|CTY|CAM|SCH|DEP|SUB|CRS|OFF|ENR|STU)\d{6}\b"
)


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _user_table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _single_column_unique_indexes(
    connection: sqlite3.Connection, table_name: str
) -> set[str]:
    unique_columns: set[str] = set()
    for index in connection.execute(f"PRAGMA index_list({_quote(table_name)})"):
        if not index[2]:
            continue
        index_columns = connection.execute(
            f"PRAGMA index_info({_quote(index[1])})"
        ).fetchall()
        if len(index_columns) == 1:
            unique_columns.add(index_columns[0][2])
    return unique_columns


def inspect_database_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    table_names = _user_table_names(connection)
    if not table_names:
        raise ValueError("database contains no user tables")

    tables: list[dict[str, Any]] = []
    identifier_locations: set[tuple[str, str]] = set()
    for table_name in table_names:
        table_info = connection.execute(
            f"PRAGMA table_info({_quote(table_name)})"
        ).fetchall()
        if not table_info:
            raise ValueError(f"table has no columns: {table_name}")
        columns = [row[1] for row in table_info]
        primary_keys = [row[1] for row in sorted(table_info, key=lambda row: row[5]) if row[5]]
        if len(primary_keys) != 1:
            raise ValueError(f"table must have exactly one primary key: {table_name}")
        identifier_columns = _single_column_unique_indexes(connection, table_name)
        identifier_columns.add(primary_keys[0])
        identifier_locations.update((table_name, column) for column in identifier_columns)
        tables.append(
            {
                "name": table_name,
                "columns": columns,
                "column_types": {row[1]: row[2].upper() for row in table_info},
                "primary_key": primary_keys[0],
                "identifier_columns": sorted(identifier_columns, key=columns.index),
            }
        )

    relationships: list[dict[str, str]] = []
    foreign_key_sources: set[tuple[str, str]] = set()
    for table in tables:
        source_table = table["name"]
        for foreign_key in connection.execute(
            f"PRAGMA foreign_key_list({_quote(source_table)})"
        ):
            source_column = foreign_key[3]
            source = (source_table, source_column)
            if source in foreign_key_sources:
                raise ValueError(f"duplicate foreign key for {source_table}.{source_column}")
            foreign_key_sources.add(source)
            target = (foreign_key[2], foreign_key[4])
            if target not in identifier_locations:
                raise ValueError(
                    f"foreign key target is not a unique identifier: {target[0]}.{target[1]}"
                )
            relationships.append(
                {
                    "type": "FOREIGN_KEY",
                    "source_table": source_table,
                    "source_column": source_column,
                    "target_table": target[0],
                    "target_column": target[1],
                }
            )
    relationships.sort(
        key=lambda relationship: (
            relationship["source_table"], relationship["source_column"]
        )
    )
    identifier_count = sum(len(table["identifier_columns"]) for table in tables)
    total_column_count = sum(len(table["columns"]) for table in tables)
    return {
        "tables": tables,
        "relationships": relationships,
        "table_count": len(tables),
        "total_schema_column_count": total_column_count,
        "identifier_schema_column_count": identifier_count,
        "experimental_schema_column_count": total_column_count - identifier_count,
        "schema_relationship_count": len(relationships),
        "schema_foreign_key_count": len(relationships),
        "schema_cross_table_foreign_key_count": sum(
            relationship["source_table"] != relationship["target_table"]
            for relationship in relationships
        ),
        "schema_intra_table_relation_count": sum(
            relationship["source_table"] == relationship["target_table"]
            for relationship in relationships
        ),
    }


def _physical_table_name(positions: list[int]) -> str:
    return "__".join(
        SEMANTIC_ENTITY_SPECS[position]["entity_type"] for position in positions
    )


def _physical_column_name(
    position: int, logical_name: str, positions: list[int]
) -> str:
    if len(positions) == 1:
        return logical_name
    entity_type = SEMANTIC_ENTITY_SPECS[position]["entity_type"]
    return f"{entity_type}__{logical_name}"


def _physical_column_descriptors(
    positions: list[int],
) -> list[tuple[int, str, str, str]]:
    descriptors: list[tuple[int, str, str, str]] = []
    for position in positions:
        spec = SEMANTIC_ENTITY_SPECS[position]
        descriptors.append(
            (
                position,
                "identifier",
                spec["id_field"],
                _physical_column_name(position, spec["id_field"], positions),
            )
        )
        descriptors.extend(
            (
                position,
                "attribute",
                attribute_name,
                _physical_column_name(position, attribute_name, positions),
            )
            for attribute_name, _ in spec["attributes"]
        )
        if position > 0:
            descriptors.append(
                (
                    position,
                    "relation",
                    spec["foreign_key"],
                    _physical_column_name(
                        position, spec["foreign_key"], positions
                    ),
                )
            )
    return descriptors


def _validated_position_partition(
    database_manifest: dict[str, Any],
) -> list[list[int]]:
    partition = database_manifest.get("position_partition")
    if not isinstance(partition, list) or not partition:
        raise ValueError("database manifest position_partition is missing or invalid")
    if len(partition) != database_manifest.get("table_count"):
        raise ValueError("database manifest position partition does not match T")
    if not all(
        isinstance(group, list)
        and group
        and all(isinstance(position, int) for position in group)
        for group in partition
    ):
        raise ValueError("database manifest position partition is invalid")
    flattened = [position for group in partition for position in group]
    if flattened != list(range(len(SEMANTIC_ENTITY_SPECS))):
        raise ValueError("database manifest position partition is incomplete")
    return partition


def _read_logical_entities(
    connection: sqlite3.Connection,
    partition: list[list[int]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    actual_table_names = set(_user_table_names(connection))
    expected_table_names = {_physical_table_name(group) for group in partition}
    if actual_table_names != expected_table_names:
        raise ValueError("physical database tables do not match the manifest partition")

    physical_groups: list[dict[str, Any]] = []
    entities_by_position: dict[int, dict[str, dict[str, Any]]] = {
        position: {} for position in range(len(SEMANTIC_ENTITY_SPECS))
    }
    seen_identifiers: set[str] = set()
    physical_row_count = 0
    for group_index, positions in enumerate(partition, start=1):
        table_name = _physical_table_name(positions)
        descriptors = _physical_column_descriptors(positions)
        expected_columns = [descriptor[3] for descriptor in descriptors]
        actual_columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({_quote(table_name)})")
        ]
        if actual_columns != expected_columns:
            raise ValueError(
                f"physical columns for {table_name} do not match semantic positions"
            )
        selected_columns = ", ".join(_quote(column) for column in expected_columns)
        group_rows: list[list[dict[str, Any]]] = []
        for row in connection.execute(
            f"SELECT {selected_columns} FROM {_quote(table_name)} ORDER BY rowid"
        ):
            if any(value is None for value in row):
                raise ValueError("database contains a null logical value")
            if not all(
                isinstance(value, (str, int)) and not isinstance(value, bool)
                for value in row
            ):
                raise ValueError("logical values must be text or integers")
            row_entities = {
                position: {
                    "position": position,
                    "entity_type": SEMANTIC_ENTITY_SPECS[position]["entity_type"],
                    "entity_id": None,
                    "attributes": {},
                    "relation_name": None,
                    "relation_target_id": None,
                }
                for position in positions
            }
            for (position, kind, logical_name, _), value in zip(
                descriptors, row, strict=True
            ):
                entity = row_entities[position]
                if kind == "identifier":
                    if not isinstance(value, str) or not value:
                        raise ValueError("entity identifiers must be non-empty text")
                    entity["entity_id"] = value
                elif kind == "attribute":
                    entity["attributes"][logical_name] = value
                else:
                    if not isinstance(value, str) or not value:
                        raise ValueError("relation targets must be non-empty text")
                    entity["relation_name"] = logical_name
                    entity["relation_target_id"] = value

            ordered_entities = [row_entities[position] for position in positions]
            for entity in ordered_entities:
                entity_id = entity["entity_id"]
                if entity_id in seen_identifiers:
                    raise ValueError("entity identifiers must be globally unique")
                seen_identifiers.add(entity_id)
                position = entity["position"]
                expected_attributes = [
                    name for name, _ in SEMANTIC_ENTITY_SPECS[position]["attributes"]
                ]
                if list(entity["attributes"]) != expected_attributes:
                    raise ValueError("entity attributes do not match the semantic schema")
                entities_by_position[position][entity_id] = entity
            group_rows.append(ordered_entities)
            physical_row_count += 1
        physical_groups.append(
            {
                "index": group_index,
                "table_name": table_name,
                "positions": positions,
                "entity_types": [
                    SEMANTIC_ENTITY_SPECS[position]["entity_type"]
                    for position in positions
                ],
                "rows": group_rows,
            }
        )

    for position, entities in entities_by_position.items():
        if not entities:
            raise ValueError(f"no entities found at semantic position {position}")
        if position == 0:
            continue
        for entity in entities.values():
            target_id = entity["relation_target_id"]
            if target_id not in entities_by_position[position - 1]:
                raise ValueError("relation target does not resolve to its semantic entity")

    schema = inspect_database_schema(connection)
    metadata = {
        "physical_row_count": physical_row_count,
        "source_identifier_count": len(seen_identifiers),
        **{
            key: schema[key]
            for key in (
                "table_count",
                "total_schema_column_count",
                "identifier_schema_column_count",
                "experimental_schema_column_count",
                "schema_relationship_count",
                "schema_foreign_key_count",
                "schema_cross_table_foreign_key_count",
                "schema_intra_table_relation_count",
            )
        },
    }
    return physical_groups, entities_by_position, metadata


def _human_join(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _indefinite_article(value: str) -> str:
    return "an" if value[:1].lower() in "aeiou" else "a"


def _possessive(value: str) -> str:
    return f"{value}'" if value.endswith("s") else f"{value}'s"


def _assign_natural_anchors(
    entities_by_position: dict[int, dict[str, dict[str, Any]]],
) -> None:
    for position, spec in enumerate(SEMANTIC_ENTITY_SPECS):
        entity_type = spec["entity_type"]
        anchors: set[str] = set()
        for entity in entities_by_position[position].values():
            if entity_type in NATURAL_IDENTIFIER_FIELDS:
                anchor = entity["attributes"][NATURAL_IDENTIFIER_FIELDS[entity_type]]
            elif entity_type == "course_offering":
                course = entities_by_position[position - 1][
                    entity["relation_target_id"]
                ]
                anchor = (
                    f"{entity['attributes']['section_label']} of "
                    f"{course['attributes']['course_title']}"
                )
            elif entity_type == "enrollment":
                offering = entities_by_position[position - 1][
                    entity["relation_target_id"]
                ]
                anchor = (
                    f"{entity['attributes']['academic_term']} enrollment for "
                    f"{offering['natural_anchor']}"
                )
            else:
                raise ValueError(f"no natural anchor rule for {entity_type}")
            if not isinstance(anchor, str) or not anchor.strip():
                raise ValueError(f"natural anchor for {entity_type} is invalid")
            if anchor in anchors:
                raise ValueError(f"natural anchors for {entity_type} are ambiguous")
            anchors.add(anchor)
            entity["natural_anchor"] = anchor


def _attribute_fact(entity: dict[str, Any], field: str) -> tuple[str, ...]:
    value = entity["attributes"][field]
    value_type = "integer" if isinstance(value, int) else "text"
    return (
        "attribute",
        entity["entity_type"],
        entity["entity_id"],
        field,
        value_type,
        str(value),
    )


def _relation_fact(entity: dict[str, Any]) -> tuple[str, ...]:
    return (
        "relation",
        entity["entity_type"],
        entity["entity_id"],
        entity["relation_name"],
        entity["relation_target_id"],
    )


def _expected_facts(
    entities_by_position: dict[int, dict[str, dict[str, Any]]],
) -> list[tuple[str, ...]]:
    facts: list[tuple[str, ...]] = []
    for position in range(len(SEMANTIC_ENTITY_SPECS)):
        for entity in entities_by_position[position].values():
            facts.extend(
                _attribute_fact(entity, field) for field in entity["attributes"]
            )
            if position > 0:
                facts.append(_relation_fact(entity))
    return facts


def _render_entity_facts(
    entity: dict[str, Any],
    entities_by_position: dict[int, dict[str, dict[str, Any]]],
) -> tuple[list[str], list[tuple[str, ...]]]:
    position = entity["position"]
    entity_type = entity["entity_type"]
    attributes = entity["attributes"]
    anchor = entity["natural_anchor"]
    target = None
    if position > 0:
        target = entities_by_position[position - 1][entity["relation_target_id"]]

    sentences: list[str]
    attribute_fields: tuple[str, ...]
    if entity_type == "continent":
        sentences = [
            f"{anchor} is a continent with a {attributes['climate_band']} climate band."
        ]
        attribute_fields = ("continent_name", "climate_band")
    elif entity_type == "country":
        sentences = [
            f"{anchor} is a country that uses the {attributes['currency_name']}.",
            f"{anchor} belongs to {target['natural_anchor']}.",
        ]
        attribute_fields = ("country_name", "currency_name")
    elif entity_type == "region":
        administrative_type = attributes["administrative_type"]
        sentences = [
            (
                f"{anchor} is {_indefinite_article(administrative_type)} "
                f"{administrative_type} in {target['natural_anchor']}."
            )
        ]
        attribute_fields = ("region_name", "administrative_type")
    elif entity_type == "city":
        sentences = [
            f"{anchor} is a city with population band {attributes['population_band']}.",
            f"{anchor} is located in {target['natural_anchor']}.",
        ]
        attribute_fields = ("city_name", "population_band")
    elif entity_type == "campus":
        campus_type = attributes["campus_type"]
        sentences = [
            (
                f"{anchor} is {_indefinite_article(campus_type)} "
                f"{campus_type} campus located in {target['natural_anchor']}."
            )
        ]
        attribute_fields = ("campus_name", "campus_type")
    elif entity_type == "school":
        period = attributes["founding_period"]
        if period.startswith("before "):
            founding_sentence = f"{anchor} was founded {period}."
        elif period.startswith("since "):
            founding_sentence = f"{anchor} was founded in the period {period}."
        else:
            founding_sentence = f"{anchor} was founded in {period}."
        sentences = [
            founding_sentence,
            f"{anchor} is located at {target['natural_anchor']}.",
        ]
        attribute_fields = ("school_name", "founding_period")
    elif entity_type == "department":
        sentences = [
            f"{anchor} focuses on {attributes['focus_area']}.",
            f"{anchor} belongs to {target['natural_anchor']}.",
        ]
        attribute_fields = ("department_name", "focus_area")
    elif entity_type == "subject":
        level = attributes["subject_level"]
        sentences = [
            (
                f"{anchor} is {_indefinite_article(level)} {level} subject in the "
                f"{attributes['discipline_group']} discipline group."
            ),
            f"{anchor} belongs to {target['natural_anchor']}.",
        ]
        attribute_fields = (
            "subject_name",
            "subject_level",
            "discipline_group",
        )
    elif entity_type == "course":
        sentences = [
            (
                f"{anchor} is a {attributes['credit_hours']}-credit "
                f"{attributes['delivery_mode']} course."
            ),
            f"{anchor} belongs to {target['natural_anchor']}.",
        ]
        attribute_fields = ("course_title", "credit_hours", "delivery_mode")
    elif entity_type == "course_offering":
        sentences = [
            (
                f"{anchor} meets {attributes['meeting_period']} in "
                f"{attributes['room_label']}."
            )
        ]
        attribute_fields = ("section_label", "meeting_period", "room_label")
    elif entity_type == "enrollment":
        sentences = [
            (
                f"The {anchor} has final grade {attributes['final_grade']} and "
                f"status {attributes['enrollment_status']}."
            )
        ]
        attribute_fields = (
            "academic_term",
            "final_grade",
            "enrollment_status",
        )
    elif entity_type == "student":
        sentences = [
            (
                f"{anchor} is a student in their {attributes['study_year']} with "
                f"{attributes['scholarship_status']}."
            ),
            (
                f"{_possessive(anchor)} primary enrollment is the "
                f"{target['natural_anchor']}."
            ),
        ]
        attribute_fields = ("full_name", "study_year", "scholarship_status")
    else:
        raise ValueError(f"no natural-language template for {entity_type}")

    covered_facts = [
        _attribute_fact(entity, field) for field in attribute_fields
    ]
    if position > 0:
        covered_facts.append(_relation_fact(entity))
    return sentences, covered_facts


def _group_heading(entity_types: list[str]) -> str:
    labels = [entity_type.replace("_", " ").title() for entity_type in entity_types]
    return f"{_human_join(labels)} Records"


def _group_description(entity_types: list[str]) -> str:
    labels = [entity_type.replace("_", " ") for entity_type in entity_types]
    joined = _human_join(labels)
    if len(labels) == 1:
        return f"{joined.capitalize()} information has its own physical record group."
    return f"{joined.capitalize()} information is kept together in one physical record group."


def build_readable_database_book(
    database_path: str | Path,
    database_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    path = Path(database_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"database file is missing or empty: {path}")
    partition = _validated_position_partition(database_manifest)
    connection = sqlite3.connect(path)
    try:
        physical_groups, entities_by_position, metadata = _read_logical_entities(
            connection, partition
        )
    finally:
        connection.close()
    _assign_natural_anchors(entities_by_position)

    lines = [
        "The Academic Database Book",
        "",
        *SCHEMA_ENTITY_DESCRIPTIONS,
        "",
        *SCHEMA_RELATION_DESCRIPTIONS,
        "",
        (
            f"This edition presents the database in {len(physical_groups)} "
            "physical record groups."
        ),
        *[
            _group_description(group["entity_types"])
            for group in physical_groups
        ],
    ]
    covered_facts: list[tuple[str, ...]] = []
    instance_sentence_count = 0
    for group in physical_groups:
        lines.extend(("", _group_heading(group["entity_types"]), ""))
        for row_entities in group["rows"]:
            for entity in row_entities:
                sentences, entity_coverage = _render_entity_facts(
                    entity, entities_by_position
                )
                lines.extend(sentences)
                covered_facts.extend(entity_coverage)
                instance_sentence_count += len(sentences)
            lines.append("")
    book = "\n".join(lines).rstrip() + "\n"

    expected_facts = _expected_facts(entities_by_position)
    if Counter(covered_facts) != Counter(expected_facts):
        raise RuntimeError(
            "natural-language serialization did not cover every database fact exactly once"
        )
    raw_identifier = RAW_ENTITY_IDENTIFIER.search(book)
    if raw_identifier:
        raise RuntimeError(
            f"natural-language serialization exposed identifier {raw_identifier.group(0)}"
        )
    attribute_fact_count = sum(fact[0] == "attribute" for fact in expected_facts)
    relation_fact_count = sum(fact[0] == "relation" for fact in expected_facts)
    logical_fact_count = len(expected_facts)
    metadata.update(
        {
            "logical_fact_occurrences": logical_fact_count,
            "attribute_fact_occurrences": attribute_fact_count,
            "relation_fact_occurrences": relation_fact_count,
            "logical_fact_coverage_sha256": hash_json_object(
                sorted(expected_facts)
            ),
            "logical_entity_count": sum(
                len(entities) for entities in entities_by_position.values()
            ),
            "instance_sentence_count": instance_sentence_count,
            "schema_description_sentence_count": (
                len(SCHEMA_ENTITY_DESCRIPTIONS)
                + len(SCHEMA_RELATION_DESCRIPTIONS)
            ),
            "record_organization_sentence_count": 1 + len(physical_groups),
            "physical_record_groups": [
                {
                    "group_index": group["index"],
                    "physical_table": group["table_name"],
                    "semantic_positions": group["positions"],
                    "entity_types": group["entity_types"],
                    "row_count": len(group["rows"]),
                }
                for group in physical_groups
            ],
        }
    )
    return book, metadata


def serialize_database_cpt(
    config: dict[str, Any],
    database_path: str | Path,
    database_manifest_path: str | Path,
    train_text_path: str | Path,
    *,
    readable_book_path: str | Path | None = None,
    expected_table_count: int | None = None,
    expected_logical_fact_count: int | None = None,
) -> dict[str, Any]:
    database_path = Path(database_path)
    database_manifest_path = Path(database_manifest_path)
    train_text_path = Path(train_text_path)
    readable_book_path = (
        Path(readable_book_path)
        if readable_book_path is not None
        else train_text_path.with_name("book_readable.txt")
    )
    for artifact_path, label in (
        (database_path, "database"),
        (database_manifest_path, "database manifest"),
    ):
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} is missing or empty: {artifact_path}")

    database_manifest = read_json(database_manifest_path)
    database_sha256 = hash_file(database_path)
    if database_manifest.get("database_sha256") != database_sha256:
        raise ValueError("database does not match its manifest hash")
    requested_n = database_manifest.get("requested_N")
    table_count = database_manifest.get("table_count")
    if (
        isinstance(requested_n, bool)
        or not isinstance(requested_n, int)
        or requested_n <= 0
    ):
        raise ValueError("database manifest requested_N must be a positive integer")
    if database_manifest.get("actual_logical_fact_count") != requested_n:
        raise ValueError("database manifest logical fact counts are inconsistent")
    if (
        isinstance(table_count, bool)
        or not isinstance(table_count, int)
        or table_count <= 0
    ):
        raise ValueError("database manifest table_count must be a positive integer")
    if database_manifest.get("T") != table_count:
        raise ValueError("database manifest T and table_count are inconsistent")
    if expected_table_count is not None and table_count != expected_table_count:
        raise ValueError(
            f"database manifest T={table_count} does not match requested "
            f"T={expected_table_count}"
        )
    if (
        expected_logical_fact_count is not None
        and requested_n != expected_logical_fact_count
    ):
        raise ValueError(
            f"database manifest N={requested_n} does not match requested "
            f"N={expected_logical_fact_count}"
        )

    readable_book, metadata = build_readable_database_book(
        database_path, database_manifest
    )
    expected = {
        "table_count": table_count,
        "logical_fact_occurrences": requested_n,
        "attribute_fact_occurrences": database_manifest.get("attribute_fact_count"),
        "relation_fact_occurrences": database_manifest.get("relation_fact_count"),
        "physical_row_count": database_manifest.get("physical_row_count"),
        "source_identifier_count": database_manifest.get("identifier_field_count"),
        "total_schema_column_count": database_manifest.get("schema_column_count"),
        "experimental_schema_column_count": database_manifest.get(
            "experimental_facts_per_chain"
        ),
        "identifier_schema_column_count": database_manifest.get(
            "identifier_fields_per_chain"
        ),
        "schema_relationship_count": database_manifest.get(
            "relation_facts_per_chain"
        ),
        "schema_foreign_key_count": database_manifest.get(
            "schema_foreign_key_count"
        ),
        "schema_cross_table_foreign_key_count": database_manifest.get(
            "cross_table_fk_edge_count"
        ),
        "schema_intra_table_relation_count": database_manifest.get(
            "intra_table_fk_edge_count"
        ),
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise ValueError(f"readable database book {key} does not match its manifest")

    fact_exposure = config["training"]["fact_exposure"]
    if (
        isinstance(fact_exposure, bool)
        or not isinstance(fact_exposure, int)
        or fact_exposure <= 0
    ):
        raise ValueError("training.fact_exposure must be a positive integer")
    train_text = readable_book * fact_exposure
    write_text(readable_book_path, readable_book)
    write_text(train_text_path, train_text)
    return {
        "format_version": SERIALIZATION_FORMAT_VERSION,
        "serialization_style": SERIALIZATION_STYLE,
        "experiment_name": config["experiment"]["name"],
        "T": table_count,
        "requested_N": requested_n,
        "logical_content_sha256": database_manifest["logical_content_sha256"],
        "source_database_sha256": database_sha256,
        "source_database_manifest_sha256": hash_file(database_manifest_path),
        "fact_exposure": fact_exposure,
        "readable_book_copy_count_in_train_text": fact_exposure,
        "logical_facts_per_exposure": metadata["logical_fact_occurrences"],
        "attribute_facts_per_exposure": metadata["attribute_fact_occurrences"],
        "relation_facts_per_exposure": metadata["relation_fact_occurrences"],
        "serialized_logical_fact_occurrences": (
            metadata["logical_fact_occurrences"] * fact_exposure
        ),
        "logical_fact_coverage_sha256": metadata[
            "logical_fact_coverage_sha256"
        ],
        "identifiers_in_readable_book": 0,
        "source_identifier_count": metadata["source_identifier_count"],
        "physical_rows_per_exposure": metadata["physical_row_count"],
        "logical_entities_per_exposure": metadata["logical_entity_count"],
        "instance_sentence_count_per_exposure": metadata[
            "instance_sentence_count"
        ],
        "schema_description_sentence_count_per_exposure": metadata[
            "schema_description_sentence_count"
        ],
        "record_organization_sentence_count_per_exposure": metadata[
            "record_organization_sentence_count"
        ],
        "table_count": metadata["table_count"],
        "physical_record_group_count": len(metadata["physical_record_groups"]),
        "physical_record_groups": metadata["physical_record_groups"],
        "total_schema_column_count": metadata["total_schema_column_count"],
        "experimental_schema_column_count": metadata[
            "experimental_schema_column_count"
        ],
        "identifier_schema_column_count": metadata[
            "identifier_schema_column_count"
        ],
        "schema_relationship_count_per_exposure": metadata[
            "schema_relationship_count"
        ],
        "schema_foreign_key_count_per_exposure": metadata[
            "schema_foreign_key_count"
        ],
        "schema_intra_table_relation_count_per_exposure": metadata[
            "schema_intra_table_relation_count"
        ],
        "serialization_order": {
            "record_groups": "database-manifest position_partition order",
            "rows": "ascending SQLite rowid within each physical record group",
            "entities": "ascending semantic position within each physical row",
            "exposures": "identical copies of book_readable.txt",
        },
        "readable_book_sha256": hash_text(readable_book),
        "readable_book_byte_count": len(readable_book.encode("utf-8")),
        "readable_book_character_count": len(readable_book),
        "readable_book_line_count": readable_book.count("\n"),
        "train_text_sha256": hash_text(train_text),
        "train_text_byte_count": len(train_text.encode("utf-8")),
        "train_text_character_count": len(train_text),
        "train_text_line_count": train_text.count("\n"),
    }

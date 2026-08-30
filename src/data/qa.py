import re
import sqlite3
from pathlib import Path
from typing import Any

from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, write_jsonl


QA_FORMAT_VERSION = 1
QUESTION_TEMPLATE_VERSION = 1
ID_DIGEST_LENGTH = 32

ENTITY_COLUMN_PATTERN = re.compile(r"^p(\d+)_entity_id$")
ATTRIBUTE_COLUMN_PATTERN = re.compile(r"^p(\d+)_(attribute_\d+)$")
RELATION_COLUMN_PATTERN = re.compile(r"^p(\d+)_previous_entity_id$")


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _user_table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]


def _stable_id(prefix: str, semantic_key: list[Any]) -> str:
    digest = hash_json_object(semantic_key)[:ID_DIGEST_LENGTH]
    return f"{prefix}_{digest}"


def _attribute_sort_key(slot: str) -> int:
    prefix, separator, number = slot.rpartition("_")
    if prefix != "attribute" or separator != "_" or not number.isdigit():
        raise ValueError(f"invalid attribute slot: {slot}")
    return int(number)


def _assign_once(record: dict[str, Any], key: str, value: str, context: str) -> None:
    if key in record:
        raise ValueError(f"duplicate logical field {key} in {context}")
    if not isinstance(value, str):
        raise ValueError(f"logical value must be text in {context}")
    record[key] = value


def reconstruct_logical_chains(
    connection: sqlite3.Connection, latent_positions: int
) -> list[dict[str, Any]]:
    if (
        isinstance(latent_positions, bool)
        or not isinstance(latent_positions, int)
        or latent_positions <= 0
    ):
        raise ValueError("latent_positions must be a positive integer")

    table_names = _user_table_names(connection)
    if not table_names:
        raise ValueError("database contains no user tables")

    aligned_rowids: list[int] | None = None
    logical_rows: list[list[dict[str, Any]]] = []
    for table_name in table_names:
        table_info = connection.execute(
            f"PRAGMA table_info({_quote(table_name)})"
        ).fetchall()
        columns = [row[1] for row in table_info]
        if not columns:
            raise ValueError(f"table has no columns: {table_name}")
        selected_columns = ", ".join(_quote(column) for column in columns)
        rows = connection.execute(
            f"SELECT rowid, {selected_columns} "
            f"FROM {_quote(table_name)} ORDER BY rowid"
        ).fetchall()
        rowids = [row[0] for row in rows]
        if aligned_rowids is None:
            aligned_rowids = rowids
            logical_rows = [
                [
                    {"attributes": {}}
                    for _ in range(latent_positions)
                ]
                for _ in rows
            ]
        elif rowids != aligned_rowids:
            raise ValueError("physical tables do not have aligned rowid sequences")

        for logical_row, row in zip(logical_rows, rows, strict=True):
            for column, value in zip(columns, row[1:], strict=True):
                entity_match = ENTITY_COLUMN_PATTERN.fullmatch(column)
                attribute_match = ATTRIBUTE_COLUMN_PATTERN.fullmatch(column)
                relation_match = RELATION_COLUMN_PATTERN.fullmatch(column)
                if entity_match is not None:
                    position = int(entity_match.group(1))
                    field = "entity_id"
                elif attribute_match is not None:
                    position = int(attribute_match.group(1))
                    field = attribute_match.group(2)
                elif relation_match is not None:
                    position = int(relation_match.group(1))
                    field = "previous_entity_id"
                else:
                    raise ValueError(f"unrecognized logical column: {column}")
                if not 0 <= position < latent_positions:
                    raise ValueError(f"column position is out of range: {column}")
                if field.startswith("attribute_"):
                    _assign_once(
                        logical_row[position]["attributes"],
                        field,
                        value,
                        column,
                    )
                else:
                    _assign_once(logical_row[position], field, value, column)

    chains: list[dict[str, Any]] = []
    all_entity_ids: set[str] = set()
    for logical_row in logical_rows:
        attribute_count = 0
        previous_entity_ids: list[str | None] = []
        entities: list[dict[str, Any]] = []
        for position, position_data in enumerate(logical_row):
            entity_id = position_data.get("entity_id")
            if not isinstance(entity_id, str):
                raise ValueError(f"missing entity ID at position {position}")
            if entity_id in all_entity_ids:
                raise ValueError(f"duplicate entity ID in database: {entity_id}")
            all_entity_ids.add(entity_id)
            attributes = position_data["attributes"]
            ordered_attributes = {
                slot: attributes[slot]
                for slot in sorted(attributes, key=_attribute_sort_key)
            }
            if "attribute_0" not in ordered_attributes:
                raise ValueError(f"missing attribute_0 at position {position}")
            attribute_count += len(ordered_attributes)
            entities.append(
                {
                    "entity_id": entity_id,
                    "attributes": ordered_attributes,
                }
            )

            previous_entity_id = position_data.get("previous_entity_id")
            if position == 0:
                if previous_entity_id is not None:
                    raise ValueError("position 0 must not have a previous relation")
                previous_entity_ids.append(None)
            else:
                if not isinstance(previous_entity_id, str):
                    raise ValueError(f"missing previous relation at position {position}")
                previous_entity_ids.append(previous_entity_id)

        if attribute_count != 17:
            raise ValueError("each logical chain must contain exactly 17 attributes")
        for position in range(1, latent_positions):
            if previous_entity_ids[position] != entities[position - 1]["entity_id"]:
                raise ValueError(
                    f"broken previous-entity relation at position {position}"
                )
        chains.append(
            {
                "entities": entities,
                "previous_entity_ids": previous_entity_ids,
            }
        )
    return chains


def load_verified_logical_chains(
    database_path: str | Path,
    database_manifest_path: str | Path,
    *,
    expected_table_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database_path = Path(database_path)
    database_manifest_path = Path(database_manifest_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"database file not found: {database_path}")
    if not database_manifest_path.is_file():
        raise FileNotFoundError(
            f"database manifest not found: {database_manifest_path}"
        )

    manifest = read_json(database_manifest_path)
    database_sha256 = hash_file(database_path)
    if manifest.get("database_sha256") != database_sha256:
        raise ValueError("database does not match its manifest hash")
    table_count = manifest.get("table_count")
    if isinstance(table_count, bool) or not isinstance(table_count, int) or table_count <= 0:
        raise ValueError("database manifest table_count must be a positive integer")
    if manifest.get("T") != table_count:
        raise ValueError("database manifest T and table_count are inconsistent")
    if expected_table_count is not None and table_count != expected_table_count:
        raise ValueError("database table count does not match the requested condition")
    requested_n = manifest.get("requested_N")
    if isinstance(requested_n, bool) or not isinstance(requested_n, int) or requested_n <= 0:
        raise ValueError("database manifest requested_N must be a positive integer")
    if requested_n != manifest.get("actual_logical_fact_count"):
        raise ValueError("database manifest logical fact counts are inconsistent")

    connection = sqlite3.connect(database_path)
    try:
        if len(_user_table_names(connection)) != table_count:
            raise ValueError("physical table count does not match the manifest")
        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_violations:
            raise ValueError(f"foreign-key violations: {foreign_key_violations}")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SQLite integrity check failed")
        chains = reconstruct_logical_chains(
            connection, latent_positions=manifest["latent_positions"]
        )
    finally:
        connection.close()

    if len(chains) != manifest.get("selected_chain_count"):
        raise ValueError("reconstructed chain count does not match the manifest")
    logical_fact_count = sum(
        len(chain["entities"])
        + sum(len(entity["attributes"]) for entity in chain["entities"])
        + len(chain["previous_entity_ids"])
        - 1
        for chain in chains
    )
    if logical_fact_count != requested_n:
        raise ValueError("reconstructed logical fact count does not match requested_N")
    return chains, manifest


def generate_qa_records(
    chains: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {
        "H0": [],
        "H1": [],
        "H2": [],
        "H3": [],
    }
    attribute_ids: dict[tuple[str, str], str] = {}
    relation_ids: dict[str, str] = {}
    semantic_ids: dict[str, tuple[Any, ...]] = {}

    def register_id(record_id: str, semantic_key: tuple[Any, ...]) -> None:
        existing_key = semantic_ids.get(record_id)
        if existing_key is not None and existing_key != semantic_key:
            raise ValueError(f"QA ID collision: {record_id}")
        if existing_key is not None:
            raise ValueError(f"duplicate logical QA: {semantic_key}")
        semantic_ids[record_id] = semantic_key

    for chain in chains:
        entities = chain["entities"]
        previous_entity_ids = chain["previous_entity_ids"]
        for entity in entities:
            entity_id = entity["entity_id"]
            for slot, value in entity["attributes"].items():
                semantic_key = ("h0_attribute", entity_id, slot)
                record_id = _stable_id("h0_attr", list(semantic_key))
                register_id(record_id, semantic_key)
                attribute_ids[(entity_id, slot)] = record_id
                records["H0"].append(
                    {
                        "id": record_id,
                        "hop": 0,
                        "type": "attribute",
                        "question": f"What is {slot} of entity {entity_id}?",
                        "answer": value,
                    }
                )
        for position in range(1, len(entities)):
            source_entity_id = entities[position]["entity_id"]
            semantic_key = ("h0_relation", source_entity_id)
            record_id = _stable_id("h0_rel", list(semantic_key))
            register_id(record_id, semantic_key)
            relation_ids[source_entity_id] = record_id
            records["H0"].append(
                {
                    "id": record_id,
                    "hop": 0,
                    "type": "relation",
                    "question": (
                        "Which entity is immediately previous to entity "
                        f"{source_entity_id}?"
                    ),
                    "answer": previous_entity_ids[position],
                }
            )

        for hop in (1, 2, 3):
            times = {1: "once", 2: "two times", 3: "three times"}[hop]
            for source_position in range(hop, len(entities)):
                source_entity_id = entities[source_position]["entity_id"]
                target_entity = entities[source_position - hop]
                semantic_key = (
                    "relational",
                    hop,
                    source_entity_id,
                    "attribute_0",
                )
                record_id = _stable_id(f"h{hop}", list(semantic_key))
                register_id(record_id, semantic_key)
                support_fact_ids = [
                    relation_ids[entities[position]["entity_id"]]
                    for position in range(
                        source_position, source_position - hop, -1
                    )
                ]
                support_fact_ids.append(
                    attribute_ids[(target_entity["entity_id"], "attribute_0")]
                )
                records[f"H{hop}"].append(
                    {
                        "id": record_id,
                        "hop": hop,
                        "type": "relational",
                        "question": (
                            f"Starting from entity {source_entity_id}, follow the "
                            f"previous-entity relation {times}. What is attribute_0 "
                            "of the reached entity?"
                        ),
                        "answer": target_entity["attributes"]["attribute_0"],
                        "support_fact_ids": support_fact_ids,
                    }
                )

    validate_support_references(records)
    return records


def validate_support_references(
    records: dict[str, list[dict[str, Any]]]
) -> None:
    h0_ids = {record["id"] for record in records["H0"]}
    if len(h0_ids) != len(records["H0"]):
        raise ValueError("duplicate H0 IDs")
    all_ids: set[str] = set(h0_ids)
    for hop in (1, 2, 3):
        hop_records = records[f"H{hop}"]
        hop_ids = {record["id"] for record in hop_records}
        if len(hop_ids) != len(hop_records) or all_ids.intersection(hop_ids):
            raise ValueError(f"duplicate or colliding H{hop} IDs")
        all_ids.update(hop_ids)
        for record in hop_records:
            support_ids = record.get("support_fact_ids")
            if not isinstance(support_ids, list) or len(support_ids) != hop + 1:
                raise ValueError(f"H{hop} record has an invalid support count")
            if any(support_id not in h0_ids for support_id in support_ids):
                raise ValueError(f"H{hop} record references an unknown H0 support")


def generate_condition_qa(
    config: dict[str, Any],
    database_path: str | Path,
    database_manifest_path: str | Path,
    output_paths: dict[str, str | Path],
    *,
    expected_table_count: int,
    expected_logical_fact_count: int | None = None,
) -> dict[str, Any]:
    chains, database_manifest = load_verified_logical_chains(
        database_path,
        database_manifest_path,
        expected_table_count=expected_table_count,
    )
    if (
        expected_logical_fact_count is not None
        and database_manifest["requested_N"] != expected_logical_fact_count
    ):
        raise ValueError("database logical fact count does not match the condition")
    records = generate_qa_records(chains)
    required_outputs = {"H0", "H1", "H2", "H3"}
    if set(output_paths) != required_outputs:
        raise ValueError("output_paths must contain exactly H0, H1, H2, and H3")
    for hop_name in ("H0", "H1", "H2", "H3"):
        write_jsonl(output_paths[hop_name], records[hop_name])

    h0_attribute_count = sum(
        record["type"] == "attribute" for record in records["H0"]
    )
    h0_relation_count = sum(
        record["type"] == "relation" for record in records["H0"]
    )
    return {
        "format_version": QA_FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "T": database_manifest["T"],
        "requested_N": database_manifest["requested_N"],
        "selected_chain_count": database_manifest["selected_chain_count"],
        "logical_content_sha256": database_manifest["logical_content_sha256"],
        "source_database_sha256": hash_file(database_path),
        "source_database_manifest_sha256": hash_file(database_manifest_path),
        "h0_definition": "attribute_and_direct_relation_facts",
        "h0_attribute_count": h0_attribute_count,
        "h0_relation_count": h0_relation_count,
        "H0_count": len(records["H0"]),
        "H1_count": len(records["H1"]),
        "H2_count": len(records["H2"]),
        "H3_count": len(records["H3"]),
        "H1_supports_per_question": 2,
        "H2_supports_per_question": 3,
        "H3_supports_per_question": 4,
        "question_template_version": QUESTION_TEMPLATE_VERSION,
        "generation_order": (
            "physical insertion order; positions ascending; attributes ascending"
        ),
        "H0_sha256": hash_file(output_paths["H0"]),
        "H1_sha256": hash_file(output_paths["H1"]),
        "H2_sha256": hash_file(output_paths["H2"]),
        "H3_sha256": hash_file(output_paths["H3"]),
        "total_QA_count": sum(len(hop_records) for hop_records in records.values()),
    }

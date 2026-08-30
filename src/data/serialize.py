import re
import sqlite3
from pathlib import Path
from typing import Any

from utils.hashing import hash_file, hash_text
from utils.io import read_json, write_text


SERIALIZATION_FORMAT_VERSION = 1
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


def inspect_database_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    table_names = _user_table_names(connection)
    if not table_names:
        raise ValueError("database contains no user tables")

    tables: list[dict[str, Any]] = []
    column_locations: dict[str, str] = {}
    foreign_keys: dict[tuple[str, str], tuple[str, str]] = {}

    for table_name in table_names:
        table_info = connection.execute(
            f"PRAGMA table_info({_quote(table_name)})"
        ).fetchall()
        columns = [row[1] for row in table_info]
        if not columns:
            raise ValueError(f"table has no columns: {table_name}")
        for column in columns:
            if column in column_locations:
                raise ValueError(f"logical column occurs in multiple tables: {column}")
            column_locations[column] = table_name

        primary_key_columns = [
            row[1] for row in sorted(table_info, key=lambda row: row[5]) if row[5]
        ]
        if len(primary_key_columns) != 1:
            raise ValueError(f"table must have exactly one primary key: {table_name}")

        table_foreign_keys = connection.execute(
            f"PRAGMA foreign_key_list({_quote(table_name)})"
        ).fetchall()
        for foreign_key in table_foreign_keys:
            source_column = foreign_key[3]
            key = (table_name, source_column)
            if key in foreign_keys:
                raise ValueError(f"duplicate foreign key for {table_name}.{source_column}")
            foreign_keys[key] = (foreign_key[2], foreign_key[4])

        tables.append(
            {
                "name": table_name,
                "columns": columns,
                "primary_key": primary_key_columns[0],
            }
        )

    relationships: list[dict[str, str | int]] = []
    for table in tables:
        source_table = table["name"]
        for source_column in table["columns"]:
            match = RELATION_COLUMN_PATTERN.fullmatch(source_column)
            if match is None:
                continue
            source_position = int(match.group(1))
            if source_position <= 0:
                raise ValueError(f"invalid relation column: {source_column}")
            target_column = f"p{source_position - 1:02d}_entity_id"
            target_table = column_locations.get(target_column)
            if target_table is None:
                raise ValueError(f"relation target column is missing: {target_column}")

            foreign_key_target = foreign_keys.get((source_table, source_column))
            if source_table == target_table:
                if foreign_key_target is not None:
                    raise ValueError("intra-table relation must not be a foreign key")
                relationship_type = "RELATION"
            else:
                if foreign_key_target != (target_table, target_column):
                    raise ValueError(
                        f"cross-table relation lacks the expected foreign key: "
                        f"{source_table}.{source_column}"
                    )
                relationship_type = "FOREIGN_KEY"

            relationships.append(
                {
                    "source_position": source_position,
                    "type": relationship_type,
                    "source_table": source_table,
                    "source_column": source_column,
                    "target_table": target_table,
                    "target_column": target_column,
                }
            )

    relationships.sort(key=lambda relationship: relationship["source_position"])
    relationship_foreign_keys = {
        (relationship["source_table"], relationship["source_column"])
        for relationship in relationships
        if relationship["type"] == "FOREIGN_KEY"
    }
    if set(foreign_keys) != relationship_foreign_keys:
        raise ValueError("database contains an unexpected foreign key")
    if [relationship["source_position"] for relationship in relationships] != list(
        range(1, len(relationships) + 1)
    ):
        raise ValueError("database relationships are not complete and contiguous")

    return {
        "tables": tables,
        "relationships": relationships,
        "table_count": len(tables),
        "total_schema_column_count": sum(
            len(table["columns"]) for table in tables
        ),
        "schema_relationship_count": len(relationships),
        "schema_foreign_key_count": sum(
            relationship["type"] == "FOREIGN_KEY"
            for relationship in relationships
        ),
        "schema_intra_table_relation_count": sum(
            relationship["type"] == "RELATION"
            for relationship in relationships
        ),
    }


def build_database_serialization_block(
    database_path: str | Path,
) -> tuple[str, dict[str, Any]]:
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"database file not found: {path}")

    connection = sqlite3.connect(path)
    try:
        schema = inspect_database_schema(connection)
        lines = ["BEGIN_DATABASE", "SCHEMA"]
        for table in schema["tables"]:
            lines.append(f"TABLE {table['name']}")
            lines.append(f"COLUMNS {' | '.join(table['columns'])}")
            lines.append(
                f"PRIMARY_KEY {table['name']}.{table['primary_key']}"
            )

        for relationship in schema["relationships"]:
            lines.append(
                f"{relationship['type']} "
                f"{relationship['source_table']}.{relationship['source_column']} -> "
                f"{relationship['target_table']}.{relationship['target_column']}"
            )

        lines.append("DATA")
        logical_fact_occurrences = 0
        physical_row_count = 0
        for table in schema["tables"]:
            table_name = table["name"]
            columns = table["columns"]
            selected_columns = ", ".join(_quote(column) for column in columns)
            rows = connection.execute(
                f"SELECT {selected_columns} FROM {_quote(table_name)} ORDER BY rowid"
            )
            for row in rows:
                if any(value is None for value in row):
                    raise ValueError("database contains a null logical value")
                if not all(isinstance(value, str) for value in row):
                    raise ValueError("all serialized logical values must be text")
                fields = [
                    f"{column}={value}"
                    for column, value in zip(columns, row, strict=True)
                ]
                lines.append(f"ROW {table_name} | {' | '.join(fields)}")
                logical_fact_occurrences += len(fields)
                physical_row_count += 1

        lines.append("END_DATABASE")
        block = "\n".join(lines) + "\n"
    finally:
        connection.close()

    metadata = {
        "logical_fact_occurrences": logical_fact_occurrences,
        "physical_row_count": physical_row_count,
        "row_line_count": physical_row_count,
        "table_count": schema["table_count"],
        "total_schema_column_count": schema["total_schema_column_count"],
        "schema_relationship_count": schema["schema_relationship_count"],
        "schema_foreign_key_count": schema["schema_foreign_key_count"],
        "schema_intra_table_relation_count": schema[
            "schema_intra_table_relation_count"
        ],
    }
    return block, metadata


def serialize_database_cpt(
    config: dict[str, Any],
    database_path: str | Path,
    database_manifest_path: str | Path,
    train_text_path: str | Path,
) -> dict[str, Any]:
    database_path = Path(database_path)
    database_manifest_path = Path(database_manifest_path)
    train_text_path = Path(train_text_path)
    if not database_manifest_path.is_file():
        raise FileNotFoundError(
            f"database manifest not found: {database_manifest_path}"
        )

    database_manifest = read_json(database_manifest_path)
    database_sha256 = hash_file(database_path)
    if database_manifest.get("database_sha256") != database_sha256:
        raise ValueError("database does not match its manifest hash")

    requested_n = database_manifest.get("requested_N")
    actual_n = database_manifest.get("actual_logical_fact_count")
    table_count = database_manifest.get("table_count")
    if isinstance(requested_n, bool) or not isinstance(requested_n, int) or requested_n <= 0:
        raise ValueError("database manifest requested_N must be a positive integer")
    if actual_n != requested_n:
        raise ValueError("database manifest logical fact counts are inconsistent")
    if isinstance(table_count, bool) or not isinstance(table_count, int) or table_count <= 0:
        raise ValueError("database manifest table_count must be a positive integer")
    if database_manifest.get("T") != table_count:
        raise ValueError("database manifest T and table_count are inconsistent")

    fact_exposure = config["training"]["fact_exposure"]
    if (
        isinstance(fact_exposure, bool)
        or not isinstance(fact_exposure, int)
        or fact_exposure <= 0
    ):
        raise ValueError("training.fact_exposure must be a positive integer")

    block, block_metadata = build_database_serialization_block(database_path)
    if block_metadata["table_count"] != table_count:
        raise ValueError("physical table count does not match the database manifest")
    if block_metadata["logical_fact_occurrences"] != requested_n:
        raise ValueError("serialized logical fact count does not match requested_N")
    if block_metadata["physical_row_count"] != database_manifest.get(
        "physical_row_count"
    ):
        raise ValueError("physical row count does not match the database manifest")
    if block_metadata["total_schema_column_count"] != database_manifest.get(
        "atomic_facts_per_chain"
    ):
        raise ValueError("schema column count does not match atomic facts per chain")
    expected_relationship_count = database_manifest.get("relation_facts_per_chain")
    if block_metadata["schema_relationship_count"] != expected_relationship_count:
        raise ValueError("schema relationship count does not match the manifest")
    expected_foreign_key_count = database_manifest.get("cross_table_fk_edge_count")
    if block_metadata["schema_foreign_key_count"] != expected_foreign_key_count:
        raise ValueError("schema foreign-key count does not match the manifest")
    if block_metadata["schema_intra_table_relation_count"] != (
        expected_relationship_count - expected_foreign_key_count
    ):
        raise ValueError("schema intra-table relation count does not match the manifest")

    train_text = block * fact_exposure
    write_text(train_text_path, train_text)
    train_bytes = train_text.encode("utf-8")

    return {
        "format_version": SERIALIZATION_FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "T": table_count,
        "requested_N": requested_n,
        "logical_content_sha256": database_manifest["logical_content_sha256"],
        "source_database_sha256": database_sha256,
        "source_database_manifest_sha256": hash_file(database_manifest_path),
        "fact_exposure": fact_exposure,
        "logical_facts_per_exposure": block_metadata[
            "logical_fact_occurrences"
        ],
        "serialized_logical_fact_occurrences": (
            block_metadata["logical_fact_occurrences"] * fact_exposure
        ),
        "physical_rows_per_exposure": block_metadata["physical_row_count"],
        "serialized_row_line_count": block_metadata["row_line_count"]
        * fact_exposure,
        "table_count": block_metadata["table_count"],
        "total_schema_column_count": block_metadata[
            "total_schema_column_count"
        ],
        "schema_relationship_count_per_exposure": block_metadata[
            "schema_relationship_count"
        ],
        "schema_foreign_key_count_per_exposure": block_metadata[
            "schema_foreign_key_count"
        ],
        "schema_intra_table_relation_count_per_exposure": block_metadata[
            "schema_intra_table_relation_count"
        ],
        "serialization_order": {
            "tables": "ascending table name",
            "rows": "ascending SQLite rowid; rowid is not serialized",
            "columns": "physical column order",
            "exposures": "identical complete database blocks",
        },
        "train_text_sha256": hash_text(train_text),
        "train_text_byte_count": len(train_bytes),
        "train_text_character_count": len(train_text),
        "train_text_line_count": train_text.count("\n"),
    }

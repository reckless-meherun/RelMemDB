import sqlite3
from pathlib import Path
from typing import Any

from utils.hashing import hash_file, hash_text
from utils.io import read_json, write_text


SERIALIZATION_FORMAT_VERSION = 2


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
            lines.append(f"PRIMARY_KEY {table['name']}.{table['primary_key']}")
        for relationship in schema["relationships"]:
            lines.append(
                f"FOREIGN_KEY {relationship['source_table']}.{relationship['source_column']} -> "
                f"{relationship['target_table']}.{relationship['target_column']}"
            )

        lines.append("DATA")
        logical_fact_occurrences = 0
        identifier_occurrences = 0
        physical_row_count = 0
        for table in schema["tables"]:
            table_name = table["name"]
            columns = table["columns"]
            identifiers = set(table["identifier_columns"])
            selected_columns = ", ".join(_quote(column) for column in columns)
            rows = connection.execute(
                f"SELECT {selected_columns} FROM {_quote(table_name)} ORDER BY rowid"
            )
            for row in rows:
                if any(value is None for value in row):
                    raise ValueError("database contains a null logical value")
                if not all(isinstance(value, (str, int)) and not isinstance(value, bool) for value in row):
                    raise ValueError("serialized logical values must be text or integers")
                fields = [
                    f"{column}={value}" for column, value in zip(columns, row, strict=True)
                ]
                lines.append(f"ROW {table_name} | {' | '.join(fields)}")
                identifier_occurrences += len(identifiers)
                logical_fact_occurrences += len(columns) - len(identifiers)
                physical_row_count += 1
        lines.append("END_DATABASE")
        block = "\n".join(lines) + "\n"
    finally:
        connection.close()

    metadata = {
        "logical_fact_occurrences": logical_fact_occurrences,
        "identifier_occurrences": identifier_occurrences,
        "stored_value_occurrences": logical_fact_occurrences + identifier_occurrences,
        "physical_row_count": physical_row_count,
        "row_line_count": physical_row_count,
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
        raise FileNotFoundError(f"database manifest not found: {database_manifest_path}")

    database_manifest = read_json(database_manifest_path)
    database_sha256 = hash_file(database_path)
    if database_manifest.get("database_sha256") != database_sha256:
        raise ValueError("database does not match its manifest hash")
    requested_n = database_manifest.get("requested_N")
    table_count = database_manifest.get("table_count")
    if isinstance(requested_n, bool) or not isinstance(requested_n, int) or requested_n <= 0:
        raise ValueError("database manifest requested_N must be a positive integer")
    if database_manifest.get("actual_logical_fact_count") != requested_n:
        raise ValueError("database manifest logical fact counts are inconsistent")
    if isinstance(table_count, bool) or not isinstance(table_count, int) or table_count <= 0:
        raise ValueError("database manifest table_count must be a positive integer")
    if database_manifest.get("T") != table_count:
        raise ValueError("database manifest T and table_count are inconsistent")

    block, metadata = build_database_serialization_block(database_path)
    expected = {
        "table_count": table_count,
        "logical_fact_occurrences": requested_n,
        "physical_row_count": database_manifest.get("physical_row_count"),
        "total_schema_column_count": database_manifest.get("schema_column_count"),
        "experimental_schema_column_count": database_manifest.get("experimental_facts_per_chain"),
        "identifier_schema_column_count": database_manifest.get("identifier_fields_per_chain"),
        "schema_relationship_count": database_manifest.get("relation_facts_per_chain"),
        "schema_foreign_key_count": database_manifest.get("schema_foreign_key_count"),
        "schema_cross_table_foreign_key_count": database_manifest.get("cross_table_fk_edge_count"),
        "schema_intra_table_relation_count": database_manifest.get("intra_table_fk_edge_count"),
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise ValueError(f"serialized database {key} does not match its manifest")

    fact_exposure = config["training"]["fact_exposure"]
    if isinstance(fact_exposure, bool) or not isinstance(fact_exposure, int) or fact_exposure <= 0:
        raise ValueError("training.fact_exposure must be a positive integer")
    train_text = block * fact_exposure
    write_text(train_text_path, train_text)
    return {
        "format_version": SERIALIZATION_FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "T": table_count,
        "requested_N": requested_n,
        "logical_content_sha256": database_manifest["logical_content_sha256"],
        "source_database_sha256": database_sha256,
        "source_database_manifest_sha256": hash_file(database_manifest_path),
        "fact_exposure": fact_exposure,
        "logical_facts_per_exposure": metadata["logical_fact_occurrences"],
        "serialized_logical_fact_occurrences": metadata["logical_fact_occurrences"] * fact_exposure,
        "identifier_occurrences_per_exposure": metadata["identifier_occurrences"],
        "physical_rows_per_exposure": metadata["physical_row_count"],
        "serialized_row_line_count": metadata["row_line_count"] * fact_exposure,
        "table_count": metadata["table_count"],
        "total_schema_column_count": metadata["total_schema_column_count"],
        "experimental_schema_column_count": metadata["experimental_schema_column_count"],
        "identifier_schema_column_count": metadata["identifier_schema_column_count"],
        "schema_relationship_count_per_exposure": metadata["schema_relationship_count"],
        "schema_foreign_key_count_per_exposure": metadata["schema_foreign_key_count"],
        "schema_intra_table_relation_count_per_exposure": metadata["schema_intra_table_relation_count"],
        "serialization_order": {
            "tables": "ascending table name",
            "rows": "ascending SQLite rowid; rowid is not serialized",
            "columns": "physical column order",
            "exposures": "identical complete database blocks",
        },
        "train_text_sha256": hash_text(train_text),
        "train_text_byte_count": len(train_text.encode("utf-8")),
        "train_text_character_count": len(train_text),
        "train_text_line_count": train_text.count("\n"),
    }

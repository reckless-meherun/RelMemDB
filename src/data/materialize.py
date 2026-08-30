import sqlite3
from pathlib import Path
from typing import Any

from utils.hashing import hash_json_object


DATABASE_FORMAT_VERSION = 1


def partition_latent_positions(
    latent_positions: int, table_count: int
) -> list[list[int]]:
    for value, name in (
        (latent_positions, "latent_positions"),
        (table_count, "table_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if table_count > latent_positions:
        raise ValueError("table_count must not exceed latent_positions")

    base_size, remainder = divmod(latent_positions, table_count)
    groups: list[list[int]] = []
    start = 0
    for table_index in range(table_count):
        group_size = base_size + (1 if table_index < remainder else 0)
        group = list(range(start, start + group_size))
        groups.append(group)
        start += group_size

    flattened = [position for group in groups for position in group]
    if flattened != list(range(latent_positions)) or any(not group for group in groups):
        raise RuntimeError("invalid latent-position partition")
    return groups


def _table_name(table_index: int) -> str:
    return f"table_{table_index:02d}"


def _entity_column(position: int) -> str:
    return f"p{position:02d}_entity_id"


def _relation_column(position: int) -> str:
    return f"p{position:02d}_previous_entity_id"


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _relation_targets(chain: dict[str, Any]) -> dict[int, str]:
    targets: dict[int, str] = {}
    entities = chain["entities"]
    for relation in chain["relations"]:
        source_position = relation["source_position"]
        target_position = relation["target_position"]
        if source_position != target_position + 1:
            raise ValueError("master-world relation does not connect adjacent positions")
        if relation["source_entity_id"] != entities[source_position]["entity_id"]:
            raise ValueError("master-world relation source does not match its entity")
        if relation["target_entity_id"] != entities[target_position]["entity_id"]:
            raise ValueError("master-world relation target does not match its entity")
        targets[source_position] = relation["target_entity_id"]
    return targets


def _columns_for_group(
    reference_chain: dict[str, Any], positions: list[int]
) -> list[str]:
    columns: list[str] = []
    for position in positions:
        entity = reference_chain["entities"][position]
        if entity["latent_position"] != position:
            raise ValueError("master-world entities are not ordered by latent position")
        columns.append(_entity_column(position))
        columns.extend(
            f"p{position:02d}_{attribute['slot']}"
            for attribute in entity["attributes"]
        )
        if position > 0:
            columns.append(_relation_column(position))
    return columns


def _row_for_group(chain: dict[str, Any], positions: list[int]) -> tuple[str, ...]:
    relation_targets = _relation_targets(chain)
    values: list[str] = []
    for position in positions:
        entity = chain["entities"][position]
        if entity["latent_position"] != position:
            raise ValueError("master-world entities are not ordered by latent position")
        values.append(entity["entity_id"])
        values.extend(attribute["value"] for attribute in entity["attributes"])
        if position > 0:
            if position not in relation_targets:
                raise ValueError(f"missing master-world relation for position {position}")
            values.append(relation_targets[position])
    return tuple(values)


def _create_table_sql(
    reference_chain: dict[str, Any],
    positions: list[int],
    table_index: int,
) -> str:
    table_name = _table_name(table_index)
    columns = _columns_for_group(reference_chain, positions)
    definitions = [f"{_quote(column)} TEXT NOT NULL" for column in columns]
    definitions.append(f"PRIMARY KEY ({_quote(_entity_column(positions[-1]))})")

    for position in positions:
        if position > positions[0]:
            definitions.append(
                f"CHECK ({_quote(_relation_column(position))} = "
                f"{_quote(_entity_column(position - 1))})"
            )

    if table_index > 0:
        first_position = positions[0]
        previous_position = first_position - 1
        definitions.append(
            f"FOREIGN KEY ({_quote(_relation_column(first_position))}) "
            f"REFERENCES {_quote(_table_name(table_index - 1))} "
            f"({_quote(_entity_column(previous_position))})"
        )

    return f"CREATE TABLE {_quote(table_name)} ({', '.join(definitions)})"


def _validate_selected_chains(
    chains: list[dict[str, Any]], latent_positions: int
) -> None:
    expected_positions = list(range(latent_positions))
    reference_attribute_slots: list[list[str]] | None = None
    for chain in chains:
        entities = chain.get("entities")
        if not isinstance(entities, list) or len(entities) != latent_positions:
            raise ValueError("each selected chain must contain every latent position")
        positions = [entity.get("latent_position") for entity in entities]
        if positions != expected_positions:
            raise ValueError("selected chain positions must be complete and ordered")
        attribute_slots = [
            [attribute["slot"] for attribute in entity["attributes"]]
            for entity in entities
        ]
        if reference_attribute_slots is None:
            reference_attribute_slots = attribute_slots
        elif attribute_slots != reference_attribute_slots:
            raise ValueError("attribute slots must be consistent across selected chains")
        relation_targets = _relation_targets(chain)
        if sorted(relation_targets) != list(range(1, latent_positions)):
            raise ValueError("each selected chain must contain all adjacent relations")


def _check_database_integrity(connection: sqlite3.Connection) -> None:
    foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_violations:
        raise RuntimeError(f"foreign-key violations: {foreign_key_violations}")
    integrity_result = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity_result != ("ok",):
        raise RuntimeError(f"SQLite integrity check failed: {integrity_result}")


def materialize_database(
    master_world: dict[str, Any],
    table_count: int,
    logical_fact_count: int,
    output_path: str | Path,
) -> dict[str, Any]:
    if (
        isinstance(logical_fact_count, bool)
        or not isinstance(logical_fact_count, int)
        or logical_fact_count <= 0
    ):
        raise ValueError("logical_fact_count must be a positive integer")

    construction = master_world["construction"]
    latent_positions = construction["latent_positions"]
    atomic_facts_per_chain = construction["atomic_facts_per_chain"]
    if logical_fact_count % atomic_facts_per_chain != 0:
        raise ValueError("logical_fact_count must be divisible by atomic_facts_per_chain")
    selected_chain_count = logical_fact_count // atomic_facts_per_chain
    available_chains = master_world["chains"]
    if selected_chain_count > len(available_chains):
        raise ValueError("master world does not contain enough chains")

    selected_chains = available_chains[:selected_chain_count]
    _validate_selected_chains(selected_chains, latent_positions)
    partition = partition_latent_positions(latent_positions, table_count)

    database_path = Path(output_path)
    if not database_path.parent.is_dir():
        raise FileNotFoundError(
            f"database parent directory does not exist: {database_path.parent}"
        )
    database_path.write_bytes(b"")

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("BEGIN")

        reference_chain = selected_chains[0]
        for table_index, positions in enumerate(partition):
            table_name = _table_name(table_index)
            columns = _columns_for_group(reference_chain, positions)
            connection.execute(
                _create_table_sql(reference_chain, positions, table_index)
            )
            placeholders = ", ".join("?" for _ in columns)
            insert_sql = (
                f"INSERT INTO {_quote(table_name)} "
                f"({', '.join(_quote(column) for column in columns)}) "
                f"VALUES ({placeholders})"
            )
            rows = [_row_for_group(chain, positions) for chain in selected_chains]
            connection.executemany(insert_sql, rows)

        connection.commit()
        if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
            raise RuntimeError("SQLite foreign-key enforcement is not enabled")
        _check_database_integrity(connection)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    relation_facts_per_chain = construction["relation_facts_per_chain"]
    cross_table_edges = table_count - 1
    rows_per_table = {
        _table_name(table_index): selected_chain_count
        for table_index in range(table_count)
    }
    entity_identifier_facts_per_chain = construction[
        "entity_identifier_facts_per_chain"
    ]
    attribute_facts_per_chain = construction["attribute_facts_per_chain"]
    return {
        "format_version": DATABASE_FORMAT_VERSION,
        "table_count": table_count,
        "requested_logical_fact_count": logical_fact_count,
        "actual_logical_fact_count": selected_chain_count
        * atomic_facts_per_chain,
        "selected_chain_count": selected_chain_count,
        "latent_positions": latent_positions,
        "atomic_facts_per_chain": atomic_facts_per_chain,
        "entity_identifier_facts_per_chain": entity_identifier_facts_per_chain,
        "attribute_facts_per_chain": attribute_facts_per_chain,
        "relation_facts_per_chain": relation_facts_per_chain,
        "entity_identifier_fact_count": entity_identifier_facts_per_chain
        * selected_chain_count,
        "attribute_fact_count": attribute_facts_per_chain * selected_chain_count,
        "relation_fact_count": relation_facts_per_chain * selected_chain_count,
        "cross_table_fk_edge_count": cross_table_edges,
        "physical_row_count": selected_chain_count * table_count,
        "rows_per_table": rows_per_table,
        "position_partition": partition,
        "cross_table_fk_instance_count": cross_table_edges
        * selected_chain_count,
        "intra_table_relation_instance_count": (
            relation_facts_per_chain - cross_table_edges
        )
        * selected_chain_count,
        "logical_content_sha256": hash_json_object(selected_chains),
    }


def build_database_manifest(
    config: dict[str, Any],
    materialization: dict[str, Any],
    *,
    sweep: str,
    master_world_sha256: str,
    configuration_sha256: str,
    database_sha256: str,
) -> dict[str, Any]:
    if not isinstance(sweep, str) or not sweep:
        raise ValueError("sweep must be a non-empty string")

    return {
        "format_version": materialization["format_version"],
        "experiment_name": config["experiment"]["name"],
        "sweep": sweep,
        "T": materialization["table_count"],
        "requested_N": materialization["requested_logical_fact_count"],
        "actual_logical_fact_count": materialization[
            "actual_logical_fact_count"
        ],
        "selected_chain_count": materialization["selected_chain_count"],
        "latent_positions": materialization["latent_positions"],
        "atomic_facts_per_chain": materialization["atomic_facts_per_chain"],
        "entity_identifier_facts_per_chain": materialization[
            "entity_identifier_facts_per_chain"
        ],
        "attribute_facts_per_chain": materialization[
            "attribute_facts_per_chain"
        ],
        "relation_facts_per_chain": materialization[
            "relation_facts_per_chain"
        ],
        "entity_identifier_fact_count": materialization[
            "entity_identifier_fact_count"
        ],
        "attribute_fact_count": materialization["attribute_fact_count"],
        "relation_fact_count": materialization["relation_fact_count"],
        "table_count": materialization["table_count"],
        "cross_table_fk_edge_count": materialization[
            "cross_table_fk_edge_count"
        ],
        "physical_row_count": materialization["physical_row_count"],
        "rows_per_table": materialization["rows_per_table"],
        "position_partition": materialization["position_partition"],
        "master_world_sha256": master_world_sha256,
        "configuration_sha256": configuration_sha256,
        "logical_content_sha256": materialization["logical_content_sha256"],
        "database_sha256": database_sha256,
        "cross_table_fk_instance_count": materialization[
            "cross_table_fk_instance_count"
        ],
        "intra_table_relation_instance_count": materialization[
            "intra_table_relation_instance_count"
        ],
    }

import sqlite3
from pathlib import Path
from typing import Any

from data.world import SEMANTIC_ENTITY_SPECS
from utils.hashing import hash_json_object


DATABASE_FORMAT_VERSION = 2


def partition_latent_positions(
    latent_positions: int, table_count: int
) -> list[list[int]]:
    for value, name in ((latent_positions, "latent_positions"), (table_count, "table_count")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if table_count > latent_positions:
        raise ValueError("table_count must not exceed latent_positions")

    base_size, remainder = divmod(latent_positions, table_count)
    groups: list[list[int]] = []
    start = 0
    for table_index in range(table_count):
        group_size = base_size + (1 if table_index < remainder else 0)
        groups.append(list(range(start, start + group_size)))
        start += group_size
    if [position for group in groups for position in group] != list(range(latent_positions)):
        raise RuntimeError("invalid latent-position partition")
    return groups


def _quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _table_name(positions: list[int]) -> str:
    return "__".join(SEMANTIC_ENTITY_SPECS[position]["entity_type"] for position in positions)


def _column_name(position: int, logical_name: str, positions: list[int]) -> str:
    if len(positions) == 1:
        return logical_name
    entity_type = SEMANTIC_ENTITY_SPECS[position]["entity_type"]
    return f"{entity_type}__{logical_name}"


def _column_descriptors(positions: list[int]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for position in positions:
        spec = SEMANTIC_ENTITY_SPECS[position]
        descriptors.append(
            {
                "position": position,
                "kind": "identifier",
                "logical_name": spec["id_field"],
                "name": _column_name(position, spec["id_field"], positions),
                "type": "TEXT",
            }
        )
        descriptors.extend(
            {
                "position": position,
                "kind": "attribute",
                "logical_name": attribute_name,
                "name": _column_name(position, attribute_name, positions),
                "type": sqlite_type,
            }
            for attribute_name, sqlite_type in spec["attributes"]
        )
        if position > 0:
            foreign_key = spec["foreign_key"]
            descriptors.append(
                {
                    "position": position,
                    "kind": "relation",
                    "logical_name": foreign_key,
                    "name": _column_name(position, foreign_key, positions),
                    "type": "TEXT",
                }
            )
    return descriptors


def _layout(partition: list[list[int]]) -> dict[int, dict[str, Any]]:
    layout: dict[int, dict[str, Any]] = {}
    for positions in partition:
        table_name = _table_name(positions)
        for position in positions:
            layout[position] = {"table": table_name, "positions": positions}
    return layout


def _create_table_sql(positions: list[int], layout: dict[int, dict[str, Any]]) -> str:
    table_name = _table_name(positions)
    descriptors = _column_descriptors(positions)
    definitions = [
        f"{_quote(descriptor['name'])} {descriptor['type']} NOT NULL"
        for descriptor in descriptors
    ]
    primary_position = positions[-1]
    primary_spec = SEMANTIC_ENTITY_SPECS[primary_position]
    primary_column = _column_name(primary_position, primary_spec["id_field"], positions)
    definitions.append(f"PRIMARY KEY ({_quote(primary_column)})")

    for position in positions[:-1]:
        spec = SEMANTIC_ENTITY_SPECS[position]
        identifier_column = _column_name(position, spec["id_field"], positions)
        definitions.append(f"UNIQUE ({_quote(identifier_column)})")

    for source_position in positions:
        if source_position == 0:
            continue
        source_spec = SEMANTIC_ENTITY_SPECS[source_position]
        source_column = _column_name(source_position, source_spec["foreign_key"], positions)
        target_position = source_position - 1
        target_layout = layout[target_position]
        target_positions = target_layout["positions"]
        target_spec = SEMANTIC_ENTITY_SPECS[target_position]
        target_column = _column_name(target_position, target_spec["id_field"], target_positions)
        definitions.append(
            f"FOREIGN KEY ({_quote(source_column)}) REFERENCES "
            f"{_quote(target_layout['table'])} ({_quote(target_column)})"
        )
    return f"CREATE TABLE {_quote(table_name)} ({', '.join(definitions)})"


def _relation_targets(chain: dict[str, Any]) -> dict[int, str]:
    entities = chain["entities"]
    targets: dict[int, str] = {}
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


def _row_for_group(chain: dict[str, Any], positions: list[int]) -> tuple[Any, ...]:
    relation_targets = _relation_targets(chain)
    values: list[Any] = []
    for position in positions:
        entity = chain["entities"][position]
        values.append(entity["entity_id"])
        values.extend(attribute["value"] for attribute in entity["attributes"])
        if position > 0:
            values.append(relation_targets[position])
    return tuple(values)


def _validate_selected_chains(chains: list[dict[str, Any]]) -> None:
    if not chains:
        raise ValueError("at least one chain must be selected")
    expected_positions = list(range(len(SEMANTIC_ENTITY_SPECS)))
    seen_identifiers: set[str] = set()
    for chain in chains:
        entities = chain.get("entities")
        if not isinstance(entities, list) or len(entities) != len(SEMANTIC_ENTITY_SPECS):
            raise ValueError("each selected chain must contain every semantic position")
        if [entity.get("position") for entity in entities] != expected_positions:
            raise ValueError("selected chain positions must be complete and ordered")
        for position, (entity, spec) in enumerate(zip(entities, SEMANTIC_ENTITY_SPECS, strict=True)):
            if entity.get("entity_type") != spec["entity_type"]:
                raise ValueError(f"wrong entity type at position {position}")
            expected_attributes = [name for name, _ in spec["attributes"]]
            if [attribute.get("name") for attribute in entity.get("attributes", [])] != expected_attributes:
                raise ValueError(f"wrong attributes for {spec['entity_type']}")
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or entity_id in seen_identifiers:
                raise ValueError("entity identifiers must be non-empty and globally unique")
            seen_identifiers.add(entity_id)
        if sorted(_relation_targets(chain)) != list(range(1, len(SEMANTIC_ENTITY_SPECS))):
            raise ValueError("each selected chain must contain all adjacent relations")


def _check_database_integrity(connection: sqlite3.Connection) -> None:
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"foreign-key violations: {violations}")
    result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError(f"SQLite integrity check failed: {result}")


def materialize_database(
    master_world: dict[str, Any],
    table_count: int,
    logical_fact_count: int,
    output_path: str | Path,
) -> dict[str, Any]:
    if isinstance(logical_fact_count, bool) or not isinstance(logical_fact_count, int) or logical_fact_count <= 0:
        raise ValueError("logical_fact_count must be a positive integer")
    construction = master_world["construction"]
    facts_per_chain = construction["experimental_facts_per_chain"]
    if logical_fact_count % facts_per_chain != 0:
        raise ValueError("logical_fact_count must be divisible by experimental_facts_per_chain")
    selected_chain_count = logical_fact_count // facts_per_chain
    selected_chains = master_world["chains"][:selected_chain_count]
    if len(selected_chains) != selected_chain_count:
        raise ValueError("master world does not contain enough chains")
    _validate_selected_chains(selected_chains)

    partition = partition_latent_positions(len(SEMANTIC_ENTITY_SPECS), table_count)
    layout = _layout(partition)
    database_path = Path(output_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.write_bytes(b"")

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("BEGIN")
        for positions in partition:
            connection.execute(_create_table_sql(positions, layout))
        for positions in partition:
            table_name = _table_name(positions)
            columns = [descriptor["name"] for descriptor in _column_descriptors(positions)]
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {_quote(table_name)} "
                f"({', '.join(_quote(column) for column in columns)}) VALUES ({placeholders})",
                [_row_for_group(chain, positions) for chain in selected_chains],
            )
        connection.commit()
        _check_database_integrity(connection)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    rows_per_table = {_table_name(positions): selected_chain_count for positions in partition}
    relation_facts_per_chain = construction["relation_facts_per_chain"]
    cross_table_edges = table_count - 1
    return {
        "format_version": DATABASE_FORMAT_VERSION,
        "table_count": table_count,
        "requested_logical_fact_count": logical_fact_count,
        "actual_logical_fact_count": selected_chain_count * facts_per_chain,
        "selected_chain_count": selected_chain_count,
        "latent_positions": len(SEMANTIC_ENTITY_SPECS),
        "experimental_facts_per_chain": facts_per_chain,
        "identifier_fields_per_chain": construction["identifier_fields_per_chain"],
        "attribute_facts_per_chain": construction["attribute_facts_per_chain"],
        "relation_facts_per_chain": relation_facts_per_chain,
        "identifier_field_count": construction["identifier_fields_per_chain"] * selected_chain_count,
        "attribute_fact_count": construction["attribute_facts_per_chain"] * selected_chain_count,
        "relation_fact_count": relation_facts_per_chain * selected_chain_count,
        "identifiers_counted_as_experimental_facts": False,
        "stored_value_count": (facts_per_chain + construction["identifier_fields_per_chain"]) * selected_chain_count,
        "schema_column_count": facts_per_chain + construction["identifier_fields_per_chain"],
        "schema_foreign_key_count": relation_facts_per_chain,
        "cross_table_fk_edge_count": cross_table_edges,
        "intra_table_fk_edge_count": relation_facts_per_chain - cross_table_edges,
        "physical_row_count": selected_chain_count * table_count,
        "rows_per_table": rows_per_table,
        "position_partition": partition,
        "cross_table_fk_instance_count": cross_table_edges * selected_chain_count,
        "intra_table_fk_instance_count": (relation_facts_per_chain - cross_table_edges) * selected_chain_count,
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
        **{
            key: value
            for key, value in materialization.items()
            if key not in {"format_version", "requested_logical_fact_count"}
        },
        "master_world_sha256": master_world_sha256,
        "configuration_sha256": configuration_sha256,
        "database_sha256": database_sha256,
    }

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from config import ConfigError, load_config, validate_config
from data.materialize import (
    build_database_manifest,
    materialize_database,
    partition_latent_positions,
)
from data.qa import (
    generate_condition_qa,
    generate_qa_records,
    load_verified_logical_chains,
)
from data.serialize import (
    build_database_serialization_block,
    serialize_database_cpt,
)
from data.world import build_master_world, derive_master_world_counts
from utils.hashing import hash_file, hash_json_object
from utils.io import read_text, write_json
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


@pytest.fixture(scope="module")
def default_world(default_config: dict[str, Any]) -> dict[str, Any]:
    return build_master_world(default_config)


def _shape(world: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            chain["chain_index"],
            tuple(
                (
                    entity["latent_position"],
                    tuple(attribute["slot"] for attribute in entity["attributes"]),
                )
                for entity in chain["entities"]
            ),
            tuple(
                (relation["source_position"], relation["target_position"])
                for relation in chain["relations"]
            ),
        )
        for chain in world["chains"]
    ]


def _entity_ids(world: dict[str, Any]) -> list[str]:
    return [
        entity["entity_id"]
        for chain in world["chains"]
        for entity in chain["entities"]
    ]


def _attribute_values(world: dict[str, Any]) -> list[str]:
    return [
        attribute["value"]
        for chain in world["chains"]
        for entity in chain["entities"]
        for attribute in entity["attributes"]
    ]


def _assert_valid_chain(chain: dict[str, Any]) -> None:
    entities = chain["entities"]
    relations = chain["relations"]
    assert len(entities) == 12
    assert [entity["latent_position"] for entity in entities] == list(range(12))

    entity_ids = [entity["entity_id"] for entity in entities]
    assert len(entity_ids) == len(set(entity_ids)) == 12
    assert all(entity["attributes"] for entity in entities)
    assert sum(len(entity["attributes"]) for entity in entities) == 17

    assert len(relations) == 11
    known_ids = set(entity_ids)
    for relation in relations:
        source_position = relation["source_position"]
        target_position = relation["target_position"]
        assert source_position == target_position + 1
        assert relation["source_entity_id"] == entities[source_position]["entity_id"]
        assert relation["target_entity_id"] == entities[target_position]["entity_id"]
        assert relation["source_entity_id"] in known_ids
        assert relation["target_entity_id"] in known_ids


def test_determinism(small_config: dict[str, Any]) -> None:
    world_a = build_master_world(small_config)
    world_b = build_master_world(small_config)

    assert world_a == world_b
    assert hash_json_object(world_a) == hash_json_object(world_b)


def test_seed_sensitivity(small_config: dict[str, Any]) -> None:
    world_a = build_master_world(small_config)
    changed_seed_config = deepcopy(small_config)
    changed_seed_config["experiment"]["seed"] += 1
    world_b = build_master_world(changed_seed_config)

    assert _shape(world_a) == _shape(world_b)
    assert _entity_ids(world_a) != _entity_ids(world_b)
    assert _attribute_values(world_a) != _attribute_values(world_b)
    assert hash_json_object(world_a) != hash_json_object(world_b)


def test_chain_structure_and_atomic_fact_accounting(
    small_world: dict[str, Any],
) -> None:
    for chain in small_world["chains"]:
        _assert_valid_chain(chain)
        entity_fact_count = len(chain["entities"])
        attribute_fact_count = sum(
            len(entity["attributes"]) for entity in chain["entities"]
        )
        relation_fact_count = len(chain["relations"])
        assert (entity_fact_count, attribute_fact_count, relation_fact_count) == (
            12,
            17,
            11,
        )
        assert entity_fact_count + attribute_fact_count + relation_fact_count == 40


def test_default_world_counts(default_world: dict[str, Any]) -> None:
    construction = default_world["construction"]
    assert len(default_world["chains"]) == 1_000
    assert construction["total_chains"] == 1_000
    assert construction["total_logical_atomic_facts"] == 40_000
    assert construction["entity_identifier_facts_per_chain"] == 12
    assert construction["attribute_facts_per_chain"] == 17
    assert construction["relation_facts_per_chain"] == 11
    assert construction["atomic_facts_per_chain"] == 40


def test_nested_n_prefixes(default_world: dict[str, Any]) -> None:
    chains = default_world["chains"]
    n5k = chains[:125]
    n10k = chains[:250]
    n20k = chains[:500]
    n40k = chains[:1000]

    assert n5k == n10k[:125]
    assert n10k == n20k[:250]
    assert n20k == n40k[:500]
    assert len(n5k) < len(n10k) < len(n20k) < len(n40k)


def test_opaque_values_are_globally_unique(default_world: dict[str, Any]) -> None:
    entity_ids = _entity_ids(default_world)
    attribute_values = _attribute_values(default_world)

    assert len(entity_ids) == len(set(entity_ids)) == 12_000
    assert len(attribute_values) == len(set(attribute_values)) == 17_000
    assert set(entity_ids).isdisjoint(attribute_values)


def test_no_broken_relation_references(default_world: dict[str, Any]) -> None:
    for chain in default_world["chains"]:
        entity_ids = {entity["entity_id"] for entity in chain["entities"]}
        for relation in chain["relations"]:
            assert relation["source_entity_id"] in entity_ids
            assert relation["target_entity_id"] in entity_ids


def test_master_world_is_schema_independent(default_world: dict[str, Any]) -> None:
    serialized = json.dumps(default_world, sort_keys=True).lower()
    for forbidden_term in (
        "t4",
        "t8",
        "t12",
        "table_name",
        "sqlite",
        "physical_schema",
    ):
        assert forbidden_term not in serialized


def test_default_config_master_world_constraints(
    default_config: dict[str, Any],
) -> None:
    data = default_config["data"]
    counts = derive_master_world_counts(default_config)
    configured_n_values = [
        data["t_sweep"]["fact_count"],
        *data["n_sweep"]["fact_counts"],
        data["optional_n40k"]["fact_count"],
    ]

    assert data["master_world"] == {
        "latent_positions": 12,
        "atomic_facts_per_chain": 40,
    }
    assert max(data["t_sweep"]["table_counts"]) <= counts["latent_positions"]
    assert max(data["hops"]) + 1 <= counts["latent_positions"]
    assert configured_n_values == [10_000, 5_000, 10_000, 20_000, 40_000]
    assert all(
        fact_count % counts["atomic_facts_per_chain"] == 0
        for fact_count in configured_n_values
    )
    assert counts["maximum_supported_configured_n"] == 40_000


@pytest.mark.parametrize(
    ("field", "value"),
    (("latent_positions", 11), ("atomic_facts_per_chain", 39)),
)
def test_invalid_master_world_config_is_rejected(
    default_config: dict[str, Any], field: str, value: int
) -> None:
    invalid_config = deepcopy(default_config)
    invalid_config["data"]["master_world"][field] = value
    with pytest.raises(ConfigError):
        validate_config(invalid_config)


def _user_tables(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]


def _database_logical_values(database_path: Path) -> Counter[str]:
    values: Counter[str] = Counter()
    with sqlite3.connect(database_path) as connection:
        for table_name in _user_tables(connection):
            for row in connection.execute(f'SELECT * FROM "{table_name}"'):
                values.update(row)
    return values


def _master_world_logical_values(chains: list[dict[str, Any]]) -> Counter[str]:
    values: Counter[str] = Counter()
    for chain in chains:
        for entity in chain["entities"]:
            values.update([entity["entity_id"]])
            values.update(attribute["value"] for attribute in entity["attributes"])
        values.update(
            relation["target_entity_id"] for relation in chain["relations"]
        )
    return values


def test_position_partitioning() -> None:
    t4 = partition_latent_positions(12, 4)
    t8 = partition_latent_positions(12, 8)
    t12 = partition_latent_positions(12, 12)

    assert t4 == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    assert [len(group) for group in t4] == [3, 3, 3, 3]
    assert [len(group) for group in t8] == [2, 2, 2, 2, 1, 1, 1, 1]
    assert [len(group) for group in t12] == [1] * 12

    for partition in (t4, t8, t12):
        assert [position for group in partition for position in group] == list(range(12))
        assert all(group == list(range(group[0], group[-1] + 1)) for group in partition)


@pytest.mark.parametrize("table_count", (4, 8, 12))
def test_physical_schema_rows_facts_and_integrity(
    tmp_path: Path,
    small_world: dict[str, Any],
    table_count: int,
) -> None:
    database_path = tmp_path / f"t{table_count}.sqlite"
    logical_fact_count = 200
    selected_chain_count = logical_fact_count // 40
    metadata = materialize_database(
        small_world,
        table_count=table_count,
        logical_fact_count=logical_fact_count,
        output_path=database_path,
    )

    with sqlite3.connect(database_path) as connection:
        tables = _user_tables(connection)
        assert len(tables) == table_count
        assert sum(
            len(connection.execute(f'PRAGMA foreign_key_list("{table}")').fetchall())
            for table in tables
        ) == table_count - 1

        total_rows = 0
        total_logical_values = 0
        for table_index, (table, positions) in enumerate(
            zip(tables, metadata["position_partition"], strict=True)
        ):
            row_count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            primary_key_columns = [column[1] for column in columns if column[5]]
            assert row_count == selected_chain_count
            assert primary_key_columns == [f"p{positions[-1]:02d}_entity_id"]
            total_rows += row_count
            total_logical_values += row_count * len(columns)

            foreign_keys = connection.execute(
                f'PRAGMA foreign_key_list("{table}")'
            ).fetchall()
            if table_index == 0:
                assert foreign_keys == []
            else:
                assert len(foreign_keys) == 1
                assert foreign_keys[0][2] == tables[table_index - 1]

        assert total_rows == table_count * selected_chain_count
        assert total_logical_values == logical_fact_count
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    assert metadata["actual_logical_fact_count"] == logical_fact_count
    assert metadata["physical_row_count"] == table_count * selected_chain_count


def test_same_logical_content_across_table_counts(
    tmp_path: Path, small_world: dict[str, Any]
) -> None:
    logical_fact_count = 200
    expected_values = _master_world_logical_values(small_world["chains"][:5])
    hashes: set[str] = set()

    for table_count in (4, 8, 12):
        database_path = tmp_path / f"same_content_t{table_count}.sqlite"
        metadata = materialize_database(
            small_world,
            table_count=table_count,
            logical_fact_count=logical_fact_count,
            output_path=database_path,
        )
        hashes.add(metadata["logical_content_sha256"])
        assert _database_logical_values(database_path) == expected_values

    assert len(hashes) == 1


def test_nested_n_materializations(
    tmp_path: Path, default_world: dict[str, Any]
) -> None:
    materialized_values: list[Counter[str]] = []
    for logical_fact_count in (5_000, 10_000, 20_000):
        database_path = tmp_path / f"n{logical_fact_count}.sqlite"
        metadata = materialize_database(
            default_world,
            table_count=8,
            logical_fact_count=logical_fact_count,
            output_path=database_path,
        )
        selected_chains = logical_fact_count // 40
        values = _database_logical_values(database_path)
        assert values == _master_world_logical_values(
            default_world["chains"][:selected_chains]
        )
        assert sum(values.values()) == logical_fact_count
        assert metadata["selected_chain_count"] == selected_chains
        materialized_values.append(values)

    n5k, n10k, n20k = materialized_values
    assert n5k != n10k and n5k <= n10k
    assert n10k != n20k and n10k <= n20k


@pytest.mark.parametrize("table_count", (4, 8, 12))
def test_stored_relations_match_master_world(
    tmp_path: Path,
    small_world: dict[str, Any],
    table_count: int,
) -> None:
    database_path = tmp_path / f"relations_t{table_count}.sqlite"
    metadata = materialize_database(
        small_world,
        table_count=table_count,
        logical_fact_count=200,
        output_path=database_path,
    )
    selected_chains = small_world["chains"][:5]

    with sqlite3.connect(database_path) as connection:
        for table_index, positions in enumerate(metadata["position_partition"]):
            table_name = f"table_{table_index:02d}"
            for position in positions:
                if position == 0:
                    continue
                stored = dict(
                    connection.execute(
                        f'SELECT "p{position:02d}_entity_id", '
                        f'"p{position:02d}_previous_entity_id" '
                        f'FROM "{table_name}"'
                    ).fetchall()
                )
                expected = {
                    chain["entities"][position]["entity_id"]: chain["relations"][
                        position - 1
                    ]["target_entity_id"]
                    for chain in selected_chains
                }
                assert stored == expected


def test_condition_manifest_accounting_and_hashes(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    database_path = tmp_path / "manifest.sqlite"
    metadata = materialize_database(
        small_world,
        table_count=4,
        logical_fact_count=200,
        output_path=database_path,
    )
    manifest = build_database_manifest(
        default_config,
        metadata,
        sweep="t_sweep",
        master_world_sha256="master-hash",
        configuration_sha256="config-hash",
        database_sha256=hash_file(database_path),
    )

    assert manifest["T"] == manifest["table_count"] == 4
    assert manifest["requested_N"] == manifest["actual_logical_fact_count"] == 200
    assert manifest["selected_chain_count"] == 5
    assert manifest["entity_identifier_fact_count"] == 60
    assert manifest["attribute_fact_count"] == 85
    assert manifest["relation_fact_count"] == 55
    assert (
        manifest["entity_identifier_fact_count"]
        + manifest["attribute_fact_count"]
        + manifest["relation_fact_count"]
        == manifest["actual_logical_fact_count"]
    )
    assert manifest["cross_table_fk_edge_count"] == 3
    assert manifest["physical_row_count"] == 20
    assert set(manifest["rows_per_table"].values()) == {5}
    assert manifest["position_partition"] == partition_latent_positions(12, 4)
    assert manifest["cross_table_fk_instance_count"] == 15
    assert manifest["intra_table_relation_instance_count"] == 40
    assert (
        manifest["cross_table_fk_instance_count"]
        + manifest["intra_table_relation_instance_count"]
        == 11 * manifest["selected_chain_count"]
    )
    assert manifest["master_world_sha256"] == "master-hash"
    assert manifest["configuration_sha256"] == "config-hash"
    assert manifest["database_sha256"] == hash_file(database_path)


def test_n10k_reuses_t8_path() -> None:
    assert n_sweep_database_dir(10_000) == t_sweep_database_dir(8)
    assert not (n_sweep_database_dir(5_000).parent / "N10K").exists()


def test_optional_n40k_placeholders_remain_empty(
    default_config: dict[str, Any],
) -> None:
    condition_dir = n_sweep_database_dir(40_000)
    assert default_config["data"]["optional_n40k"]["enabled"] is False
    assert (condition_dir / "database.sqlite").stat().st_size == 0
    assert (condition_dir / "manifest.json").stat().st_size == 0
    assert (condition_dir / "cpt" / "train.txt").stat().st_size == 0
    assert (condition_dir / "cpt" / "manifest.json").stat().st_size == 0


def _serialized_values(serialization_block: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for line in serialization_block.splitlines():
        if not line.startswith("ROW "):
            continue
        for field in line.split(" | ")[1:]:
            _, separator, value = field.partition("=")
            assert separator == "="
            values[value] += 1
    return values


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


def test_cpt_serialization_is_deterministic(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    (
        database_path,
        database_manifest_path,
        train_text_path,
        cpt_manifest_path,
        first_manifest,
    ) = _build_serialized_test_condition(tmp_path, default_config, small_world)
    first_train_bytes = train_text_path.read_bytes()
    first_manifest_bytes = cpt_manifest_path.read_bytes()

    second_manifest = serialize_database_cpt(
        default_config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        train_text_path=train_text_path,
    )
    write_json(cpt_manifest_path, second_manifest)

    assert second_manifest == first_manifest
    assert train_text_path.read_bytes() == first_train_bytes
    assert cpt_manifest_path.read_bytes() == first_manifest_bytes


def test_cpt_exact_exposure_and_repeated_blocks(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    database_path, _, train_text_path, _, manifest = (
        _build_serialized_test_condition(tmp_path, default_config, small_world)
    )
    block, metadata = build_database_serialization_block(database_path)
    train_text = read_text(train_text_path)
    fact_exposure = default_config["training"]["fact_exposure"]

    assert metadata["logical_fact_occurrences"] == 200
    assert sum(
        line.count("=")
        for line in block.splitlines()
        if line.startswith("ROW ")
    ) == 200
    assert train_text == block * fact_exposure
    assert [
        train_text[index * len(block) : (index + 1) * len(block)]
        for index in range(fact_exposure)
    ] == [block] * fact_exposure
    assert manifest["serialized_logical_fact_occurrences"] == 200 * fact_exposure
    assert manifest["serialized_row_line_count"] == (
        metadata["physical_row_count"] * fact_exposure
    )


def test_cpt_data_fidelity(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    database_path, _, _, _, _ = _build_serialized_test_condition(
        tmp_path, default_config, small_world
    )
    block, _ = build_database_serialization_block(database_path)
    assert _serialized_values(block) == _database_logical_values(database_path)


@pytest.mark.parametrize(
    ("table_count", "foreign_key_count", "relation_count"),
    ((4, 3, 8), (8, 7, 4), (12, 11, 0)),
)
def test_cpt_schema_relationship_counts(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
    table_count: int,
    foreign_key_count: int,
    relation_count: int,
) -> None:
    database_path, _, _, _, manifest = _build_serialized_test_condition(
        tmp_path,
        default_config,
        small_world,
        table_count=table_count,
    )
    block, metadata = build_database_serialization_block(database_path)

    assert sum(line.startswith("FOREIGN_KEY ") for line in block.splitlines()) == (
        foreign_key_count
    )
    assert sum(line.startswith("RELATION ") for line in block.splitlines()) == (
        relation_count
    )
    assert metadata["schema_relationship_count"] == 11
    assert manifest["schema_relationship_count_per_exposure"] == 11
    assert manifest["schema_foreign_key_count_per_exposure"] == foreign_key_count
    assert (
        manifest["schema_intra_table_relation_count_per_exposure"]
        == relation_count
    )


def test_cpt_same_n10k_information_across_t() -> None:
    value_multisets = []
    logical_hashes = []
    for table_count in (4, 8, 12):
        condition_dir = t_sweep_database_dir(table_count)
        block, metadata = build_database_serialization_block(
            condition_dir / "database.sqlite"
        )
        database_manifest = json.loads(
            read_text(condition_dir / "manifest.json")
        )
        assert metadata["logical_fact_occurrences"] == 10_000
        value_multisets.append(_serialized_values(block))
        logical_hashes.append(database_manifest["logical_content_sha256"])

    assert value_multisets[0] == value_multisets[1] == value_multisets[2]
    assert len(set(logical_hashes)) == 1


def test_cpt_nested_n_content() -> None:
    conditions = (
        n_sweep_database_dir(5_000),
        t_sweep_database_dir(8),
        n_sweep_database_dir(20_000),
    )
    value_multisets = [
        _serialized_values(
            build_database_serialization_block(
                condition_dir / "database.sqlite"
            )[0]
        )
        for condition_dir in conditions
    ]
    n5k, n10k, n20k = value_multisets
    assert n5k != n10k and n5k <= n10k
    assert n10k != n20k and n10k <= n20k


def test_cpt_has_no_metadata_or_sql_leakage(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    _, _, train_text_path, _, _ = _build_serialized_test_condition(
        tmp_path, default_config, small_world
    )
    train_text = read_text(train_text_path)
    lowercase_text = train_text.lower()

    for forbidden in (
        "chain_index",
        "seed",
        "sha256",
        "master-hash",
        "configuration-hash",
        str(PROJECT_ROOT).lower(),
        "t_sweep",
        "n_sweep",
        "n10k",
        "n5k",
        "n20k",
        "t4/n10k",
        "t8/n10k",
        "t12/n10k",
    ):
        assert forbidden not in lowercase_text
    for sql_statement in ("CREATE TABLE", "SELECT", "INSERT", "PRAGMA"):
        assert sql_statement not in train_text
    assert "rowid" not in lowercase_text


def test_cpt_manifest_accounting(
    tmp_path: Path,
    default_config: dict[str, Any],
    small_world: dict[str, Any],
) -> None:
    (
        database_path,
        database_manifest_path,
        train_text_path,
        _,
        manifest,
    ) = _build_serialized_test_condition(tmp_path, default_config, small_world)
    train_text = read_text(train_text_path)

    assert manifest["format_version"] == 1
    assert manifest["T"] == manifest["table_count"] == 4
    assert manifest["requested_N"] == manifest["logical_facts_per_exposure"] == 200
    assert manifest["fact_exposure"] == default_config["training"]["fact_exposure"]
    assert manifest["serialized_logical_fact_occurrences"] == 800
    assert manifest["physical_rows_per_exposure"] == 20
    assert manifest["serialized_row_line_count"] == 80
    assert manifest["total_schema_column_count"] == 40
    assert manifest["schema_relationship_count_per_exposure"] == 11
    assert manifest["source_database_sha256"] == hash_file(database_path)
    assert manifest["source_database_manifest_sha256"] == hash_file(
        database_manifest_path
    )
    assert manifest["train_text_sha256"] == hash_file(train_text_path)
    assert manifest["train_text_byte_count"] == len(train_text.encode("utf-8"))
    assert manifest["train_text_character_count"] == len(train_text)
    assert manifest["train_text_line_count"] == train_text.count("\n")


def test_no_duplicate_n10k_cpt_directory() -> None:
    assert not (n_sweep_database_dir(5_000).parent / "N10K").exists()


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


def test_preflight_qa_placeholders_remain_empty() -> None:
    preflight_dir = EXP01_QA_DIR / "preflight_relational_qa"
    assert all(path.stat().st_size == 0 for path in preflight_dir.rglob("*") if path.is_file())

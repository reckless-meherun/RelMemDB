from copy import deepcopy
import json
from typing import Any

import pytest

from config import ConfigError, load_config, validate_config
from data.world import build_master_world, derive_master_world_counts
from utils.hashing import hash_json_object


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

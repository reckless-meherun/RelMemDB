from typing import Any

from config import validate_config
from utils.hashing import hash_text


FORMAT_VERSION = 1
TOKEN_HEX_LENGTH = 32


def derive_master_world_counts(config: dict[str, Any]) -> dict[str, int]:
    data = config["data"]
    construction = data["master_world"]
    latent_positions = construction["latent_positions"]
    atomic_facts_per_chain = construction["atomic_facts_per_chain"]
    entity_facts_per_chain = latent_positions
    relation_facts_per_chain = latent_positions - 1
    attribute_facts_per_chain = (
        atomic_facts_per_chain
        - entity_facts_per_chain
        - relation_facts_per_chain
    )
    configured_fact_counts = [
        data["t_sweep"]["fact_count"],
        *data["n_sweep"]["fact_counts"],
        data["optional_n40k"]["fact_count"],
    ]
    maximum_configured_n = max(configured_fact_counts)
    total_chains = maximum_configured_n // atomic_facts_per_chain

    return {
        "latent_positions": latent_positions,
        "atomic_facts_per_chain": atomic_facts_per_chain,
        "entity_identifier_facts_per_chain": entity_facts_per_chain,
        "attribute_facts_per_chain": attribute_facts_per_chain,
        "relation_facts_per_chain": relation_facts_per_chain,
        "total_chains": total_chains,
        "total_logical_atomic_facts": total_chains * atomic_facts_per_chain,
        "maximum_supported_configured_n": maximum_configured_n,
    }


def _opaque_token(
    *,
    prefix: str,
    namespace: str,
    seed: int,
    chain_index: int,
    latent_position: int,
    slot: int | None = None,
) -> str:
    components = [
        "relmemdb-master-world-v1",
        namespace,
        f"seed={seed}",
        f"chain={chain_index}",
        f"position={latent_position}",
    ]
    if slot is not None:
        components.append(f"slot={slot}")
    digest = hash_text("|".join(components))
    return f"{prefix}_{digest[:TOKEN_HEX_LENGTH]}"


def _attribute_counts_by_position(
    latent_positions: int, attribute_facts_per_chain: int
) -> list[int]:
    base_count, extra_count = divmod(attribute_facts_per_chain, latent_positions)
    return [
        base_count + (1 if position < extra_count else 0)
        for position in range(latent_positions)
    ]


def _build_chain(
    *,
    seed: int,
    chain_index: int,
    latent_positions: int,
    attribute_facts_per_chain: int,
) -> dict[str, Any]:
    attribute_counts = _attribute_counts_by_position(
        latent_positions, attribute_facts_per_chain
    )
    entities: list[dict[str, Any]] = []

    for position, attribute_count in enumerate(attribute_counts):
        entity_id = _opaque_token(
            prefix="e",
            namespace="entity-identifier",
            seed=seed,
            chain_index=chain_index,
            latent_position=position,
        )
        attributes = [
            {
                "slot": f"attribute_{slot}",
                "value": _opaque_token(
                    prefix="v",
                    namespace="attribute-value",
                    seed=seed,
                    chain_index=chain_index,
                    latent_position=position,
                    slot=slot,
                ),
            }
            for slot in range(attribute_count)
        ]
        entities.append(
            {
                "latent_position": position,
                "entity_id": entity_id,
                "attributes": attributes,
            }
        )

    relations = [
        {
            "relation_type": "adjacent_reference",
            "source_position": position,
            "target_position": position - 1,
            "source_entity_id": entities[position]["entity_id"],
            "target_entity_id": entities[position - 1]["entity_id"],
        }
        for position in range(1, latent_positions)
    ]

    return {
        "chain_index": chain_index,
        "entities": entities,
        "relations": relations,
    }


def build_master_world(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    counts = derive_master_world_counts(config)
    seed = config["experiment"]["seed"]
    chains = [
        _build_chain(
            seed=seed,
            chain_index=chain_index,
            latent_positions=counts["latent_positions"],
            attribute_facts_per_chain=counts["attribute_facts_per_chain"],
        )
        for chain_index in range(counts["total_chains"])
    ]

    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": seed,
        "construction": {
            "schema_topology": config["data"]["schema_topology"],
            **counts,
        },
        "chains": chains,
    }


def build_master_world_manifest(
    config: dict[str, Any], *, configuration_sha256: str, world_sha256: str
) -> dict[str, Any]:
    counts = derive_master_world_counts(config)
    total_chains = counts["total_chains"]

    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": config["experiment"]["seed"],
        "schema_topology": config["data"]["schema_topology"],
        "latent_positions": counts["latent_positions"],
        "atomic_facts_per_chain": counts["atomic_facts_per_chain"],
        "entity_identifier_facts_per_chain": counts[
            "entity_identifier_facts_per_chain"
        ],
        "attribute_facts_per_chain": counts["attribute_facts_per_chain"],
        "relation_facts_per_chain": counts["relation_facts_per_chain"],
        "total_chain_count": total_chains,
        "total_logical_atomic_fact_count": counts["total_logical_atomic_facts"],
        "total_entity_identifier_count": (
            total_chains * counts["entity_identifier_facts_per_chain"]
        ),
        "total_attribute_fact_count": (
            total_chains * counts["attribute_facts_per_chain"]
        ),
        "total_relation_fact_count": (
            total_chains * counts["relation_facts_per_chain"]
        ),
        "maximum_supported_configured_n": counts[
            "maximum_supported_configured_n"
        ],
        "configuration_sha256": configuration_sha256,
        "world_json_sha256": world_sha256,
    }

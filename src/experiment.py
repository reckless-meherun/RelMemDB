"""Small, shared representation of a first-experiment T/N/L condition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.hashing import hash_file
from utils.io import read_json
from utils.paths import EXP01_GENERATED_DATABASES_DIR


@dataclass(frozen=True)
class ExperimentCondition:
    """The independently variable database and model dimensions."""

    table_count: int
    fact_count: int
    layers: int

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        table_count: int,
        fact_count: int,
        layers: int,
        master_world_manifest_path: str | Path | None = None,
    ) -> ExperimentCondition:
        for value, name in (
            (table_count, "table_count"),
            (fact_count, "fact_count"),
            (layers, "layers"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        master = config["data"]["master_world"]
        latent_positions = master["latent_positions"]
        facts_per_chain = master["experimental_facts_per_chain"]
        if table_count > latent_positions:
            raise ValueError(f"table_count must be between 1 and {latent_positions}")
        if fact_count % facts_per_chain:
            raise ValueError(
                "fact_count must be divisible by "
                "data.master_world.experimental_facts_per_chain"
            )

        manifest_path = (
            Path(master_world_manifest_path)
            if master_world_manifest_path
            else (EXP01_GENERATED_DATABASES_DIR / "master_world" / "manifest.json")
        )
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"master-world manifest is missing or empty: {manifest_path}"
            )
        manifest = read_json(manifest_path)
        total_chains = manifest.get("total_chains")
        if isinstance(total_chains, bool) or not isinstance(total_chains, int):
            raise ValueError("master-world manifest total_chains is invalid")
        required_chains = fact_count // facts_per_chain
        if required_chains > total_chains:
            raise ValueError(
                f"fact_count requires {required_chains} chains but the master world "
                f"contains only {total_chains}"
            )
        return cls(table_count=table_count, fact_count=fact_count, layers=layers)

    @property
    def fact_count_label(self) -> str:
        if self.fact_count % 1000 == 0:
            return f"N{self.fact_count // 1000}K"
        return f"N{self.fact_count}"

    @property
    def label(self) -> str:
        return f"T{self.table_count}_{self.fact_count_label}_L{self.layers}"

    @property
    def checkpoint_suffix(self) -> str:
        return self.label.lower()

    def as_metadata(self) -> dict[str, int]:
        return {
            "table_count": self.table_count,
            "fact_count": self.fact_count,
            "layers": self.layers,
        }


def qa_reference_values(config: dict[str, Any]) -> tuple[int, int]:
    reference = config.get("data", {}).get("qa_reference")
    if not isinstance(reference, dict):
        raise ValueError("data.qa_reference configuration is required")
    table_count = reference.get("table_count")
    fact_count = reference.get("fact_count")
    if (
        isinstance(table_count, bool)
        or not isinstance(table_count, int)
        or table_count <= 0
        or isinstance(fact_count, bool)
        or not isinstance(fact_count, int)
        or fact_count <= 0
    ):
        raise ValueError("data.qa_reference must contain positive T and N values")
    return table_count, fact_count


def verify_checkpoint_layers(
    checkpoint: str | Path, requested_layers: int
) -> dict[str, Any]:
    """Authenticate the declared transformer depth without loading model weights."""
    if (
        isinstance(requested_layers, bool)
        or not isinstance(requested_layers, int)
        or requested_layers <= 0
    ):
        raise ValueError("requested layers must be a positive integer")
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint / "config.json"
    if not checkpoint.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"local checkpoint is missing: {checkpoint}")
    checkpoint_config = read_json(config_path)
    declared = checkpoint_config.get("n_layer")
    if declared is None:
        declared = checkpoint_config.get("num_hidden_layers")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        raise ValueError(
            f"checkpoint config does not declare a valid transformer layer count: "
            f"{config_path}"
        )
    if declared != requested_layers:
        raise ValueError(
            f"requested L{requested_layers} but checkpoint actually has L{declared}: "
            f"{checkpoint}"
        )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_config_sha256": hash_file(config_path),
        "requested_layers": requested_layers,
        "actual_layers": declared,
    }

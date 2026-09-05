"""Small, shared representation of a first-experiment T/N/L condition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.hashing import hash_file
from utils.io import read_json
from utils.paths import BASE_MODELS_DIR, EXP01_GENERATED_DATABASES_DIR

MODEL_REGISTRY = {
    "gpt2": {"relative_path": "gpt2", "native_layers": 12},
}


def configured_model_layers(config: dict[str, Any]) -> int:
    model = config.get("model", {})
    layers = model.get("layers", model.get("native_layers"))
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError("model layers/native_layers must be a positive integer")
    return layers


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


def resolve_model_checkpoint(
    model_name: str = "gpt2", *, source_checkpoint: str | Path | None = None
) -> tuple[Path, int]:
    """Resolve a local-only causal-LM base checkpoint and its native depth."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model must be a non-empty registry name")
    explicit_override = source_checkpoint is not None
    if explicit_override:
        checkpoint = Path(source_checkpoint)
    else:
        entry = MODEL_REGISTRY.get(model_name)
        if entry is None:
            available = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"unsupported model {model_name!r}; registered models: {available}")
        checkpoint = BASE_MODELS_DIR / entry["relative_path"]
    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "config.json"
    if not checkpoint.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"local model checkpoint is missing: {checkpoint}")
    checkpoint_config = read_json(config_path)
    native_layers = checkpoint_config.get(
        "n_layer", checkpoint_config.get("num_hidden_layers")
    )
    if isinstance(native_layers, bool) or not isinstance(native_layers, int) or native_layers <= 0:
        raise ValueError(f"model checkpoint does not declare native layer depth: {config_path}")
    registered = MODEL_REGISTRY.get(model_name)
    if not explicit_override and registered and registered["native_layers"] != native_layers:
        raise ValueError(
            f"registry says {model_name} has L{registered['native_layers']} but local checkpoint has L{native_layers}"
        )
    return checkpoint, native_layers


def load_exp2_dataset_condition(training_data_dir: str | Path) -> dict[str, Any]:
    bundle = Path(training_data_dir).resolve()
    manifest_path = bundle / "manifest.json"
    database_path = bundle / "database.sqlite"
    cpt_manifest_path = bundle / "cpt" / "manifest.json"
    for path, label in (
        (manifest_path, "dataset manifest"),
        (database_path, "dataset database"),
        (cpt_manifest_path, "CPT manifest"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
    manifest = read_json(manifest_path)
    cpt_manifest = read_json(cpt_manifest_path)
    if manifest.get("experiment_name") != "exp02_capacity_boundary":
        raise ValueError("training-data bundle is not an Experiment-2 dataset")
    if manifest.get("database_sha256") != hash_file(database_path):
        raise ValueError("training-data database hash does not match its manifest")
    selected = manifest.get("selected_tables")
    if not isinstance(selected, list) or not selected or manifest.get("T") != len(selected):
        raise ValueError("training-data selected-table/T metadata is invalid")
    if (
        cpt_manifest.get("experiment_name") != manifest.get("experiment_name")
        or cpt_manifest.get("T") != manifest.get("T")
        or cpt_manifest.get("requested_N") != manifest.get("requested_N")
        or cpt_manifest.get("selected_tables") != selected
        or cpt_manifest.get("source_database_sha256") != hash_file(database_path)
        or cpt_manifest.get("source_database_manifest_sha256") != hash_file(manifest_path)
    ):
        raise ValueError("CPT manifest is incompatible with its Experiment-2 dataset bundle")
    fact_count = manifest.get("requested_N")
    layers = manifest.get("facts_per_selected_chain")
    chains = manifest.get("selected_chain_count")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (fact_count, layers, chains)):
        raise ValueError("training-data N/facts-per-chain/chain metadata is invalid")
    if fact_count != layers * chains:
        raise ValueError("training-data logical-fact accounting is inconsistent")
    return {
        "bundle": bundle,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "database": database_path,
        "cpt_dir": bundle / "cpt",
        "cpt_manifest": cpt_manifest_path,
        "T": len(selected),
        "N": fact_count,
        "selected_tables": selected,
    }

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "exp01_first_feasibility.yaml"
)

REQUIRED_SECTIONS = {
    "experiment",
    "model",
    "data",
    "training",
    "evaluation",
    "preflight",
    "layer_study",
}


class ConfigError(ValueError):
    """Raised when an experiment configuration is structurally invalid."""


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _require_int_list(value: Any, name: str, *, allow_zero: bool) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    validator = _require_non_negative_int if allow_zero else _require_positive_int
    return [validator(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing required field: {section_name}.{key}")
    return section[key]


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_SECTIONS - config.keys())
    if missing:
        raise ConfigError(f"missing required configuration sections: {', '.join(missing)}")

    sections = {
        name: _require_mapping(config[name], name) for name in REQUIRED_SECTIONS
    }
    experiment = sections["experiment"]
    model = sections["model"]
    data = sections["data"]
    training = sections["training"]
    evaluation = sections["evaluation"]
    preflight = sections["preflight"]
    layer_study = sections["layer_study"]

    name = _required(experiment, "name", "experiment")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("experiment.name must be a non-empty string")
    _require_non_negative_int(_required(experiment, "seed", "experiment"), "experiment.seed")

    for key in ("layers", "hidden_size", "attention_heads", "context_length"):
        _require_positive_int(_required(model, key, "model"), f"model.{key}")
    for key in ("name", "precision"):
        value = _required(model, key, "model")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"model.{key} must be a non-empty string")

    t_sweep = _require_mapping(_required(data, "t_sweep", "data"), "data.t_sweep")
    n_sweep = _require_mapping(_required(data, "n_sweep", "data"), "data.n_sweep")
    optional_n40k = _require_mapping(
        _required(data, "optional_n40k", "data"), "data.optional_n40k"
    )
    master_world = _require_mapping(
        _required(data, "master_world", "data"), "data.master_world"
    )
    t_sweep_fact_count = _require_positive_int(
        _required(t_sweep, "fact_count", "data.t_sweep"),
        "data.t_sweep.fact_count",
    )
    table_counts = _require_int_list(
        _required(t_sweep, "table_counts", "data.t_sweep"),
        "data.t_sweep.table_counts",
        allow_zero=False,
    )
    n_sweep_table_count = _require_positive_int(
        _required(n_sweep, "table_count", "data.n_sweep"),
        "data.n_sweep.table_count",
    )
    fact_counts = _require_int_list(
        _required(n_sweep, "fact_counts", "data.n_sweep"),
        "data.n_sweep.fact_counts",
        allow_zero=False,
    )
    hops = _require_int_list(
        _required(data, "hops", "data"), "data.hops", allow_zero=True
    )
    for key in ("same_master_world", "nested_n_subsets", "reuse_t8_n10k"):
        _require_bool(_required(data, key, "data"), f"data.{key}")
    topology = _required(data, "schema_topology", "data")
    if not isinstance(topology, str) or not topology.strip():
        raise ConfigError("data.schema_topology must be a non-empty string")
    _require_bool(
        _required(optional_n40k, "enabled", "data.optional_n40k"),
        "data.optional_n40k.enabled",
    )
    optional_fact_count = _require_positive_int(
        _required(optional_n40k, "fact_count", "data.optional_n40k"),
        "data.optional_n40k.fact_count",
    )
    latent_positions = _require_positive_int(
        _required(master_world, "latent_positions", "data.master_world"),
        "data.master_world.latent_positions",
    )
    descriptive_facts_per_chain = _require_positive_int(
        _required(
            master_world, "descriptive_facts_per_chain", "data.master_world"
        ),
        "data.master_world.descriptive_facts_per_chain",
    )
    relation_facts_per_chain = _require_positive_int(
        _required(master_world, "relation_facts_per_chain", "data.master_world"),
        "data.master_world.relation_facts_per_chain",
    )
    experimental_facts_per_chain = _require_positive_int(
        _required(
            master_world, "experimental_facts_per_chain", "data.master_world"
        ),
        "data.master_world.experimental_facts_per_chain",
    )
    identifier_fields_per_chain = _require_positive_int(
        _required(master_world, "identifier_fields_per_chain", "data.master_world"),
        "data.master_world.identifier_fields_per_chain",
    )
    canonical_target = _require_mapping(
        _required(data, "canonical_target", "data"), "data.canonical_target"
    )
    canonical_table_count = _require_positive_int(
        _required(canonical_target, "table_count", "data.canonical_target"),
        "data.canonical_target.table_count",
    )
    canonical_fact_count = _require_positive_int(
        _required(canonical_target, "fact_count", "data.canonical_target"),
        "data.canonical_target.fact_count",
    )
    largest_table_count = max([*table_counts, n_sweep_table_count])
    if latent_positions < largest_table_count:
        raise ConfigError(
            "data.master_world.latent_positions must be at least the largest "
            "configured table count"
        )
    if relation_facts_per_chain != latent_positions - 1:
        raise ConfigError(
            "data.master_world.relation_facts_per_chain must equal "
            "latent_positions - 1 for a chain topology"
        )
    if identifier_fields_per_chain != latent_positions:
        raise ConfigError(
            "data.master_world.identifier_fields_per_chain must equal "
            "latent_positions"
        )
    if experimental_facts_per_chain != (
        descriptive_facts_per_chain + relation_facts_per_chain
    ):
        raise ConfigError(
            "data.master_world.experimental_facts_per_chain must equal the sum of "
            "descriptive and relation facts; identifier fields are excluded"
        )
    if (
        latent_positions,
        descriptive_facts_per_chain,
        relation_facts_per_chain,
        experimental_facts_per_chain,
        identifier_fields_per_chain,
    ) != (12, 29, 11, 40, 12):
        raise ConfigError(
            "the academic master world requires 12 positions, 29 descriptive "
            "facts, 11 relations, 40 experimental facts, and 12 identifier fields"
        )
    if (canonical_table_count, canonical_fact_count) != (12, 10_000):
        raise ConfigError("data.canonical_target must be T12/N10K")
    configured_fact_counts = [t_sweep_fact_count, *fact_counts, optional_fact_count]
    for fact_count in configured_fact_counts:
        if fact_count % experimental_facts_per_chain != 0:
            raise ConfigError(
                f"configured fact count {fact_count} must be exactly divisible by "
                "data.master_world.experimental_facts_per_chain"
            )

    if data["reuse_t8_n10k"]:
        if t_sweep_fact_count != 10_000:
            raise ConfigError(
                "data.t_sweep.fact_count must be 10000 when "
                "data.reuse_t8_n10k is enabled"
            )
        if n_sweep_table_count != 8:
            raise ConfigError(
                "data.n_sweep.table_count must be 8 when "
                "data.reuse_t8_n10k is enabled"
            )
        if 10_000 not in fact_counts:
            raise ConfigError(
                "data.n_sweep.fact_counts must include 10000 "
                "when data.reuse_t8_n10k is enabled"
            )
        if 8 not in table_counts:
            raise ConfigError(
                "data.t_sweep.table_counts must include 8 "
                "when data.reuse_t8_n10k is enabled"
            )

    _require_positive_int(
        _required(training, "fact_exposure", "training"), "training.fact_exposure"
    )
    warmup_ratio = _require_number(
        _required(training, "warmup_ratio", "training"), "training.warmup_ratio"
    )
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ConfigError("training.warmup_ratio must be between 0 and 1")
    learning_rate = _require_number(
        _required(training, "learning_rate", "training"), "training.learning_rate"
    )
    if learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    max_grad_norm = _require_number(
        _required(training, "max_grad_norm", "training"), "training.max_grad_norm"
    )
    if max_grad_norm <= 0:
        raise ConfigError("training.max_grad_norm must be positive")
    training_context = _require_positive_int(
        _required(training, "context_length", "training"), "training.context_length"
    )
    if training_context != model["context_length"]:
        raise ConfigError("model and training context lengths must match")
    for key in ("optimizer", "scheduler", "precision"):
        value = _required(training, key, "training")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"training.{key} must be a non-empty string")
    if training["precision"] != model["precision"]:
        raise ConfigError("model and training precision must match")

    for key in ("decoding", "primary_metric"):
        value = _required(evaluation, key, "evaluation")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"evaluation.{key} must be a non-empty string")
    temperature = _require_number(
        _required(evaluation, "temperature", "evaluation"), "evaluation.temperature"
    )
    if temperature < 0:
        raise ConfigError("evaluation.temperature must be non-negative")

    preflight_hops = _require_int_list(
        _required(preflight, "relational_hops", "preflight"),
        "preflight.relational_hops",
        allow_zero=False,
    )
    _require_positive_int(
        _required(preflight, "baseline_examples_per_hop", "preflight"),
        "preflight.baseline_examples_per_hop",
    )
    for key in ("baseline_strict_em_threshold", "copy_control_threshold"):
        threshold = _require_number(
            _required(preflight, key, "preflight"), f"preflight.{key}"
        )
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"preflight.{key} must be between 0 and 1")
    for key in ("max_input_length", "max_new_tokens"):
        _require_positive_int(
            _required(preflight, key, "preflight"), f"preflight.{key}"
        )
    skill_config = _require_mapping(
        _required(preflight, "skill_training", "preflight"),
        "preflight.skill_training",
    )
    primitive_config = _require_mapping(
        _required(preflight, "primitive_training", "preflight"),
        "preflight.primitive_training",
    )
    for key in (
        "train_examples_per_type",
        "validation_examples_per_type",
        "epochs",
        "batch_size",
        "max_length",
    ):
        _require_positive_int(
            _required(primitive_config, key, "preflight.primitive_training"),
            f"preflight.primitive_training.{key}",
        )
    _require_non_negative_int(
        _required(primitive_config, "seed", "preflight.primitive_training"),
        "preflight.primitive_training.seed",
    )
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = _require_number(
            _required(primitive_config, key, "preflight.primitive_training"),
            f"preflight.primitive_training.{key}",
        )
        if value < 0 or (key != "weight_decay" and value == 0):
            raise ConfigError(f"preflight.primitive_training.{key} must be positive")
    primitive_warmup = _require_number(
        _required(primitive_config, "warmup_ratio", "preflight.primitive_training"),
        "preflight.primitive_training.warmup_ratio",
    )
    if not 0.0 <= primitive_warmup <= 1.0:
        raise ConfigError(
            "preflight.primitive_training.warmup_ratio must be between 0 and 1"
        )
    for key in ("optimizer", "scheduler"):
        value = _required(primitive_config, key, "preflight.primitive_training")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"preflight.primitive_training.{key} must be a non-empty string"
            )
    if primitive_config["max_length"] != preflight["max_input_length"]:
        raise ConfigError("preflight primitive-training and input lengths must match")
    for key in (
        "train_examples_per_hop",
        "validation_examples_per_hop",
        "epochs",
        "batch_size",
        "max_length",
    ):
        _require_positive_int(
            _required(skill_config, key, "preflight.skill_training"),
            f"preflight.skill_training.{key}",
        )
    _require_non_negative_int(
        _required(skill_config, "seed", "preflight.skill_training"),
        "preflight.skill_training.seed",
    )
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = _require_number(
            _required(skill_config, key, "preflight.skill_training"),
            f"preflight.skill_training.{key}",
        )
        if value < 0 or (key != "weight_decay" and value == 0):
            raise ConfigError(f"preflight.skill_training.{key} must be positive")
    skill_warmup = _require_number(
        _required(skill_config, "warmup_ratio", "preflight.skill_training"),
        "preflight.skill_training.warmup_ratio",
    )
    if not 0.0 <= skill_warmup <= 1.0:
        raise ConfigError(
            "preflight.skill_training.warmup_ratio must be between 0 and 1"
        )
    for key in ("optimizer", "scheduler"):
        value = _required(skill_config, key, "preflight.skill_training")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"preflight.skill_training.{key} must be a non-empty string"
            )
    if skill_config["max_length"] != preflight["max_input_length"]:
        raise ConfigError("preflight skill-training and input lengths must match")
    if latent_positions < max([*hops, *preflight_hops]) + 1:
        raise ConfigError(
            "data.master_world.latent_positions must support the largest requested hop"
        )
    skill_training = _required(
        preflight, "generic_relational_skill_training", "preflight"
    )
    if not isinstance(skill_training, str) or not skill_training.strip():
        raise ConfigError(
            "preflight.generic_relational_skill_training must be a non-empty string"
        )
    qa_range = _require_mapping(
        _required(preflight, "qa_examples_range", "preflight"),
        "preflight.qa_examples_range",
    )
    minimum = _require_positive_int(
        _required(qa_range, "minimum", "preflight.qa_examples_range"),
        "preflight.qa_examples_range.minimum",
    )
    maximum = _require_positive_int(
        _required(qa_range, "maximum", "preflight.qa_examples_range"),
        "preflight.qa_examples_range.maximum",
    )
    if minimum > maximum:
        raise ConfigError("preflight QA example minimum must not exceed maximum")

    _require_bool(
        _required(layer_study, "enabled", "layer_study"), "layer_study.enabled"
    )
    _require_int_list(
        _required(layer_study, "layers", "layer_study"),
        "layer_study.layers",
        allow_zero=False,
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {config_path}: {exc}") from exc

    config = _require_mapping(loaded, "configuration root")
    validate_config(config)
    return config

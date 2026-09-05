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
    "target_sft",
    "evaluation",
    "preflight",
    "layer_study",
}

EXP02_NAME = "exp02_capacity_boundary"


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
    experiment_value = config.get("experiment")
    if isinstance(experiment_value, dict) and experiment_value.get("name") == EXP02_NAME:
        _validate_exp02_config(config)
        return
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
    target_sft = sections["target_sft"]
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
    qa_reference = _require_mapping(
        _required(data, "qa_reference", "data"), "data.qa_reference"
    )
    qa_reference_table_count = _require_positive_int(
        _required(qa_reference, "table_count", "data.qa_reference"),
        "data.qa_reference.table_count",
    )
    qa_reference_fact_count = _require_positive_int(
        _required(qa_reference, "fact_count", "data.qa_reference"),
        "data.qa_reference.fact_count",
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
    if (qa_reference_table_count, qa_reference_fact_count) != (12, 10_000):
        raise ConfigError("data.qa_reference must identify the immutable T12/N10K benchmark")
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
    _require_positive_int(
        _required(training, "cpt_batch_size", "training"),
        "training.cpt_batch_size",
    )
    _require_positive_int(
        _required(training, "cpt_epochs", "training"),
        "training.cpt_epochs",
    )
    _require_positive_int(
        _required(training, "gradient_accumulation_steps", "training"),
        "training.gradient_accumulation_steps",
    )
    _require_non_negative_int(
        _required(training, "dataloader_workers", "training"),
        "training.dataloader_workers",
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
    weight_decay = _require_number(
        _required(training, "weight_decay", "training"), "training.weight_decay"
    )
    if weight_decay < 0:
        raise ConfigError("training.weight_decay must be non-negative")
    epsilon = _require_number(
        _required(training, "epsilon", "training"), "training.epsilon"
    )
    if epsilon <= 0:
        raise ConfigError("training.epsilon must be positive")
    betas = _required(training, "betas", "training")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("training.betas must contain exactly two numbers")
    for index, beta in enumerate(betas):
        beta_value = _require_number(beta, f"training.betas[{index}]")
        if not 0.0 <= beta_value < 1.0:
            raise ConfigError(f"training.betas[{index}] must be in [0, 1)")
    max_grad_norm = _require_number(
        _required(training, "max_grad_norm", "training"), "training.max_grad_norm"
    )
    if max_grad_norm <= 0:
        raise ConfigError("training.max_grad_norm must be positive")
    _require_positive_int(
        _required(training, "context_length", "training"), "training.context_length"
    )
    for key in ("optimizer", "scheduler", "precision"):
        value = _required(training, key, "training")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"training.{key} must be a non-empty string")
    if training["optimizer"].lower() != "adamw":
        raise ConfigError("training.optimizer must be adamw")
    if training["scheduler"].lower() not in {
        "constant",
        "constant_with_warmup",
        "cosine",
        "linear",
    }:
        raise ConfigError(
            "training.scheduler must be constant, constant_with_warmup, cosine, "
            "or linear"
        )
    if training["precision"].lower() not in {"bf16", "fp16", "fp32"}:
        raise ConfigError("training.precision must be bf16, fp16, or fp32")
    for key in (
        "shuffle",
        "gradient_checkpointing",
        "fused_optimizer",
        "pin_memory",
        "drop_last",
    ):
        _require_bool(_required(training, key, "training"), f"training.{key}")

    dataset_dir = _required(target_sft, "dataset_dir", "target_sft")
    if dataset_dir != "target_sft":
        raise ConfigError("target_sft.dataset_dir must be target_sft")
    training_split = _required(target_sft, "training_split", "target_sft")
    if training_split != "train":
        raise ConfigError(
            "target_sft.training_split must be train; validation and test are held out"
        )
    dev_split = _required(target_sft, "dev_split", "target_sft")
    if dev_split != "dev":
        raise ConfigError(
            "target_sft.dev_split must be dev; validation and test are held out"
        )
    early_stopping_patience = _require_positive_int(
        _required(target_sft, "early_stopping_patience", "target_sft"),
        "target_sft.early_stopping_patience",
    )
    if early_stopping_patience != 3:
        raise ConfigError("target_sft.early_stopping_patience must be 3")
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "context_length",
    ):
        _require_positive_int(
            _required(target_sft, key, "target_sft"), f"target_sft.{key}"
        )
    _require_non_negative_int(
        _required(target_sft, "dataloader_workers", "target_sft"),
        "target_sft.dataloader_workers",
    )
    target_sft_lr = _require_number(
        _required(target_sft, "learning_rate", "target_sft"),
        "target_sft.learning_rate",
    )
    if target_sft_lr <= 0:
        raise ConfigError("target_sft.learning_rate must be positive")
    target_sft_weight_decay = _require_number(
        _required(target_sft, "weight_decay", "target_sft"),
        "target_sft.weight_decay",
    )
    if target_sft_weight_decay < 0:
        raise ConfigError("target_sft.weight_decay must be non-negative")
    target_sft_epsilon = _require_number(
        _required(target_sft, "epsilon", "target_sft"), "target_sft.epsilon"
    )
    if target_sft_epsilon <= 0:
        raise ConfigError("target_sft.epsilon must be positive")
    target_sft_betas = _required(target_sft, "betas", "target_sft")
    if not isinstance(target_sft_betas, list) or len(target_sft_betas) != 2:
        raise ConfigError("target_sft.betas must contain exactly two numbers")
    for index, beta in enumerate(target_sft_betas):
        beta_value = _require_number(beta, f"target_sft.betas[{index}]")
        if not 0.0 <= beta_value < 1.0:
            raise ConfigError(f"target_sft.betas[{index}] must be in [0, 1)")
    target_sft_warmup = _require_number(
        _required(target_sft, "warmup_ratio", "target_sft"),
        "target_sft.warmup_ratio",
    )
    if not 0.0 <= target_sft_warmup <= 1.0:
        raise ConfigError("target_sft.warmup_ratio must be between 0 and 1")
    target_sft_max_grad_norm = _require_number(
        _required(target_sft, "max_grad_norm", "target_sft"),
        "target_sft.max_grad_norm",
    )
    if target_sft_max_grad_norm <= 0:
        raise ConfigError("target_sft.max_grad_norm must be positive")
    for key, expected in (
        ("optimizer", "adamw"),
        ("scheduler", "cosine"),
        ("precision", "bf16"),
    ):
        value = _required(target_sft, key, "target_sft")
        if not isinstance(value, str) or value.lower() != expected:
            raise ConfigError(f"target_sft.{key} must be {expected}")
    for key in (
        "shuffle",
        "gradient_checkpointing",
        "fused_optimizer",
        "pin_memory",
        "drop_last",
        "answer_only_loss",
        "supervise_eos",
    ):
        _require_bool(
            _required(target_sft, key, "target_sft"), f"target_sft.{key}"
        )
    if target_sft["drop_last"]:
        raise ConfigError(
            "target_sft.drop_last must be false so every QA record is trained"
        )
    if not target_sft["answer_only_loss"]:
        raise ConfigError("target_sft.answer_only_loss must be true")
    if not target_sft["supervise_eos"]:
        raise ConfigError("target_sft.supervise_eos must be true")

    for key in ("decoding", "primary_metric"):
        value = _required(evaluation, key, "evaluation")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"evaluation.{key} must be a non-empty string")
    temperature = _require_number(
        _required(evaluation, "temperature", "evaluation"), "evaluation.temperature"
    )
    if temperature != 0.0:
        raise ConfigError("evaluation.temperature must be zero for greedy decoding")
    if evaluation["decoding"].lower() != "greedy":
        raise ConfigError("evaluation.decoding must be greedy")
    if evaluation["primary_metric"] != "normalized_exact_match":
        raise ConfigError(
            "evaluation.primary_metric must be normalized_exact_match"
        )
    _require_positive_int(
        _required(evaluation, "batch_size", "evaluation"),
        "evaluation.batch_size",
    )
    evaluation_context = _require_positive_int(
        _required(evaluation, "context_length", "evaluation"),
        "evaluation.context_length",
    )
    if evaluation_context != model["context_length"]:
        raise ConfigError("model and evaluation context lengths must match")
    if (
        _require_positive_int(
            _required(evaluation, "max_new_tokens", "evaluation"),
            "evaluation.max_new_tokens",
        )
        != 64
    ):
        raise ConfigError("evaluation.max_new_tokens must be 64")

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


def _validate_exp02_config(config: dict[str, Any]) -> None:
    """Validate Exp2 without importing Exp1's fixed T/N sweep assumptions."""
    required = {"experiment", "model", "data", "training", "target_sft", "evaluation"}
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError(f"missing required configuration sections: {', '.join(missing)}")
    experiment = _require_mapping(config["experiment"], "experiment")
    model = _require_mapping(config["model"], "model")
    data = _require_mapping(config["data"], "data")
    training = _require_mapping(config["training"], "training")
    target_sft = _require_mapping(config["target_sft"], "target_sft")
    evaluation = _require_mapping(config["evaluation"], "evaluation")
    if experiment.get("name") != EXP02_NAME:
        raise ConfigError(f"experiment.name must be {EXP02_NAME}")
    _require_non_negative_int(_required(experiment, "seed", "experiment"), "experiment.seed")
    for key in ("name", "precision"):
        value = _required(model, key, "model")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"model.{key} must be a non-empty string")
    for key in ("native_layers", "hidden_size", "attention_heads", "context_length"):
        _require_positive_int(_required(model, key, "model"), f"model.{key}")
    if "t_sweep" in data or "n_sweep" in data:
        raise ConfigError("Experiment 2 must not define fixed T or N sweep arrays")
    if data.get("schema_topology") != "chain":
        raise ConfigError("data.schema_topology must be chain")
    construction = _require_mapping(_required(data, "master_world", "data"), "data.master_world")
    expected = {
        "latent_positions": 12,
        "descriptive_facts_per_chain": 29,
        "relation_facts_per_chain": 11,
        "experimental_facts_per_chain": 40,
        "identifier_fields_per_chain": 12,
    }
    for key, expected_value in expected.items():
        if _require_positive_int(_required(construction, key, "data.master_world"), f"data.master_world.{key}") != expected_value:
            raise ConfigError(f"data.master_world.{key} must be {expected_value}")
    canonical = _require_mapping(_required(data, "canonical_source", "data"), "data.canonical_source")
    path = _required(canonical, "path", "data.canonical_source")
    if not isinstance(path, str) or not path.strip():
        raise ConfigError("data.canonical_source.path must be a non-empty string")

    # Exp2 deliberately reuses the proven optimization/evaluation semantics.  Keep
    # this validation compact and focused on fields consumed by the shared code.
    for key in ("fact_exposure", "cpt_batch_size", "cpt_epochs", "gradient_accumulation_steps", "context_length"):
        _require_positive_int(_required(training, key, "training"), f"training.{key}")
    for key in ("dataloader_workers",):
        _require_non_negative_int(_required(training, key, "training"), f"training.{key}")
    for key in ("shuffle", "gradient_checkpointing", "fused_optimizer", "pin_memory", "drop_last"):
        _require_bool(_required(training, key, "training"), f"training.{key}")
    for key in ("learning_rate", "epsilon", "max_grad_norm"):
        if _require_number(_required(training, key, "training"), f"training.{key}") <= 0:
            raise ConfigError(f"training.{key} must be positive")
    if _require_number(_required(training, "weight_decay", "training"), "training.weight_decay") < 0:
        raise ConfigError("training.weight_decay must be non-negative")
    warmup = _require_number(_required(training, "warmup_ratio", "training"), "training.warmup_ratio")
    if not 0 <= warmup <= 1:
        raise ConfigError("training.warmup_ratio must be between 0 and 1")
    betas = _required(training, "betas", "training")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("training.betas must contain exactly two numbers")
    for key in ("optimizer", "scheduler", "precision"):
        if not isinstance(_required(training, key, "training"), str):
            raise ConfigError(f"training.{key} must be text")

    if target_sft.get("dataset_dir") != "target_sft" or target_sft.get("training_split") != "train" or target_sft.get("dev_split") != "dev":
        raise ConfigError("Experiment-2 target_sft must use target_sft/train and target_sft/dev")
    for key in ("batch_size", "gradient_accumulation_steps", "epochs", "context_length", "early_stopping_patience"):
        _require_positive_int(_required(target_sft, key, "target_sft"), f"target_sft.{key}")
    for key in ("dataloader_workers",):
        _require_non_negative_int(_required(target_sft, key, "target_sft"), f"target_sft.{key}")
    for key in ("shuffle", "gradient_checkpointing", "fused_optimizer", "pin_memory", "drop_last", "answer_only_loss", "supervise_eos"):
        _require_bool(_required(target_sft, key, "target_sft"), f"target_sft.{key}")
    if target_sft["drop_last"] or not target_sft["answer_only_loss"] or not target_sft["supervise_eos"]:
        raise ConfigError("target SFT requires drop_last=false, answer_only_loss=true, supervise_eos=true")
    for key in ("learning_rate", "epsilon", "max_grad_norm"):
        if _require_number(_required(target_sft, key, "target_sft"), f"target_sft.{key}") <= 0:
            raise ConfigError(f"target_sft.{key} must be positive")
    if _require_number(_required(target_sft, "weight_decay", "target_sft"), "target_sft.weight_decay") < 0:
        raise ConfigError("target_sft.weight_decay must be non-negative")
    for key in ("betas", "optimizer", "scheduler", "precision", "warmup_ratio"):
        _required(target_sft, key, "target_sft")
    for key in ("batch_size", "context_length", "max_new_tokens"):
        _require_positive_int(_required(evaluation, key, "evaluation"), f"evaluation.{key}")
    if evaluation.get("decoding") != "greedy" or evaluation.get("temperature") != 0.0:
        raise ConfigError("Experiment-2 evaluation must use greedy decoding at temperature 0")
    if evaluation.get("primary_metric") != "normalized_exact_match":
        raise ConfigError("evaluation.primary_metric must be normalized_exact_match")


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

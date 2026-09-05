from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from data.serialize import SERIALIZATION_FORMAT_VERSION, SERIALIZATION_STYLE
from experiment import configured_model_layers, verify_checkpoint_layers
from utils.hashing import hash_file
from utils.io import read_json, read_text, write_json, write_jsonl, write_yaml

SUPPORTED_CPT_OPTIMIZERS = {"adamw"}
SUPPORTED_CPT_SCHEDULERS = {
    "constant",
    "constant_with_warmup",
    "cosine",
    "linear",
}
SUPPORTED_CPT_PRECISIONS = {"bf16", "fp16", "fp32"}


class CPTArtifactError(ValueError):
    """Raised when CPT provenance files are missing or inconsistent."""


def _require_nonempty_file(path: str | Path, label: str) -> Path:
    artifact_path = Path(path)
    if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {artifact_path}")
    return artifact_path


def verify_cpt_artifacts(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    database_path: str | Path,
    database_manifest_path: str | Path,
    readable_book_path: str | Path,
    train_text_path: str | Path,
    cpt_manifest_path: str | Path,
) -> dict[str, Any]:
    database_path = _require_nonempty_file(database_path, "database")
    database_manifest_path = _require_nonempty_file(
        database_manifest_path, "database manifest"
    )
    readable_book_path = _require_nonempty_file(readable_book_path, "CPT readable book")
    train_text_path = _require_nonempty_file(train_text_path, "CPT train text")
    cpt_manifest_path = _require_nonempty_file(cpt_manifest_path, "CPT manifest")

    database_manifest = read_json(database_manifest_path)
    cpt_manifest = read_json(cpt_manifest_path)
    database_sha256 = hash_file(database_path)
    database_manifest_sha256 = hash_file(database_manifest_path)
    readable_book_sha256 = hash_file(readable_book_path)
    train_text_sha256 = hash_file(train_text_path)
    cpt_manifest_sha256 = hash_file(cpt_manifest_path)

    if database_manifest.get("database_sha256") != database_sha256:
        raise CPTArtifactError("database hash does not match its manifest")
    for manifest, label in (
        (database_manifest, "database manifest"),
        (cpt_manifest, "CPT manifest"),
    ):
        if (
            manifest.get("T") != table_count
            or manifest.get("table_count") != table_count
        ):
            raise CPTArtifactError(f"{label} T metadata does not match T={table_count}")
        if manifest.get("requested_N") != fact_count:
            raise CPTArtifactError(f"{label} N metadata does not match N={fact_count}")
    if database_manifest.get("actual_logical_fact_count") != fact_count:
        raise CPTArtifactError(
            "database manifest actual logical fact count is inconsistent"
        )
    if cpt_manifest.get("format_version") != SERIALIZATION_FORMAT_VERSION:
        raise CPTArtifactError("CPT serialization format version is unsupported")
    if cpt_manifest.get("serialization_style") != SERIALIZATION_STYLE:
        raise CPTArtifactError("CPT serialization style is unsupported")

    fact_exposure = config["training"]["fact_exposure"]
    if cpt_manifest.get("fact_exposure") != fact_exposure:
        raise CPTArtifactError("CPT fact exposure does not match the experiment config")
    if cpt_manifest.get("readable_book_copy_count_in_train_text") != fact_exposure:
        raise CPTArtifactError(
            "CPT readable-book copy count does not match fact exposure"
        )
    if cpt_manifest.get("source_database_sha256") != database_sha256:
        raise CPTArtifactError("CPT manifest source database hash is inconsistent")
    if cpt_manifest.get("source_database_manifest_sha256") != database_manifest_sha256:
        raise CPTArtifactError(
            "CPT manifest source database-manifest hash is inconsistent"
        )
    if cpt_manifest.get("train_text_sha256") != train_text_sha256:
        raise CPTArtifactError("CPT train-text hash does not match its manifest")
    if cpt_manifest.get("readable_book_sha256") != readable_book_sha256:
        raise CPTArtifactError("CPT readable-book hash does not match its manifest")
    if cpt_manifest.get("logical_facts_per_exposure") != fact_count:
        raise CPTArtifactError("CPT logical facts per exposure are inconsistent")
    if cpt_manifest.get("serialized_logical_fact_occurrences") != (
        fact_count * fact_exposure
    ):
        raise CPTArtifactError("CPT serialized logical-fact accounting is inconsistent")
    if database_manifest.get("experiment_mode") == "selected_canonical_tables":
        if cpt_manifest.get("experiment_name") != database_manifest.get("experiment_name"):
            raise CPTArtifactError("CPT experiment identity does not match the database")
        if cpt_manifest.get("selected_tables") != database_manifest.get("selected_tables"):
            raise CPTArtifactError("CPT selected tables do not match the database")
        if cpt_manifest.get("facts_per_selected_chain") != database_manifest.get("facts_per_selected_chain"):
            raise CPTArtifactError("CPT facts-per-selected-chain metadata is inconsistent")
        if cpt_manifest.get("logical_content_sha256") != database_manifest.get("logical_content_sha256"):
            raise CPTArtifactError("CPT logical-content provenance is inconsistent")

    readable_book_bytes = readable_book_path.read_bytes()
    train_bytes = train_text_path.read_bytes()
    try:
        readable_book = readable_book_bytes.decode("utf-8")
        train_text = train_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CPTArtifactError(
            "CPT readable book or train text is not valid UTF-8"
        ) from exc
    expected_statistics = {
        "readable_book_byte_count": len(readable_book_bytes),
        "readable_book_character_count": len(readable_book),
        "readable_book_line_count": readable_book.count("\n"),
        "train_text_byte_count": len(train_bytes),
        "train_text_character_count": len(train_text),
        "train_text_line_count": train_text.count("\n"),
    }
    for key, actual_value in expected_statistics.items():
        if cpt_manifest.get(key) != actual_value:
            raise CPTArtifactError(f"CPT manifest {key} is inconsistent")

    if train_bytes != readable_book_bytes * fact_exposure:
        raise CPTArtifactError(
            "CPT train text is not exactly fact_exposure copies of the readable book"
        )

    return {
        "T": table_count,
        "N": fact_count,
        "source_database_sha256": database_sha256,
        "database_manifest_sha256": database_manifest_sha256,
        "readable_book_sha256": readable_book_sha256,
        "cpt_train_text_sha256": train_text_sha256,
        "cpt_manifest_sha256": cpt_manifest_sha256,
        "fact_exposure": fact_exposure,
        "readable_book_copy_count_in_train_text": fact_exposure,
        "serialization_style": SERIALIZATION_STYLE,
        "readable_book_byte_count": len(readable_book_bytes),
        "train_text_byte_count": len(train_bytes),
        "experiment_name": database_manifest.get("experiment_name"),
        "selected_tables": database_manifest.get("selected_tables"),
        "training_data_dir": str(database_manifest_path.parent.resolve()),
        "logical_content_sha256": database_manifest.get("logical_content_sha256"),
    }


def chunk_token_ids(
    token_ids: list[int], *, context_length: int, pad_token_id: int
) -> tuple[list[dict[str, list[int] | int]], dict[str, int]]:
    if not token_ids:
        raise ValueError("the CPT corpus tokenization is empty")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError("token_ids must contain only integers")

    examples: list[dict[str, list[int] | int]] = []
    for start in range(0, len(token_ids), context_length):
        real_tokens = token_ids[start : start + context_length]
        padding = context_length - len(real_tokens)
        examples.append(
            {
                "input_ids": [*real_tokens, *([pad_token_id] * padding)],
                "attention_mask": [*([1] * len(real_tokens)), *([0] * padding)],
                "labels": [*real_tokens, *([-100] * padding)],
                "real_token_count": len(real_tokens),
            }
        )
    supervised_tokens = sum(
        sum(label != -100 for label in example["labels"]) for example in examples
    )
    if supervised_tokens != len(token_ids):
        raise RuntimeError("CPT chunking lost or duplicated supervised tokens")
    final_remainder = len(token_ids) % context_length
    statistics = {
        "total_tokens": len(token_ids),
        "supervised_tokens": supervised_tokens,
        "sequence_count": len(examples),
        "context_length": context_length,
        "final_partial_sequence_size": final_remainder,
        "final_sequence_real_token_count": examples[-1]["real_token_count"],
        "padding_token_count": len(examples) * context_length - len(token_ids),
    }
    return examples, statistics


def tokenize_cpt_corpus(
    text: str, tokenizer: Any, *, context_length: int
) -> tuple[list[dict[str, list[int] | int]], dict[str, int]]:
    if not isinstance(text, str) or not text:
        raise ValueError("CPT corpus text must be non-empty")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad token before CPT chunking")
    return chunk_token_ids(
        token_ids, context_length=context_length, pad_token_id=pad_token_id
    )


def collate_cpt_examples(
    examples: list[dict[str, list[int] | int]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty CPT batch")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("missing required dependency: torch") from exc
    return {
        key: torch.tensor([example[key] for example in examples], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


def enable_full_parameter_training(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("CPT must train every model parameter")
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def build_cpt_training_plan(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    sequence_count: int,
    layers: int | None = None,
) -> dict[str, Any]:
    training = config["training"]
    if (
        isinstance(sequence_count, bool)
        or not isinstance(sequence_count, int)
        or sequence_count <= 0
    ):
        raise ValueError("sequence_count must be positive")

    def positive_int(key: str) -> int:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
        return value

    def non_negative_int(key: str) -> int:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training.{key} must be a non-negative integer")
        return value

    def boolean(key: str) -> bool:
        value = training.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"training.{key} must be a boolean")
        return value

    def number(key: str, *, positive: bool, allow_zero: bool = False) -> float:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"training.{key} must be a number")
        numeric_value = float(value)
        if positive and (numeric_value < 0 or (numeric_value == 0 and not allow_zero)):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"training.{key} must be {qualifier}")
        return numeric_value

    fact_exposure = positive_int("fact_exposure")
    batch_size = positive_int("cpt_batch_size")
    epochs = positive_int("cpt_epochs")
    gradient_accumulation_steps = positive_int("gradient_accumulation_steps")
    context_length = positive_int("context_length")
    dataloader_workers = non_negative_int("dataloader_workers")
    learning_rate = number("learning_rate", positive=True)
    weight_decay = number("weight_decay", positive=True, allow_zero=True)
    epsilon = number("epsilon", positive=True)
    warmup_ratio = number("warmup_ratio", positive=False)
    max_grad_norm = number("max_grad_norm", positive=True)
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("training.warmup_ratio must be between 0 and 1")

    betas = training.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("training.betas must contain exactly two numbers")
    normalized_betas: list[float] = []
    for index, beta in enumerate(betas):
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise ValueError(f"training.betas[{index}] must be a number")
        normalized_beta = float(beta)
        if not 0.0 <= normalized_beta < 1.0:
            raise ValueError(f"training.betas[{index}] must be in [0, 1)")
        normalized_betas.append(normalized_beta)

    optimizer = str(training.get("optimizer", "")).lower()
    scheduler = str(training.get("scheduler", "")).lower()
    precision = str(training.get("precision", "")).lower()
    if optimizer not in SUPPORTED_CPT_OPTIMIZERS:
        raise ValueError(
            f"unsupported training.optimizer={optimizer!r}; supported values: "
            f"{sorted(SUPPORTED_CPT_OPTIMIZERS)}"
        )
    if scheduler not in SUPPORTED_CPT_SCHEDULERS:
        raise ValueError(
            f"unsupported training.scheduler={scheduler!r}; supported values: "
            f"{sorted(SUPPORTED_CPT_SCHEDULERS)}"
        )
    if precision not in SUPPORTED_CPT_PRECISIONS:
        raise ValueError(
            f"unsupported training.precision={precision!r}; supported values: "
            f"{sorted(SUPPORTED_CPT_PRECISIONS)}"
        )

    shuffle = boolean("shuffle")
    gradient_checkpointing = boolean("gradient_checkpointing")
    fused_optimizer = boolean("fused_optimizer")
    pin_memory = boolean("pin_memory")
    drop_last = boolean("drop_last")
    seed = config.get("experiment", {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment.seed must be a non-negative integer")

    if drop_last:
        micro_batches_per_epoch = sequence_count // batch_size
        trained_sequence_count_per_epoch = micro_batches_per_epoch * batch_size
    else:
        micro_batches_per_epoch = math.ceil(sequence_count / batch_size)
        trained_sequence_count_per_epoch = sequence_count
    if micro_batches_per_epoch == 0:
        raise ValueError(
            "training.drop_last would discard every CPT sequence; reduce "
            "training.cpt_batch_size or disable drop_last"
        )
    steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    optimizer_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(optimizer_steps * warmup_ratio)
    return {
        "stage": "cpt",
        "T": table_count,
        "N": fact_count,
        "L": configured_model_layers(config) if layers is None else layers,
        "seed": seed,
        "epochs": epochs,
        "passes_over_serialized_corpus": epochs,
        "fact_exposure": fact_exposure,
        "effective_fact_exposure": fact_exposure * epochs,
        "context_length": context_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "shuffle": shuffle,
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_workers": dataloader_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "trained_sequence_count_per_epoch": trained_sequence_count_per_epoch,
        "dropped_sequences_per_epoch": (
            sequence_count - trained_sequence_count_per_epoch
        ),
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": normalized_betas,
        "epsilon": epsilon,
        "fused_optimizer_requested": fused_optimizer,
        "fused_optimizer_actually_used": None,
        "scheduler": scheduler,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "precision": precision,
        "sequence_count": sequence_count,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": optimizer_steps,
        "optimizer_steps": optimizer_steps,
    }


def _iterate_cpt_batches(loader: Any, epochs: int) -> Iterator[tuple[int, int, Any]]:
    for epoch in range(1, epochs + 1):
        for step_in_epoch, batch in enumerate(loader, start=1):
            yield epoch, step_in_epoch, batch


def _seeded_dataloader_generator(torch_module: Any, seed: int) -> Any:
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def _build_adamw_optimizer(
    torch_module: Any, parameters: Any, plan: dict[str, Any]
) -> tuple[Any, bool, str | None]:
    parameter_list = list(parameters)
    optimizer_kwargs = {
        "lr": plan["learning_rate"],
        "weight_decay": plan["weight_decay"],
        "betas": tuple(plan["betas"]),
        "eps": plan["epsilon"],
    }
    if not plan["fused_optimizer_requested"]:
        return (
            torch_module.optim.AdamW(parameter_list, **optimizer_kwargs),
            False,
            None,
        )
    try:
        optimizer = torch_module.optim.AdamW(
            parameter_list, fused=True, **optimizer_kwargs
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        optimizer = torch_module.optim.AdamW(parameter_list, **optimizer_kwargs)
        return optimizer, False, f"{type(exc).__name__}: {exc}"
    actually_used = bool(getattr(optimizer, "defaults", {}).get("fused", True))
    return optimizer, actually_used, None


def _configure_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if enabled:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "the source model does not support gradient checkpointing"
            )
        model.gradient_checkpointing_enable()
        return
    if getattr(model, "is_gradient_checkpointing", False):
        if not hasattr(model, "gradient_checkpointing_disable"):
            raise RuntimeError("the source model cannot disable gradient checkpointing")
        model.gradient_checkpointing_disable()


def _load_model_and_tokenizer(source_checkpoint: Path) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("missing required dependency: transformers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(source_checkpoint), local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(source_checkpoint), local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"unable to load source checkpoint {source_checkpoint}: {exc}"
        ) from exc
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("source tokenizer defines neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run_cpt_training(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    layers: int | None = None,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    run_config_path: str | Path,
    train_log_path: str | Path,
    database_path: str | Path,
    database_manifest_path: str | Path,
    readable_book_path: str | Path,
    train_text_path: str | Path,
    cpt_manifest_path: str | Path,
) -> dict[str, Any]:
    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    run_config_path = Path(run_config_path)
    train_log_path = Path(train_log_path)
    if (
        not source_checkpoint.is_dir()
        or not (source_checkpoint / "config.json").is_file()
    ):
        raise FileNotFoundError(f"source checkpoint is missing: {source_checkpoint}")
    requested_layers = configured_model_layers(config) if layers is None else layers
    layer_provenance = verify_checkpoint_layers(source_checkpoint, requested_layers)
    if source_checkpoint.resolve() == output_checkpoint.resolve():
        raise ValueError("source and final CPT checkpoint paths must differ")
    if output_checkpoint.exists():
        if not output_checkpoint.is_dir() or any(output_checkpoint.iterdir()):
            raise FileExistsError(
                f"final CPT checkpoint path is not an empty directory: "
                f"{output_checkpoint}"
            )

    provenance = verify_cpt_artifacts(
        config,
        table_count=table_count,
        fact_count=fact_count,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        readable_book_path=readable_book_path,
        train_text_path=train_text_path,
        cpt_manifest_path=cpt_manifest_path,
    )
    started = time.perf_counter()
    model, tokenizer = _load_model_and_tokenizer(source_checkpoint)
    examples, token_statistics = tokenize_cpt_corpus(
        read_text(train_text_path),
        tokenizer,
        context_length=config["training"]["context_length"],
    )
    book_token_count = len(
        tokenizer.encode(read_text(readable_book_path), add_special_tokens=False)
    )
    token_statistics["book_token_count"] = book_token_count
    token_statistics["train_token_count"] = token_statistics["total_tokens"]
    plan = build_cpt_training_plan(
        config,
        table_count=table_count,
        fact_count=fact_count,
        sequence_count=token_statistics["sequence_count"],
        layers=requested_layers,
    )
    model_context_limit = getattr(model.config, "max_position_embeddings", None)
    if model_context_limit is None:
        model_context_limit = getattr(model.config, "n_positions", None)
    if (
        isinstance(model_context_limit, int)
        and plan["context_length"] > model_context_limit
    ):
        raise ValueError(
            f"training.context_length={plan['context_length']} exceeds the source "
            f"model limit of {model_context_limit}"
        )

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("missing required CPT training dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CPT training requires a CUDA device")
    if plan["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "training.precision=bf16 requires a CUDA device with BF16 support"
        )
    device = torch.device("cuda")
    seed = plan["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = model.to(device)
    model.train()
    previous_use_cache = model.config.use_cache
    model.config.use_cache = False
    _configure_gradient_checkpointing(model, plan["gradient_checkpointing"])
    parameter_counts = enable_full_parameter_training(model)
    total_parameters = parameter_counts["total_parameters"]
    trainable_parameters = parameter_counts["trainable_parameters"]

    dataloader_generator = (
        _seeded_dataloader_generator(torch, seed) if plan["shuffle"] else None
    )
    loader = DataLoader(
        examples,
        batch_size=plan["batch_size"],
        shuffle=plan["shuffle"],
        collate_fn=collate_cpt_examples,
        pin_memory=plan["pin_memory"],
        drop_last=plan["drop_last"],
        num_workers=plan["dataloader_workers"],
        persistent_workers=plan["dataloader_workers"] > 0,
        generator=dataloader_generator,
    )
    if len(loader) != plan["micro_batches_per_epoch"]:
        raise RuntimeError("CPT DataLoader batch accounting is inconsistent")
    optimizer, fused_optimizer_used, fused_fallback_reason = _build_adamw_optimizer(
        torch, model.parameters(), plan
    )
    plan["fused_optimizer_actually_used"] = fused_optimizer_used
    plan["fused_optimizer_fallback_reason"] = fused_fallback_reason
    scheduler = get_scheduler(
        plan["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=plan["warmup_steps"],
        num_training_steps=plan["total_optimizer_steps"],
    )
    run_record = {
        "experiment": config["experiment"]["name"],
        "model": config["model"]["name"],
        "run_timestamp": config.get("_runtime", {}).get("run_timestamp"),
        "T": table_count,
        "N": fact_count,
        "L": requested_layers,
        "experiment_condition": {
            "table_count": table_count,
            "fact_count": fact_count,
            "layers": requested_layers,
            "selected_tables": provenance.get("selected_tables"),
        },
        "source_checkpoint": str(source_checkpoint),
        "final_checkpoint_path": str(output_checkpoint),
        "source_checkpoint_config_sha256": hash_file(source_checkpoint / "config.json"),
        "checkpoint_layer_verification": layer_provenance,
        "tokenizer_identity": getattr(
            tokenizer, "name_or_path", tokenizer.__class__.__name__
        ),
        "model_identity": getattr(
            model.config, "_name_or_path", model.__class__.__name__
        ),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "full_parameter_training": True,
        "target_qa_used": False,
        "checkpoint_selection": "final_state_after_configured_cpt_epochs",
        "epochs": plan["epochs"],
        "steps_per_epoch": plan["steps_per_epoch"],
        "optimizer_steps": plan["optimizer_steps"],
        "effective_fact_exposure": plan["effective_fact_exposure"],
        "provenance": provenance,
        "tokenization": token_statistics,
        "training": plan,
    }
    write_yaml(run_config_path, run_record)

    step_records: list[dict[str, Any]] = []
    weighted_loss = 0.0
    loss_token_count = 0
    observed_supervised_tokens = 0
    optimizer_step = 0
    accumulation_micro_batches = 0
    accumulation_supervised_tokens = 0
    accumulation_weighted_loss = 0.0
    accumulation_loss_tokens = 0
    accumulation_steps = plan["gradient_accumulation_steps"]
    final_accumulation_remainder = plan["micro_batches_per_epoch"] % accumulation_steps
    autocast_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(plan["precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=plan["precision"] == "fp16")
    optimizer.zero_grad(set_to_none=True)
    for epoch, micro_batch_in_epoch, batch in _iterate_cpt_batches(
        loader, plan["epochs"]
    ):
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        supervised_tokens = int((batch["labels"] != -100).sum().item())
        observed_supervised_tokens += supervised_tokens
        is_final_micro_batch = micro_batch_in_epoch == plan["micro_batches_per_epoch"]
        in_final_partial_accumulation = (
            final_accumulation_remainder > 0
            and micro_batch_in_epoch
            > plan["micro_batches_per_epoch"] - final_accumulation_remainder
        )
        accumulation_divisor = (
            final_accumulation_remainder
            if in_final_partial_accumulation
            else accumulation_steps
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            output = model(**batch)
            loss = output.loss
            scaled_loss = loss / accumulation_divisor
        scaler.scale(scaled_loss).backward()
        contributing_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        detached_loss = float(loss.detach().item())
        weighted_loss += detached_loss * contributing_tokens
        loss_token_count += contributing_tokens
        accumulation_micro_batches += 1
        accumulation_supervised_tokens += supervised_tokens
        accumulation_weighted_loss += detached_loss * contributing_tokens
        accumulation_loss_tokens += contributing_tokens
        should_step = (
            micro_batch_in_epoch % accumulation_steps == 0 or is_final_micro_batch
        )
        if not should_step:
            continue

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), plan["max_grad_norm"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        step_records.append(
            {
                "record_type": "optimizer_step",
                "step": optimizer_step,
                "epoch": epoch,
                "step_in_epoch": math.ceil(micro_batch_in_epoch / accumulation_steps),
                "last_micro_batch_in_epoch": micro_batch_in_epoch,
                "accumulated_micro_batches": accumulation_micro_batches,
                "loss": accumulation_weighted_loss / accumulation_loss_tokens,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm),
                "supervised_tokens": accumulation_supervised_tokens,
            }
        )
        accumulation_micro_batches = 0
        accumulation_supervised_tokens = 0
        accumulation_weighted_loss = 0.0
        accumulation_loss_tokens = 0

    if len(step_records) != plan["optimizer_steps"]:
        raise RuntimeError("CPT optimizer-step accounting is inconsistent")
    expected_supervised_tokens = token_statistics["supervised_tokens"] * plan["epochs"]
    if (
        not plan["drop_last"]
        and observed_supervised_tokens != expected_supervised_tokens
    ):
        raise RuntimeError(
            "CPT training did not consume every supervised token once per epoch"
        )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.mkdir(parents=True, exist_ok=True)
    _configure_gradient_checkpointing(model, False)
    model.config.use_cache = previous_use_cache
    model.save_pretrained(output_checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(output_checkpoint)
    runtime_seconds = time.perf_counter() - started
    summary = {
        "record_type": "summary",
        **run_record,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "optimizer_steps": len(step_records),
        "training_loss": weighted_loss / loss_token_count,
        "loss_contributing_shifted_tokens": loss_token_count,
        "observed_supervised_tokens": observed_supervised_tokens,
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_checkpoint / "training_metadata.json", summary)
    write_jsonl(
        train_log_path,
        [{"record_type": "configuration", **run_record}, *step_records, summary],
    )
    return summary

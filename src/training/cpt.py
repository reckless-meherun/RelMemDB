from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

from data.serialize import SERIALIZATION_FORMAT_VERSION, SERIALIZATION_STYLE
from utils.hashing import hash_file
from utils.io import read_json, read_text, write_json, write_jsonl, write_yaml

CPT_EPOCHS = 1
ADAMW_DEFAULT_WEIGHT_DECAY = 0.01
ADAMW_DEFAULT_BETAS = (0.9, 0.999)
ADAMW_DEFAULT_EPSILON = 1e-8


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
    readable_book_path = _require_nonempty_file(
        readable_book_path, "CPT readable book"
    )
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
        if manifest.get("T") != table_count or manifest.get("table_count") != table_count:
            raise CPTArtifactError(f"{label} T metadata does not match T={table_count}")
        if manifest.get("requested_N") != fact_count:
            raise CPTArtifactError(f"{label} N metadata does not match N={fact_count}")
    if database_manifest.get("actual_logical_fact_count") != fact_count:
        raise CPTArtifactError("database manifest actual logical fact count is inconsistent")
    if cpt_manifest.get("format_version") != SERIALIZATION_FORMAT_VERSION:
        raise CPTArtifactError("CPT serialization format version is unsupported")
    if cpt_manifest.get("serialization_style") != SERIALIZATION_STYLE:
        raise CPTArtifactError("CPT serialization style is unsupported")

    fact_exposure = config["training"]["fact_exposure"]
    if cpt_manifest.get("fact_exposure") != fact_exposure:
        raise CPTArtifactError("CPT fact exposure does not match the experiment config")
    if cpt_manifest.get("readable_book_copy_count_in_train_text") != fact_exposure:
        raise CPTArtifactError("CPT readable-book copy count does not match fact exposure")
    if cpt_manifest.get("source_database_sha256") != database_sha256:
        raise CPTArtifactError("CPT manifest source database hash is inconsistent")
    if (
        cpt_manifest.get("source_database_manifest_sha256")
        != database_manifest_sha256
    ):
        raise CPTArtifactError("CPT manifest source database-manifest hash is inconsistent")
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

    readable_book_bytes = readable_book_path.read_bytes()
    train_bytes = train_text_path.read_bytes()
    try:
        readable_book = readable_book_bytes.decode("utf-8")
        train_text = train_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CPTArtifactError("CPT readable book or train text is not valid UTF-8") from exc
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
    }


def chunk_token_ids(
    token_ids: list[int], *, context_length: int, pad_token_id: int
) -> tuple[list[dict[str, list[int] | int]], dict[str, int]]:
    if not token_ids:
        raise ValueError("the CPT corpus tokenization is empty")
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids):
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
    config: dict[str, Any], *, table_count: int, fact_count: int, sequence_count: int
) -> dict[str, Any]:
    training = config["training"]
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    required = {
        "fact_exposure": 4,
        "context_length": 256,
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "precision": "bf16",
    }
    for key, expected_value in required.items():
        actual_value = training[key]
        if key in {"optimizer", "scheduler", "precision"}:
            actual_value = str(actual_value).lower()
        if actual_value != expected_value:
            raise ValueError(f"CPT requires training.{key}={expected_value!r}")
    if config["experiment"]["seed"] != 2025:
        raise ValueError("CPT requires experiment.seed=2025")
    batch_size = training["cpt_batch_size"]
    optimizer_steps = math.ceil(sequence_count / batch_size)
    warmup_steps = math.ceil(optimizer_steps * training["warmup_ratio"])
    return {
        "stage": "cpt",
        "T": table_count,
        "N": fact_count,
        "seed": config["experiment"]["seed"],
        "epochs": CPT_EPOCHS,
        "passes_over_serialized_corpus": 1,
        "fact_exposure": training["fact_exposure"],
        "context_length": training["context_length"],
        "batch_size": batch_size,
        "shuffle": False,
        "optimizer": "AdamW",
        "learning_rate": training["learning_rate"],
        "weight_decay": ADAMW_DEFAULT_WEIGHT_DECAY,
        "betas": list(ADAMW_DEFAULT_BETAS),
        "epsilon": ADAMW_DEFAULT_EPSILON,
        "scheduler": "cosine",
        "warmup_ratio": training["warmup_ratio"],
        "warmup_steps": warmup_steps,
        "max_grad_norm": training["max_grad_norm"],
        "precision": "bf16",
        "sequence_count": sequence_count,
        "optimizer_steps": optimizer_steps,
    }


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
    if not source_checkpoint.is_dir() or not (
        source_checkpoint / "config.json"
    ).is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {source_checkpoint}")
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
    plan = build_cpt_training_plan(
        config,
        table_count=table_count,
        fact_count=fact_count,
        sequence_count=token_statistics["sequence_count"],
    )

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("missing required CPT training dependencies") from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CPT training requires a CUDA device with BF16 support")
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
    parameter_counts = enable_full_parameter_training(model)
    total_parameters = parameter_counts["total_parameters"]
    trainable_parameters = parameter_counts["trainable_parameters"]

    loader = DataLoader(
        examples,
        batch_size=plan["batch_size"],
        shuffle=False,
        collate_fn=collate_cpt_examples,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=plan["learning_rate"],
        weight_decay=plan["weight_decay"],
        betas=tuple(plan["betas"]),
        eps=plan["epsilon"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=plan["warmup_steps"],
        num_training_steps=plan["optimizer_steps"],
    )
    run_record = {
        "experiment": config["experiment"]["name"],
        "source_checkpoint": str(source_checkpoint),
        "final_checkpoint_path": str(output_checkpoint),
        "source_checkpoint_config_sha256": hash_file(
            source_checkpoint / "config.json"
        ),
        "tokenizer_identity": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "model_identity": getattr(model.config, "_name_or_path", model.__class__.__name__),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "full_parameter_training": True,
        "target_qa_used": False,
        "checkpoint_selection": "final_state_after_one_corpus_pass",
        "provenance": provenance,
        "tokenization": token_statistics,
        "training": plan,
    }
    write_yaml(run_config_path, run_record)

    step_records: list[dict[str, Any]] = []
    weighted_loss = 0.0
    loss_token_count = 0
    observed_supervised_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        observed_supervised_tokens += int((batch["labels"] != -100).sum().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), plan["max_grad_norm"]
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        contributing_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        weighted_loss += float(loss.detach().item()) * contributing_tokens
        loss_token_count += contributing_tokens
        step_records.append(
            {
                "record_type": "optimizer_step",
                "step": step,
                "loss": float(loss.detach().item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm),
                "supervised_tokens": int((batch["labels"] != -100).sum().item()),
            }
        )

    if len(step_records) != plan["optimizer_steps"]:
        raise RuntimeError("CPT optimizer-step accounting is inconsistent")
    if observed_supervised_tokens != token_statistics["supervised_tokens"]:
        raise RuntimeError("CPT training did not consume every supervised token once")

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.mkdir(parents=True, exist_ok=True)
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
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_checkpoint / "training_metadata.json", summary)
    write_jsonl(
        train_log_path,
        [{"record_type": "configuration", **run_record}, *step_records, summary],
    )
    return summary

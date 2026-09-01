"""Closed-book supervised fine-tuning on verified target-database QA records."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

from evaluation.inference import (
    HOP_NAMES,
    PROMPT_TEMPLATE,
    format_question_prompt,
    load_verified_qa_split,
)
from training.cpt import enable_full_parameter_training
from utils.hashing import hash_file
from utils.io import read_json, write_json, write_jsonl, write_yaml

LEGAL_TARGET_SFT_SPLIT = "validation"


def load_target_sft_records(
    qa_condition_dir: str | Path,
    *,
    training_split: str,
    table_count: int,
    fact_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the authenticated validation QA only; the test directory is never read."""
    if training_split != LEGAL_TARGET_SFT_SPLIT:
        raise ValueError(
            "target SFT may use only the validation QA split; the test split is "
            "held out and forbidden"
        )
    condition_dir = Path(qa_condition_dir)
    split_dir = condition_dir / LEGAL_TARGET_SFT_SPLIT
    records, verified = load_verified_qa_split(
        split_dir,
        split=LEGAL_TARGET_SFT_SPLIT,
        expected_table_count=table_count,
        expected_fact_count=fact_count,
    )
    split_manifest_path = condition_dir / "split_manifest.json"
    split_manifest = read_json(split_manifest_path)
    qa_manifest = verified["qa_manifest"]
    if split_manifest.get("target_qa_training_generated") is not False:
        raise ValueError(
            "split_manifest.json must declare that target-QA training data was not generated"
        )
    if split_manifest.get("split_method_version") != (
        "chain_order_3_reserved_1_validation_1_test_v1"
    ):
        raise ValueError("unsupported target-QA chain split method")
    if qa_manifest.get("chain_count") != split_manifest.get(
        "validation_chain_count"
    ):
        raise ValueError("validation chain count is inconsistent across QA manifests")
    if (table_count, fact_count) == (12, 10_000):
        actual_chain_counts = (
            split_manifest.get("reserved_chain_count"),
            split_manifest.get("validation_chain_count"),
            split_manifest.get("test_chain_count"),
        )
        if actual_chain_counts != (150, 50, 50):
            raise ValueError(
                "canonical target SFT requires the 150/50/50 chain split"
            )

    hop_counts = Counter(f"H{record['hop']}" for record in records)
    counts = {hop_name: hop_counts.get(hop_name, 0) for hop_name in HOP_NAMES}
    if counts != {
        hop_name: qa_manifest["counts"][hop_name]["final_retained_count"]
        for hop_name in HOP_NAMES
    }:
        raise ValueError("loaded target-SFT hop counts are inconsistent")
    return records, {
        "training_split": LEGAL_TARGET_SFT_SPLIT,
        "total_examples": len(records),
        "hop_counts": counts,
        "qa_manifest_sha256": verified["qa_manifest_sha256"],
        "split_manifest_sha256": verified["qa_split_manifest_sha256"],
        "input_file_sha256": verified["input_hashes"],
        "source_database_sha256": qa_manifest["source_database_sha256"],
        "source_database_manifest_sha256": qa_manifest[
            "source_database_manifest_sha256"
        ],
        "zero_context": True,
        "test_split_used": False,
    }


def encode_target_sft_example(
    record: dict[str, Any], tokenizer: Any, *, context_length: int
) -> dict[str, Any]:
    """Encode evaluator-compatible question prompting with answer-only labels."""
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("target_sft.context_length must be a positive integer")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    question = record.get("question")
    gold_answer = record.get("gold_answer")
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ValueError("target-SFT gold_answer must be non-empty text")
    prompt = format_question_prompt(question)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = list(tokenizer.encode(gold_answer, add_special_tokens=False))
    if not answer_ids:
        raise ValueError("target-SFT gold_answer tokenization is empty")
    input_ids = [*prompt_ids, *answer_ids, tokenizer.eos_token_id]
    if len(input_ids) > context_length:
        raise ValueError(
            f"target-SFT record {record.get('id', '<unknown>')} has "
            f"{len(input_ids)} tokens and exceeds context length {context_length}; "
            "truncation is forbidden"
        )
    labels = [-100] * len(prompt_ids) + [*answer_ids, tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt": prompt,
        "prompt_length": len(prompt_ids),
        "answer_token_count": len(answer_ids),
        "supervised_token_count": len(answer_ids) + 1,
        "sequence_length": len(input_ids),
        "record_id": record.get("id"),
        "hop": record.get("hop"),
    }


def collate_target_sft_examples(
    examples: list[dict[str, Any]], *, pad_token_id: int
) -> dict[str, Any]:
    """Right-pad causal-LM examples while masking every padding label."""
    if not examples:
        raise ValueError("cannot collate an empty target-SFT batch")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("target SFT requires torch") from exc
    width = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(example["attention_mask"] + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_target_sft_training_plan(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    example_count: int,
) -> dict[str, Any]:
    """Validate target-SFT settings and calculate exact update accounting."""
    settings = config.get("target_sft")
    if not isinstance(settings, dict):
        raise ValueError("target_sft configuration section is required")
    if (
        isinstance(example_count, bool)
        or not isinstance(example_count, int)
        or example_count <= 0
    ):
        raise ValueError("example_count must be a positive integer")

    def positive_int(key: str) -> int:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"target_sft.{key} must be a positive integer")
        return value

    def non_negative_int(key: str) -> int:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"target_sft.{key} must be a non-negative integer")
        return value

    def boolean(key: str) -> bool:
        value = settings.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"target_sft.{key} must be a boolean")
        return value

    def number(key: str, *, allow_zero: bool = False) -> float:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"target_sft.{key} must be a number")
        value = float(value)
        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"target_sft.{key} must be {qualifier}")
        return value

    if settings.get("training_split") != LEGAL_TARGET_SFT_SPLIT:
        raise ValueError("target_sft.training_split must be validation")
    batch_size = positive_int("batch_size")
    accumulation_steps = positive_int("gradient_accumulation_steps")
    epochs = positive_int("epochs")
    context_length = positive_int("context_length")
    dataloader_workers = non_negative_int("dataloader_workers")
    learning_rate = number("learning_rate")
    weight_decay = number("weight_decay", allow_zero=True)
    epsilon = number("epsilon")
    warmup_ratio = number("warmup_ratio", allow_zero=True)
    max_grad_norm = number("max_grad_norm")
    if warmup_ratio > 1.0:
        raise ValueError("target_sft.warmup_ratio must be between 0 and 1")
    betas = settings.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("target_sft.betas must contain exactly two numbers")
    normalized_betas: list[float] = []
    for index, beta in enumerate(betas):
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise ValueError(f"target_sft.betas[{index}] must be a number")
        normalized_beta = float(beta)
        if not 0.0 <= normalized_beta < 1.0:
            raise ValueError(f"target_sft.betas[{index}] must be in [0, 1)")
        normalized_betas.append(normalized_beta)
    for key, expected in (
        ("optimizer", "adamw"),
        ("scheduler", "cosine"),
        ("precision", "bf16"),
    ):
        if str(settings.get(key, "")).lower() != expected:
            raise ValueError(f"target_sft.{key} must be {expected}")
    shuffle = boolean("shuffle")
    gradient_checkpointing = boolean("gradient_checkpointing")
    fused_optimizer = boolean("fused_optimizer")
    pin_memory = boolean("pin_memory")
    drop_last = boolean("drop_last")
    answer_only_loss = boolean("answer_only_loss")
    supervise_eos = boolean("supervise_eos")
    if drop_last:
        raise ValueError("target_sft.drop_last must be false to train on all QA records")
    if not answer_only_loss or not supervise_eos:
        raise ValueError("target SFT requires answer-only loss and supervised EOS")
    seed = config.get("experiment", {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment.seed must be a non-negative integer")

    microbatches_per_epoch = math.ceil(example_count / batch_size)
    optimizer_steps_per_epoch = math.ceil(
        microbatches_per_epoch / accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = math.ceil(total_optimizer_steps * warmup_ratio)
    return {
        "stage": "target-sft",
        "T": table_count,
        "N": fact_count,
        "training_split": LEGAL_TARGET_SFT_SPLIT,
        "example_count": example_count,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "epochs": epochs,
        "context_length": context_length,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": normalized_betas,
        "epsilon": epsilon,
        "scheduler": "cosine",
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "precision": "bf16",
        "shuffle": shuffle,
        "gradient_checkpointing": gradient_checkpointing,
        "fused_optimizer_requested": fused_optimizer,
        "fused_optimizer_actually_used": None,
        "dataloader_workers": dataloader_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "answer_only_loss": True,
        "supervise_eos": True,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "optimizer_steps": total_optimizer_steps,
        "seed": seed,
        "test_split_used": False,
    }


def seeded_dataloader_generator(torch_module: Any, seed: int) -> Any:
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def build_target_sft_optimizer(
    torch_module: Any, parameters: Any, plan: dict[str, Any]
) -> tuple[Any, bool, str | None]:
    parameter_list = list(parameters)
    kwargs = {
        "lr": plan["learning_rate"],
        "weight_decay": plan["weight_decay"],
        "betas": tuple(plan["betas"]),
        "eps": plan["epsilon"],
    }
    if not plan["fused_optimizer_requested"]:
        return torch_module.optim.AdamW(parameter_list, **kwargs), False, None
    try:
        optimizer = torch_module.optim.AdamW(parameter_list, fused=True, **kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        return (
            torch_module.optim.AdamW(parameter_list, **kwargs),
            False,
            f"{type(exc).__name__}: {exc}",
        )
    return optimizer, bool(optimizer.defaults.get("fused", True)), None


def configure_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if enabled:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("source model does not support gradient checkpointing")
        model.gradient_checkpointing_enable()
    elif getattr(model, "is_gradient_checkpointing", False):
        if not hasattr(model, "gradient_checkpointing_disable"):
            raise RuntimeError("source model cannot disable gradient checkpointing")
        model.gradient_checkpointing_disable()


def ensure_target_sft_outputs_available(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    run_config_path: str | Path,
    train_log_path: str | Path,
) -> None:
    source = Path(source_checkpoint)
    output = Path(output_checkpoint)
    if source.resolve() == output.resolve():
        raise ValueError("source and target-SFT checkpoint paths must differ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"target-SFT checkpoint path is not an empty directory: {output}"
        )
    for path in (Path(run_config_path), Path(train_log_path)):
        if path.exists():
            raise FileExistsError(f"target-SFT run artifact already exists: {path}")


def _load_local_model_and_tokenizer(source_checkpoint: Path) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("target SFT requires transformers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source_checkpoint, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            source_checkpoint, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"unable to load local source checkpoint {source_checkpoint}: {exc}"
        ) from exc
    if tokenizer.eos_token_id is None:
        raise ValueError("source tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run_target_sft_training(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    run_config_path: str | Path,
    train_log_path: str | Path,
    qa_condition_dir: str | Path,
) -> dict[str, Any]:
    """Run full-parameter closed-book SFT without loading any held-out QA."""
    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    run_config_path = Path(run_config_path)
    train_log_path = Path(train_log_path)
    if not source_checkpoint.is_dir() or not (
        source_checkpoint / "config.json"
    ).is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {source_checkpoint}")
    ensure_target_sft_outputs_available(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        run_config_path=run_config_path,
        train_log_path=train_log_path,
    )
    records, provenance = load_target_sft_records(
        qa_condition_dir,
        training_split=config.get("target_sft", {}).get("training_split"),
        table_count=table_count,
        fact_count=fact_count,
    )
    plan = build_target_sft_training_plan(
        config,
        table_count=table_count,
        fact_count=fact_count,
        example_count=len(records),
    )
    started = time.perf_counter()
    model, tokenizer = _load_local_model_and_tokenizer(source_checkpoint)
    examples = [
        encode_target_sft_example(
            record, tokenizer, context_length=plan["context_length"]
        )
        for record in records
    ]
    model_context_limit = getattr(model.config, "max_position_embeddings", None)
    if model_context_limit is None:
        model_context_limit = getattr(model.config, "n_positions", None)
    if (
        isinstance(model_context_limit, int)
        and plan["context_length"] > model_context_limit
    ):
        raise ValueError(
            f"target_sft.context_length={plan['context_length']} exceeds the source "
            f"model limit of {model_context_limit}"
        )

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("target SFT requires torch and transformers") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("target SFT requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("target_sft.precision=bf16 requires BF16 CUDA support")
    device = torch.device("cuda")
    seed = plan["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device)
    model.train()
    previous_use_cache = model.config.use_cache
    model.config.use_cache = False
    configure_gradient_checkpointing(model, plan["gradient_checkpointing"])
    parameter_counts = enable_full_parameter_training(model)
    generator = (
        seeded_dataloader_generator(torch, seed) if plan["shuffle"] else None
    )
    loader = DataLoader(
        examples,
        batch_size=plan["batch_size"],
        shuffle=plan["shuffle"],
        collate_fn=partial(
            collate_target_sft_examples, pad_token_id=tokenizer.pad_token_id
        ),
        pin_memory=plan["pin_memory"],
        drop_last=False,
        num_workers=plan["dataloader_workers"],
        persistent_workers=plan["dataloader_workers"] > 0,
        generator=generator,
    )
    if len(loader) != plan["microbatches_per_epoch"]:
        raise RuntimeError("target-SFT DataLoader batch accounting is inconsistent")
    optimizer, fused_used, fused_fallback = build_target_sft_optimizer(
        torch, model.parameters(), plan
    )
    plan["fused_optimizer_actually_used"] = fused_used
    plan["fused_optimizer_fallback_reason"] = fused_fallback
    scheduler = get_scheduler(
        plan["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=plan["warmup_steps"],
        num_training_steps=plan["total_optimizer_steps"],
    )
    run_record = {
        "stage": "target-sft",
        "experiment": config["experiment"]["name"],
        "T": table_count,
        "N": fact_count,
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "source_checkpoint_config_sha256": hash_file(
            source_checkpoint / "config.json"
        ),
        "qa_training_split": LEGAL_TARGET_SFT_SPLIT,
        "total_examples": len(records),
        "hop_counts": provenance["hop_counts"],
        "qa_manifest_sha256": provenance["qa_manifest_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "input_file_sha256": provenance["input_file_sha256"],
        "prompt_format": PROMPT_TEMPLATE,
        "answer_only_loss": True,
        "eos_supervised": True,
        "full_parameter_training": True,
        "test_split_used": False,
        "checkpoint_selection": "final_state_after_configured_target_sft_epochs",
        "context_length": plan["context_length"],
        "batch_size": plan["batch_size"],
        "gradient_accumulation_steps": plan["gradient_accumulation_steps"],
        "effective_batch_size": plan["effective_batch_size"],
        "optimizer": plan["optimizer"],
        "fused_optimizer_requested": plan["fused_optimizer_requested"],
        "fused_optimizer_actually_used": plan[
            "fused_optimizer_actually_used"
        ],
        "learning_rate": plan["learning_rate"],
        "weight_decay": plan["weight_decay"],
        "scheduler": plan["scheduler"],
        "warmup_steps": plan["warmup_steps"],
        "epochs": plan["epochs"],
        "microbatches_per_epoch": plan["microbatches_per_epoch"],
        "optimizer_steps_per_epoch": plan["optimizer_steps_per_epoch"],
        "total_optimizer_steps": plan["total_optimizer_steps"],
        "precision": plan["precision"],
        "seed": plan["seed"],
        "tokenizer_identity": getattr(
            tokenizer, "name_or_path", tokenizer.__class__.__name__
        ),
        "model_identity": getattr(
            model.config, "_name_or_path", model.__class__.__name__
        ),
        "total_parameters": parameter_counts["total_parameters"],
        "trainable_parameters": parameter_counts["trainable_parameters"],
        "provenance": provenance,
        "training": plan,
    }
    write_yaml(run_config_path, run_record)

    log_records: list[dict[str, Any]] = [
        {"record_type": "configuration", **run_record}
    ]
    optimizer.zero_grad(set_to_none=True)
    total_weighted_loss = 0.0
    total_loss_tokens = 0
    optimizer_step = 0
    observed_examples = 0
    epoch_losses: list[dict[str, Any]] = []
    accumulation_steps = plan["gradient_accumulation_steps"]
    for epoch in range(1, plan["epochs"] + 1):
        epoch_weighted_loss = 0.0
        epoch_loss_tokens = 0
        epoch_examples = 0
        epoch_optimizer_steps = 0
        accumulated_loss_tokens = 0
        for microbatch, batch in enumerate(loader, start=1):
            batch_size = int(batch["input_ids"].shape[0])
            epoch_examples += batch_size
            observed_examples += batch_size
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                contributing_tokens = int(
                    (batch["labels"][:, 1:] != -100).sum().item()
                )
                if contributing_tokens <= 0:
                    raise RuntimeError("target-SFT batch has no supervised LM tokens")
                summed_loss = output.loss * contributing_tokens
            summed_loss.backward()
            detached_loss = float(output.loss.detach().item())
            epoch_weighted_loss += detached_loss * contributing_tokens
            epoch_loss_tokens += contributing_tokens
            total_weighted_loss += detached_loss * contributing_tokens
            total_loss_tokens += contributing_tokens
            accumulated_loss_tokens += contributing_tokens
            should_step = (
                microbatch % accumulation_steps == 0
                or microbatch == plan["microbatches_per_epoch"]
            )
            if not should_step:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_loss_tokens)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), plan["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            epoch_optimizer_steps += 1
            log_records.append(
                {
                    "record_type": "optimizer_step",
                    "epoch": epoch,
                    "step": optimizer_step,
                    "step_in_epoch": epoch_optimizer_steps,
                    "last_microbatch_in_epoch": microbatch,
                    "supervised_shifted_tokens": accumulated_loss_tokens,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            accumulated_loss_tokens = 0
        if accumulated_loss_tokens != 0:
            raise RuntimeError("target-SFT partial accumulation was not stepped")
        if epoch_examples != len(records):
            raise RuntimeError("target SFT did not consume every QA example in an epoch")
        if epoch_optimizer_steps != plan["optimizer_steps_per_epoch"]:
            raise RuntimeError("target-SFT per-epoch optimizer-step count is inconsistent")
        epoch_record = {
            "record_type": "epoch",
            "epoch": epoch,
            "train_loss": epoch_weighted_loss / epoch_loss_tokens,
            "examples": epoch_examples,
            "microbatches": plan["microbatches_per_epoch"],
            "optimizer_steps": epoch_optimizer_steps,
            "supervised_shifted_tokens": epoch_loss_tokens,
        }
        epoch_losses.append(epoch_record)
        log_records.append(epoch_record)
    if optimizer_step != plan["total_optimizer_steps"]:
        raise RuntimeError("target-SFT total optimizer-step accounting is inconsistent")
    if observed_examples != len(records) * plan["epochs"]:
        raise RuntimeError("target SFT did not use every example exactly once per epoch")

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.mkdir(parents=True, exist_ok=True)
    configure_gradient_checkpointing(model, False)
    model.config.use_cache = previous_use_cache
    model.save_pretrained(output_checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(output_checkpoint)
    summary = {
        "record_type": "summary",
        **run_record,
        "optimizer_steps": optimizer_step,
        "per_epoch_train_loss": epoch_losses,
        "training_loss": total_weighted_loss / total_loss_tokens,
        "observed_examples": observed_examples,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "test_split_used": False,
    }
    write_json(output_checkpoint / "training_metadata.json", summary)
    log_records.append(summary)
    write_jsonl(train_log_path, log_records)
    return summary

#!/usr/bin/env python3
"""Run Step 6A baseline inference or Step 6B skill training/evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config  # noqa: E402
from training.relational_qa import (  # noqa: E402
    PREFLIGHT_NAMESPACE,
    deterministic_json_bytes,
    encode_supervised_example,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    generate_continuations,
    generate_skill_dataset,
    load_gpt2,
    make_prediction_record,
    post_skill_decision,
    summarize_predictions,
    train_relational_skill,
    validate_prompt_lengths,
    verify_skill_isolation,
    write_baseline_dataset,
    write_skill_dataset,
)


BASELINE_DATASET_DIR = (
    PROJECT_ROOT
    / "datasets/qa/exp01_first_feasibility/preflight_relational_qa/baseline"
)
SKILL_DATASET_DIR = (
    PROJECT_ROOT
    / "datasets/qa/exp01_first_feasibility/preflight_relational_qa/skill_training"
)
BASELINE_RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/baseline/PLACEHOLDER_RUN"
)
AFTER_SKILL_RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/after_skill_training/PLACEHOLDER_RUN"
)
RUN_DIR = (
    PROJECT_ROOT
    / "runs/exp01_first_feasibility/relational_qa_training/PLACEHOLDER_RUN"
)
BASE_MODEL_DIR = PROJECT_ROOT / "models/base_models/gpt2"
SKILL_MODEL_DIR = PROJECT_ROOT / "models/trained_models/gpt2_relational_skill"
TARGET_WORLD_PATH = (
    PROJECT_ROOT
    / "datasets/generated_databases/exp01_first_feasibility/master_world/world.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _load_baseline_files(hops: list[int]) -> dict[int, list[dict]]:
    dataset: dict[int, list[dict]] = {}
    for hop in hops:
        path = BASELINE_DATASET_DIR / f"H{hop}.jsonl"
        dataset[hop] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return dataset


def _baseline_hashes(hops: list[int]) -> dict[str, str]:
    names = [*(f"H{hop}.jsonl" for hop in hops), "manifest.json"]
    return {name: _sha256(BASELINE_DATASET_DIR / name) for name in names}


def _evaluate(
    config: dict,
    dataset: dict[int, list[dict]],
    model_dir: Path,
    result_dir: Path,
    mode: str,
    dataset_manifest: dict,
) -> dict:
    import torch

    preflight = config["preflight"]
    hops = preflight["relational_hops"]
    started = time.perf_counter()
    model, tokenizer, device, precision = load_gpt2(model_dir)
    rows = [row for hop in sorted(dataset) for row in dataset[hop]]
    relational_prompts = [format_relational_prompt(row) for row in rows]
    copy_prompts = [format_copy_prompt(row["answer"]) for row in rows]
    relational_lengths = validate_prompt_lengths(
        relational_prompts, tokenizer, preflight["max_input_length"]
    )
    validate_prompt_lengths(copy_prompts, tokenizer, preflight["max_input_length"])
    generation_args = (
        model,
        tokenizer,
        device,
        preflight["max_input_length"],
        preflight["max_new_tokens"],
    )
    relational_outputs = generate_continuations(
        relational_prompts, *generation_args, batch_size=32
    )
    copy_outputs = generate_continuations(copy_prompts, *generation_args, batch_size=32)
    predictions_by_hop = {hop: [] for hop in hops}
    for row, continuation, copy_continuation in zip(
        rows, relational_outputs, copy_outputs, strict=True
    ):
        predictions_by_hop[row["hop"]].append(
            make_prediction_record(row, continuation, copy_continuation)
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    stem = "baseline" if mode == "baseline" else "after_skill_training"
    prediction_files: dict[str, dict] = {}
    for hop in sorted(predictions_by_hop):
        name = f"exp01_preflight_{stem}_H{hop}_predictions_PLACEHOLDER.jsonl"
        path = result_dir / name
        path.write_bytes(deterministic_json_bytes(predictions_by_hop[hop], jsonl=True))
        prediction_files[name] = {
            "examples": len(predictions_by_hop[hop]),
            "sha256": _sha256(path),
        }
    metrics = summarize_predictions(
        predictions_by_hop,
        float(preflight["baseline_strict_em_threshold"]),
        float(preflight["copy_control_threshold"]),
    )
    if mode == "after-skill":
        strict_by_hop = {
            hop: metrics["per_hop"][f"H{hop}"]["strict_exact_match_accuracy"]
            for hop in hops
        }
        metrics["decision"] = post_skill_decision(
            strict_by_hop, float(preflight["baseline_strict_em_threshold"])
        )
        metrics["copy_control_role"] = "diagnostic_only"
    metrics_name = f"exp01_preflight_{stem}_metrics_PLACEHOLDER.json"
    metrics_path = result_dir / metrics_name
    metrics_path.write_bytes(deterministic_json_bytes(metrics))
    run_manifest = {
        "experiment": config["experiment"]["name"],
        "mode": mode,
        "namespace": PREFLIGHT_NAMESPACE,
        "model_path": str(model_dir.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "precision": precision,
        "decoding": {"do_sample": False, "num_beams": 1},
        "maximum_prompt_tokens": max(relational_lengths),
        "dataset_manifest_sha256": _sha256(BASELINE_DATASET_DIR / "manifest.json"),
        "dataset": dataset_manifest,
        "prediction_files": prediction_files,
        "metrics_sha256": _sha256(metrics_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_name = f"exp01_preflight_{stem}_manifest_PLACEHOLDER.json"
    (result_dir / manifest_name).write_bytes(deterministic_json_bytes(run_manifest))
    for hop, values in metrics["per_hop"].items():
        print(
            f"{hop}: strict_em={values['strict_exact_match_accuracy']:.4f} "
            f"prefix_em={values['prefix_match_accuracy']:.4f} "
            f"copy_em={values['copy_control_exact_match_accuracy']:.4f}",
            flush=True,
        )
    print(
        "Overall copy-control EM: "
        f"{metrics['overall']['copy_control_exact_match_accuracy']:.4f}",
        flush=True,
    )
    print(f"Decision: {metrics['decision']}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _run_baseline(config: dict) -> dict:
    preflight = config["preflight"]
    seed = config["experiment"]["seed"]
    dataset = generate_baseline_dataset(
        seed,
        preflight["baseline_examples_per_hop"],
        preflight["relational_hops"],
    )
    manifest = write_baseline_dataset(BASELINE_DATASET_DIR, dataset, seed)
    return _evaluate(
        config, dataset, BASE_MODEL_DIR, BASELINE_RESULT_DIR, "baseline", manifest
    )


def _run_after_skill(config: dict) -> dict:
    if not (SKILL_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"skill checkpoint missing: {SKILL_MODEL_DIR}")
    dataset = _load_baseline_files(config["preflight"]["relational_hops"])
    manifest = json.loads(
        (BASELINE_DATASET_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    return _evaluate(
        config,
        dataset,
        SKILL_MODEL_DIR,
        AFTER_SKILL_RESULT_DIR,
        "after-skill",
        manifest,
    )


def _run_train_skill(config: dict) -> dict:
    import torch

    preflight = config["preflight"]
    skill_config = dict(preflight["skill_training"])
    seed = config["experiment"]["seed"]
    required = {
        "train_examples_per_hop": 5_000,
        "validation_examples_per_hop": 1_000,
        "epochs": 3,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "max_length": 256,
        "seed": seed,
    }
    if skill_config != required:
        raise ValueError(f"Step 6B requires exactly one fixed configuration: {required}")

    hops = preflight["relational_hops"]
    baseline_before = _baseline_hashes(hops)
    baseline_dataset = _load_baseline_files(hops)
    model, tokenizer, device, precision = load_gpt2(BASE_MODEL_DIR)
    skill_dataset = generate_skill_dataset(
        seed,
        skill_config["train_examples_per_hop"],
        skill_config["validation_examples_per_hop"],
        tokenizer,
        hops,
    )
    isolation = verify_skill_isolation(
        skill_dataset, baseline_dataset, TARGET_WORLD_PATH
    )
    rows_by_split = {
        split: [
            row
            for hop in sorted(skill_dataset[split])
            for row in skill_dataset[split][hop]
        ]
        for split in ("train", "validation")
    }
    prompts = {
        split: [format_relational_prompt(row) for row in rows]
        for split, rows in rows_by_split.items()
    }
    prompt_lengths = {
        split: validate_prompt_lengths(
            split_prompts, tokenizer, skill_config["max_length"]
        )
        for split, split_prompts in prompts.items()
    }
    encoded = {
        split: [
            encode_supervised_example(row, tokenizer, skill_config["max_length"])
            for row in rows
        ]
        for split, rows in rows_by_split.items()
    }
    maximum_sequence_tokens = max(
        len(row["input_ids"])
        for split_rows in encoded.values()
        for row in split_rows
    )
    maximum_prompt_tokens = max(
        length for split_lengths in prompt_lengths.values() for length in split_lengths
    )
    dataset_manifest = write_skill_dataset(
        SKILL_DATASET_DIR,
        skill_dataset,
        seed,
        isolation,
        maximum_prompt_tokens,
        maximum_sequence_tokens,
    )
    print(
        f"Generated skill data: train={len(rows_by_split['train'])} "
        f"validation={len(rows_by_split['validation'])} "
        f"max_prompt_tokens={maximum_prompt_tokens} "
        f"max_sequence_tokens={maximum_sequence_tokens}",
        flush=True,
    )
    print(f"Isolation intersections: {isolation}", flush=True)

    base_hashes_before = _directory_hashes(BASE_MODEL_DIR)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_config_path = RUN_DIR / "exp01_relqa_config_PLACEHOLDER.yaml"
    run_config_record = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "base_model_path": str(BASE_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "checkpoint_path": str(SKILL_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "precision": precision,
        "training": skill_config,
        "dataset_manifest_sha256": _sha256(SKILL_DATASET_DIR / "manifest.json"),
        "target_database_facts_used": False,
    }
    run_config_path.write_text(
        yaml.safe_dump(run_config_record, sort_keys=False), encoding="utf-8"
    )
    records, summary = train_relational_skill(
        model,
        tokenizer,
        device,
        rows_by_split["train"],
        skill_dataset["validation"],
        skill_config,
        SKILL_MODEL_DIR,
        preflight["max_new_tokens"],
    )
    train_log_path = RUN_DIR / "exp01_relqa_trainlog_PLACEHOLDER.jsonl"
    log_rows = [
        {"record_type": "configuration", **run_config_record},
        *({"record_type": "epoch", **record} for record in records),
        {"record_type": "summary", **summary},
    ]
    train_log_path.write_bytes(deterministic_json_bytes(log_rows, jsonl=True))
    if _directory_hashes(BASE_MODEL_DIR) != base_hashes_before:
        raise RuntimeError("original GPT-2 checkpoint changed during skill training")
    if _baseline_hashes(hops) != baseline_before:
        raise RuntimeError("Step 6A held-out files changed during skill training")
    del model, tokenizer, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = _run_after_skill(config)
    return {
        "dataset": dataset_manifest,
        "training": {"epochs": records, **summary},
        "after_skill": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "train-skill", "after-skill"), required=True
    )
    args = parser.parse_args()
    config = load_config()
    seed = config["experiment"]["seed"]
    random.seed(seed)
    if args.mode == "baseline":
        _run_baseline(config)
    elif args.mode == "after-skill":
        _run_after_skill(config)
    else:
        _run_train_skill(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

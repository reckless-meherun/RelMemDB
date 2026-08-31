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
    PRIMITIVE_TRAIN_NAMESPACE,
    PRIMITIVE_VALIDATION_NAMESPACE,
    curriculum_start_checkpoint,
    deterministic_json_bytes,
    encode_supervised_example,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    generate_continuations,
    generate_primitive_dataset,
    generate_skill_dataset,
    load_gpt2,
    make_prediction_record,
    post_skill_decision,
    primitive_gate_decision,
    select_best_epoch,
    step6c_decision,
    summarize_predictions,
    train_primitive_skill,
    train_relational_skill,
    validate_prompt_lengths,
    verify_skill_isolation,
    verify_step6c_isolation,
    write_baseline_dataset,
    write_primitive_dataset,
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
PRIMITIVE_DATASET_DIR = (
    PROJECT_ROOT
    / "datasets/qa/exp01_first_feasibility/preflight_relational_qa/primitive_training"
)
BASELINE_RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/baseline/PLACEHOLDER_RUN"
)
AFTER_SKILL_RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/after_skill_training/PLACEHOLDER_RUN"
)
STEP6C_RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/step6c/PLACEHOLDER_RUN"
)
RUN_DIR = (
    PROJECT_ROOT
    / "runs/exp01_first_feasibility/relational_qa_training/PLACEHOLDER_RUN"
)
STEP6C_RUN_DIR = (
    PROJECT_ROOT
    / "runs/exp01_first_feasibility/relational_qa_training/step6c/PLACEHOLDER_RUN"
)
BASE_MODEL_DIR = PROJECT_ROOT / "models/base_models/gpt2"
SKILL_MODEL_DIR = PROJECT_ROOT / "models/trained_models/gpt2_relational_skill"
PRIMITIVE_MODEL_DIR = PROJECT_ROOT / "models/trained_models/gpt2_primitive_skill"
CURRICULUM_MODEL_DIR = PROJECT_ROOT / "models/trained_models/gpt2_relational_curriculum"
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


def _skill_data_hashes() -> dict[str, str]:
    return {
        name: _sha256(SKILL_DATASET_DIR / name)
        for name in ("train.jsonl", "val.jsonl", "manifest.json")
    }


def _load_frozen_skill_files() -> dict[str, list[dict]]:
    manifest = json.loads((SKILL_DATASET_DIR / "manifest.json").read_text())
    dataset: dict[str, list[dict]] = {}
    for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl")):
        path = SKILL_DATASET_DIR / filename
        if _sha256(path) != manifest["files"][filename]["sha256"]:
            raise RuntimeError(f"frozen Step-6B file hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != manifest["files"][filename]["examples"]:
            raise RuntimeError(f"frozen Step-6B row count mismatch: {path}")
        dataset[split] = rows
    return dataset


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
    stem = {
        "baseline": "baseline",
        "after-skill": "after_skill_training",
        "after-curriculum": "after_curriculum",
    }[mode]
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
    if mode in ("after-skill", "after-curriculum"):
        strict_by_hop = {
            hop: metrics["per_hop"][f"H{hop}"]["strict_exact_match_accuracy"]
            for hop in hops
        }
        if mode == "after-curriculum":
            metrics["decision"] = step6c_decision(
                {"relation": 1.0, "attribute": 1.0},
                strict_by_hop,
                float(preflight["baseline_strict_em_threshold"]),
            )
        else:
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


def _run_step6c(config: dict) -> dict:
    """Run the gated primitive-to-relational Step-6C diagnostic curriculum."""
    import torch

    preflight = config["preflight"]
    seed = config["experiment"]["seed"]
    primitive_config = dict(preflight["primitive_training"])
    skill_config = dict(preflight["skill_training"])
    fixed_optimizer = {
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
    required_primitive = {
        "train_examples_per_type": 5_000,
        "validation_examples_per_type": 1_000,
        **fixed_optimizer,
    }
    required_relational = {
        "train_examples_per_hop": 5_000,
        "validation_examples_per_hop": 1_000,
        **fixed_optimizer,
    }
    if primitive_config != required_primitive:
        raise ValueError(
            f"Step 6C primitive phase requires exactly: {required_primitive}"
        )
    if skill_config != required_relational:
        raise ValueError(
            f"Step 6C relational phase requires exactly: {required_relational}"
        )

    hops = preflight["relational_hops"]
    baseline_before = _baseline_hashes(hops)
    skill_data_before = _skill_data_hashes()
    base_before = _directory_hashes(BASE_MODEL_DIR)
    target_before = _sha256(TARGET_WORLD_PATH)
    baseline_dataset = _load_baseline_files(hops)
    relational_dataset = _load_frozen_skill_files()

    model, tokenizer, device, precision = load_gpt2(BASE_MODEL_DIR)
    if device.type != "cuda" or precision != "bfloat16":
        raise RuntimeError("Step 6C requires BF16 CUDA")
    primitive_dataset = generate_primitive_dataset(
        seed,
        primitive_config["train_examples_per_type"],
        primitive_config["validation_examples_per_type"],
        tokenizer,
    )
    isolation = verify_step6c_isolation(
        primitive_dataset,
        relational_dataset,
        baseline_dataset,
        TARGET_WORLD_PATH,
    )
    primitive_rows_by_split = {
        split: [
            row
            for qa_type in ("relation", "attribute")
            for row in primitive_dataset[split][qa_type]
        ]
        for split in ("train", "validation")
    }
    prompts = [
        format_relational_prompt(row)
        for rows in primitive_rows_by_split.values()
        for row in rows
    ]
    prompt_lengths = validate_prompt_lengths(
        prompts, tokenizer, primitive_config["max_length"]
    )
    encoded = [
        encode_supervised_example(row, tokenizer, primitive_config["max_length"])
        for rows in primitive_rows_by_split.values()
        for row in rows
    ]
    primitive_manifest = write_primitive_dataset(
        PRIMITIVE_DATASET_DIR,
        primitive_dataset,
        seed,
        isolation,
        max(prompt_lengths),
        max(len(row["input_ids"]) for row in encoded),
    )
    print(
        "Generated primitive data: "
        f"train={len(primitive_rows_by_split['train'])} "
        f"validation={len(primitive_rows_by_split['validation'])} "
        f"max_prompt_tokens={max(prompt_lengths)} "
        f"max_sequence_tokens={max(len(row['input_ids']) for row in encoded)}",
        flush=True,
    )
    print(f"Step-6C isolation: {isolation}", flush=True)

    STEP6C_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_config_record = {
        "experiment": config["experiment"]["name"],
        "step": "6C",
        "seed": seed,
        "base_model_path": str(BASE_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "primitive_checkpoint_path": str(
            PRIMITIVE_MODEL_DIR.relative_to(PROJECT_ROOT)
        ),
        "curriculum_checkpoint_path": str(
            CURRICULUM_MODEL_DIR.relative_to(PROJECT_ROOT)
        ),
        "precision": precision,
        "primitive_training": primitive_config,
        "relational_training": skill_config,
        "primitive_manifest_sha256": _sha256(PRIMITIVE_DATASET_DIR / "manifest.json"),
        "frozen_relational_manifest_sha256": _sha256(
            SKILL_DATASET_DIR / "manifest.json"
        ),
        "step6a_baseline_used_for_training": False,
        "step6a_baseline_used_for_model_selection": False,
        "target_database_facts_used": False,
    }
    (STEP6C_RUN_DIR / "exp01_step6c_config_PLACEHOLDER.yaml").write_text(
        yaml.safe_dump(run_config_record, sort_keys=False), encoding="utf-8"
    )
    primitive_records, primitive_summary = train_primitive_skill(
        model,
        tokenizer,
        device,
        primitive_rows_by_split["train"],
        primitive_dataset["validation"],
        primitive_config,
        PRIMITIVE_MODEL_DIR,
        preflight["max_new_tokens"],
        BASE_MODEL_DIR,
    )
    selected_primitive = select_best_epoch(primitive_records)
    primitive_scores = {
        "relation": selected_primitive["validation_relation_lookup_EM"],
        "attribute": selected_primitive["validation_attribute_lookup_EM"],
    }
    primitive_gate = primitive_gate_decision(primitive_scores, 0.90)
    primitive_log = [
        {"record_type": "configuration", **run_config_record},
        *({"record_type": "epoch", **record} for record in primitive_records),
        {
            "record_type": "summary",
            **primitive_summary,
            "primitive_scores": primitive_scores,
            "primitive_gate": primitive_gate,
        },
    ]
    (STEP6C_RUN_DIR / "exp01_step6c_primitive_trainlog_PLACEHOLDER.jsonl").write_bytes(
        deterministic_json_bytes(primitive_log, jsonl=True)
    )
    report: dict = {
        "step": "6C",
        "primitive_dataset": primitive_manifest,
        "isolation": isolation,
        "primitive_training": {
            "epochs": primitive_records,
            **primitive_summary,
            "selected_scores": primitive_scores,
        },
        "primitive_gate": primitive_gate,
        "phase2_started": False,
        "step6a_baseline_used_for_training": False,
        "step6a_baseline_used_for_model_selection": False,
        "target_database_facts_used": False,
    }

    def verify_frozen_inputs() -> None:
        if _directory_hashes(BASE_MODEL_DIR) != base_before:
            raise RuntimeError("original GPT-2 checkpoint changed during Step 6C")
        if _baseline_hashes(hops) != baseline_before:
            raise RuntimeError("frozen Step-6A files changed during Step 6C")
        if _skill_data_hashes() != skill_data_before:
            raise RuntimeError("frozen corrected Step-6B files changed during Step 6C")
        if _sha256(TARGET_WORLD_PATH) != target_before:
            raise RuntimeError("target master world changed during Step 6C")

    verify_frozen_inputs()
    STEP6C_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = STEP6C_RESULT_DIR / "exp01_step6c_report_PLACEHOLDER.json"
    if primitive_gate == "primitive_skill_failure":
        report["decision"] = "primitive_skill_failure"
        report_path.write_bytes(deterministic_json_bytes(report))
        print("Primitive gate failed; relational curriculum was not started.", flush=True)
        del model, tokenizer, encoded
        gc.collect()
        torch.cuda.empty_cache()
        return report

    start_checkpoint = curriculum_start_checkpoint(
        primitive_scores, PRIMITIVE_MODEL_DIR, 0.90
    )
    if start_checkpoint != PRIMITIVE_MODEL_DIR:
        raise RuntimeError("invalid Step-6C curriculum start checkpoint")
    del model, tokenizer, encoded
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer, device, curriculum_precision = load_gpt2(start_checkpoint)
    if curriculum_precision != "bfloat16":
        raise RuntimeError("Step 6C relational curriculum requires BF16 CUDA")
    relational_validation = {
        hop: [row for row in relational_dataset["validation"] if row["hop"] == hop]
        for hop in hops
    }
    relational_records, relational_summary = train_relational_skill(
        model,
        tokenizer,
        device,
        relational_dataset["train"],
        relational_validation,
        skill_config,
        CURRICULUM_MODEL_DIR,
        preflight["max_new_tokens"],
        source_checkpoint=start_checkpoint,
    )
    report["phase2_started"] = True
    report["relational_training"] = {
        "epochs": relational_records,
        **relational_summary,
    }
    relational_log = [
        {"record_type": "configuration", **run_config_record},
        *({"record_type": "epoch", **record} for record in relational_records),
        {"record_type": "summary", **relational_summary},
    ]
    (STEP6C_RUN_DIR / "exp01_step6c_relational_trainlog_PLACEHOLDER.jsonl").write_bytes(
        deterministic_json_bytes(relational_log, jsonl=True)
    )
    verify_frozen_inputs()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    baseline_manifest = json.loads(
        (BASELINE_DATASET_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    held_out = _evaluate(
        config,
        baseline_dataset,
        CURRICULUM_MODEL_DIR,
        STEP6C_RESULT_DIR,
        "after-curriculum",
        baseline_manifest,
    )
    held_out_scores = {
        hop: held_out["per_hop"][f"H{hop}"]["strict_exact_match_accuracy"]
        for hop in hops
    }
    report["held_out_step6a"] = held_out
    report["decision"] = step6c_decision(primitive_scores, held_out_scores, 0.90)
    verify_frozen_inputs()
    report_path.write_bytes(deterministic_json_bytes(report))
    print(f"Step-6C decision: {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("baseline", "train-skill", "after-skill", "step6c"),
        required=True,
    )
    args = parser.parse_args()
    config = load_config()
    seed = config["experiment"]["seed"]
    random.seed(seed)
    if args.mode == "baseline":
        _run_baseline(config)
    elif args.mode == "after-skill":
        _run_after_skill(config)
    elif args.mode == "train-skill":
        _run_train_skill(config)
    else:
        _run_step6c(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

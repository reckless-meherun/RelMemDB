#!/usr/bin/env python3
"""Run the original GPT-2 Small Step 6A relational preflight."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config  # noqa: E402
from training.relational_qa import (  # noqa: E402
    PREFLIGHT_NAMESPACE,
    deterministic_json_bytes,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    generate_continuations,
    load_gpt2,
    make_prediction_record,
    summarize_predictions,
    validate_prompt_lengths,
    write_baseline_dataset,
)


DATASET_DIR = (
    PROJECT_ROOT
    / "datasets/qa/exp01_first_feasibility/preflight_relational_qa/baseline"
)
RESULT_DIR = (
    PROJECT_ROOT
    / "results/exp01_first_feasibility/preflight_relational_qa/baseline/PLACEHOLDER_RUN"
)
MODEL_DIR = PROJECT_ROOT / "models/base_models/gpt2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline",), required=True)
    args = parser.parse_args()
    del args

    config = load_config()
    preflight = config["preflight"]
    hops = preflight["relational_hops"]
    count = preflight["baseline_examples_per_hop"]
    if hops != [1, 2, 3] or count != 200:
        raise ValueError("Step 6A requires hops [1, 2, 3] and 200 examples per hop")
    for key in ("baseline_strict_em_threshold", "copy_control_threshold"):
        if not 0.0 <= float(preflight[key]) <= 1.0:
            raise ValueError(f"preflight.{key} must be between 0 and 1")
    for key in ("max_input_length", "max_new_tokens"):
        if isinstance(preflight[key], bool) or int(preflight[key]) <= 0:
            raise ValueError(f"preflight.{key} must be a positive integer")

    seed = config["experiment"]["seed"]
    random.seed(seed)
    dataset = generate_baseline_dataset(seed, count, hops)
    dataset_manifest = write_baseline_dataset(DATASET_DIR, dataset, seed)
    print(f"Generated deterministic baseline dataset: {sum(map(len, dataset.values()))} examples")

    try:
        model, tokenizer, device, precision = load_gpt2(MODEL_DIR)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows = [example for hop in sorted(dataset) for example in dataset[hop]]
    relational_prompts = [format_relational_prompt(example) for example in rows]
    copy_prompts = [format_copy_prompt(example["answer"]) for example in rows]
    max_input_length = preflight["max_input_length"]
    relational_lengths = validate_prompt_lengths(
        relational_prompts, tokenizer, max_input_length
    )
    validate_prompt_lengths(copy_prompts, tokenizer, max_input_length)
    lengths_by_hop = {
        hop: [
            length
            for example, length in zip(rows, relational_lengths, strict=True)
            if example["hop"] == hop
        ]
        for hop in hops
    }
    print(f"Max prompt tokens overall: {max(relational_lengths)}")
    for hop in hops:
        print(f"Max H{hop} prompt tokens: {max(lengths_by_hop[hop])}")
    generation_args = (
        model,
        tokenizer,
        device,
        max_input_length,
        preflight["max_new_tokens"],
    )
    relational_outputs = generate_continuations(relational_prompts, *generation_args)
    copy_outputs = generate_continuations(copy_prompts, *generation_args)

    predictions_by_hop = {hop: [] for hop in hops}
    for example, continuation, copy_continuation in zip(
        rows, relational_outputs, copy_outputs, strict=True
    ):
        predictions_by_hop[example["hop"]].append(
            make_prediction_record(example, continuation, copy_continuation)
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_files: dict[str, dict[str, object]] = {}
    for hop in sorted(predictions_by_hop):
        name = f"exp01_preflight_baseline_H{hop}_predictions_PLACEHOLDER.jsonl"
        path = RESULT_DIR / name
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
    metrics_path = RESULT_DIR / "exp01_preflight_baseline_metrics_PLACEHOLDER.json"
    metrics_path.write_bytes(deterministic_json_bytes(metrics))
    run_manifest = {
        "experiment": config["experiment"]["name"],
        "mode": "baseline",
        "namespace": PREFLIGHT_NAMESPACE,
        "model_id": "gpt2",
        "model_path": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "precision": precision,
        "decoding": {"do_sample": False, "num_beams": 1},
        "dataset_manifest_sha256": _sha256(DATASET_DIR / "manifest.json"),
        "dataset": dataset_manifest,
        "prediction_files": prediction_files,
        "metrics_sha256": _sha256(metrics_path),
    }
    manifest_path = RESULT_DIR / "exp01_preflight_baseline_manifest_PLACEHOLDER.json"
    manifest_path.write_bytes(deterministic_json_bytes(run_manifest))

    for hop, values in metrics["per_hop"].items():
        print(
            f"{hop}: strict_em={values['strict_exact_match_accuracy']:.4f} "
            f"prefix_em={values['prefix_match_accuracy']:.4f} "
            f"copy_em={values['copy_control_exact_match_accuracy']:.4f}"
        )
    print(
        "Overall copy-control EM: "
        f"{metrics['overall']['copy_control_exact_match_accuracy']:.4f}"
    )
    print(f"Decision: {metrics['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

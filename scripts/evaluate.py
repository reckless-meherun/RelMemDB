"""Evaluate one local checkpoint on one closed-book target-QA split."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from evaluation.inference import (
    EVALUATION_SEED,
    MAX_NEW_TOKENS,
    PROMPT_TEMPLATE,
    evaluate_with_local_checkpoint,
    load_verified_qa_split,
    prepare_result_directory,
)
from evaluation.metrics import compute_evaluation_metrics
from utils.hashing import hash_file
from utils.io import write_json, write_jsonl
from utils.paths import evaluation_result_dir, qa_condition_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic closed-book target-QA evaluation"
    )
    parser.add_argument("--table-count", type=int, required=True)
    parser.add_argument("--fact-count", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _resolve_checkpoint(path: Path) -> Path:
    checkpoint = path if path.is_absolute() else PROJECT_ROOT / path
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"local checkpoint is missing: {checkpoint}")
    return checkpoint


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    evaluation = config["evaluation"]
    batch_size = (
        evaluation["batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    checkpoint = _resolve_checkpoint(args.checkpoint)
    qa_dir = qa_condition_dir(args.table_count, args.fact_count) / args.split
    qa_records, qa_provenance = load_verified_qa_split(
        qa_dir,
        split=args.split,
        expected_table_count=args.table_count,
        expected_fact_count=args.fact_count,
    )
    output_dir = evaluation_result_dir(
        args.table_count, args.fact_count, args.split, args.run_name
    )
    prepare_result_directory(output_dir)
    predictions, model_identity = evaluate_with_local_checkpoint(
        qa_records,
        checkpoint=checkpoint,
        batch_size=batch_size,
        context_length=evaluation["context_length"],
        max_new_tokens=evaluation["max_new_tokens"],
    )
    metrics = compute_evaluation_metrics(predictions)
    checkpoint_config = checkpoint / "config.json"
    evaluation_config = {
        "experiment_name": config["experiment"]["name"],
        "T": args.table_count,
        "N": args.fact_count,
        "split": args.split,
        "run_name": args.run_name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_config_sha256": (
            hash_file(checkpoint_config) if checkpoint_config.is_file() else None
        ),
        **model_identity,
        "qa_manifest_sha256": qa_provenance["qa_manifest_sha256"],
        "qa_split_manifest_sha256": qa_provenance[
            "qa_split_manifest_sha256"
        ],
        "qa_input_hashes": qa_provenance["input_hashes"],
        "qa_record_count": len(qa_records),
        "prompt_format": PROMPT_TEMPLATE,
        "zero_context": True,
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "continuation_only": True,
            "padding_side": "left",
            "pad_token_policy": "eos_if_missing",
        },
        "max_new_tokens": MAX_NEW_TOKENS,
        "context_length": evaluation["context_length"],
        "batch_size": batch_size,
        "primary_metric": "normalized_exact_match",
        "seed": EVALUATION_SEED,
    }
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "evaluation_config.json", evaluation_config)
    print(
        f"Evaluation complete: split={args.split}, records={len(predictions)}, "
        f"normalized_EM={metrics['overall']['normalized_exact_match_accuracy']:.6f}, "
        f"output={output_dir.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()

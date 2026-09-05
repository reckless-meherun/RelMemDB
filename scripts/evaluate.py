"""Evaluate one local checkpoint on one closed-book target-QA split."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.qa_reference import verify_qa_reference_compatibility
from evaluation.inference import (
    EVALUATION_SEED,
    MAX_NEW_TOKENS,
    PROMPT_TEMPLATE,
    evaluate_with_local_checkpoint,
    load_verified_qa_split,
    prepare_result_directory,
)
from evaluation.metrics import compute_evaluation_metrics
from experiment import (
    ExperimentCondition,
    qa_reference_values,
    verify_checkpoint_layers,
)
from utils.hashing import hash_file
from utils.io import read_json, write_json, write_jsonl
from utils.paths import (
    EXP02_RESULTS_DIR,
    evaluation_result_dir,
    exp2_artifact_stem,
    qa_reference_dir,
    safe_component,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic closed-book target-QA evaluation"
    )
    parser.add_argument("--table-count", type=int)
    parser.add_argument("--fact-count", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--qa-data-dir", type=Path)
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
    if config["experiment"]["name"] == "exp02_capacity_boundary":
        _evaluate_exp2(args, config)
        return
    if any(value is None for value in (args.table_count, args.fact_count, args.layers, args.run_name)):
        raise SystemExit("ERROR: Experiment-1 evaluation requires --table-count, --fact-count, --layers, and --run-name.")
    condition = ExperimentCondition.from_config(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        layers=args.layers,
    )
    evaluation = config["evaluation"]
    batch_size = (
        evaluation["batch_size"] if args.batch_size is None else args.batch_size
    )
    checkpoint = _resolve_checkpoint(args.checkpoint)
    layer_provenance = verify_checkpoint_layers(checkpoint, condition.layers)
    reference_dir = qa_reference_dir(config)
    reference_table_count, reference_fact_count = qa_reference_values(config)
    compatibility = verify_qa_reference_compatibility(
        config,
        condition.table_count,
        condition.fact_count,
        reference_dir=reference_dir,
    )
    qa_dir = reference_dir / args.split
    qa_records, qa_provenance = load_verified_qa_split(
        qa_dir,
        split=args.split,
        expected_table_count=reference_table_count,
        expected_fact_count=reference_fact_count,
    )
    output_dir = evaluation_result_dir(
        args.table_count, args.fact_count, args.layers, args.split, args.run_name
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
        "L": args.layers,
        "experiment_condition": condition.as_metadata(),
        "current_database_condition": {
            **compatibility["current_database_condition"],
            "layers": args.layers,
        },
        "qa_reference": compatibility["qa_reference"],
        "semantic_compatibility_fingerprint": compatibility[
            "semantic_compatibility_fingerprint"
        ],
        "split": args.split,
        "run_name": args.run_name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_config_sha256": (
            hash_file(checkpoint_config) if checkpoint_config.is_file() else None
        ),
        "checkpoint_layer_verification": layer_provenance,
        **model_identity,
        "qa_manifest_sha256": qa_provenance["qa_manifest_sha256"],
        "qa_split_manifest_sha256": qa_provenance["qa_split_manifest_sha256"],
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


def _evaluate_exp2(args: argparse.Namespace, config: dict) -> None:
    if args.qa_data_dir is None:
        raise SystemExit("ERROR: --qa-data-dir is required for Experiment-2 evaluation.")
    if args.table_count is not None or args.fact_count is not None:
        raise ValueError("Experiment 2 derives T and N from --qa-data-dir")
    qa_root = args.qa_data_dir
    if not qa_root.is_absolute():
        qa_root = PROJECT_ROOT / qa_root
    qa_root = qa_root.resolve()
    if qa_root.name in {"validation", "test"}:
        if qa_root.name != args.split:
            raise ValueError("--qa-data-dir split does not match --split")
        qa_root = qa_root.parent
    root_manifest_path = qa_root / "split_manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"QA split manifest is missing: {root_manifest_path}")
    root_manifest = read_json(root_manifest_path)
    if root_manifest.get("experiment_name") != "exp02_capacity_boundary":
        raise ValueError("--qa-data-dir is not an Experiment-2 QA bundle")
    table_count = root_manifest.get("T")
    fact_count = root_manifest.get("requested_N")
    selected_tables = root_manifest.get("selected_tables")
    if not isinstance(selected_tables, list) or len(selected_tables) != table_count:
        raise ValueError("QA bundle selected-table metadata is invalid")
    checkpoint = _resolve_checkpoint(args.checkpoint)
    checkpoint_config = read_json(checkpoint / "config.json")
    actual_layers = checkpoint_config.get(
        "n_layer", checkpoint_config.get("num_hidden_layers")
    )
    layers = actual_layers if args.layers is None else args.layers
    layer_provenance = verify_checkpoint_layers(checkpoint, layers)
    qa_records, qa_provenance = load_verified_qa_split(
        qa_root / args.split, split=args.split,
        expected_table_count=table_count, expected_fact_count=fact_count,
    )
    metadata_path = checkpoint / "training_metadata.json"
    checkpoint_metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    current = checkpoint_metadata.get("current_database_condition", {})
    cpt_provenance = checkpoint_metadata.get("provenance", {})
    recorded_database_hash = current.get(
        "source_database_sha256", cpt_provenance.get("source_database_sha256")
    )
    if recorded_database_hash and recorded_database_hash != root_manifest.get("source_database_sha256"):
        raise ValueError("checkpoint and QA dataset source database provenance mismatch")
    evaluation = config["evaluation"]
    batch_size = evaluation["batch_size"] if args.batch_size is None else args.batch_size
    model_name = checkpoint_metadata.get("model", config["model"]["name"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = exp2_artifact_stem(
        model=model_name, table_count=table_count, fact_count=fact_count,
        layers=layers, timestamp=timestamp,
    )
    run_name = args.run_name or "evaluation"
    if safe_component(run_name) != run_name:
        raise ValueError("run_name must be a filesystem-safe name")
    output_dir = EXP02_RESULTS_DIR / stem / args.split / run_name
    prepare_result_directory(output_dir)
    predictions, model_identity = evaluate_with_local_checkpoint(
        qa_records, checkpoint=checkpoint, batch_size=batch_size,
        context_length=evaluation["context_length"],
        max_new_tokens=evaluation["max_new_tokens"],
    )
    metrics = compute_evaluation_metrics(predictions)
    evaluation_config = {
        "experiment_name": "exp02_capacity_boundary",
        "evaluation_timestamp": timestamp,
        "T": table_count, "N": fact_count, "L": layers, "M": model_name,
        "selected_tables": selected_tables,
        "qa_data_dir": str(qa_root),
        "qa_split_manifest_sha256": qa_provenance["qa_split_manifest_sha256"],
        "qa_manifest_sha256": qa_provenance["qa_manifest_sha256"],
        "qa_input_hashes": qa_provenance["input_hashes"],
        "source_database_sha256": root_manifest.get("source_database_sha256"),
        "source_database_manifest_sha256": root_manifest.get("source_database_manifest_sha256"),
        "source_dataset_manifest_sha256": root_manifest.get(
            "source_dataset_manifest_sha256",
            root_manifest.get("source_database_manifest_sha256"),
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_training_metadata_sha256": hash_file(metadata_path) if metadata_path.is_file() else None,
        "checkpoint_layer_verification": layer_provenance,
        "split": args.split, "run_name": run_name,
        "qa_record_count": len(qa_records),
        "prompt_format": PROMPT_TEMPLATE, "zero_context": True,
        "decoding": {"strategy": "greedy", "do_sample": False, "temperature": None},
        "max_new_tokens": evaluation["max_new_tokens"],
        "context_length": evaluation["context_length"], "batch_size": batch_size,
        "primary_metric": "normalized_exact_match", "seed": EVALUATION_SEED,
        **model_identity,
    }
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "evaluation_config.json", evaluation_config)
    print(
        f"Evaluation complete: model={model_name}, T={table_count}, N={fact_count}, "
        f"L={layers}, split={args.split}, output={output_dir.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()

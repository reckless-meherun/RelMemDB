#!/usr/bin/env python3
"""Run an explicitly selected RelMemDB training stage."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from experiment import (
    ExperimentCondition,
    load_exp2_dataset_condition,
    resolve_model_checkpoint,
    verify_checkpoint_layers,
)
from utils.io import read_json
from training.cpt import run_cpt_training
from training.target_sft import run_target_sft_training
from utils.paths import (
    TRAINED_MODELS_DIR,
    cpt_database_dir,
    cpt_run_dir,
    database_condition_dir,
    qa_reference_dir,
    target_sft_run_dir,
    EXP02_RUNS_DIR,
    exp2_artifact_stem,
)


def _fact_count_label(fact_count: int) -> str:
    return f"n{fact_count // 1000}k" if fact_count % 1000 == 0 else f"n{fact_count}"


def _resolve_source_checkpoint(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {resolved}")
    return resolved


def build_cpt_paths(
    config: dict, *, table_count: int, fact_count: int, layers: int
) -> dict[str, Path]:
    condition_dir = database_condition_dir(table_count, fact_count)
    cpt_dir = cpt_database_dir(table_count, fact_count)
    run_dir = cpt_run_dir(table_count, fact_count, layers)
    epochs = config["training"]["cpt_epochs"]
    stem = f"exp01_tsweep_T{table_count}_N{fact_count // 1000}K_L{layers}_E{epochs}"
    model_name = config["model"]["name"].replace("/", "_")
    return {
        "database": condition_dir / "database.sqlite",
        "database_manifest": condition_dir / "manifest.json",
        "readable_book": cpt_dir / "book_readable.txt",
        "train_text": cpt_dir / "train.txt",
        "cpt_manifest": cpt_dir / "manifest.json",
        "run_config": run_dir / f"{stem}_config_PLACEHOLDER.yaml",
        "train_log": run_dir / f"{stem}_trainlog_PLACEHOLDER.jsonl",
        "output_checkpoint": (
            TRAINED_MODELS_DIR
            / (
                f"{model_name}_cpt_t{table_count}_"
                f"{_fact_count_label(fact_count)}_l{layers}_e{epochs}"
            )
        ),
    }


def build_target_sft_paths(
    config: dict,
    *,
    table_count: int,
    fact_count: int,
    layers: int,
    source_checkpoint: Path,
) -> dict[str, Path]:
    settings = config["target_sft"]
    epochs = settings["epochs"]
    model_name = config["model"]["name"].replace("/", "_")
    cpt_epochs = config["training"]["cpt_epochs"]
    checkpoint_name = (
        f"{model_name}_cpt_t{table_count}_{_fact_count_label(fact_count)}_"
        f"l{layers}_e{cpt_epochs}_sft_target_e{epochs}"
    )
    run_stem = checkpoint_name
    run_dir = target_sft_run_dir(table_count, fact_count, layers)
    return {
        "qa_condition_dir": qa_reference_dir(config),
        "run_config": run_dir / f"{run_stem}_config.yaml",
        "train_log": run_dir / f"{run_stem}_trainlog.jsonl",
        "output_checkpoint": TRAINED_MODELS_DIR / checkpoint_name,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RelMemDB model training")
    parser.add_argument("--stage", required=True, choices=("cpt", "target-sft"))
    parser.add_argument("--table-count", type=int)
    parser.add_argument("--fact-count", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--training-data-dir", type=Path)
    parser.add_argument("--sft-data-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    if config["experiment"]["name"] == "exp02_capacity_boundary":
        _run_exp2(args, config)
        return
    missing = [
        name for name, value in (
            ("--table-count", args.table_count), ("--fact-count", args.fact_count),
            ("--layers", args.layers), ("--source-checkpoint", args.source_checkpoint),
        ) if value is None
    ]
    if missing:
        raise SystemExit(f"ERROR: Experiment-1 training requires {', '.join(missing)}.")
    condition = ExperimentCondition.from_config(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        layers=args.layers,
    )
    source_checkpoint = _resolve_source_checkpoint(args.source_checkpoint)
    verify_checkpoint_layers(source_checkpoint, condition.layers)
    if args.stage == "target-sft":
        paths = build_target_sft_paths(
            config,
            table_count=args.table_count,
            fact_count=args.fact_count,
            layers=args.layers,
            source_checkpoint=source_checkpoint,
        )
        summary = run_target_sft_training(
            config,
            table_count=args.table_count,
            fact_count=args.fact_count,
            layers=args.layers,
            source_checkpoint=source_checkpoint,
            output_checkpoint=paths["output_checkpoint"],
            run_config_path=paths["run_config"],
            train_log_path=paths["train_log"],
            qa_condition_dir=paths["qa_condition_dir"],
        )
        print(
            f"Target SFT complete: train_examples={summary['train_example_count']}, "
            f"dev_examples={summary['dev_example_count']}, "
            f"steps={summary['optimizer_steps']}, "
            f"selected_epoch={summary['selected_epoch']}, "
            f"loss={summary['training_loss']:.6f}, "
            f"checkpoint={summary['output_checkpoint']}"
        )
        return

    paths = build_cpt_paths(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        layers=args.layers,
    )
    summary = run_cpt_training(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        layers=args.layers,
        source_checkpoint=source_checkpoint,
        output_checkpoint=paths["output_checkpoint"],
        run_config_path=paths["run_config"],
        train_log_path=paths["train_log"],
        database_path=paths["database"],
        database_manifest_path=paths["database_manifest"],
        readable_book_path=paths["readable_book"],
        train_text_path=paths["train_text"],
        cpt_manifest_path=paths["cpt_manifest"],
    )
    print(
        f"CPT complete: steps={summary['optimizer_steps']}, "
        f"loss={summary['training_loss']:.6f}, "
        f"checkpoint={summary['final_checkpoint_path']}"
    )


def _run_exp2(args: argparse.Namespace, config: dict) -> None:
    model_name = args.model or config["model"].get("name", "gpt2")
    config["model"]["name"] = model_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    config["_runtime"] = {"run_timestamp": timestamp}
    if args.stage == "cpt":
        if args.training_data_dir is None:
            raise SystemExit("ERROR: --training-data-dir is required for Experiment-2 CPT training.")
        if args.sft_data_dir is not None:
            raise ValueError("--sft-data-dir is only valid for target-SFT training")
        condition = load_exp2_dataset_condition(args.training_data_dir)
        for supplied, actual, label in (
            (args.table_count, condition["T"], "T/--table-count"),
            (args.fact_count, condition["N"], "N/--fact-count"),
        ):
            if supplied is not None and supplied != actual:
                raise ValueError(f"optional {label} assertion does not match the dataset manifest")
        source_checkpoint, native_layers = resolve_model_checkpoint(
            model_name, source_checkpoint=args.source_checkpoint
        )
        layers = native_layers if args.layers is None else args.layers
        verify_checkpoint_layers(source_checkpoint, layers)
        stem = exp2_artifact_stem(
            model=model_name, table_count=condition["T"], fact_count=condition["N"],
            layers=layers, timestamp=timestamp,
        )
        run_dir = EXP02_RUNS_DIR / stem / "cpt"
        output_checkpoint = TRAINED_MODELS_DIR / stem
        cpt_dir = condition["cpt_dir"]
        summary = run_cpt_training(
            config, table_count=condition["T"], fact_count=condition["N"],
            layers=layers, source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            run_config_path=run_dir / "run_config.yaml",
            train_log_path=run_dir / "train_log.jsonl",
            database_path=condition["database"],
            database_manifest_path=condition["manifest_path"],
            readable_book_path=cpt_dir / "book_readable.txt",
            train_text_path=cpt_dir / "train.txt",
            cpt_manifest_path=condition["cpt_manifest"],
        )
        print(
            f"CPT complete: model={model_name}, T={condition['T']}, N={condition['N']}, "
            f"L={layers}, steps={summary['optimizer_steps']}, checkpoint={output_checkpoint}"
        )
        return

    if args.sft_data_dir is None:
        raise SystemExit("ERROR: --sft-data-dir is required for target-SFT training.")
    if args.source_checkpoint is None:
        raise SystemExit("ERROR: --source-checkpoint is required for target-SFT training.")
    sft_path = args.sft_data_dir
    if not sft_path.is_absolute():
        sft_path = PROJECT_ROOT / sft_path
    sft_path = sft_path.resolve()
    qa_condition_dir = sft_path.parent if sft_path.name == "target_sft" else sft_path
    split_manifest_path = qa_condition_dir / "target_sft" / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"target-SFT manifest is missing: {split_manifest_path}")
    split_manifest = read_json(split_manifest_path)
    if split_manifest.get("experiment_name") != "exp02_capacity_boundary":
        raise ValueError("--sft-data-dir is not an Experiment-2 SFT dataset")
    table_count = split_manifest.get("T")
    fact_count = split_manifest.get("requested_N")
    for supplied, actual, label in (
        (args.table_count, table_count, "T/--table-count"),
        (args.fact_count, fact_count, "N/--fact-count"),
    ):
        if supplied is not None and supplied != actual:
            raise ValueError(f"optional {label} assertion does not match the SFT manifest")
    source_checkpoint = _resolve_source_checkpoint(args.source_checkpoint)
    if not (source_checkpoint / "training_metadata.json").is_file():
        raise FileNotFoundError(
            "Experiment-2 target SFT requires an explicit CPT checkpoint with training_metadata.json"
        )
    _, native_layers = resolve_model_checkpoint(model_name, source_checkpoint=source_checkpoint)
    layers = native_layers if args.layers is None else args.layers
    verify_checkpoint_layers(source_checkpoint, layers)
    stem = exp2_artifact_stem(
        model=model_name, table_count=table_count, fact_count=fact_count,
        layers=layers, timestamp=timestamp,
    ) + "_sft"
    run_dir = EXP02_RUNS_DIR / stem / "target_sft"
    output_checkpoint = TRAINED_MODELS_DIR / stem
    summary = run_target_sft_training(
        config, table_count=table_count, fact_count=fact_count, layers=layers,
        source_checkpoint=source_checkpoint, output_checkpoint=output_checkpoint,
        run_config_path=run_dir / "run_config.yaml",
        train_log_path=run_dir / "train_log.jsonl",
        qa_condition_dir=qa_condition_dir,
    )
    print(
        f"Target SFT complete: model={model_name}, T={table_count}, N={fact_count}, "
        f"L={layers}, steps={summary['optimizer_steps']}, checkpoint={output_checkpoint}"
    )


if __name__ == "__main__":
    main()

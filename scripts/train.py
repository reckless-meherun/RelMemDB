#!/usr/bin/env python3
"""Run an explicitly selected RelMemDB training stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from experiment import ExperimentCondition, verify_checkpoint_layers
from training.cpt import run_cpt_training
from training.target_sft import run_target_sft_training
from utils.paths import (
    TRAINED_MODELS_DIR,
    cpt_database_dir,
    cpt_run_dir,
    database_condition_dir,
    qa_reference_dir,
    target_sft_run_dir,
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
    parser.add_argument("--table-count", required=True, type=int)
    parser.add_argument("--fact-count", required=True, type=int)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
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


if __name__ == "__main__":
    main()

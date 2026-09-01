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
from training.cpt import run_cpt_training
from utils.paths import (
    TRAINED_MODELS_DIR,
    cpt_database_dir,
    cpt_run_dir,
    database_condition_dir,
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
    config: dict, *, table_count: int, fact_count: int
) -> dict[str, Path]:
    condition_dir = database_condition_dir(table_count, fact_count)
    cpt_dir = cpt_database_dir(table_count, fact_count)
    run_dir = cpt_run_dir(table_count, fact_count)
    layer_count = config["model"]["layers"]
    stem = (
        f"exp01_tsweep_T{table_count}_N{fact_count // 1000}K_"
        f"L{layer_count}"
    )
    model_name = config["model"]["name"].replace("/", "_")
    return {
        "database": condition_dir / "database.sqlite",
        "database_manifest": condition_dir / "manifest.json",
        "train_text": cpt_dir / "train.txt",
        "cpt_manifest": cpt_dir / "manifest.json",
        "run_config": run_dir / f"{stem}_config_PLACEHOLDER.yaml",
        "train_log": run_dir / f"{stem}_trainlog_PLACEHOLDER.jsonl",
        "output_checkpoint": (
            TRAINED_MODELS_DIR
            / f"{model_name}_cpt_t{table_count}_{_fact_count_label(fact_count)}"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RelMemDB model training")
    parser.add_argument("--stage", required=True, choices=("cpt",))
    parser.add_argument("--table-count", required=True, type=int)
    parser.add_argument("--fact-count", required=True, type=int)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    source_checkpoint = _resolve_source_checkpoint(args.source_checkpoint)
    paths = build_cpt_paths(
        config, table_count=args.table_count, fact_count=args.fact_count
    )
    summary = run_cpt_training(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        source_checkpoint=source_checkpoint,
        output_checkpoint=paths["output_checkpoint"],
        run_config_path=paths["run_config"],
        train_log_path=paths["train_log"],
        database_path=paths["database"],
        database_manifest_path=paths["database_manifest"],
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

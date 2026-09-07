#!/usr/bin/env python3
"""Generate isolated factual-completion probes from Exp02-style CPT records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.cpt_test import generate_cpt_test
from utils.paths import EXP02_QA_DIR, QA_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical and held-out declarative completion probes from "
            "the exact independent records used for Experiment-2-style CPT."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    training_data_dir = args.training_data_dir
    if not training_data_dir.is_absolute():
        training_data_dir = PROJECT_ROOT / training_data_dir
    training_data_dir = training_data_dir.resolve()
    experiment_name = config["experiment"]["name"]
    qa_root = (
        EXP02_QA_DIR
        if experiment_name == "exp02_capacity_boundary"
        else QA_DIR / experiment_name
    )
    output_dir = qa_root / training_data_dir.name / "cpt_test"
    result = generate_cpt_test(
        config,
        training_data_dir=training_data_dir,
        output_dir=output_dir,
    )
    manifest = result["manifest"]
    print(
        f"CPT-test probes: records={manifest['source_cpt_record_count']}, "
        f"probes={manifest['probe_count']}"
    )
    print(f"CPT-test output: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

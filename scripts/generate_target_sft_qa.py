import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.qa import HOP_NAMES, generate_condition_qa, generate_target_sft_qa
from utils.io import read_json
from utils.paths import database_condition_dir, exp2_qa_bundle_dir, qa_condition_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic target-SFT train/dev QA from the reserved "
            "chains of one existing closed-book QA condition."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--training-data-dir", type=Path)
    parser.add_argument("--table-count", type=int)
    parser.add_argument("--fact-count", type=int)
    args = parser.parse_args()
    if (args.table_count is None) != (args.fact_count is None):
        parser.error("--table-count and --fact-count must be supplied together")
    return args


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    if config["experiment"]["name"] == "exp02_capacity_boundary":
        if args.training_data_dir is None:
            raise SystemExit("ERROR: --training-data-dir is required for Experiment-2 target-SFT QA generation.")
        if args.table_count is not None or args.fact_count is not None:
            raise ValueError("Experiment 2 derives T and N from --training-data-dir; do not pass them")
        database_dir = args.training_data_dir
        if not database_dir.is_absolute():
            database_dir = PROJECT_ROOT / database_dir
        database_dir = database_dir.resolve()
        manifest_path = database_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"training-data manifest is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("experiment_name") != "exp02_capacity_boundary":
            raise ValueError("--training-data-dir is not an Experiment-2 dataset bundle")
        selected = manifest.get("selected_tables")
        table_count = manifest.get("T")
        fact_count = manifest.get("requested_N")
        if not isinstance(selected, list) or len(selected) != table_count:
            raise ValueError("training-data selected-table metadata is invalid")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        condition_dir = exp2_qa_bundle_dir(selected, fact_count, timestamp)
        generate_condition_qa(
            config,
            database_dir / "database.sqlite",
            manifest_path,
            condition_dir,
            expected_table_count=table_count,
            expected_logical_fact_count=fact_count,
            source_training_data_dir=database_dir,
            generation_timestamp=timestamp,
        )
        result = generate_target_sft_qa(
            config,
            database_path=database_dir / "database.sqlite",
            database_manifest_path=manifest_path,
            qa_condition_dir=condition_dir,
            expected_table_count=table_count,
            expected_logical_fact_count=fact_count,
            source_training_data_dir=database_dir,
            generation_timestamp=timestamp,
        )
        print(f"Target-SFT QA output: {condition_dir.relative_to(PROJECT_ROOT)}")
        for split in ("train", "dev"):
            counts = result[f"{split}_manifest"]["retained_counts"]
            print(f"{split}: " + ", ".join(f"{hop}={counts[hop]}" for hop in HOP_NAMES))
        return
    if args.training_data_dir is not None:
        raise ValueError("--training-data-dir is reserved for Experiment 2")
    if args.table_count is None:
        raise SystemExit("ERROR: --table-count and --fact-count are required for Experiment 1.")
    database_dir = database_condition_dir(args.table_count, args.fact_count)
    condition_dir = qa_condition_dir(args.table_count, args.fact_count)
    result = generate_target_sft_qa(
        config,
        database_path=database_dir / "database.sqlite",
        database_manifest_path=database_dir / "manifest.json",
        qa_condition_dir=condition_dir,
        expected_table_count=args.table_count,
        expected_logical_fact_count=args.fact_count,
    )
    for split in ("train", "dev"):
        manifest = result[f"{split}_manifest"]
        counts = ", ".join(
            f"{hop}={manifest['counts'][hop]['final_retained_count']}"
            for hop in HOP_NAMES
        )
        print(
            f"{split}: chains={manifest['chain_count']}, "
            f"candidates={manifest['candidate_total']}, "
            f"retained={manifest['final_retained_total']} ({counts})"
        )
    print(f"Target-SFT QA output: {(condition_dir / 'target_sft').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

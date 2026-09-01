import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.qa import HOP_NAMES, generate_target_sft_qa
from utils.paths import database_condition_dir, qa_condition_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic target-SFT train/dev QA from the reserved "
            "chains of one existing closed-book QA condition."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--table-count", type=int, required=True)
    parser.add_argument("--fact-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
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

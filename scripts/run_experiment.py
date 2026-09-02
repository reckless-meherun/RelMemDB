#!/usr/bin/env python3
"""Thin, explicit stage driver for one first-experiment T/N/L condition."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.qa import load_verified_semantic_chains
from data.qa_reference import verify_qa_reference_compatibility
from experiment import ExperimentCondition, verify_checkpoint_layers
from training.cpt import verify_cpt_artifacts
from utils.io import read_json
from utils.paths import (
    TRAINED_MODELS_DIR,
    cpt_database_dir,
    database_condition_dir,
    evaluation_result_dir,
)

STAGES = (
    "prepare",
    "cpt",
    "eval-cpt",
    "sft",
    "eval-sft",
    "test-sft",
    "through-validation",
)
THROUGH_VALIDATION_STAGES = (
    "prepare",
    "cpt",
    "eval-cpt",
    "sft",
    "eval-sft",
)


def _fact_label(fact_count: int) -> str:
    return f"n{fact_count // 1000}k" if fact_count % 1000 == 0 else f"n{fact_count}"


def checkpoint_paths(config: dict, condition: ExperimentCondition) -> dict[str, Path]:
    model_name = config["model"]["name"].replace("/", "_")
    cpt_epochs = config["training"]["cpt_epochs"]
    sft_epochs = config["target_sft"]["epochs"]
    cpt_name = (
        f"{model_name}_cpt_t{condition.table_count}_"
        f"{_fact_label(condition.fact_count)}_l{condition.layers}_e{cpt_epochs}"
    )
    return {
        "cpt": TRAINED_MODELS_DIR / cpt_name,
        "sft": TRAINED_MODELS_DIR / f"{cpt_name}_sft_target_e{sft_epochs}",
    }


def stage_sequence(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    return THROUGH_VALIDATION_STAGES if stage == "through-validation" else (stage,)


def build_stage_commands(
    *,
    config_path: Path,
    config: dict,
    condition: ExperimentCondition,
    stage: str,
    source_checkpoint: Path | None,
) -> list[list[str]]:
    checkpoints = checkpoint_paths(config, condition)
    common = [
        "--table-count",
        str(condition.table_count),
        "--fact-count",
        str(condition.fact_count),
    ]
    conditioned = [*common, "--layers", str(condition.layers)]
    commands: list[list[str]] = []
    for selected in stage_sequence(stage):
        if selected == "prepare":
            commands.extend(
                [
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/generate_databases.py"),
                        *common,
                        "--config",
                        str(config_path),
                    ],
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/generate_databases.py"),
                        *common,
                        "--serialize-cpt-only",
                        "--config",
                        str(config_path),
                    ],
                ]
            )
            continue
        if source_checkpoint is None and stage != "through-validation":
            raise ValueError(f"--source-checkpoint is required for stage {selected}")
        if selected == "cpt":
            checkpoint = source_checkpoint
            assert checkpoint is not None
            commands.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/train.py"),
                    "--stage",
                    "cpt",
                    *conditioned,
                    "--source-checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config_path),
                ]
            )
        elif selected in {"eval-cpt", "eval-sft", "test-sft"}:
            if stage == "through-validation":
                checkpoint = checkpoints["cpt" if selected == "eval-cpt" else "sft"]
            else:
                checkpoint = source_checkpoint
            assert checkpoint is not None
            split = "test" if selected == "test-sft" else "validation"
            run_name = selected.replace("-", "_")
            commands.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/evaluate.py"),
                    *conditioned,
                    "--split",
                    split,
                    "--checkpoint",
                    str(checkpoint),
                    "--run-name",
                    run_name,
                    "--config",
                    str(config_path),
                ]
            )
        elif selected == "sft":
            checkpoint = (
                checkpoints["cpt"]
                if stage == "through-validation"
                else source_checkpoint
            )
            assert checkpoint is not None
            commands.append(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/train.py"),
                    "--stage",
                    "target-sft",
                    *conditioned,
                    "--source-checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config_path),
                ]
            )
    return commands


def _all_nonempty(paths: list[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _reuse_existing_artifact(
    command: list[str], config: dict, condition: ExperimentCondition
) -> bool:
    """Authenticate and reuse complete outputs; reject ambiguous partial outputs."""
    script_name = Path(command[1]).name
    database_dir = database_condition_dir(condition.table_count, condition.fact_count)
    if script_name == "generate_databases.py":
        database_files = [
            database_dir / "database.sqlite",
            database_dir / "manifest.json",
        ]
        if "--serialize-cpt-only" not in command:
            existing = [
                path for path in database_files if path.exists() and path.stat().st_size
            ]
            if not existing:
                return False
            if not _all_nonempty(database_files):
                raise FileExistsError(
                    f"partial database artifact exists for {condition.label}"
                )
            load_verified_semantic_chains(
                database_files[0],
                database_files[1],
                expected_table_count=condition.table_count,
                expected_logical_fact_count=condition.fact_count,
            )
            return True
        cpt_dir = cpt_database_dir(condition.table_count, condition.fact_count)
        cpt_files = [
            cpt_dir / "book_readable.txt",
            cpt_dir / "train.txt",
            cpt_dir / "manifest.json",
        ]
        existing = [path for path in cpt_files if path.exists() and path.stat().st_size]
        if not existing:
            return False
        if not _all_nonempty([*database_files, *cpt_files]):
            raise FileExistsError(
                f"partial CPT serialization artifact exists for {condition.label}"
            )
        verify_cpt_artifacts(
            config,
            table_count=condition.table_count,
            fact_count=condition.fact_count,
            database_path=database_files[0],
            database_manifest_path=database_files[1],
            readable_book_path=cpt_files[0],
            train_text_path=cpt_files[1],
            cpt_manifest_path=cpt_files[2],
        )
        return True

    checkpoints = checkpoint_paths(config, condition)
    if script_name == "train.py":
        key = "sft" if command[command.index("--stage") + 1] == "target-sft" else "cpt"
        output = checkpoints[key]
        if not output.exists() or (output.is_dir() and not any(output.iterdir())):
            return False
        metadata_path = output / "training_metadata.json"
        if not output.is_dir() or not metadata_path.is_file():
            raise FileExistsError(f"incomplete checkpoint exists: {output}")
        verify_checkpoint_layers(output, condition.layers)
        metadata = read_json(metadata_path)
        if (
            metadata.get("T") != condition.table_count
            or metadata.get("N") != condition.fact_count
            or metadata.get("L") != condition.layers
        ):
            raise ValueError(
                f"existing checkpoint provenance does not match {condition.label}"
            )
        if key == "sft":
            compatibility = verify_qa_reference_compatibility(
                config, condition.table_count, condition.fact_count
            )
            if metadata.get("semantic_compatibility_fingerprint") != compatibility.get(
                "semantic_compatibility_fingerprint"
            ):
                raise ValueError(
                    "existing target-SFT checkpoint QA compatibility provenance "
                    "does not match"
                )
        return True

    if script_name == "evaluate.py":
        split = command[command.index("--split") + 1]
        run_name = command[command.index("--run-name") + 1]
        checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
        output = evaluation_result_dir(
            condition.table_count,
            condition.fact_count,
            condition.layers,
            split,
            run_name,
        )
        if not output.exists() or (output.is_dir() and not any(output.iterdir())):
            return False
        config_path = output / "evaluation_config.json"
        if not output.is_dir() or not config_path.is_file():
            raise FileExistsError(f"incomplete evaluation result exists: {output}")
        metadata = read_json(config_path)
        compatibility = verify_qa_reference_compatibility(
            config, condition.table_count, condition.fact_count
        )
        if (
            metadata.get("T") != condition.table_count
            or metadata.get("N") != condition.fact_count
            or metadata.get("L") != condition.layers
            or Path(metadata.get("checkpoint_path", "")).resolve() != checkpoint
            or metadata.get("qa_reference", {}).get("table_count") != 12
            or metadata.get("qa_reference", {}).get("fact_count") != 10_000
            or metadata.get("semantic_compatibility_fingerprint")
            != compatibility.get("semantic_compatibility_fingerprint")
        ):
            raise ValueError(
                f"existing evaluation provenance does not match {condition.label}"
            )
        return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--table-count", required=True, type=int)
    parser.add_argument("--fact-count", required=True, type=int)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    condition = ExperimentCondition.from_config(
        config,
        table_count=args.table_count,
        fact_count=args.fact_count,
        layers=args.layers,
    )
    source = args.source_checkpoint
    if source is not None:
        source = (source if source.is_absolute() else PROJECT_ROOT / source).resolve()
        verify_checkpoint_layers(source, condition.layers)
    if args.stage != "prepare" and source is None:
        raise ValueError("--source-checkpoint is required unless --stage prepare")
    commands = build_stage_commands(
        config_path=config_path,
        config=config,
        condition=condition,
        stage=args.stage,
        source_checkpoint=source,
    )
    for command in commands:
        if _reuse_existing_artifact(command, config, condition):
            print(f"Reusing authenticated artifact for: {' '.join(command[1:3])}")
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

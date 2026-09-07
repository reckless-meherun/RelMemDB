from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from utils.io import read_json, write_json
from utils.paths import TRAINED_MODELS_DIR

EXP2_CHECKPOINT_NAME = re.compile(
    r"^[A-Za-z0-9_.-]+_exp02_T\d{2}_N\d+_L\d+_\d{8}_\d{6}_\d{6}(?:_sft)?$"
)
EXP2_EXPERIMENT_NAME = "exp02_capacity_boundary"
EXP3_EXPERIMENT_NAME = "exp03_continent_inverse"


def is_exp2_trained_checkpoint_dir(path: str | Path) -> bool:
    checkpoint = Path(path)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        return False
    metadata_path = checkpoint / "training_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = read_json(metadata_path)
        except (OSError, ValueError, TypeError):
            metadata = {}
        if metadata.get("experiment") == EXP2_EXPERIMENT_NAME:
            return True
    return bool(EXP2_CHECKPOINT_NAME.fullmatch(checkpoint.name))


def remove_exp2_trained_checkpoints_except(
    retain: Iterable[str | Path] = (),
    *,
    trained_models_dir: str | Path = TRAINED_MODELS_DIR,
) -> list[Path]:
    root = Path(trained_models_dir).resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"trained-models path is not a directory: {root}")
    retained = {Path(path).resolve() for path in retain}
    removed: list[Path] = []
    for candidate in sorted(root.iterdir(), key=lambda path: path.name):
        if candidate.resolve() in retained:
            continue
        if not is_exp2_trained_checkpoint_dir(candidate):
            continue
        if candidate.parent.resolve() != root:
            raise ValueError(f"refusing to remove non-child checkpoint: {candidate}")
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


def remove_failed_exp2_checkpoint(
    checkpoint: str | Path,
    *,
    trained_models_dir: str | Path = TRAINED_MODELS_DIR,
) -> bool:
    root = Path(trained_models_dir).resolve()
    target = Path(checkpoint).resolve()
    if target.parent != root:
        return False
    if not is_exp2_trained_checkpoint_dir(target):
        return False
    shutil.rmtree(target)
    return True


def retain_best_exp3_checkpoint(
    source_checkpoint: str | Path,
    destination: str | Path,
    metadata: dict[str, Any],
) -> bool:
    """Copy a strictly better final Exp03 checkpoint into its inspection slot."""
    source = Path(source_checkpoint).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise FileNotFoundError(f"Experiment-3 source checkpoint is missing: {source}")
    if metadata.get("experiment") != EXP3_EXPERIMENT_NAME:
        raise ValueError("best-checkpoint metadata is not for Experiment 3")
    score = metadata.get("EM")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("best-checkpoint EM must be numeric and in [0, 1]")
    for key in ("N", "epochs", "seed", "checkpoint_source", "run"):
        if key not in metadata:
            raise ValueError(f"best-checkpoint metadata is missing {key}")

    metadata_path = destination.parent / "best_metadata.json"
    if metadata_path.is_file():
        incumbent = read_json(metadata_path).get("EM")
        if isinstance(incumbent, (int, float)) and not isinstance(incumbent, bool):
            if float(score) <= float(incumbent):
                return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    shutil.copytree(source, temporary)
    try:
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        write_json(metadata_path, metadata)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True

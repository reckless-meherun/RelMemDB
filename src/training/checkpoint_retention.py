from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from utils.io import read_json
from utils.paths import TRAINED_MODELS_DIR

EXP2_CHECKPOINT_NAME = re.compile(
    r"^[A-Za-z0-9_.-]+_exp02_T\d{2}_N\d+_L\d+_\d{8}_\d{6}_\d{6}(?:_sft)?$"
)
EXP2_EXPERIMENT_NAME = "exp02_capacity_boundary"


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

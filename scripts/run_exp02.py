#!/usr/bin/env python3
"""Single-command orchestrator for RelMemDB Experiment 2.

This is a thin wrapper around the existing Experiment-2 scripts. It does not
reimplement database generation, CPT, QA generation, SFT, evaluation, or result
routing.

Fresh run example:

    python3 scripts/run_exp02.py \
        --tables continent \
        --fact-count 500 \
        --model gpt2 \
        --layers 12 \
        --base-model models/base_models/gpt2 \
        --cpt-epochs 40 \
        --sft-epochs 20

The normal pipeline is:
database/CPT bundle -> CPT -> QA/SFT data -> CPT validation -> SFT -> SFT validation.

The test split is never evaluated unless --evaluate-test is explicitly supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import validate_config
from data.world import facts_per_selected_chain, validate_selected_tables
from experiment import (
    load_exp2_dataset_condition,
    resolve_model_checkpoint,
    verify_checkpoint_layers,
)
from utils.hashing import hash_file
from utils.io import read_json


EXP2_NAME = "exp02_capacity_boundary"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp02_capacity_boundary.yaml"
PIPELINE_RUNS_DIR = PROJECT_ROOT / "runs" / EXP2_NAME / "pipeline_runs"
EXP2_DATASETS_DIR = (
    PROJECT_ROOT / "datasets" / "generated_databases" / EXP2_NAME
)
EXP2_QA_DIR = PROJECT_ROOT / "datasets" / "qa" / EXP2_NAME

TResult = TypeVar("TResult")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {path}")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    _require_file(path, "configuration")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return loaded


def _checkpoint_actual_layers(checkpoint: Path) -> int:
    config_path = _require_file(checkpoint / "config.json", "checkpoint config")
    payload = read_json(config_path)
    layers = payload.get("n_layer", payload.get("num_hidden_layers"))
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError(
            f"checkpoint config does not declare a valid layer count: {config_path}"
        )
    return layers


def _create_run_dir() -> Path:
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        path = PIPELINE_RUNS_DIR / _timestamp()
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError("could not allocate a unique Experiment-2 pipeline run directory")


def _runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "cpt_epochs": args.cpt_epochs,
        "sft_epochs": args.sft_epochs,
        "cpt_batch_size": args.cpt_batch_size,
        "cpt_gradient_accumulation": args.cpt_gradient_accumulation,
        "sft_batch_size": args.sft_batch_size,
        "sft_gradient_accumulation": args.sft_gradient_accumulation,
        "cpt_learning_rate": args.cpt_learning_rate,
        "sft_learning_rate": args.sft_learning_rate,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _write_resolved_config(
    *,
    base_config_path: Path,
    output_path: Path,
    model_name: str,
    native_layers: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(_load_yaml(base_config_path))
    if config.get("experiment", {}).get("name") != EXP2_NAME:
        raise ValueError(
            f"runner requires an {EXP2_NAME} config, found "
            f"{config.get('experiment', {}).get('name')!r}"
        )

    config["model"]["name"] = model_name
    config["model"]["native_layers"] = native_layers

    if "cpt_epochs" in overrides:
        config["training"]["cpt_epochs"] = overrides["cpt_epochs"]
    if "sft_epochs" in overrides:
        config["target_sft"]["epochs"] = overrides["sft_epochs"]
    if "cpt_batch_size" in overrides:
        config["training"]["cpt_batch_size"] = overrides["cpt_batch_size"]
    if "cpt_gradient_accumulation" in overrides:
        config["training"]["gradient_accumulation_steps"] = overrides[
            "cpt_gradient_accumulation"
        ]
    if "sft_batch_size" in overrides:
        config["target_sft"]["batch_size"] = overrides["sft_batch_size"]
    if "sft_gradient_accumulation" in overrides:
        config["target_sft"]["gradient_accumulation_steps"] = overrides[
            "sft_gradient_accumulation"
        ]
    if "cpt_learning_rate" in overrides:
        config["training"]["learning_rate"] = overrides["cpt_learning_rate"]
    if "sft_learning_rate" in overrides:
        config["target_sft"]["learning_rate"] = overrides["sft_learning_rate"]

    validate_config(config)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def _run_command(command: list[str]) -> str:
    """Run one existing stage, streaming and capturing combined stdout/stderr."""
    print("\n$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    output = "".join(lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


def _last_line_value(
    output: str,
    *,
    prefix: str | None = None,
    token: str | None = None,
) -> str:
    values: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if prefix is not None and line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                values.append(value)
        elif token is not None and token in line:
            value = line.rsplit(token, 1)[1].strip()
            if value:
                values.append(value)
    if not values:
        marker = prefix if prefix is not None else token
        raise RuntimeError(f"child process completed but did not report {marker!r}")
    return values[-1]


def _path_from_child(value: str) -> Path:
    return _resolve_path(value)


def _mark_stage(
    *,
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    status: str,
    command: list[str] | None = None,
    artifact: str | Path | None = None,
    error: str | None = None,
) -> None:
    record = state.setdefault("stages", {}).setdefault(stage, {})
    record["status"] = status
    if status in {"running", "reused"} and "started_at" not in record:
        record["started_at"] = _utc_iso()
    if status in {"completed", "reused", "failed"}:
        record["finished_at"] = _utc_iso()
    if command is not None:
        record["command"] = command
    if artifact is not None:
        record["artifact"] = str(artifact)
    if error is not None:
        record["error"] = error
    state["current_stage"] = None if status in {"completed", "reused"} else stage
    if status == "failed":
        state["status"] = "failed"
    _write_json_atomic(state_path, state)


def _execute_stage(
    *,
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    command: list[str],
    action: Callable[[], TResult],
    artifact_of: Callable[[TResult], str | Path | None] | None = None,
) -> TResult:
    print(f"\n=== {stage} ===", flush=True)
    _mark_stage(
        state=state,
        state_path=state_path,
        stage=stage,
        status="running",
        command=command,
    )
    try:
        result = action()
    except Exception as exc:
        _mark_stage(
            state=state,
            state_path=state_path,
            stage=stage,
            status="failed",
            command=command,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    artifact = artifact_of(result) if artifact_of is not None else None
    _mark_stage(
        state=state,
        state_path=state_path,
        stage=stage,
        status="completed",
        command=command,
        artifact=artifact,
    )
    return result


def _reuse_stage(
    *,
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    artifact: Path,
) -> None:
    print(f"\n=== {stage}: reusing authenticated artifact ===")
    print(artifact)
    _mark_stage(
        state=state,
        state_path=state_path,
        stage=stage,
        status="reused",
        artifact=artifact,
    )


def _verify_dataset_bundle(
    path: Path,
    *,
    selected_tables: tuple[str, ...],
    requested_n: int | None,
) -> dict[str, Any]:
    condition = load_exp2_dataset_condition(path)
    if tuple(condition["selected_tables"]) != selected_tables:
        raise ValueError(
            "dataset selected_tables do not match --tables: "
            f"{condition['selected_tables']} != {list(selected_tables)}"
        )
    if requested_n is not None and condition["N"] != requested_n:
        raise ValueError(
            f"dataset N={condition['N']} does not match --fact-count={requested_n}"
        )
    for artifact, label in (
        (condition["database"], "dataset database"),
        (condition["manifest_path"], "dataset manifest"),
        (condition["cpt_dir"] / "book_readable.txt", "CPT readable book"),
        (condition["cpt_dir"] / "train.txt", "CPT train text"),
        (condition["cpt_manifest"], "CPT manifest"),
    ):
        _require_file(artifact, label)
    return condition


def _normalize_qa_root(path: Path) -> Path:
    resolved = _resolve_path(path)
    if resolved.name == "target_sft":
        resolved = resolved.parent
    if resolved.name in {"validation", "test"}:
        resolved = resolved.parent
    return resolved


def _verify_qa_bundle(
    path: Path,
    *,
    dataset_condition: dict[str, Any],
    selected_tables: tuple[str, ...],
) -> dict[str, Any]:
    root = _normalize_qa_root(path)
    _require_dir(root, "QA bundle")
    root_manifest_path = _require_file(root / "split_manifest.json", "QA split manifest")
    sft_manifest_path = _require_file(
        root / "target_sft" / "split_manifest.json", "target-SFT split manifest"
    )
    _require_file(root / "validation" / "manifest.json", "validation QA manifest")
    _require_file(root / "test" / "manifest.json", "test QA manifest")
    _require_file(root / "target_sft" / "train" / "manifest.json", "SFT train manifest")
    _require_file(root / "target_sft" / "dev" / "manifest.json", "SFT dev manifest")

    root_manifest = read_json(root_manifest_path)
    sft_manifest = read_json(sft_manifest_path)
    expected_t = dataset_condition["T"]
    expected_n = dataset_condition["N"]

    for artifact, label in (
        (root_manifest, "QA split manifest"),
        (sft_manifest, "target-SFT split manifest"),
    ):
        if artifact.get("experiment_name") != EXP2_NAME:
            raise ValueError(f"{label} is not for Experiment 2")
        if artifact.get("T") != expected_t:
            raise ValueError(f"{label} T does not match the dataset")
        if artifact.get("requested_N") != expected_n:
            raise ValueError(f"{label} N does not match the dataset")
        if tuple(artifact.get("selected_tables", [])) != selected_tables:
            raise ValueError(f"{label} selected_tables do not match --tables")

    source_dir = root_manifest.get("source_training_data_dir")
    if not isinstance(source_dir, str):
        raise ValueError("QA split manifest is missing source_training_data_dir")
    if _resolve_path(source_dir) != dataset_condition["bundle"]:
        raise ValueError("QA bundle was not generated from the selected dataset bundle")

    if (
        root_manifest.get("source_database_sha256")
        != dataset_condition["manifest"].get("database_sha256")
    ):
        raise ValueError("QA source database hash does not match the dataset")

    return {
        "root": root,
        "root_manifest": root_manifest,
        "sft_manifest": sft_manifest,
        "sft_data_dir": root / "target_sft",
    }


def _find_existing_dataset_bundles(
    *,
    selected_tables: tuple[str, ...],
    requested_n: int,
    seed: int,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return authenticated matching datasets in deterministic oldest-first order."""
    search_root = EXP2_DATASETS_DIR if root is None else root
    if not search_root.is_dir():
        return []
    matches: list[dict[str, Any]] = []
    for candidate in sorted(
        (path for path in search_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        try:
            condition = _verify_dataset_bundle(
                candidate,
                selected_tables=selected_tables,
                requested_n=requested_n,
            )
        except (OSError, TypeError, ValueError, KeyError):
            continue
        if condition["manifest"].get("seed") != seed:
            continue
        matches.append(condition)
    return matches


def _find_existing_qa_bundle(
    *,
    dataset_condition: dict[str, Any],
    selected_tables: tuple[str, ...],
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Return the oldest authenticated QA bundle for the exact selected dataset."""
    search_root = EXP2_QA_DIR if root is None else root
    if not search_root.is_dir():
        return None
    for candidate in sorted(
        (path for path in search_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        try:
            return _verify_qa_bundle(
                candidate,
                dataset_condition=dataset_condition,
                selected_tables=selected_tables,
            )
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return None


def _automatic_dataset_n(
    config: dict[str, Any],
    *,
    selected_tables: tuple[str, ...],
    requested_n: int | None,
) -> int:
    if requested_n is not None:
        return requested_n
    canonical_source = config.get("data", {}).get("canonical_source", {}).get("path")
    if not isinstance(canonical_source, str) or not canonical_source:
        raise ValueError("Experiment-2 canonical source path is missing")
    manifest_path = _require_file(
        _resolve_path(canonical_source) / "manifest.json",
        "canonical database manifest",
    )
    chain_count = read_json(manifest_path).get("selected_chain_count")
    if (
        isinstance(chain_count, bool)
        or not isinstance(chain_count, int)
        or chain_count <= 0
    ):
        raise ValueError("canonical manifest selected_chain_count is invalid")
    return chain_count * facts_per_selected_chain(selected_tables)


def _selected_tables_from_checkpoint_metadata(metadata: dict[str, Any]) -> list[str] | None:
    experiment_condition = metadata.get("experiment_condition", {})
    current_condition = metadata.get("current_database_condition", {})
    provenance = metadata.get("provenance", {})
    candidates = (
        experiment_condition.get("selected_tables")
        if isinstance(experiment_condition, dict)
        else None,
        current_condition.get("selected_tables")
        if isinstance(current_condition, dict)
        else None,
        provenance.get("selected_tables") if isinstance(provenance, dict) else None,
    )
    for value in candidates:
        if isinstance(value, list):
            return value
    return None


def _verify_cpt_checkpoint(
    path: Path,
    *,
    dataset_condition: dict[str, Any],
    selected_tables: tuple[str, ...],
    model_name: str,
    layers: int,
) -> dict[str, Any]:
    checkpoint = _resolve_path(path)
    _require_dir(checkpoint, "CPT checkpoint")
    _require_file(checkpoint / "config.json", "CPT checkpoint config")
    metadata_path = _require_file(
        checkpoint / "training_metadata.json", "CPT training metadata"
    )
    verify_checkpoint_layers(checkpoint, layers)
    metadata = read_json(metadata_path)

    expected = {
        "experiment": EXP2_NAME,
        "stage": "cpt",
        "model": model_name,
        "T": dataset_condition["T"],
        "N": dataset_condition["N"],
        "L": layers,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"CPT checkpoint {key} mismatch: expected {value!r}, "
                f"found {metadata.get(key)!r}"
            )

    checkpoint_tables = _selected_tables_from_checkpoint_metadata(metadata)
    if checkpoint_tables is not None and tuple(checkpoint_tables) != selected_tables:
        raise ValueError("CPT checkpoint selected_tables do not match the dataset")

    provenance = metadata.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("CPT checkpoint provenance must be a JSON object")
    expected_database_hash = dataset_condition["manifest"].get("database_sha256")
    recorded_database_hash = provenance.get("source_database_sha256")
    if recorded_database_hash is not None and recorded_database_hash != expected_database_hash:
        raise ValueError("CPT checkpoint source database hash does not match the dataset")

    expected_manifest_hash = hash_file(dataset_condition["manifest_path"])
    recorded_manifest_hash = provenance.get(
        "database_manifest_sha256",
        provenance.get("source_database_manifest_sha256"),
    )
    if recorded_manifest_hash is not None and recorded_manifest_hash != expected_manifest_hash:
        raise ValueError("CPT checkpoint dataset-manifest hash does not match the dataset")

    return metadata


def _verify_sft_checkpoint(
    path: Path,
    *,
    qa: dict[str, Any],
    cpt_checkpoint: Path,
    selected_tables: tuple[str, ...],
    model_name: str,
    layers: int,
    table_count: int,
    fact_count: int,
) -> dict[str, Any]:
    checkpoint = _resolve_path(path)
    _require_dir(checkpoint, "SFT checkpoint")
    _require_file(checkpoint / "config.json", "SFT checkpoint config")
    metadata_path = _require_file(
        checkpoint / "training_metadata.json", "SFT training metadata"
    )
    verify_checkpoint_layers(checkpoint, layers)
    metadata = read_json(metadata_path)

    expected = {
        "experiment": EXP2_NAME,
        "stage": "target-sft",
        "model": model_name,
        "T": table_count,
        "N": fact_count,
        "L": layers,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"SFT checkpoint {key} mismatch: expected {value!r}, "
                f"found {metadata.get(key)!r}"
            )

    source_checkpoint = metadata.get("source_checkpoint")
    if not isinstance(source_checkpoint, str):
        raise ValueError("SFT checkpoint metadata is missing source_checkpoint")
    if _resolve_path(source_checkpoint) != cpt_checkpoint:
        raise ValueError("SFT checkpoint was not trained from the selected CPT checkpoint")

    checkpoint_tables = _selected_tables_from_checkpoint_metadata(metadata)
    if checkpoint_tables is not None and tuple(checkpoint_tables) != selected_tables:
        raise ValueError("SFT checkpoint selected_tables do not match the QA/dataset condition")

    expected_sft_manifest_hash = hash_file(
        qa["root"] / "target_sft" / "split_manifest.json"
    )
    recorded_sft_manifest_hash = metadata.get("target_sft_split_manifest_sha256")
    if (
        recorded_sft_manifest_hash is not None
        and recorded_sft_manifest_hash != expected_sft_manifest_hash
    ):
        raise ValueError("SFT checkpoint target-SFT manifest hash does not match the QA bundle")

    return metadata


def _verify_result_dir(path: Path) -> Path:
    result = _resolve_path(path)
    _require_dir(result, "evaluation result")
    _require_file(result / "evaluation_config.json", "evaluation config")
    _require_file(result / "metrics.json", "evaluation metrics")
    _require_file(result / "predictions.jsonl", "evaluation predictions")
    return result


def _infer_resume_inputs(args: argparse.Namespace) -> None:
    """Infer only exact upstream paths recorded by supplied downstream artifacts."""
    if args.sft_checkpoint is not None:
        sft_checkpoint = _resolve_path(args.sft_checkpoint)
        metadata_path = _require_file(
            sft_checkpoint / "training_metadata.json", "SFT training metadata"
        )
        metadata = read_json(metadata_path)
        if args.cpt_checkpoint is None:
            source = metadata.get("source_checkpoint")
            if isinstance(source, str) and source:
                args.cpt_checkpoint = _resolve_path(source)
        if args.qa_path is None:
            qa_reference = metadata.get("qa_reference", {})
            if isinstance(qa_reference, dict):
                candidate = qa_reference.get("path")
                if isinstance(candidate, str) and candidate:
                    args.qa_path = _resolve_path(candidate)
            if args.qa_path is None:
                candidate = metadata.get("sft_dataset_path")
                if isinstance(candidate, str) and candidate:
                    candidate_path = _resolve_path(candidate)
                    args.qa_path = (
                        candidate_path.parent
                        if candidate_path.name == "target_sft"
                        else candidate_path
                    )

    if args.qa_path is not None and args.dataset_path is None:
        qa_root = _normalize_qa_root(_resolve_path(args.qa_path))
        manifest_path = _require_file(qa_root / "split_manifest.json", "QA split manifest")
        source = read_json(manifest_path).get("source_training_data_dir")
        if isinstance(source, str) and source:
            args.dataset_path = _resolve_path(source)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full RelMemDB Experiment-2 pipeline with one command."
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        required=True,
        help="canonical tables to expose; T is inferred",
    )
    parser.add_argument(
        "--fact-count",
        type=_positive_int,
        help="exact exposed logical fact count N; omit for baseline-N behavior",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--layers",
        type=_positive_int,
        help="architecture depth; defaults to the local model/checkpoint native depth",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help="explicit local base checkpoint; optional for registered models",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="base Experiment-2 YAML config",
    )

    parser.add_argument("--cpt-epochs", type=_positive_int)
    parser.add_argument("--sft-epochs", type=_positive_int)
    parser.add_argument("--cpt-batch-size", type=_positive_int)
    parser.add_argument("--cpt-gradient-accumulation", type=_positive_int)
    parser.add_argument("--sft-batch-size", type=_positive_int)
    parser.add_argument("--sft-gradient-accumulation", type=_positive_int)
    parser.add_argument("--cpt-learning-rate", type=_positive_float)
    parser.add_argument("--sft-learning-rate", type=_positive_float)

    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--qa-path", type=Path)
    parser.add_argument("--cpt-checkpoint", type=Path)
    parser.add_argument("--sft-checkpoint", type=Path)

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="explicitly evaluate the final SFT checkpoint on test after validation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_tables = validate_selected_tables(args.tables)
    args.config = _resolve_path(args.config)

    _infer_resume_inputs(args)

    if args.dataset_path is None and args.cpt_checkpoint is not None:
        raise ValueError(
            "--cpt-checkpoint reuse requires --dataset-path (or --qa-path/"
            "--sft-checkpoint from which the exact dataset can be authenticated)"
        )

    if args.cpt_checkpoint is not None:
        cpt_checkpoint_input = _resolve_path(args.cpt_checkpoint)
        native_layers = _checkpoint_actual_layers(cpt_checkpoint_input)
        resolved_base_model: Path | None = (
            _resolve_path(args.base_model) if args.base_model is not None else None
        )
    else:
        resolved_base_model, native_layers = resolve_model_checkpoint(
            args.model,
            source_checkpoint=(
                _resolve_path(args.base_model) if args.base_model is not None else None
            ),
        )

    layers = native_layers if args.layers is None else args.layers
    if args.cpt_checkpoint is None:
        assert resolved_base_model is not None
        verify_checkpoint_layers(resolved_base_model, layers)
    else:
        verify_checkpoint_layers(cpt_checkpoint_input, layers)

    run_dir = _create_run_dir()
    state_path = run_dir / "pipeline_state.json"
    resolved_config_path = run_dir / "resolved_config.yaml"
    overrides = _runtime_overrides(args)
    resolved_config = _write_resolved_config(
        base_config_path=args.config,
        output_path=resolved_config_path,
        model_name=args.model,
        native_layers=native_layers,
        overrides=overrides,
    )

    state: dict[str, Any] = {
        "experiment_name": EXP2_NAME,
        "status": "running",
        "created_at": _utc_iso(),
        "current_stage": None,
        "selected_tables": list(selected_tables),
        "T": len(selected_tables),
        "N": args.fact_count,
        "model": args.model,
        "layers": layers,
        "base_config_path": str(args.config),
        "resolved_config_path": str(resolved_config_path),
        "base_model_path": (
            str(resolved_base_model) if resolved_base_model is not None else None
        ),
        "cpt_epochs": resolved_config["training"]["cpt_epochs"],
        "sft_epochs": resolved_config["target_sft"]["epochs"],
        "training_overrides": overrides,
        "evaluate_test": args.evaluate_test,
        "dataset_path": None,
        "cpt_checkpoint_path": None,
        "qa_path": None,
        "sft_data_path": None,
        "sft_checkpoint_path": None,
        "cpt_validation_result_path": None,
        "sft_validation_result_path": None,
        "sft_test_result_path": None,
        "stages": {},
    }
    _write_json_atomic(state_path, state)

    try:
        if args.dataset_path is None:
            expected_n = _automatic_dataset_n(
                resolved_config,
                selected_tables=selected_tables,
                requested_n=args.fact_count,
            )
            candidates = _find_existing_dataset_bundles(
                selected_tables=selected_tables,
                requested_n=expected_n,
                seed=resolved_config["experiment"]["seed"],
            )
            for candidate in candidates:
                compatible_qa = _find_existing_qa_bundle(
                    dataset_condition=candidate,
                    selected_tables=selected_tables,
                )
                if compatible_qa is not None:
                    args.dataset_path = candidate["bundle"]
                    args.qa_path = compatible_qa["root"]
                    break
            if args.dataset_path is None and candidates:
                args.dataset_path = candidates[0]["bundle"]

        if args.dataset_path is not None:
            dataset_path = _resolve_path(args.dataset_path)
            dataset_condition = _verify_dataset_bundle(
                dataset_path,
                selected_tables=selected_tables,
                requested_n=args.fact_count,
            )
            _reuse_stage(
                state=state,
                state_path=state_path,
                stage="generate_dataset",
                artifact=dataset_path,
            )
        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_databases.py"),
                "--config",
                str(resolved_config_path),
                "--tables",
                *selected_tables,
            ]
            if args.fact_count is not None:
                command.extend(["--fact-count", str(args.fact_count)])

            def generate_dataset() -> tuple[Path, dict[str, Any]]:
                output = _run_command(command)
                path = _path_from_child(_last_line_value(output, prefix="Output: "))
                condition = _verify_dataset_bundle(
                    path,
                    selected_tables=selected_tables,
                    requested_n=args.fact_count,
                )
                return path, condition

            dataset_path, dataset_condition = _execute_stage(
                state=state,
                state_path=state_path,
                stage="generate_dataset",
                command=command,
                action=generate_dataset,
                artifact_of=lambda result: result[0],
            )

        state["T"] = dataset_condition["T"]
        state["N"] = dataset_condition["N"]
        state["dataset_path"] = str(dataset_condition["bundle"])
        _write_json_atomic(state_path, state)

        if args.cpt_checkpoint is not None:
            cpt_checkpoint = _resolve_path(args.cpt_checkpoint)
            _verify_cpt_checkpoint(
                cpt_checkpoint,
                dataset_condition=dataset_condition,
                selected_tables=selected_tables,
                model_name=args.model,
                layers=layers,
            )
            _reuse_stage(
                state=state,
                state_path=state_path,
                stage="cpt",
                artifact=cpt_checkpoint,
            )
        else:
            assert resolved_base_model is not None
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--config",
                str(resolved_config_path),
                "--stage",
                "cpt",
                "--training-data-dir",
                str(dataset_condition["bundle"]),
                "--model",
                args.model,
                "--layers",
                str(layers),
                "--source-checkpoint",
                str(resolved_base_model),
            ]

            def train_cpt() -> Path:
                output = _run_command(command)
                checkpoint = _path_from_child(
                    _last_line_value(output, token="checkpoint=")
                )
                _verify_cpt_checkpoint(
                    checkpoint,
                    dataset_condition=dataset_condition,
                    selected_tables=selected_tables,
                    model_name=args.model,
                    layers=layers,
                )
                return checkpoint

            cpt_checkpoint = _execute_stage(
                state=state,
                state_path=state_path,
                stage="cpt",
                command=command,
                action=train_cpt,
                artifact_of=lambda path: path,
            )

        state["cpt_checkpoint_path"] = str(cpt_checkpoint)
        _write_json_atomic(state_path, state)

        if args.qa_path is None:
            existing_qa = _find_existing_qa_bundle(
                dataset_condition=dataset_condition,
                selected_tables=selected_tables,
            )
            if existing_qa is not None:
                args.qa_path = existing_qa["root"]

        if args.qa_path is not None:
            qa = _verify_qa_bundle(
                _resolve_path(args.qa_path),
                dataset_condition=dataset_condition,
                selected_tables=selected_tables,
            )
            _reuse_stage(
                state=state,
                state_path=state_path,
                stage="generate_qa",
                artifact=qa["root"],
            )
        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_target_sft_qa.py"),
                "--config",
                str(resolved_config_path),
                "--training-data-dir",
                str(dataset_condition["bundle"]),
            ]

            def generate_qa() -> dict[str, Any]:
                output = _run_command(command)
                root = _path_from_child(
                    _last_line_value(output, prefix="Target-SFT QA output: ")
                )
                return _verify_qa_bundle(
                    root,
                    dataset_condition=dataset_condition,
                    selected_tables=selected_tables,
                )

            qa = _execute_stage(
                state=state,
                state_path=state_path,
                stage="generate_qa",
                command=command,
                action=generate_qa,
                artifact_of=lambda value: value["root"],
            )

        state["qa_path"] = str(qa["root"])
        state["sft_data_path"] = str(qa["sft_data_dir"])
        _write_json_atomic(state_path, state)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(resolved_config_path),
            "--qa-data-dir",
            str(qa["root"]),
            "--checkpoint",
            str(cpt_checkpoint),
            "--layers",
            str(layers),
            "--split",
            "validation",
        ]

        def eval_cpt() -> Path:
            output = _run_command(command)
            result = _path_from_child(_last_line_value(output, token="output="))
            return _verify_result_dir(result)

        cpt_validation_result = _execute_stage(
            state=state,
            state_path=state_path,
            stage="eval_cpt_validation",
            command=command,
            action=eval_cpt,
            artifact_of=lambda path: path,
        )
        state["cpt_validation_result_path"] = str(cpt_validation_result)
        _write_json_atomic(state_path, state)

        if args.sft_checkpoint is not None:
            sft_checkpoint = _resolve_path(args.sft_checkpoint)
            _verify_sft_checkpoint(
                sft_checkpoint,
                qa=qa,
                cpt_checkpoint=cpt_checkpoint,
                selected_tables=selected_tables,
                model_name=args.model,
                layers=layers,
                table_count=dataset_condition["T"],
                fact_count=dataset_condition["N"],
            )
            _reuse_stage(
                state=state,
                state_path=state_path,
                stage="target_sft",
                artifact=sft_checkpoint,
            )
        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--config",
                str(resolved_config_path),
                "--stage",
                "target-sft",
                "--sft-data-dir",
                str(qa["sft_data_dir"]),
                "--source-checkpoint",
                str(cpt_checkpoint),
                "--model",
                args.model,
                "--layers",
                str(layers),
            ]

            def train_sft() -> Path:
                output = _run_command(command)
                checkpoint = _path_from_child(
                    _last_line_value(output, token="checkpoint=")
                )
                _verify_sft_checkpoint(
                    checkpoint,
                    qa=qa,
                    cpt_checkpoint=cpt_checkpoint,
                    selected_tables=selected_tables,
                    model_name=args.model,
                    layers=layers,
                    table_count=dataset_condition["T"],
                    fact_count=dataset_condition["N"],
                )
                return checkpoint

            sft_checkpoint = _execute_stage(
                state=state,
                state_path=state_path,
                stage="target_sft",
                command=command,
                action=train_sft,
                artifact_of=lambda path: path,
            )

        state["sft_checkpoint_path"] = str(sft_checkpoint)
        _write_json_atomic(state_path, state)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(resolved_config_path),
            "--qa-data-dir",
            str(qa["root"]),
            "--checkpoint",
            str(sft_checkpoint),
            "--layers",
            str(layers),
            "--split",
            "validation",
        ]

        def eval_sft_validation() -> Path:
            output = _run_command(command)
            result = _path_from_child(_last_line_value(output, token="output="))
            return _verify_result_dir(result)

        sft_validation_result = _execute_stage(
            state=state,
            state_path=state_path,
            stage="eval_sft_validation",
            command=command,
            action=eval_sft_validation,
            artifact_of=lambda path: path,
        )
        state["sft_validation_result_path"] = str(sft_validation_result)
        _write_json_atomic(state_path, state)

        if args.evaluate_test:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(resolved_config_path),
                "--qa-data-dir",
                str(qa["root"]),
                "--checkpoint",
                str(sft_checkpoint),
                "--layers",
                str(layers),
                "--split",
                "test",
            ]

            def eval_sft_test() -> Path:
                output = _run_command(command)
                result = _path_from_child(_last_line_value(output, token="output="))
                return _verify_result_dir(result)

            sft_test_result = _execute_stage(
                state=state,
                state_path=state_path,
                stage="eval_sft_test",
                command=command,
                action=eval_sft_test,
                artifact_of=lambda path: path,
            )
            state["sft_test_result_path"] = str(sft_test_result)
            _write_json_atomic(state_path, state)

        state["status"] = "completed"
        state["current_stage"] = None
        state["completed_at"] = _utc_iso()
        _write_json_atomic(state_path, state)

    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = {
            "time": _utc_iso(),
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json_atomic(state_path, state)
        print("\nExperiment 2 pipeline FAILED.", file=sys.stderr)
        print(f"Run state preserved at: {state_path}", file=sys.stderr)
        raise

    print(
        "\n"
        "Experiment 2 complete\n"
        "---------------------\n"
        f"Tables: {' '.join(selected_tables)}\n"
        f"T: {dataset_condition['T']}\n"
        f"N: {dataset_condition['N']}\n"
        f"Model: {args.model}\n"
        f"Layers: {layers}\n"
        f"CPT epochs: {resolved_config['training']['cpt_epochs']}\n"
        f"SFT epochs: {resolved_config['target_sft']['epochs']}\n"
        "\n"
        f"Dataset:\n{dataset_condition['bundle']}\n\n"
        f"CPT checkpoint:\n{cpt_checkpoint}\n\n"
        f"QA:\n{qa['root']}\n\n"
        f"SFT dataset:\n{qa['sft_data_dir']}\n\n"
        f"SFT checkpoint:\n{sft_checkpoint}\n\n"
        f"CPT validation result:\n{cpt_validation_result}\n\n"
        f"SFT validation result:\n{sft_validation_result}\n\n"
        f"Run state:\n{state_path}\n"
    )
    if args.evaluate_test:
        print(f"\nSFT test result:\n{state['sft_test_result_path']}")


if __name__ == "__main__":
    main()

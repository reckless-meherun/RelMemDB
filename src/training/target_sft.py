"""Closed-book supervised fine-tuning on verified target-database QA records."""

from __future__ import annotations

import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

from data.qa import RAW_ENTITY_IDENTIFIER, answer_is_in_question, normalize_for_leakage
from data.qa_reference import verify_qa_reference_compatibility
from evaluation.inference import (
    HOP_NAMES,
    PROMPT_TEMPLATE,
    format_question_prompt,
    generate_prediction_records,
)
from evaluation.metrics import compute_evaluation_metrics
from experiment import configured_model_layers, qa_reference_values, verify_checkpoint_layers
from training.cpt import enable_full_parameter_training
from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, read_jsonl, write_json, write_jsonl, write_yaml
from utils.paths import database_condition_dir, qa_reference_dir

TARGET_SFT_DATASET_DIR = "target_sft"
TARGET_SFT_TRAIN_SPLIT = "train"
TARGET_SFT_DEV_SPLIT = "dev"
TARGET_SFT_SPLIT_METHOD_VERSION = "reserved_order_9_train_1_dev_v1"
CANONICAL_TARGET_SFT_COUNTS = {
    "train": {
        "chain_count": 135,
        "total_examples": 6378,
        "hop_counts": {"H0": 3485, "H1": 1168, "H2": 917, "H3": 808},
    },
    "dev": {
        "chain_count": 15,
        "total_examples": 709,
        "hop_counts": {"H0": 388, "H1": 130, "H2": 102, "H3": 89},
    },
}
REQUIRED_CROSS_PARTITION_PAIRS = {
    "train__dev",
    "train__validation",
    "train__test",
    "dev__validation",
    "dev__test",
}


def _require_nonempty_file(
    path: Path, label: str, *, allow_empty: bool = False
) -> Path:
    if not path.is_file() or (path.stat().st_size == 0 and not allow_empty):
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_zero_overlap_audit(split_manifest: dict[str, Any], field: str) -> None:
    audit = split_manifest.get(field)
    if not isinstance(audit, dict) or not REQUIRED_CROSS_PARTITION_PAIRS <= set(audit):
        raise ValueError(f"target-SFT {field} is missing required partition pairs")
    invalid = {
        pair: count
        for pair, count in audit.items()
        if isinstance(count, bool) or not isinstance(count, int) or count != 0
    }
    if invalid:
        raise ValueError(
            f"target-SFT {field} must contain only zero overlaps: {invalid}"
        )


def _validate_target_sft_record(
    record: dict[str, Any], *, split: str, hop: int
) -> None:
    h0_fields = {
        "id",
        "split",
        "hop",
        "question",
        "gold_answer",
        "fact_type",
        "source_entity_type",
        "target_entity_type",
        "target_field",
    }
    expected_fields = (
        h0_fields if hop == 0 else h0_fields - {"fact_type"} | {"support_fact_ids"}
    )
    if set(record) != expected_fields:
        raise ValueError(f"target-SFT {split} H{hop} record schema is invalid")
    if record.get("split") != split or record.get("hop") != hop:
        raise ValueError(f"target-SFT {split} H{hop} metadata is inconsistent")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError(f"target-SFT {split} H{hop} record ID is invalid")
    for field in ("question", "gold_answer"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"target-SFT {split} H{hop} {field} is invalid")
        if RAW_ENTITY_IDENTIFIER.search(value):
            raise ValueError(f"target-SFT {split} H{hop} exposes a raw database ID")
    if answer_is_in_question(record["question"], record["gold_answer"]):
        raise ValueError(f"target-SFT {split} H{hop} retains answer leakage")
    if hop == 0 and record["fact_type"] not in {"attribute", "relation"}:
        raise ValueError(f"target-SFT {split} H0 fact_type is invalid")


def _load_authenticated_target_sft_split(
    *,
    dataset_path: Path,
    split: str,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
    expected_table_count: int,
    expected_fact_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = _require_nonempty_file(
        dataset_path / split / "manifest.json", f"target-SFT {split} manifest"
    )
    manifest_sha256 = hash_file(manifest_path)
    if manifest_sha256 != split_manifest.get(f"{split}_manifest_sha256"):
        raise ValueError(
            f"target-SFT {split} manifest hash does not match split manifest"
        )
    manifest = read_json(manifest_path)
    expected_chain_indices = split_manifest.get(f"{split}_chain_indices")
    expected_metadata = {
        "format_version": split_manifest.get("format_version"),
        "experiment_name": split_manifest.get("experiment_name"),
        "T": expected_table_count,
        "requested_N": expected_fact_count,
        "N": expected_fact_count,
        "split": split,
        "chain_count": split_manifest.get(f"{split}_chain_count"),
        "chain_indices": expected_chain_indices,
        "source_database_sha256": split_manifest.get("source_database_sha256"),
        "source_database_manifest_sha256": split_manifest.get(
            "source_database_manifest_sha256"
        ),
        "source_evaluation_split_manifest_sha256": split_manifest.get(
            "source_evaluation_split_manifest_sha256"
        ),
        "sft_split_method_version": TARGET_SFT_SPLIT_METHOD_VERSION,
        "question_template_version": split_manifest.get("question_template_version"),
        "zero_context": True,
        "selected_tables": split_manifest.get("selected_tables"),
        "selected_positions": split_manifest.get("selected_positions"),
        "source_training_data_dir": split_manifest.get("source_training_data_dir"),
        "generation_timestamp": split_manifest.get("generation_timestamp"),
    }
    if split_manifest.get("experiment_name") == "exp02_capacity_boundary":
        expected_metadata["source_dataset_manifest_sha256"] = split_manifest.get(
            "source_dataset_manifest_sha256"
        )
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(f"target-SFT {split} manifest {key} is inconsistent")
    if manifest.get("chain_indices_sha256") != hash_json_object(expected_chain_indices):
        raise ValueError(f"target-SFT {split} chain assignment hash is inconsistent")
    output_hashes = manifest.get("output_file_hashes")
    counts = manifest.get("counts")
    if not isinstance(output_hashes, dict) or not isinstance(counts, dict):
        raise ValueError(f"target-SFT {split} manifest hashes or counts are missing")

    records_by_hop: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for hop, hop_name in enumerate(HOP_NAMES):
        path = _require_nonempty_file(
            dataset_path / split / f"{hop_name}.jsonl",
            f"target-SFT {split} {hop_name}",
            allow_empty=(
                split_manifest.get("experiment_name") == "exp02_capacity_boundary"
                and counts.get(hop_name, {}).get("final_retained_count") == 0
            ),
        )
        actual_hash = hash_file(path)
        if output_hashes.get(path.name) != actual_hash:
            raise ValueError(f"target-SFT {split} {hop_name} hash is inconsistent")
        hop_records = read_jsonl(path)
        expected_count = counts.get(hop_name, {}).get("final_retained_count")
        if len(hop_records) != expected_count:
            raise ValueError(f"target-SFT {split} {hop_name} count is inconsistent")
        if manifest.get("retained_counts", {}).get(hop_name) != expected_count:
            raise ValueError(
                f"target-SFT {split} {hop_name} retained count is inconsistent"
            )
        for record in hop_records:
            _validate_target_sft_record(record, split=split, hop=hop)
            if record["id"] in seen_ids:
                raise ValueError(f"target-SFT {split} record IDs must be unique")
            seen_ids.add(record["id"])
        records_by_hop[hop_name] = hop_records
        input_hashes[hop_name] = actual_hash

    h0_ids = {record["id"] for record in records_by_hop["H0"]}
    for hop, hop_name in enumerate(HOP_NAMES[1:], start=1):
        for record in records_by_hop[hop_name]:
            support_ids = record["support_fact_ids"]
            if len(support_ids) != hop + 1 or any(
                support_id not in h0_ids for support_id in support_ids
            ):
                raise ValueError(
                    f"target-SFT {split} {hop_name} support closure is invalid"
                )
    records = [record for hop_name in HOP_NAMES for record in records_by_hop[hop_name]]
    if len(records) != manifest.get("final_retained_total"):
        raise ValueError(f"target-SFT {split} retained total is inconsistent")
    hop_counts = {hop: len(records_by_hop[hop]) for hop in HOP_NAMES}
    return records, {
        "split": split,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
        "chain_count": manifest["chain_count"],
        "chain_indices": manifest["chain_indices"],
        "total_examples": len(records),
        "hop_counts": hop_counts,
        "input_file_sha256": input_hashes,
        "target_sft_split_manifest_sha256": split_manifest_sha256,
    }


def load_target_sft_dataset(
    qa_condition_dir: str | Path,
    *,
    dataset_dir: str,
    training_split: str,
    dev_split: str,
    table_count: int,
    fact_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and load only dedicated target-SFT train and dev records."""
    if dataset_dir != TARGET_SFT_DATASET_DIR:
        raise ValueError("target SFT dataset_dir must be target_sft")
    if training_split != TARGET_SFT_TRAIN_SPLIT or dev_split != TARGET_SFT_DEV_SPLIT:
        raise ValueError(
            "target SFT may load only target_sft/train and target_sft/dev; "
            "validation and test are forbidden"
        )
    dataset_path = Path(qa_condition_dir) / dataset_dir
    split_manifest_path = _require_nonempty_file(
        dataset_path / "split_manifest.json", "target-SFT split manifest"
    )
    split_manifest_sha256 = hash_file(split_manifest_path)
    split_manifest = read_json(split_manifest_path)
    if split_manifest.get("experiment_name") == "exp02_capacity_boundary":
        expected_root = {
            "format_version": 2,
            "T": table_count,
            "N": fact_count,
            "requested_N": fact_count,
            "source_evaluation_split_manifest": "../split_manifest.json",
            "question_template_version": "semantic_academic_closed_book_v1",
            "sft_split_method_version": TARGET_SFT_SPLIT_METHOD_VERSION,
            "zero_context": True,
            "target_qa_training_generated": True,
            "deterministic_generation": True,
            "runtime_llm_used": False,
            "immutable_evaluation_artifacts_unchanged": True,
        }
        for key, expected in expected_root.items():
            if split_manifest.get(key) != expected:
                raise ValueError(f"Experiment-2 target-SFT split manifest {key} is inconsistent")
        selected = split_manifest.get("selected_tables")
        if not isinstance(selected, list) or len(selected) != table_count:
            raise ValueError("Experiment-2 target-SFT selected-table metadata is invalid")
        if split_manifest.get("source_dataset_manifest_sha256") != split_manifest.get(
            "source_database_manifest_sha256"
        ):
            raise ValueError("Experiment-2 source dataset-manifest provenance is inconsistent")
        assignments = {
            split: split_manifest.get(f"{split}_chain_indices")
            for split in ("train", "dev")
        }
        for split, indices in assignments.items():
            if not isinstance(indices, list) or len(indices) != split_manifest.get(f"{split}_chain_count"):
                raise ValueError(f"Experiment-2 target-SFT {split} chain assignment is invalid")
            expected_hash = hash_json_object(indices)
            if (
                split_manifest.get(f"{split}_chain_indices_sha256") != expected_hash
                or split_manifest.get("chain_assignment_hashes", {}).get(split) != expected_hash
            ):
                raise ValueError(f"Experiment-2 target-SFT {split} chain assignment hash is invalid")
        if split_manifest.get("target_sft_chain_assignments_sha256") != hash_json_object(assignments):
            raise ValueError("Experiment-2 target-SFT combined chain assignment hash is invalid")
        partition_sets = {
            "train": set(assignments["train"]),
            "dev": set(assignments["dev"]),
            "validation": set(split_manifest.get("validation_chain_indices", [])),
            "test": set(split_manifest.get("test_chain_indices", [])),
        }
        if any(
            partition_sets[left] & partition_sets[right]
            for index, left in enumerate(partition_sets)
            for right in list(partition_sets)[index + 1 :]
        ):
            raise ValueError("Experiment-2 target-SFT chain partitions overlap")
        for audit_field in (
            "chain_overlap_counts", "qa_id_overlap_counts", "question_overlap_counts",
            "exact_question_overlap_counts", "normalized_question_overlap_counts",
            "normalized_qa_pair_overlap_counts",
        ):
            _require_zero_overlap_audit(split_manifest, audit_field)
        qa_root = Path(qa_condition_dir)
        evaluation_manifest_path = _require_nonempty_file(
            qa_root / "split_manifest.json", "Experiment-2 evaluation split manifest"
        )
        if hash_file(evaluation_manifest_path) != split_manifest.get("source_evaluation_split_manifest_sha256"):
            raise ValueError("Experiment-2 SFT evaluation-manifest provenance mismatch")
        for field in (
            "source_database_sha256", "source_database_manifest_sha256",
            "source_evaluation_split_manifest_sha256", "train_manifest_sha256",
            "dev_manifest_sha256", "train_chain_indices_sha256",
            "dev_chain_indices_sha256", "target_sft_chain_assignments_sha256",
        ):
            if not _is_sha256(split_manifest.get(field)):
                raise ValueError(f"Experiment-2 target-SFT split manifest {field} is invalid")
        train_records, train_provenance = _load_authenticated_target_sft_split(
            dataset_path=dataset_path, split=training_split,
            split_manifest=split_manifest, split_manifest_sha256=split_manifest_sha256,
            expected_table_count=table_count, expected_fact_count=fact_count,
        )
        dev_records, dev_provenance = _load_authenticated_target_sft_split(
            dataset_path=dataset_path, split=dev_split,
            split_manifest=split_manifest, split_manifest_sha256=split_manifest_sha256,
            expected_table_count=table_count, expected_fact_count=fact_count,
        )
        train_ids = {record["id"] for record in train_records}
        dev_ids = {record["id"] for record in dev_records}
        if train_ids & dev_ids:
            raise ValueError("Experiment-2 target-SFT train/dev QA IDs overlap")
        train_questions = {normalize_for_leakage(record["question"]) for record in train_records}
        dev_questions = {normalize_for_leakage(record["question"]) for record in dev_records}
        if train_questions & dev_questions:
            raise ValueError("Experiment-2 target-SFT train/dev normalized questions overlap")
        train_pairs = {
            (normalize_for_leakage(record["question"]), normalize_for_leakage(record["gold_answer"]))
            for record in train_records
        }
        dev_pairs = {
            (normalize_for_leakage(record["question"]), normalize_for_leakage(record["gold_answer"]))
            for record in dev_records
        }
        if train_pairs & dev_pairs:
            raise ValueError("Experiment-2 target-SFT train/dev normalized QA pairs overlap")
        source_dir_value = split_manifest.get("source_training_data_dir")
        if not isinstance(source_dir_value, str) or not source_dir_value:
            raise ValueError("Experiment-2 target-SFT source training-data path is missing")
        source_dir = Path(source_dir_value).resolve()
        database_path = _require_nonempty_file(source_dir / "database.sqlite", "Experiment-2 source database")
        database_manifest_path = _require_nonempty_file(source_dir / "manifest.json", "Experiment-2 source database manifest")
        if hash_file(database_path) != split_manifest["source_database_sha256"]:
            raise ValueError("Experiment-2 target-SFT source database hash mismatch")
        if hash_file(database_manifest_path) != split_manifest["source_database_manifest_sha256"]:
            raise ValueError("Experiment-2 target-SFT source database manifest hash mismatch")
        return train_records, dev_records, {
            "dataset_path": str(dataset_path.resolve()),
            "qa_condition_dir": str(Path(qa_condition_dir).resolve()),
            "source_training_data_dir": str(source_dir),
            "selected_tables": selected,
            "target_sft_split_manifest_sha256": split_manifest_sha256,
            "train_manifest_sha256": train_provenance["manifest_sha256"],
            "dev_manifest_sha256": dev_provenance["manifest_sha256"],
            "source_database_sha256": split_manifest["source_database_sha256"],
            "source_database_manifest_sha256": split_manifest["source_database_manifest_sha256"],
            "train": train_provenance,
            "dev": dev_provenance,
            "zero_context": True,
            "validation_split_used": False,
            "test_split_used": False,
        }
    expected_root = {
        "format_version": 2,
        "experiment_name": "exp01_first_feasibility",
        "T": table_count,
        "N": fact_count,
        "requested_N": fact_count,
        "source_evaluation_split_manifest": "../split_manifest.json",
        "question_template_version": "semantic_academic_closed_book_v1",
        "sft_split_method_version": TARGET_SFT_SPLIT_METHOD_VERSION,
        "original_reserved_chain_count": 150,
        "train_chain_count": 135,
        "dev_chain_count": 15,
        "zero_context": True,
        "target_qa_training_generated": True,
        "deterministic_generation": True,
        "runtime_llm_used": False,
        "immutable_evaluation_artifacts_unchanged": True,
    }
    for key, expected in expected_root.items():
        if split_manifest.get(key) != expected:
            raise ValueError(f"target-SFT split manifest {key} is inconsistent")
    for field in (
        "source_database_sha256",
        "source_database_manifest_sha256",
        "source_evaluation_split_manifest_sha256",
        "train_manifest_sha256",
        "dev_manifest_sha256",
        "train_chain_indices_sha256",
        "dev_chain_indices_sha256",
        "target_sft_chain_assignments_sha256",
    ):
        if not _is_sha256(split_manifest.get(field)):
            raise ValueError(f"target-SFT split manifest {field} is invalid")
    for split in (TARGET_SFT_TRAIN_SPLIT, TARGET_SFT_DEV_SPLIT):
        indices = split_manifest.get(f"{split}_chain_indices")
        if not isinstance(indices, list) or len(indices) != split_manifest.get(
            f"{split}_chain_count"
        ):
            raise ValueError(f"target-SFT {split} chain assignment is invalid")
        expected_hash = split_manifest.get(f"{split}_chain_indices_sha256")
        if expected_hash != hash_json_object(indices):
            raise ValueError(f"target-SFT {split} chain assignment hash is invalid")
        if (
            split_manifest.get("chain_assignment_hashes", {}).get(split)
            != expected_hash
        ):
            raise ValueError(f"target-SFT {split} recorded assignment hash is invalid")
    assignments = {
        "train": split_manifest["train_chain_indices"],
        "dev": split_manifest["dev_chain_indices"],
    }
    if split_manifest["target_sft_chain_assignments_sha256"] != hash_json_object(
        assignments
    ):
        raise ValueError("target-SFT combined chain assignment hash is invalid")
    train_chains = set(split_manifest["train_chain_indices"])
    dev_chains = set(split_manifest["dev_chain_indices"])
    validation_chains = set(split_manifest.get("validation_chain_indices", []))
    test_chains = set(split_manifest.get("test_chain_indices", []))
    if train_chains & dev_chains or len(train_chains | dev_chains) != 150:
        raise ValueError("target-SFT train/dev chain partition is invalid")
    if train_chains & validation_chains or train_chains & test_chains:
        raise ValueError("target-SFT train chains overlap held-out chains")
    if dev_chains & validation_chains or dev_chains & test_chains:
        raise ValueError("target-SFT dev chains overlap held-out chains")
    if len(validation_chains) != 50 or len(test_chains) != 50:
        raise ValueError("target-SFT held-out chain audit counts are invalid")
    if validation_chains & test_chains:
        raise ValueError("target-SFT validation/test chain audits overlap")
    for audit_field in (
        "chain_overlap_counts",
        "qa_id_overlap_counts",
        "question_overlap_counts",
        "exact_question_overlap_counts",
        "normalized_question_overlap_counts",
        "normalized_qa_pair_overlap_counts",
    ):
        _require_zero_overlap_audit(split_manifest, audit_field)

    database_dir = database_condition_dir(table_count, fact_count)
    database_path = _require_nonempty_file(
        database_dir / "database.sqlite", "target-SFT source database"
    )
    database_manifest_path = _require_nonempty_file(
        database_dir / "manifest.json", "target-SFT source database manifest"
    )
    database_sha256 = hash_file(database_path)
    database_manifest_sha256 = hash_file(database_manifest_path)
    if database_sha256 != split_manifest.get("source_database_sha256"):
        raise ValueError("target-SFT source database hash is inconsistent")
    if database_manifest_sha256 != split_manifest.get(
        "source_database_manifest_sha256"
    ):
        raise ValueError("target-SFT source database manifest hash is inconsistent")
    database_manifest = read_json(database_manifest_path)
    if database_manifest.get("database_sha256") != database_sha256:
        raise ValueError("target-SFT database does not match its database manifest")
    if (
        database_manifest.get("T") != table_count
        or database_manifest.get("requested_N") != fact_count
    ):
        raise ValueError("target-SFT source database condition is inconsistent")

    train_records, train_provenance = _load_authenticated_target_sft_split(
        dataset_path=dataset_path,
        split=training_split,
        split_manifest=split_manifest,
        split_manifest_sha256=split_manifest_sha256,
        expected_table_count=table_count,
        expected_fact_count=fact_count,
    )
    dev_records, dev_provenance = _load_authenticated_target_sft_split(
        dataset_path=dataset_path,
        split=dev_split,
        split_manifest=split_manifest,
        split_manifest_sha256=split_manifest_sha256,
        expected_table_count=table_count,
        expected_fact_count=fact_count,
    )
    train_ids = {record["id"] for record in train_records}
    dev_ids = {record["id"] for record in dev_records}
    if train_ids & dev_ids:
        raise ValueError("target-SFT train/dev QA IDs overlap")
    train_questions = {
        normalize_for_leakage(record["question"]) for record in train_records
    }
    dev_questions = {
        normalize_for_leakage(record["question"]) for record in dev_records
    }
    if train_questions & dev_questions:
        raise ValueError("target-SFT train/dev normalized questions overlap")
    train_pairs = {
        (
            normalize_for_leakage(record["question"]),
            normalize_for_leakage(record["gold_answer"]),
        )
        for record in train_records
    }
    dev_pairs = {
        (
            normalize_for_leakage(record["question"]),
            normalize_for_leakage(record["gold_answer"]),
        )
        for record in dev_records
    }
    if train_pairs & dev_pairs:
        raise ValueError("target-SFT train/dev normalized QA pairs overlap")

    if (table_count, fact_count) == (12, 10_000):
        for split, provenance in (
            ("train", train_provenance),
            ("dev", dev_provenance),
        ):
            expected = CANONICAL_TARGET_SFT_COUNTS[split]
            for key in ("chain_count", "total_examples", "hop_counts"):
                if provenance[key] != expected[key]:
                    raise ValueError(
                        f"canonical target-SFT {split} {key} is inconsistent"
                    )
    return (
        train_records,
        dev_records,
        {
            "dataset_path": str(dataset_path),
            "training_split": training_split,
            "dev_split": dev_split,
            "target_sft_split_manifest_sha256": split_manifest_sha256,
            "train_manifest_sha256": train_provenance["manifest_sha256"],
            "dev_manifest_sha256": dev_provenance["manifest_sha256"],
            "source_database_sha256": database_sha256,
            "source_database_manifest_sha256": database_manifest_sha256,
            "train": train_provenance,
            "dev": dev_provenance,
            "zero_context": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
    )


def encode_target_sft_example(
    record: dict[str, Any], tokenizer: Any, *, context_length: int
) -> dict[str, Any]:
    """Encode evaluator-compatible question prompting with answer-only labels."""
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("target_sft.context_length must be a positive integer")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    question = record.get("question")
    gold_answer = record.get("gold_answer")
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ValueError("target-SFT gold_answer must be non-empty text")
    prompt = format_question_prompt(question)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = list(tokenizer.encode(gold_answer, add_special_tokens=False))
    if not answer_ids:
        raise ValueError("target-SFT gold_answer tokenization is empty")
    input_ids = [*prompt_ids, *answer_ids, tokenizer.eos_token_id]
    if len(input_ids) > context_length:
        raise ValueError(
            f"target-SFT record {record.get('id', '<unknown>')} has "
            f"{len(input_ids)} tokens and exceeds context length {context_length}; "
            "truncation is forbidden"
        )
    labels = [-100] * len(prompt_ids) + [*answer_ids, tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt": prompt,
        "prompt_length": len(prompt_ids),
        "answer_token_count": len(answer_ids),
        "supervised_token_count": len(answer_ids) + 1,
        "sequence_length": len(input_ids),
        "record_id": record.get("id"),
        "hop": record.get("hop"),
    }


def collate_target_sft_examples(
    examples: list[dict[str, Any]], *, pad_token_id: int
) -> dict[str, Any]:
    """Right-pad causal-LM examples while masking every padding label."""
    if not examples:
        raise ValueError("cannot collate an empty target-SFT batch")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("target SFT requires torch") from exc
    width = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(example["attention_mask"] + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_target_sft_training_plan(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    example_count: int,
    layers: int | None = None,
) -> dict[str, Any]:
    """Validate target-SFT settings and calculate exact update accounting."""
    settings = config.get("target_sft")
    if not isinstance(settings, dict):
        raise ValueError("target_sft configuration section is required")
    if (
        isinstance(example_count, bool)
        or not isinstance(example_count, int)
        or example_count <= 0
    ):
        raise ValueError("example_count must be a positive integer")

    def positive_int(key: str) -> int:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"target_sft.{key} must be a positive integer")
        return value

    def non_negative_int(key: str) -> int:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"target_sft.{key} must be a non-negative integer")
        return value

    def boolean(key: str) -> bool:
        value = settings.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"target_sft.{key} must be a boolean")
        return value

    def number(key: str, *, allow_zero: bool = False) -> float:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"target_sft.{key} must be a number")
        value = float(value)
        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"target_sft.{key} must be {qualifier}")
        return value

    if settings.get("dataset_dir") != TARGET_SFT_DATASET_DIR:
        raise ValueError("target_sft.dataset_dir must be target_sft")
    if settings.get("training_split") != TARGET_SFT_TRAIN_SPLIT:
        raise ValueError("target_sft.training_split must be train")
    if settings.get("dev_split") != TARGET_SFT_DEV_SPLIT:
        raise ValueError("target_sft.dev_split must be dev")
    batch_size = positive_int("batch_size")
    accumulation_steps = positive_int("gradient_accumulation_steps")
    epochs = positive_int("epochs")
    early_stopping_patience = positive_int("early_stopping_patience")
    if early_stopping_patience != 3:
        raise ValueError("target_sft.early_stopping_patience must be 3")
    context_length = positive_int("context_length")
    dataloader_workers = non_negative_int("dataloader_workers")
    learning_rate = number("learning_rate")
    weight_decay = number("weight_decay", allow_zero=True)
    epsilon = number("epsilon")
    warmup_ratio = number("warmup_ratio", allow_zero=True)
    max_grad_norm = number("max_grad_norm")
    if warmup_ratio > 1.0:
        raise ValueError("target_sft.warmup_ratio must be between 0 and 1")
    betas = settings.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("target_sft.betas must contain exactly two numbers")
    normalized_betas: list[float] = []
    for index, beta in enumerate(betas):
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise ValueError(f"target_sft.betas[{index}] must be a number")
        normalized_beta = float(beta)
        if not 0.0 <= normalized_beta < 1.0:
            raise ValueError(f"target_sft.betas[{index}] must be in [0, 1)")
        normalized_betas.append(normalized_beta)
    for key, expected in (
        ("optimizer", "adamw"),
        ("scheduler", "cosine"),
        ("precision", "bf16"),
    ):
        if str(settings.get(key, "")).lower() != expected:
            raise ValueError(f"target_sft.{key} must be {expected}")
    shuffle = boolean("shuffle")
    gradient_checkpointing = boolean("gradient_checkpointing")
    fused_optimizer = boolean("fused_optimizer")
    pin_memory = boolean("pin_memory")
    drop_last = boolean("drop_last")
    answer_only_loss = boolean("answer_only_loss")
    supervise_eos = boolean("supervise_eos")
    if drop_last:
        raise ValueError(
            "target_sft.drop_last must be false to train on all QA records"
        )
    if not answer_only_loss or not supervise_eos:
        raise ValueError("target SFT requires answer-only loss and supervised EOS")
    seed = config.get("experiment", {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment.seed must be a non-negative integer")

    microbatches_per_epoch = math.ceil(example_count / batch_size)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = math.ceil(total_optimizer_steps * warmup_ratio)
    return {
        "stage": "target-sft",
        "T": table_count,
        "N": fact_count,
        "L": configured_model_layers(config) if layers is None else layers,
        "dataset_dir": TARGET_SFT_DATASET_DIR,
        "training_split": TARGET_SFT_TRAIN_SPLIT,
        "dev_split": TARGET_SFT_DEV_SPLIT,
        "early_stopping_patience": early_stopping_patience,
        "example_count": example_count,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "epochs": epochs,
        "context_length": context_length,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": normalized_betas,
        "epsilon": epsilon,
        "scheduler": "cosine",
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "precision": "bf16",
        "shuffle": shuffle,
        "gradient_checkpointing": gradient_checkpointing,
        "fused_optimizer_requested": fused_optimizer,
        "fused_optimizer_actually_used": None,
        "dataloader_workers": dataloader_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "answer_only_loss": True,
        "supervise_eos": True,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "steps_per_epoch": optimizer_steps_per_epoch,
        "maximum_optimizer_steps": total_optimizer_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "optimizer_steps": total_optimizer_steps,
        "seed": seed,
        "validation_split_used": False,
        "test_split_used": False,
    }


def seeded_dataloader_generator(torch_module: Any, seed: int) -> Any:
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def build_target_sft_optimizer(
    torch_module: Any, parameters: Any, plan: dict[str, Any]
) -> tuple[Any, bool, str | None]:
    parameter_list = list(parameters)
    kwargs = {
        "lr": plan["learning_rate"],
        "weight_decay": plan["weight_decay"],
        "betas": tuple(plan["betas"]),
        "eps": plan["epsilon"],
    }
    if not plan["fused_optimizer_requested"]:
        return torch_module.optim.AdamW(parameter_list, **kwargs), False, None
    try:
        optimizer = torch_module.optim.AdamW(parameter_list, fused=True, **kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        return (
            torch_module.optim.AdamW(parameter_list, **kwargs),
            False,
            f"{type(exc).__name__}: {exc}",
        )
    return optimizer, bool(optimizer.defaults.get("fused", True)), None


def configure_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if enabled:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("source model does not support gradient checkpointing")
        model.gradient_checkpointing_enable()
    elif getattr(model, "is_gradient_checkpointing", False):
        if not hasattr(model, "gradient_checkpointing_disable"):
            raise RuntimeError("source model cannot disable gradient checkpointing")
        model.gradient_checkpointing_disable()


def ensure_target_sft_outputs_available(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    run_config_path: str | Path,
    train_log_path: str | Path,
) -> None:
    source = Path(source_checkpoint)
    output = Path(output_checkpoint)
    if source.resolve() == output.resolve():
        raise ValueError("source and target-SFT checkpoint paths must differ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"target-SFT checkpoint path is not an empty directory: {output}"
        )
    for path in (Path(run_config_path), Path(train_log_path)):
        if path.exists():
            raise FileExistsError(f"target-SFT run artifact already exists: {path}")


def _load_local_model_and_tokenizer(source_checkpoint: Path) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("target SFT requires transformers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source_checkpoint, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            source_checkpoint, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"unable to load local source checkpoint {source_checkpoint}: {exc}"
        ) from exc
    if tokenizer.eos_token_id is None:
        raise ValueError("source tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def summarize_target_sft_dev_metrics(
    prediction_records: list[dict[str, Any]],
    *,
    dev_answer_only_loss: float,
) -> dict[str, Any]:
    """Return the exact dev metrics used for target-SFT model selection."""
    if not math.isfinite(dev_answer_only_loss) or dev_answer_only_loss < 0:
        raise ValueError("dev answer-only loss must be finite and non-negative")
    metrics = compute_evaluation_metrics(prediction_records)

    def normalized_accuracy(group: dict[str, Any]) -> float:
        value = group.get("normalized_exact_match_accuracy")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("target-SFT dev normalized exact match is missing")
        return float(value)

    return {
        "dev_answer_only_loss": float(dev_answer_only_loss),
        "dev_overall_normalized_exact_match": normalized_accuracy(metrics["overall"]),
        **{
            f"dev_{hop}_normalized_exact_match": normalized_accuracy(
                metrics["by_hop"][hop]
            )
            for hop in HOP_NAMES
        },
        "dev_H0_attribute_normalized_exact_match": normalized_accuracy(
            metrics["h0_by_fact_type"]["attribute"]
        ),
        "dev_H0_relation_normalized_exact_match": normalized_accuracy(
            metrics["h0_by_fact_type"]["relation"]
        ),
    }


def target_sft_epoch_is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None
) -> bool:
    """Compare epochs by dev EM, then dev loss, then earlier epoch."""

    def selection_key(record: dict[str, Any]) -> tuple[float, float, int]:
        epoch = record.get("epoch")
        em = record.get("dev_overall_normalized_exact_match")
        loss = record.get("dev_answer_only_loss")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("target-SFT selection epoch must be positive")
        if isinstance(em, bool) or not isinstance(em, (int, float)):
            raise ValueError("target-SFT selection EM must be numeric")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError("target-SFT selection loss must be numeric")
        if not math.isfinite(float(em)) or not 0.0 <= float(em) <= 1.0:
            raise ValueError("target-SFT selection EM must be in [0, 1]")
        if not math.isfinite(float(loss)) or float(loss) < 0:
            raise ValueError("target-SFT selection loss must be non-negative")
        return (-float(em), float(loss), epoch)

    candidate_key = selection_key(candidate)
    if incumbent is None:
        return True
    return candidate_key < selection_key(incumbent)


def update_target_sft_selection(
    candidate: dict[str, Any],
    *,
    incumbent: dict[str, Any] | None,
    completed_epochs_without_improvement: int,
    patience: int,
) -> tuple[dict[str, Any], int, bool, bool]:
    """Update best-epoch state and report whether patience is exhausted."""
    if (
        isinstance(completed_epochs_without_improvement, bool)
        or not isinstance(completed_epochs_without_improvement, int)
        or completed_epochs_without_improvement < 0
    ):
        raise ValueError("completed epochs without improvement must be non-negative")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("early-stopping patience must be positive")
    improved = target_sft_epoch_is_better(candidate, incumbent)
    if improved:
        return candidate, 0, True, False
    if incumbent is None:
        raise RuntimeError("target-SFT selection has no incumbent")
    completed_epochs_without_improvement += 1
    return (
        incumbent,
        completed_epochs_without_improvement,
        False,
        completed_epochs_without_improvement >= patience,
    )


def evaluate_target_sft_dev(
    *,
    model: Any,
    tokenizer: Any,
    dev_records: list[dict[str, Any]],
    dev_loader: Any,
    torch_module: Any,
    device: Any,
    generation_batch_size: int,
    generation_context_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure answer-only loss and greedy closed-book EM on target_sft/dev."""
    model.eval()
    loss_sum = 0.0
    loss_tokens = 0
    with torch_module.inference_mode():
        for batch in dev_loader:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                output = model(**batch)
            contributing_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if contributing_tokens <= 0:
                raise RuntimeError("target-SFT dev batch has no supervised LM tokens")
            loss_sum += float(output.loss.detach().item()) * contributing_tokens
            loss_tokens += contributing_tokens
    if loss_tokens <= 0:
        raise RuntimeError("target-SFT dev evaluation has no supervised LM tokens")
    previous_padding_side = tokenizer.padding_side
    try:
        prediction_records = generate_prediction_records(
            dev_records,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            batch_size=generation_batch_size,
            context_length=generation_context_length,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
        model.train()
    return summarize_target_sft_dev_metrics(
        prediction_records,
        dev_answer_only_loss=loss_sum / loss_tokens,
    )


def _save_best_target_sft_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    output_checkpoint: Path,
    previous_use_cache: bool,
) -> None:
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.mkdir(parents=True, exist_ok=True)
    training_use_cache = model.config.use_cache
    try:
        model.config.use_cache = previous_use_cache
        model.save_pretrained(output_checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(output_checkpoint)
    finally:
        model.config.use_cache = training_use_cache


def run_target_sft_training(
    config: dict[str, Any],
    *,
    table_count: int,
    fact_count: int,
    layers: int | None = None,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    run_config_path: str | Path,
    train_log_path: str | Path,
    qa_condition_dir: str | Path,
) -> dict[str, Any]:
    """Run full-parameter closed-book SFT without loading any held-out QA."""
    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    run_config_path = Path(run_config_path)
    train_log_path = Path(train_log_path)
    if (
        not source_checkpoint.is_dir()
        or not (source_checkpoint / "config.json").is_file()
    ):
        raise FileNotFoundError(f"source checkpoint is missing: {source_checkpoint}")
    requested_layers = configured_model_layers(config) if layers is None else layers
    layer_provenance = verify_checkpoint_layers(source_checkpoint, requested_layers)
    ensure_target_sft_outputs_available(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        run_config_path=run_config_path,
        train_log_path=train_log_path,
    )
    settings = config.get("target_sft", {})
    is_exp2 = config["experiment"]["name"] == "exp02_capacity_boundary"
    configured_reference_dir = None if is_exp2 else qa_reference_dir(config)
    if not is_exp2 and Path(qa_condition_dir).resolve() != configured_reference_dir.resolve():
        raise ValueError("target SFT must use the configured immutable data.qa_reference directory")
    reference_table_count, reference_fact_count = (
        (table_count, fact_count) if is_exp2 else qa_reference_values(config)
    )
    train_records, dev_records, provenance = load_target_sft_dataset(
        qa_condition_dir,
        dataset_dir=settings.get("dataset_dir"),
        training_split=settings.get("training_split"),
        dev_split=settings.get("dev_split"),
        table_count=reference_table_count,
        fact_count=reference_fact_count,
    )
    if is_exp2:
        compatibility = {
            "current_database_condition": {
                "T": table_count, "N": fact_count,
                "selected_tables": provenance.get("selected_tables"),
                "source_database_sha256": provenance["source_database_sha256"],
                "source_database_manifest_sha256": provenance["source_database_manifest_sha256"],
            },
            "qa_reference": {"path": str(Path(qa_condition_dir).resolve())},
            "semantic_compatibility_fingerprint": hash_json_object({
                "database": provenance["source_database_sha256"],
                "manifest": provenance["source_database_manifest_sha256"],
            }),
        }
        checkpoint_metadata_path = source_checkpoint / "training_metadata.json"
        if not checkpoint_metadata_path.is_file():
            raise FileNotFoundError(
                "Experiment-2 target SFT requires a CPT checkpoint with training_metadata.json"
            )
        checkpoint_metadata = read_json(checkpoint_metadata_path)
        if checkpoint_metadata.get("model") != config["model"]["name"]:
            raise ValueError("source CPT checkpoint model identity is incompatible")
        checkpoint_provenance = checkpoint_metadata.get("provenance", {})
        for field in ("source_database_sha256", "database_manifest_sha256"):
            expected = (
                provenance["source_database_sha256"]
                if field == "source_database_sha256"
                else provenance["source_database_manifest_sha256"]
            )
            if checkpoint_provenance.get(field) != expected:
                raise ValueError(f"source CPT checkpoint provenance mismatch for {field}")
    else:
        compatibility = verify_qa_reference_compatibility(
            config, table_count, fact_count, reference_dir=qa_condition_dir
        )
    plan = build_target_sft_training_plan(
        config,
        table_count=table_count,
        fact_count=fact_count,
        example_count=len(train_records),
        layers=requested_layers,
    )
    started = time.perf_counter()
    model, tokenizer = _load_local_model_and_tokenizer(source_checkpoint)
    train_examples = [
        encode_target_sft_example(
            record, tokenizer, context_length=plan["context_length"]
        )
        for record in train_records
    ]
    dev_examples = [
        encode_target_sft_example(
            record, tokenizer, context_length=plan["context_length"]
        )
        for record in dev_records
    ]
    model_context_limit = getattr(model.config, "max_position_embeddings", None)
    if model_context_limit is None:
        model_context_limit = getattr(model.config, "n_positions", None)
    if (
        isinstance(model_context_limit, int)
        and plan["context_length"] > model_context_limit
    ):
        raise ValueError(
            f"target_sft.context_length={plan['context_length']} exceeds the source "
            f"model limit of {model_context_limit}"
        )

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("target SFT requires torch and transformers") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("target SFT requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("target_sft.precision=bf16 requires BF16 CUDA support")
    device = torch.device("cuda")
    seed = plan["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device)
    model.train()
    previous_use_cache = model.config.use_cache
    model.config.use_cache = False
    configure_gradient_checkpointing(model, plan["gradient_checkpointing"])
    parameter_counts = enable_full_parameter_training(model)
    generator = seeded_dataloader_generator(torch, seed) if plan["shuffle"] else None
    train_loader = DataLoader(
        train_examples,
        batch_size=plan["batch_size"],
        shuffle=plan["shuffle"],
        collate_fn=partial(
            collate_target_sft_examples, pad_token_id=tokenizer.pad_token_id
        ),
        pin_memory=plan["pin_memory"],
        drop_last=False,
        num_workers=plan["dataloader_workers"],
        persistent_workers=plan["dataloader_workers"] > 0,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_examples,
        batch_size=plan["batch_size"],
        shuffle=False,
        collate_fn=partial(
            collate_target_sft_examples, pad_token_id=tokenizer.pad_token_id
        ),
        pin_memory=plan["pin_memory"],
        drop_last=False,
        num_workers=plan["dataloader_workers"],
        persistent_workers=plan["dataloader_workers"] > 0,
    )
    if len(train_loader) != plan["microbatches_per_epoch"]:
        raise RuntimeError("target-SFT DataLoader batch accounting is inconsistent")
    optimizer, fused_used, fused_fallback = build_target_sft_optimizer(
        torch, model.parameters(), plan
    )
    plan["fused_optimizer_actually_used"] = fused_used
    plan["fused_optimizer_fallback_reason"] = fused_fallback
    scheduler = get_scheduler(
        plan["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=plan["warmup_steps"],
        num_training_steps=plan["total_optimizer_steps"],
    )
    run_record = {
        "stage": "target-sft",
        "experiment": config["experiment"]["name"],
        "model": config["model"]["name"],
        "run_timestamp": config.get("_runtime", {}).get("run_timestamp"),
        "T": table_count,
        "N": fact_count,
        "L": requested_layers,
        "current_database_condition": {
            **compatibility["current_database_condition"],
            "layers": requested_layers,
        },
        "qa_reference": compatibility["qa_reference"],
        "semantic_compatibility_fingerprint": compatibility[
            "semantic_compatibility_fingerprint"
        ],
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "source_checkpoint_config_sha256": hash_file(source_checkpoint / "config.json"),
        "checkpoint_layer_verification": layer_provenance,
        "sft_dataset_path": provenance["dataset_path"],
        "qa_training_split": TARGET_SFT_TRAIN_SPLIT,
        "qa_dev_split": TARGET_SFT_DEV_SPLIT,
        "train_example_count": len(train_records),
        "dev_example_count": len(dev_records),
        "total_examples": len(train_records),
        "train_hop_counts": provenance["train"]["hop_counts"],
        "dev_hop_counts": provenance["dev"]["hop_counts"],
        "train_chain_count": provenance["train"]["chain_count"],
        "dev_chain_count": provenance["dev"]["chain_count"],
        "target_sft_split_manifest_sha256": provenance[
            "target_sft_split_manifest_sha256"
        ],
        "train_manifest_sha256": provenance["train_manifest_sha256"],
        "dev_manifest_sha256": provenance["dev_manifest_sha256"],
        "train_input_file_sha256": provenance["train"]["input_file_sha256"],
        "dev_input_file_sha256": provenance["dev"]["input_file_sha256"],
        "prompt_format": PROMPT_TEMPLATE,
        "answer_only_loss": True,
        "eos_supervised": True,
        "full_parameter_training": True,
        "validation_split_used": False,
        "test_split_used": False,
        "checkpoint_selection": (
            "highest target_sft/dev overall normalized exact match; then lower "
            "dev answer-only loss; then earlier epoch"
        ),
        "early_stopping_patience": plan["early_stopping_patience"],
        "context_length": plan["context_length"],
        "batch_size": plan["batch_size"],
        "gradient_accumulation_steps": plan["gradient_accumulation_steps"],
        "effective_batch_size": plan["effective_batch_size"],
        "optimizer": plan["optimizer"],
        "fused_optimizer_requested": plan["fused_optimizer_requested"],
        "fused_optimizer_actually_used": plan["fused_optimizer_actually_used"],
        "learning_rate": plan["learning_rate"],
        "weight_decay": plan["weight_decay"],
        "scheduler": plan["scheduler"],
        "warmup_steps": plan["warmup_steps"],
        "epochs": plan["epochs"],
        "microbatches_per_epoch": plan["microbatches_per_epoch"],
        "optimizer_steps_per_epoch": plan["optimizer_steps_per_epoch"],
        "maximum_optimizer_steps": plan["maximum_optimizer_steps"],
        "total_optimizer_steps": plan["total_optimizer_steps"],
        "precision": plan["precision"],
        "seed": plan["seed"],
        "tokenizer_identity": getattr(
            tokenizer, "name_or_path", tokenizer.__class__.__name__
        ),
        "model_identity": getattr(
            model.config, "_name_or_path", model.__class__.__name__
        ),
        "total_parameters": parameter_counts["total_parameters"],
        "trainable_parameters": parameter_counts["trainable_parameters"],
        "provenance": provenance,
        "training": plan,
    }
    write_yaml(run_config_path, run_record)

    log_records: list[dict[str, Any]] = [{"record_type": "configuration", **run_record}]
    optimizer.zero_grad(set_to_none=True)
    total_weighted_loss = 0.0
    total_loss_tokens = 0
    optimizer_step = 0
    observed_examples = 0
    epoch_records: list[dict[str, Any]] = []
    best_epoch_record: dict[str, Any] | None = None
    completed_epochs_without_improvement = 0
    early_stopped = False
    accumulation_steps = plan["gradient_accumulation_steps"]
    for epoch in range(1, plan["epochs"] + 1):
        epoch_weighted_loss = 0.0
        epoch_loss_tokens = 0
        epoch_examples = 0
        epoch_optimizer_steps = 0
        accumulated_loss_tokens = 0
        for microbatch, batch in enumerate(train_loader, start=1):
            batch_size = int(batch["input_ids"].shape[0])
            epoch_examples += batch_size
            observed_examples += batch_size
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                contributing_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                if contributing_tokens <= 0:
                    raise RuntimeError("target-SFT batch has no supervised LM tokens")
                summed_loss = output.loss * contributing_tokens
            summed_loss.backward()
            detached_loss = float(output.loss.detach().item())
            epoch_weighted_loss += detached_loss * contributing_tokens
            epoch_loss_tokens += contributing_tokens
            total_weighted_loss += detached_loss * contributing_tokens
            total_loss_tokens += contributing_tokens
            accumulated_loss_tokens += contributing_tokens
            should_step = (
                microbatch % accumulation_steps == 0
                or microbatch == plan["microbatches_per_epoch"]
            )
            if not should_step:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_loss_tokens)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), plan["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            epoch_optimizer_steps += 1
            log_records.append(
                {
                    "record_type": "optimizer_step",
                    "epoch": epoch,
                    "step": optimizer_step,
                    "step_in_epoch": epoch_optimizer_steps,
                    "last_microbatch_in_epoch": microbatch,
                    "supervised_shifted_tokens": accumulated_loss_tokens,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            accumulated_loss_tokens = 0
        if accumulated_loss_tokens != 0:
            raise RuntimeError("target-SFT partial accumulation was not stepped")
        if epoch_examples != len(train_records):
            raise RuntimeError(
                "target SFT did not consume every QA example in an epoch"
            )
        if epoch_optimizer_steps != plan["optimizer_steps_per_epoch"]:
            raise RuntimeError(
                "target-SFT per-epoch optimizer-step count is inconsistent"
            )
        epoch_record = {
            "record_type": "epoch",
            "epoch": epoch,
            "train_loss": epoch_weighted_loss / epoch_loss_tokens,
            "train_examples": epoch_examples,
            "train_microbatches": plan["microbatches_per_epoch"],
            "train_optimizer_steps": epoch_optimizer_steps,
            "train_supervised_shifted_tokens": epoch_loss_tokens,
        }
        epoch_record.update(
            evaluate_target_sft_dev(
                model=model,
                tokenizer=tokenizer,
                dev_records=dev_records,
                dev_loader=dev_loader,
                torch_module=torch,
                device=device,
                generation_batch_size=config["evaluation"]["batch_size"],
                generation_context_length=config["evaluation"]["context_length"],
                max_new_tokens=config["evaluation"]["max_new_tokens"],
            )
        )
        (
            best_epoch_record,
            completed_epochs_without_improvement,
            improved,
            should_stop,
        ) = update_target_sft_selection(
            epoch_record,
            incumbent=best_epoch_record,
            completed_epochs_without_improvement=(completed_epochs_without_improvement),
            patience=plan["early_stopping_patience"],
        )
        epoch_record["improved_best_checkpoint"] = improved
        epoch_record["completed_epochs_without_improvement"] = (
            completed_epochs_without_improvement
        )
        if improved:
            _save_best_target_sft_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_checkpoint=output_checkpoint,
                previous_use_cache=previous_use_cache,
            )
        epoch_records.append(epoch_record)
        log_records.append(epoch_record)
        if should_stop:
            early_stopped = True
            break
    completed_epochs = len(epoch_records)
    expected_actual_steps = plan["optimizer_steps_per_epoch"] * completed_epochs
    if optimizer_step != expected_actual_steps:
        raise RuntimeError(
            "target-SFT actual optimizer-step accounting is inconsistent"
        )
    if optimizer_step > plan["maximum_optimizer_steps"]:
        raise RuntimeError("target-SFT exceeded its maximum optimizer-step count")
    if observed_examples != len(train_records) * completed_epochs:
        raise RuntimeError(
            "target SFT did not use every example exactly once per epoch"
        )
    if best_epoch_record is None:
        raise RuntimeError("target-SFT model selection did not select a checkpoint")
    for epoch_record in epoch_records:
        epoch_record["selected_as_best"] = (
            epoch_record["epoch"] == best_epoch_record["epoch"]
        )

    configure_gradient_checkpointing(model, False)
    model.config.use_cache = previous_use_cache
    summary = {
        "record_type": "summary",
        **run_record,
        "actual_optimizer_steps": optimizer_step,
        "optimizer_steps": optimizer_step,
        "maximum_optimizer_steps": plan["maximum_optimizer_steps"],
        "completed_epochs": completed_epochs,
        "per_epoch_metrics": epoch_records,
        "per_epoch_train_loss": [
            {"epoch": record["epoch"], "train_loss": record["train_loss"]}
            for record in epoch_records
        ],
        "per_epoch_dev_loss": [
            {
                "epoch": record["epoch"],
                "dev_answer_only_loss": record["dev_answer_only_loss"],
            }
            for record in epoch_records
        ],
        "training_loss": total_weighted_loss / total_loss_tokens,
        "observed_examples": observed_examples,
        "selected_epoch": best_epoch_record["epoch"],
        "early_stopped": early_stopped,
        "best_dev_normalized_exact_match": best_epoch_record[
            "dev_overall_normalized_exact_match"
        ],
        "best_dev_answer_only_loss": best_epoch_record["dev_answer_only_loss"],
        "completed_epochs_without_improvement": (completed_epochs_without_improvement),
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "validation_split_used": False,
        "test_split_used": False,
    }
    write_json(output_checkpoint / "training_metadata.json", summary)
    log_records.append(summary)
    write_jsonl(train_log_path, log_records)
    return summary

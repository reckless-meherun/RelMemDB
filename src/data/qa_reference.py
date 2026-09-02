"""Authentication and semantic compatibility for the immutable QA benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.qa import load_verified_semantic_chains, normalize_for_leakage
from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, read_jsonl
from utils.paths import database_condition_dir, qa_reference_dir


class QAReferenceCompatibilityError(ValueError):
    """Raised when a current database cannot answer the fixed QA benchmark."""


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _chain_partitions(
    reference_dir: Path,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    root_path = _require_file(
        reference_dir / "split_manifest.json", "QA split manifest"
    )
    sft_path = _require_file(
        reference_dir / "target_sft" / "split_manifest.json",
        "target-SFT split manifest",
    )
    root = read_json(root_path)
    sft = read_json(sft_path)
    if (root.get("T"), root.get("requested_N")) != (12, 10_000):
        raise QAReferenceCompatibilityError("canonical QA manifest is not T12/N10K")
    if root.get("zero_context") is not True or sft.get("zero_context") is not True:
        raise QAReferenceCompatibilityError("canonical QA must be zero-context")
    if sft.get("source_evaluation_split_manifest_sha256") != hash_file(root_path):
        raise QAReferenceCompatibilityError(
            "target-SFT provenance does not authenticate the evaluation split manifest"
        )
    for field in ("source_database_sha256", "source_database_manifest_sha256"):
        if sft.get(field) != root.get(field):
            raise QAReferenceCompatibilityError(
                f"canonical QA manifests disagree on {field}"
            )

    records_by_partition: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev"):
        split_dir = reference_dir / "target_sft" / split
        manifest_path = _require_file(
            split_dir / "manifest.json", f"target-SFT {split} manifest"
        )
        if hash_file(manifest_path) != sft.get(f"{split}_manifest_sha256"):
            raise QAReferenceCompatibilityError(
                f"target-SFT {split} manifest hash mismatch"
            )
        manifest = read_json(manifest_path)
        h0_ids: set[str] = set()
        relational_records: list[dict[str, Any]] = []
        total = 0
        for hop in range(4):
            hop_name = f"H{hop}"
            path = _require_file(split_dir / f"{hop_name}.jsonl", hop_name)
            if hash_file(path) != manifest.get("output_file_hashes", {}).get(path.name):
                raise QAReferenceCompatibilityError(
                    f"target-SFT {split} {hop_name} hash mismatch"
                )
            records = read_jsonl(path)
            expected_count = (
                manifest.get("counts", {}).get(hop_name, {}).get("final_retained_count")
            )
            if len(records) != expected_count:
                raise QAReferenceCompatibilityError(
                    f"target-SFT {split} {hop_name} count mismatch"
                )
            for record in records:
                if record.get("split") != split or record.get("hop") != hop:
                    raise QAReferenceCompatibilityError(
                        f"target-SFT {split} {hop_name} record metadata mismatch"
                    )
            if hop == 0:
                h0_ids = {record.get("id") for record in records}
                if len(h0_ids) != len(records):
                    raise QAReferenceCompatibilityError(
                        f"target-SFT {split} H0 IDs are not unique"
                    )
            else:
                relational_records.extend(records)
            total += len(records)
        if total != manifest.get("final_retained_total"):
            raise QAReferenceCompatibilityError(
                f"target-SFT {split} retained total mismatch"
            )
        if any(
            support_id not in h0_ids
            for record in relational_records
            for support_id in record.get("support_fact_ids", [])
        ):
            raise QAReferenceCompatibilityError(
                f"target-SFT {split} support closure is invalid"
            )
        records_by_partition[split] = [
            *[record for record in read_jsonl(split_dir / "H0.jsonl")],
            *relational_records,
        ]

    for split in ("validation", "test"):
        split_dir = reference_dir / split
        manifest_path = _require_file(split_dir / "manifest.json", f"{split} manifest")
        if hash_file(manifest_path) != root.get(f"{split}_manifest_sha256"):
            raise QAReferenceCompatibilityError(f"{split} manifest hash mismatch")
        manifest = read_json(manifest_path)
        records: list[dict[str, Any]] = []
        h0_ids: set[str] = set()
        for hop in range(4):
            hop_name = f"H{hop}"
            path = _require_file(split_dir / f"{hop_name}.jsonl", hop_name)
            if hash_file(path) != manifest.get("output_file_hashes", {}).get(path.name):
                raise QAReferenceCompatibilityError(f"{split} {hop_name} hash mismatch")
            hop_records = read_jsonl(path)
            if len(hop_records) != manifest.get("counts", {}).get(hop_name, {}).get(
                "final_retained_count"
            ):
                raise QAReferenceCompatibilityError(
                    f"{split} {hop_name} count mismatch"
                )
            if hop == 0:
                h0_ids = {record.get("id") for record in hop_records}
            elif any(
                support_id not in h0_ids
                for record in hop_records
                for support_id in record.get("support_fact_ids", [])
            ):
                raise QAReferenceCompatibilityError(
                    f"{split} {hop_name} support closure is invalid"
                )
            records.extend(hop_records)
        if len(records) != manifest.get("final_retained_total"):
            raise QAReferenceCompatibilityError(f"{split} retained total mismatch")
        records_by_partition[split] = records

    overlap_keys = {
        "qa ID": lambda record: record["id"],
        "normalized question": lambda record: normalize_for_leakage(record["question"]),
        "normalized QA pair": lambda record: (
            normalize_for_leakage(record["question"]),
            normalize_for_leakage(record["gold_answer"]),
        ),
    }
    partition_names = list(records_by_partition)
    for label, key_function in overlap_keys.items():
        values = {
            name: {key_function(record) for record in records}
            for name, records in records_by_partition.items()
        }
        for offset, left in enumerate(partition_names):
            for right in partition_names[offset + 1 :]:
                if values[left] & values[right]:
                    raise QAReferenceCompatibilityError(
                        f"canonical QA cross-partition {label} overlap: {left}/{right}"
                    )

    partitions = {
        "train": sft.get("train_chain_indices"),
        "dev": sft.get("dev_chain_indices"),
        "validation": root.get("validation_chain_indices"),
        "test": root.get("test_chain_indices"),
    }
    expected_counts = {"train": 135, "dev": 15, "validation": 50, "test": 50}
    for name, indices in partitions.items():
        if (
            not isinstance(indices, list)
            or len(indices) != expected_counts[name]
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise QAReferenceCompatibilityError(
                f"canonical QA {name} chain assignment is invalid"
            )
    names = list(partitions)
    for offset, left in enumerate(names):
        for right in names[offset + 1 :]:
            if set(partitions[left]) & set(partitions[right]):
                raise QAReferenceCompatibilityError(
                    f"canonical QA chain partitions overlap: {left}/{right}"
                )
    required = set().union(*(set(indices) for indices in partitions.values()))
    if len(required) != root.get("total_chain_count") or required != set(range(250)):
        raise QAReferenceCompatibilityError(
            "canonical QA partitions do not cover exactly the recorded 250 chains"
        )

    immutable_hashes = sft.get("immutable_evaluation_artifact_hashes_after")
    if not isinstance(immutable_hashes, dict) or immutable_hashes != sft.get(
        "immutable_evaluation_artifact_hashes_before"
    ):
        raise QAReferenceCompatibilityError(
            "canonical QA immutable-artifact audit is inconsistent"
        )
    for relative_name, expected_hash in immutable_hashes.items():
        artifact_path = _require_file(reference_dir / relative_name, relative_name)
        if hash_file(artifact_path) != expected_hash:
            raise QAReferenceCompatibilityError(
                f"canonical QA immutable artifact hash mismatch: {relative_name}"
            )
    for audit_name in (
        "chain_overlap_counts",
        "qa_id_overlap_counts",
        "exact_question_overlap_counts",
        "normalized_question_overlap_counts",
        "normalized_qa_pair_overlap_counts",
    ):
        audit = sft.get(audit_name)
        if not isinstance(audit, dict) or any(value != 0 for value in audit.values()):
            raise QAReferenceCompatibilityError(
                f"canonical QA {audit_name} is missing or nonzero"
            )
    artifact_hashes = dict(immutable_hashes)
    artifact_hashes["target_sft/split_manifest.json"] = hash_file(sft_path)
    for split in ("train", "dev"):
        split_dir = reference_dir / "target_sft" / split
        artifact_hashes[f"target_sft/{split}/manifest.json"] = hash_file(
            split_dir / "manifest.json"
        )
        for hop in range(4):
            relative_name = f"target_sft/{split}/H{hop}.jsonl"
            artifact_hashes[relative_name] = hash_file(reference_dir / relative_name)
    return partitions, {
        "qa_reference_split_manifest_sha256": hash_file(root_path),
        "target_sft_split_manifest_sha256": hash_file(sft_path),
        "source_database_sha256": root["source_database_sha256"],
        "source_database_manifest_sha256": root["source_database_manifest_sha256"],
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }


def required_qa_chain_indices(reference_dir: str | Path) -> list[int]:
    partitions, _ = _chain_partitions(Path(reference_dir))
    return sorted(set().union(*(set(indices) for indices in partitions.values())))


def _semantic_chain_payload(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "chain_index": chain["chain_index"],
        "entities": [
            {
                "position": entity["position"],
                "entity_type": entity["entity_type"],
                "entity_id": entity["entity_id"],
                "attributes": entity["attributes"],
                "relation_name": entity["relation_name"],
                "relation_target_id": entity["relation_target_id"],
                "natural_anchor": entity["natural_anchor"],
            }
            for entity in chain["entities"]
        ],
    }


def verify_qa_reference_compatibility(
    config: dict[str, Any],
    current_table_count: int,
    current_fact_count: int,
    *,
    current_database_path: str | Path | None = None,
    current_database_manifest_path: str | Path | None = None,
    reference_database_path: str | Path | None = None,
    reference_database_manifest_path: str | Path | None = None,
    reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prove fixed-QA semantics are present in the current physical database."""
    reference_dir = Path(reference_dir) if reference_dir else qa_reference_dir(config)
    partitions, qa_provenance = _chain_partitions(reference_dir)
    required_indices = sorted(
        set().union(*(set(indices) for indices in partitions.values()))
    )
    reference_condition = config["data"]["qa_reference"]
    reference_table_count = reference_condition["table_count"]
    reference_fact_count = reference_condition["fact_count"]
    canonical_db_dir = database_condition_dir(
        reference_table_count, reference_fact_count
    )
    current_db_dir = database_condition_dir(current_table_count, current_fact_count)
    reference_database_path = (
        Path(reference_database_path)
        if reference_database_path
        else canonical_db_dir / "database.sqlite"
    )
    reference_database_manifest_path = (
        Path(reference_database_manifest_path)
        if reference_database_manifest_path
        else canonical_db_dir / "manifest.json"
    )
    current_database_path = (
        Path(current_database_path)
        if current_database_path
        else current_db_dir / "database.sqlite"
    )
    current_database_manifest_path = (
        Path(current_database_manifest_path)
        if current_database_manifest_path
        else current_db_dir / "manifest.json"
    )

    canonical_chains, canonical_manifest = load_verified_semantic_chains(
        reference_database_path,
        reference_database_manifest_path,
        expected_table_count=reference_table_count,
        expected_logical_fact_count=reference_fact_count,
    )
    if hash_file(reference_database_path) != qa_provenance["source_database_sha256"]:
        raise QAReferenceCompatibilityError(
            "canonical QA source database hash is inconsistent"
        )
    if (
        hash_file(reference_database_manifest_path)
        != qa_provenance["source_database_manifest_sha256"]
    ):
        raise QAReferenceCompatibilityError(
            "canonical QA source database manifest hash is inconsistent"
        )
    current_chains, current_manifest = load_verified_semantic_chains(
        current_database_path,
        current_database_manifest_path,
        expected_table_count=current_table_count,
        expected_logical_fact_count=current_fact_count,
    )
    if required_indices and required_indices[-1] >= len(current_chains):
        raise QAReferenceCompatibilityError(
            "current database does not contain all chains required by the canonical "
            f"QA benchmark (requires chain {required_indices[-1]}, contains "
            f"indices 0-{len(current_chains) - 1})"
        )

    canonical_payload = [
        _semantic_chain_payload(canonical_chains[index]) for index in required_indices
    ]
    current_payload = [
        _semantic_chain_payload(current_chains[index]) for index in required_indices
    ]
    canonical_semantic_hash = hash_json_object(canonical_payload)
    current_semantic_hash = hash_json_object(current_payload)
    if current_semantic_hash != canonical_semantic_hash:
        mismatch = next(
            index
            for index, (expected, actual) in enumerate(
                zip(canonical_payload, current_payload, strict=True)
            )
            if expected != actual
        )
        raise QAReferenceCompatibilityError(
            "current database semantic content does not match the canonical QA "
            f"benchmark at required chain {required_indices[mismatch]}"
        )
    fingerprint = hash_json_object(
        {
            "format_version": 1,
            "required_chain_indices": required_indices,
            "canonical_semantic_content_sha256": canonical_semantic_hash,
            "qa_reference_split_manifest_sha256": qa_provenance[
                "qa_reference_split_manifest_sha256"
            ],
        }
    )
    return {
        "compatible": True,
        "qa_reference": {
            "table_count": reference_table_count,
            "fact_count": reference_fact_count,
            "directory": str(reference_dir),
            **qa_provenance,
        },
        "current_database_condition": {
            "table_count": current_table_count,
            "fact_count": current_fact_count,
            "database_sha256": hash_file(current_database_path),
            "database_manifest_sha256": hash_file(current_database_manifest_path),
            "chain_count": current_manifest["selected_chain_count"],
        },
        "required_chain_count": len(required_indices),
        "required_chain_indices_sha256": hash_json_object(required_indices),
        "semantic_content_sha256": canonical_semantic_hash,
        "semantic_compatibility_fingerprint": fingerprint,
        "physical_database_sha_equality_required": False,
        "canonical_database_logical_content_sha256": canonical_manifest.get(
            "logical_content_sha256"
        ),
        "current_database_logical_content_sha256": current_manifest.get(
            "logical_content_sha256"
        ),
    }

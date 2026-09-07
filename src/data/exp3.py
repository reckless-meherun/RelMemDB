"""Experiment-3 standalone continent data and QA generation."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.serialize import serialize_database_cpt
from data.world import _identifier, _natural_name_candidate
from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, read_jsonl, write_json, write_jsonl

EXP3_NAME = "exp03_continent_inverse"
EXP3_MODE = "standalone_canonical_table"
EXP3_QUESTION_TEMPLATE_VERSION = "exp03_continent_tasks_v1"
EXP3_SFT_SPLIT_METHOD_VERSION = "all_continent_rows_train_v1"
HOP_NAMES = ("H0", "H1", "H2", "H3")

CLIMATE_BANDS = (
    "Equatorial",
    "Tropical Rainforest",
    "Tropical Monsoon",
    "Tropical Savanna",
    "Hot Desert",
    "Cold Desert",
    "Semi-Arid",
    "Mediterranean",
    "Humid Subtropical",
    "Oceanic",
    "Humid Continental",
    "Subarctic",
    "Tundra",
    "Ice Cap",
    "Highland",
    "Temperate",
    "Polar",
)


def validate_exp3_fact_count(fact_count: int) -> int:
    if isinstance(fact_count, bool) or not isinstance(fact_count, int) or fact_count <= 0:
        raise ValueError("Experiment-3 N/--fact-count must be a positive integer")
    row_count, remainder = divmod(fact_count, 2)
    if remainder:
        raise ValueError(
            f"Experiment-3 N={fact_count} is invalid: each continent row contributes "
            "exactly 2 semantic facts, so N must be divisible by 2"
        )
    maximum = len(CLIMATE_BANDS) * 2
    if row_count > maximum:
        raise ValueError(
            f"Experiment-3 N={fact_count} requires {row_count} continent rows, but "
            f"17 climate bands with at most 2 continents each support at most "
            f"{maximum} rows (N={maximum * 2})"
        )
    return row_count


def build_exp3_continent_rows(seed: int, fact_count: int) -> list[dict[str, str]]:
    """Build deterministic canonical continent rows without constructing other tables."""
    row_count = validate_exp3_fact_count(fact_count)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Experiment-3 seed must be a non-negative integer")
    used_ids: set[str] = set()
    used_suffixes: set[str] = set()
    used_names: set[str] = set()
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        climate_band = CLIMATE_BANDS[index // 2]
        continent_id = _identifier(
            seed, index, 0, "CTN", used_ids, used_suffixes
        )
        normalized_band = climate_band.casefold()
        for attempt in range(1_000_000):
            continent_name = _natural_name_candidate(
                "continent_name", seed, index, attempt
            )
            if continent_name in used_names:
                continue
            if normalized_band in continent_name.casefold():
                continue
            used_names.add(continent_name)
            break
        else:
            raise RuntimeError("could not generate a unique readable continent name")
        rows.append(
            {
                "continent_id": continent_id,
                "continent_name": continent_name,
                "climate_band": climate_band,
            }
        )
    validate_exp3_rows(rows, fact_count=fact_count)
    return rows


def validate_exp3_rows(rows: list[dict[str, str]], *, fact_count: int) -> None:
    expected_rows = validate_exp3_fact_count(fact_count)
    if len(rows) != expected_rows:
        raise ValueError("Experiment-3 row count does not equal N / 2")
    ids = [row.get("continent_id") for row in rows]
    names = [row.get("continent_name") for row in rows]
    bands = [row.get("climate_band") for row in rows]
    if any(not isinstance(value, str) or not value for value in (*ids, *names, *bands)):
        raise ValueError("every Experiment-3 row value must be non-empty text")
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment-3 continent_id values must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Experiment-3 continent_name values must be unique")
    if any(band not in CLIMATE_BANDS for band in bands):
        raise ValueError("Experiment-3 row contains an unknown climate band")
    if any(count > 2 for count in Counter(bands).values()):
        raise ValueError("a climate band may be assigned to at most 2 continents")


def _load_rows(database_path: Path) -> list[dict[str, str]]:
    with sqlite3.connect(database_path) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        if table_names != ["continent"]:
            raise ValueError("Experiment-3 database must contain only continent")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(continent)")]
        if columns != ["continent_id", "continent_name", "climate_band"]:
            raise ValueError("Experiment-3 continent schema is not canonical")
        values = connection.execute(
            "SELECT continent_id, continent_name, climate_band "
            "FROM continent ORDER BY rowid"
        ).fetchall()
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Experiment-3 SQLite integrity check failed")
    return [
        {
            "continent_id": row[0],
            "continent_name": row[1],
            "climate_band": row[2],
        }
        for row in values
    ]


def materialize_exp3_dataset(
    config: dict[str, Any], output_dir: str | Path, *, fact_count: int
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Experiment-3 dataset: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = build_exp3_continent_rows(config["experiment"]["seed"], fact_count)
    database_path = output_dir / "database.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE continent ("
            "continent_id TEXT NOT NULL PRIMARY KEY, "
            "continent_name TEXT NOT NULL, "
            "climate_band TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO continent "
            "(continent_id, continent_name, climate_band) VALUES (?, ?, ?)",
            [
                (row["continent_id"], row["continent_name"], row["climate_band"])
                for row in rows
            ],
        )
    row_count = len(rows)
    manifest = {
        "format_version": 3,
        "experiment_name": EXP3_NAME,
        "experiment_mode": EXP3_MODE,
        "T": 1,
        "N": fact_count,
        "table_count": 1,
        "requested_N": fact_count,
        "actual_logical_fact_count": fact_count,
        "selected_chain_count": row_count,
        "selected_tables": ["continent"],
        "selected_positions": [0],
        "support_tables": [],
        "position_partition": [[0]],
        "facts_per_selected_chain": 2,
        "experimental_facts_per_chain": 2,
        "attribute_facts_per_chain": 2,
        "relation_facts_per_chain": 0,
        "attribute_fact_count": fact_count,
        "relation_fact_count": 0,
        "identifier_fields_per_chain": 1,
        "identifier_field_count": row_count,
        "identifier_count": row_count,
        "identifiers_counted_as_experimental_facts": False,
        "schema_column_count": 3,
        "schema_foreign_key_count": 0,
        "physical_table_count": 1,
        "physical_row_count": row_count,
        "exposed_row_count": row_count,
        "rows_per_table": {"continent": row_count},
        "exposed_rows_per_table": {"continent": row_count},
        "row_counts": {"continent": row_count},
        "seed": config["experiment"]["seed"],
        "artifact_path": str(output_dir.resolve()),
        "climate_bands": list(CLIMATE_BANDS),
        "max_rows_per_climate_band": 2,
        "logical_content_sha256": hash_json_object(rows),
        "database_sha256": hash_file(database_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    cpt_dir = output_dir / "cpt"
    cpt_manifest = serialize_database_cpt(
        config,
        database_path,
        output_dir / "manifest.json",
        None,
        readable_book_path=cpt_dir / "book_readable.txt",
        expected_table_count=1,
        expected_logical_fact_count=fact_count,
    )
    write_json(cpt_dir / "manifest.json", cpt_manifest)
    return manifest


def verify_exp3_dataset(
    output_dir: str | Path, *, fact_count: int, seed: int
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    database_path = output_dir / "database.sqlite"
    manifest_path = output_dir / "manifest.json"
    cpt_manifest_path = output_dir / "cpt" / "manifest.json"
    for path in (
        database_path,
        manifest_path,
        output_dir / "cpt" / "book_readable.txt",
        cpt_manifest_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Experiment-3 artifact is missing or empty: {path}")
    if (output_dir / "cpt" / "train.txt").exists():
        raise ValueError("Experiment-3 must use independent CPT records, not train.txt")
    manifest = read_json(manifest_path)
    expected = {
        "experiment_name": EXP3_NAME,
        "experiment_mode": EXP3_MODE,
        "T": 1,
        "table_count": 1,
        "N": fact_count,
        "requested_N": fact_count,
        "actual_logical_fact_count": fact_count,
        "selected_tables": ["continent"],
        "selected_positions": [0],
        "position_partition": [[0]],
        "seed": seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Experiment-3 dataset manifest {key} is inconsistent")
    if manifest.get("database_sha256") != hash_file(database_path):
        raise ValueError("Experiment-3 database hash does not match its manifest")
    rows = _load_rows(database_path)
    validate_exp3_rows(rows, fact_count=fact_count)
    if manifest.get("logical_content_sha256") != hash_json_object(rows):
        raise ValueError("Experiment-3 logical content hash is inconsistent")
    cpt_manifest = read_json(cpt_manifest_path)
    if (
        cpt_manifest.get("experiment_name") != EXP3_NAME
        or cpt_manifest.get("requested_N") != fact_count
        or cpt_manifest.get("selected_tables") != ["continent"]
    ):
        raise ValueError("Experiment-3 CPT manifest is inconsistent")
    return {
        "root": output_dir,
        "database": database_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cpt_dir": output_dir / "cpt",
        "cpt_manifest": cpt_manifest_path,
        "rows": rows,
        "T": 1,
        "N": fact_count,
    }


def _qa_record(
    *, row: dict[str, str], split: str
) -> dict[str, Any]:
    identity = [EXP3_NAME, "sft", row["continent_id"]]
    return {
        "id": f"exp3_sft_{hash_json_object(identity)[:32]}",
        "split": split,
        "hop": 0,
        "question": (
            f"What climate band does {row['continent_name']} belong to?"
        ),
        "gold_answer": row["climate_band"],
        "fact_type": "attribute",
        "source_entity_type": "continent",
        "target_entity_type": "continent",
        "target_field": "climate_band",
    }


def build_exp3_sft_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records = [_qa_record(row=row, split="train") for row in rows]
    if len(records) != len(rows):
        raise RuntimeError("Experiment-3 SFT count is inconsistent")
    if len({record["id"] for record in records}) != len(rows):
        raise RuntimeError("Experiment-3 SFT records are not one-per-continent")
    return records


def build_exp3_aggregation_records(
    rows: list[dict[str, str]], *, split: str
) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("Experiment-3 aggregation split must be validation or test")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["climate_band"]].append(row)
    records: list[dict[str, Any]] = []
    for band in CLIMATE_BANDS:
        members = grouped.get(band, [])
        if not members:
            continue
        members.sort(key=lambda row: row["continent_id"])
        names = [row["continent_name"] for row in members]
        if not 1 <= len(names) <= 2:
            raise RuntimeError("Experiment-3 aggregation cardinality must be 1 or 2")
        records.append(
            {
                "id": (
                    f"exp3_{split}_aggregate_"
                    f"{hash_json_object([EXP3_NAME, split, band])[:32]}"
                ),
                "split": split,
                "hop": 0,
                "question": f"Which continents belong to {band}?",
                "gold_answer": ", ".join(names),
                "fact_type": "attribute",
                "source_entity_type": "continent",
                "target_entity_type": "continent",
                "target_field": "continent_name",
            }
        )
    if len(records) != len(grouped):
        raise RuntimeError("Experiment-3 must have one aggregation QA per used band")
    return records


def _counts(h0_count: int) -> dict[str, dict[str, int]]:
    return {
        hop: {
            "candidate_count": h0_count if hop == "H0" else 0,
            "leakage_filtered_count": 0,
            "post_leakage_count": h0_count if hop == "H0" else 0,
            "support_closure_filtered_count": 0,
            "final_retained_count": h0_count if hop == "H0" else 0,
        }
        for hop in HOP_NAMES
    }


def _write_qa_split(
    root: Path,
    *,
    split: str,
    records: list[dict[str, Any]],
    base: dict[str, Any],
) -> dict[str, Any]:
    split_dir = root / split
    by_hop = {"H0": records, "H1": [], "H2": [], "H3": []}
    paths = {hop: split_dir / f"{hop}.jsonl" for hop in HOP_NAMES}
    for hop, path in paths.items():
        write_jsonl(path, by_hop[hop])
    manifest = {
        **base,
        "split": split,
        "chain_count": 0,
        "chain_indices": [],
        "counts": _counts(len(records)),
        "candidate_total": len(records),
        "final_retained_total": len(records),
        "excluded_items": [],
        "output_file_hashes": {path.name: hash_file(path) for path in paths.values()},
    }
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def generate_exp3_qa(
    dataset_dir: str | Path, output_dir: str | Path, *, fact_count: int, seed: int
) -> dict[str, Any]:
    dataset = verify_exp3_dataset(dataset_dir, fact_count=fact_count, seed=seed)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Experiment-3 QA: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = dataset["rows"]
    source_database_hash = hash_file(dataset["database"])
    source_manifest_hash = hash_file(dataset["manifest_path"])
    base = {
        "format_version": 2,
        "experiment_name": EXP3_NAME,
        "T": 1,
        "requested_N": fact_count,
        "source_database_sha256": source_database_hash,
        "source_database_manifest_sha256": source_manifest_hash,
        "source_dataset_manifest_sha256": source_manifest_hash,
        "question_template_version": EXP3_QUESTION_TEMPLATE_VERSION,
        "zero_context": True,
        "selected_tables": ["continent"],
        "selected_positions": [0],
        "source_training_data_dir": str(Path(dataset_dir).resolve()),
        "generation_timestamp": None,
    }
    aggregation = {
        split: build_exp3_aggregation_records(rows, split=split)
        for split in ("validation", "test")
    }
    for split in ("validation", "test"):
        _write_qa_split(
            output_dir, split=split, records=aggregation[split], base=base
        )
    row_indices = list(range(len(rows)))
    root_manifest = {
        **base,
        "total_chain_count": len(rows),
        "reserved_chain_count": len(rows),
        "validation_chain_count": 0,
        "test_chain_count": 0,
        "reserved_chain_indices": row_indices,
        "validation_chain_indices": [],
        "test_chain_indices": [],
        "target_qa_training_generated": False,
        "evaluation_operation": "climate_band_to_continent_names",
        "validation_manifest_sha256": hash_file(output_dir / "validation" / "manifest.json"),
        "test_manifest_sha256": hash_file(output_dir / "test" / "manifest.json"),
    }
    write_json(output_dir / "split_manifest.json", root_manifest)

    sft_records = build_exp3_sft_records(rows)
    sft_dir = output_dir / "target_sft"
    train_dir = sft_dir / "train"
    sft_paths = {hop: train_dir / f"{hop}.jsonl" for hop in HOP_NAMES}
    for hop, path in sft_paths.items():
        write_jsonl(path, sft_records if hop == "H0" else [])
    train_manifest = {
        **base,
        "N": fact_count,
        "split": "train",
        "chain_count": len(rows),
        "chain_indices": row_indices,
        "chain_indices_sha256": hash_json_object(row_indices),
        "source_evaluation_split_manifest_sha256": hash_file(
            output_dir / "split_manifest.json"
        ),
        "sft_split_method_version": EXP3_SFT_SPLIT_METHOD_VERSION,
        "counts": _counts(len(sft_records)),
        "retained_counts": {
            hop: len(sft_records) if hop == "H0" else 0 for hop in HOP_NAMES
        },
        "final_retained_total": len(sft_records),
        "output_file_hashes": {
            path.name: hash_file(path) for path in sft_paths.values()
        },
    }
    write_json(train_dir / "manifest.json", train_manifest)
    pairs = {"train__validation": 0, "train__test": 0, "validation__test": 0}
    assignments = {"train": row_indices}
    sft_manifest = {
        **base,
        "N": fact_count,
        "source_evaluation_split_manifest": "../split_manifest.json",
        "source_evaluation_split_manifest_sha256": hash_file(
            output_dir / "split_manifest.json"
        ),
        "sft_split_method_version": EXP3_SFT_SPLIT_METHOD_VERSION,
        "target_qa_training_generated": True,
        "deterministic_generation": True,
        "runtime_llm_used": False,
        "immutable_evaluation_artifacts_unchanged": True,
        "train_chain_count": len(rows),
        "train_chain_indices": row_indices,
        "train_chain_indices_sha256": hash_json_object(row_indices),
        "validation_chain_indices": [],
        "test_chain_indices": [],
        "chain_assignment_hashes": {"train": hash_json_object(row_indices)},
        "target_sft_chain_assignments_sha256": hash_json_object(assignments),
        "train_manifest_sha256": hash_file(train_dir / "manifest.json"),
        "sft_operation": "continent_name_to_climate_band",
        **{
            field: dict(pairs)
            for field in (
                "chain_overlap_counts",
                "qa_id_overlap_counts",
                "question_overlap_counts",
                "exact_question_overlap_counts",
                "normalized_question_overlap_counts",
                "normalized_qa_pair_overlap_counts",
            )
        },
    }
    write_json(sft_dir / "split_manifest.json", sft_manifest)
    return {
        "root": output_dir.resolve(),
        "sft_data_dir": sft_dir.resolve(),
        "root_manifest": root_manifest,
        "sft_manifest": sft_manifest,
    }


def verify_exp3_qa(
    qa_dir: str | Path,
    *,
    dataset_dir: str | Path,
    fact_count: int,
    seed: int,
) -> dict[str, Any]:
    dataset = verify_exp3_dataset(dataset_dir, fact_count=fact_count, seed=seed)
    qa_dir = Path(qa_dir).resolve()
    root_path = qa_dir / "split_manifest.json"
    sft_path = qa_dir / "target_sft" / "split_manifest.json"
    root = read_json(root_path)
    sft = read_json(sft_path)
    for manifest in (root, sft):
        if (
            manifest.get("experiment_name") != EXP3_NAME
            or manifest.get("T") != 1
            or manifest.get("requested_N") != fact_count
            or manifest.get("selected_tables") != ["continent"]
            or manifest.get("source_database_sha256") != hash_file(dataset["database"])
            or manifest.get("source_database_manifest_sha256")
            != hash_file(dataset["manifest_path"])
        ):
            raise ValueError("Experiment-3 QA provenance is inconsistent")
    sft_records = read_jsonl(qa_dir / "target_sft" / "train" / "H0.jsonl")
    expected_sft = build_exp3_sft_records(dataset["rows"])
    if sft_records != expected_sft or len(sft_records) != fact_count // 2:
        raise ValueError("Experiment-3 SFT QA is inconsistent")
    expected_bands = len({row["climate_band"] for row in dataset["rows"]})
    for split in ("validation", "test"):
        records = read_jsonl(qa_dir / split / "H0.jsonl")
        if records != build_exp3_aggregation_records(dataset["rows"], split=split):
            raise ValueError(f"Experiment-3 {split} QA is inconsistent")
        if len(records) != expected_bands:
            raise ValueError(f"Experiment-3 {split} count is inconsistent")
        if any(not 1 <= len(record["gold_answer"].split(", ")) <= 2 for record in records):
            raise ValueError(f"Experiment-3 {split} answer cardinality is invalid")
    return {
        "root": qa_dir,
        "sft_data_dir": qa_dir / "target_sft",
        "root_manifest": root,
        "sft_manifest": sft,
    }

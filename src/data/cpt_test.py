from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from data.serialize import (
    EXP2_CPT_EXAMPLE_METHOD_VERSION,
    SCHEMA_ENTITY_DESCRIPTIONS,
    SCHEMA_RELATION_DESCRIPTIONS,
    build_selected_cpt_records,
)
from experiment import load_exp2_dataset_condition
from utils.hashing import hash_file, hash_json_object, hash_text
from utils.io import read_json, read_jsonl, write_json, write_jsonl

CPT_TEST_FORMAT_VERSION = 1
CPT_TEST_METHOD_VERSION = "canonical_and_heldout_declarative_completion_v1"
CPT_TEST_PROBE_TYPES = ("canonical_completion", "heldout_declarative_completion")
SUPPORTED_EXPERIMENTS = frozenset(
    {"exp02_capacity_boundary", "exp03_continent_inverse"}
)


def _assert_isolated_probe(prompt: str, source_fact: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("CPT-test prompt must be non-empty")
    if "\n" in prompt or "?" in prompt:
        raise ValueError("CPT-test prompt must be one isolated declarative prefix")
    prohibited = (
        "The Academic Database Book",
        " Records",
        *SCHEMA_ENTITY_DESCRIPTIONS,
        *SCHEMA_RELATION_DESCRIPTIONS,
    )
    if any(value in prompt for value in prohibited):
        raise ValueError("CPT-test prompt contains book schema or header context")
    if "\n" in source_fact:
        raise ValueError("CPT-test source fact must contain exactly one record")


def build_exp2_cpt_test_probes(
    cpt_records: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    probes: list[dict[str, Any]] = []
    for source_record in cpt_records:
        source_id = source_record["id"]
        source_fact = source_record["text"]
        for probe_type, prompt_field in (
            ("canonical_completion", "canonical_prompt"),
            ("heldout_declarative_completion", "held_out_prompt"),
        ):
            prompt = source_record[prompt_field]
            gold_answer = source_record["gold_answer"]
            _assert_isolated_probe(prompt, source_fact)
            if probe_type == "canonical_completion":
                if not source_fact.startswith(prompt + gold_answer):
                    raise RuntimeError(
                        "canonical CPT-test prompt is not a source-record prefix"
                    )
            else:
                held_out_statement = f"{prompt}{gold_answer}."
                if held_out_statement == source_fact:
                    raise RuntimeError(
                        "held-out CPT-test template duplicates its training template"
                    )
            identity = {
                "method_version": CPT_TEST_METHOD_VERSION,
                "seed": seed,
                "source_record_id": source_id,
                "probe_type": probe_type,
            }
            probes.append(
                {
                    "id": f"cpt_probe_{hash_json_object(identity)[:32]}",
                    "probe_type": probe_type,
                    "entity": source_record["entity"],
                    "attribute": source_record["attribute"],
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "source_fact": source_fact,
                    "source_fact_sha256": hash_text(source_fact),
                    "source_cpt_record_id": source_id,
                }
            )
    counts = Counter(probe["source_cpt_record_id"] for probe in probes)
    if set(counts) != {record["id"] for record in cpt_records} or any(
        count != len(CPT_TEST_PROBE_TYPES) for count in counts.values()
    ):
        raise RuntimeError("CPT-test probes do not cover every CPT record twice")
    return probes


def _load_cpt_test_condition(training_data_dir: str | Path) -> dict[str, Any]:
    training_data_dir = Path(training_data_dir).resolve()
    manifest_path = training_data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"database manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    experiment_name = manifest.get("experiment_name")
    if experiment_name == "exp02_capacity_boundary":
        return load_exp2_dataset_condition(training_data_dir)
    if experiment_name != "exp03_continent_inverse":
        raise ValueError("CPT-test dataset is not from Experiment 2 or Experiment 3")
    database = training_data_dir / "database.sqlite"
    cpt_dir = training_data_dir / "cpt"
    cpt_manifest = cpt_dir / "manifest.json"
    readable_book = cpt_dir / "book_readable.txt"
    for path in (database, cpt_manifest, readable_book):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CPT-test source artifact is missing: {path}")
    if manifest.get("selected_tables") != ["continent"]:
        raise ValueError("Experiment-3 CPT-test source must select only continent")
    return {
        "bundle": training_data_dir,
        "database": database,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cpt_dir": cpt_dir,
        "cpt_manifest": cpt_manifest,
        "T": manifest.get("T"),
        "N": manifest.get("N"),
        "selected_tables": manifest.get("selected_tables"),
    }


def generate_cpt_test(
    config: dict[str, Any],
    *,
    training_data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    experiment_name = config.get("experiment", {}).get("name")
    if experiment_name not in SUPPORTED_EXPERIMENTS:
        raise ValueError("CPT-test generation supports only Experiment 2 or Experiment 3")
    output_dir = Path(output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite CPT-test output: {output_dir}")

    condition = _load_cpt_test_condition(training_data_dir)
    database_manifest = condition["manifest"]
    if database_manifest.get("experiment_name") != experiment_name:
        raise ValueError("CPT-test config experiment does not match the dataset")
    if database_manifest.get("seed") != config["experiment"]["seed"]:
        raise ValueError("CPT-test config seed does not match the dataset seed")
    cpt_manifest = read_json(condition["cpt_manifest"])
    cpt_records, record_metadata = build_selected_cpt_records(
        condition["database"], database_manifest
    )
    if (
        cpt_manifest.get("cpt_example_method_version")
        != EXP2_CPT_EXAMPLE_METHOD_VERSION
        or cpt_manifest.get("cpt_example_count") != record_metadata["record_count"]
        or cpt_manifest.get("cpt_examples_sha256")
        != record_metadata["records_sha256"]
        or cpt_manifest.get("cpt_example_logical_fact_count")
        != record_metadata["logical_fact_count"]
        or cpt_manifest.get("cpt_example_logical_fact_coverage_sha256")
        != record_metadata["logical_fact_coverage_sha256"]
    ):
        raise ValueError("CPT records do not match the authenticated CPT manifest")

    seed = config["experiment"]["seed"]
    probes = build_exp2_cpt_test_probes(cpt_records, seed=seed)
    probes_path = output_dir / "probes.jsonl"
    write_jsonl(probes_path, probes)
    manifest = {
        "format_version": CPT_TEST_FORMAT_VERSION,
        "experiment_name": experiment_name,
        "method_version": CPT_TEST_METHOD_VERSION,
        "seed": seed,
        "T": condition["T"],
        "N": condition["N"],
        "selected_tables": condition["selected_tables"],
        "source_training_data_dir": str(condition["bundle"]),
        "source_database_sha256": hash_file(condition["database"]),
        "source_database_manifest_sha256": hash_file(condition["manifest_path"]),
        "source_cpt_manifest_sha256": hash_file(condition["cpt_manifest"]),
        "source_readable_book_sha256": hash_file(
            condition["cpt_dir"] / "book_readable.txt"
        ),
        "source_cpt_example_method_version": record_metadata["method_version"],
        "source_cpt_record_count": record_metadata["record_count"],
        "source_cpt_records_sha256": record_metadata["records_sha256"],
        "source_cpt_logical_fact_count": record_metadata["logical_fact_count"],
        "source_cpt_logical_fact_coverage_sha256": record_metadata[
            "logical_fact_coverage_sha256"
        ],
        "probe_count": len(probes),
        "probe_counts": {
            probe_type: sum(probe["probe_type"] == probe_type for probe in probes)
            for probe_type in CPT_TEST_PROBE_TYPES
        },
        "probe_source_record_ids_sha256": hash_json_object(
            [probe["source_cpt_record_id"] for probe in probes]
        ),
        "probes_sha256": hash_file(probes_path),
        "same_underlying_facts_as_cpt": True,
        "book_context_in_prompts": False,
        "natural_language_questions": False,
        "deterministic_generation": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return {"output_dir": output_dir, "manifest": manifest, "probes": probes}


def generate_exp2_cpt_test(
    config: dict[str, Any],
    *,
    training_data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if config.get("experiment", {}).get("name") != "exp02_capacity_boundary":
        raise ValueError("CPT-test generation is only supported for Experiment 2")
    return generate_cpt_test(
        config, training_data_dir=training_data_dir, output_dir=output_dir
    )


def verify_cpt_test(
    *,
    training_data_dir: str | Path,
    cpt_test_dir: str | Path,
    expected_experiment: str | None = None,
) -> dict[str, Any]:
    condition = _load_cpt_test_condition(training_data_dir)
    experiment_name = condition["manifest"].get("experiment_name")
    if expected_experiment is not None and experiment_name != expected_experiment:
        raise ValueError("CPT-test dataset experiment does not match")
    cpt_test_dir = Path(cpt_test_dir)
    manifest_path = cpt_test_dir / "manifest.json"
    probes_path = cpt_test_dir / "probes.jsonl"
    if not manifest_path.is_file() or not probes_path.is_file():
        raise FileNotFoundError("CPT-test manifest or probes are missing")
    manifest = read_json(manifest_path)
    probes = read_jsonl(probes_path)
    cpt_records, record_metadata = build_selected_cpt_records(
        condition["database"], condition["manifest"]
    )
    source_by_id = {record["id"]: record for record in cpt_records}
    cpt_manifest = read_json(condition["cpt_manifest"])
    expected = {
        "format_version": CPT_TEST_FORMAT_VERSION,
        "experiment_name": experiment_name,
        "method_version": CPT_TEST_METHOD_VERSION,
        "seed": condition["manifest"]["seed"],
        "T": condition["T"],
        "N": condition["N"],
        "selected_tables": condition["selected_tables"],
        "source_training_data_dir": str(condition["bundle"]),
        "source_database_sha256": hash_file(condition["database"]),
        "source_database_manifest_sha256": hash_file(condition["manifest_path"]),
        "source_cpt_manifest_sha256": hash_file(condition["cpt_manifest"]),
        "source_readable_book_sha256": hash_file(
            condition["cpt_dir"] / "book_readable.txt"
        ),
        "source_cpt_example_method_version": record_metadata["method_version"],
        "source_cpt_record_count": record_metadata["record_count"],
        "source_cpt_records_sha256": record_metadata["records_sha256"],
        "source_cpt_logical_fact_count": record_metadata["logical_fact_count"],
        "source_cpt_logical_fact_coverage_sha256": record_metadata[
            "logical_fact_coverage_sha256"
        ],
        "probe_count": len(probes),
        "probe_counts": {
            probe_type: sum(
                probe.get("probe_type") == probe_type for probe in probes
            )
            for probe_type in CPT_TEST_PROBE_TYPES
        },
        "probe_source_record_ids_sha256": hash_json_object(
            [probe.get("source_cpt_record_id") for probe in probes]
        ),
        "probes_sha256": hash_file(probes_path),
        "same_underlying_facts_as_cpt": True,
        "book_context_in_prompts": False,
        "natural_language_questions": False,
        "deterministic_generation": True,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"CPT-test manifest {field} is inconsistent")
    if (
        cpt_manifest.get("cpt_example_method_version")
        != record_metadata["method_version"]
        or cpt_manifest.get("cpt_example_count") != record_metadata["record_count"]
        or cpt_manifest.get("cpt_examples_sha256")
        != record_metadata["records_sha256"]
        or cpt_manifest.get("cpt_example_logical_fact_count")
        != record_metadata["logical_fact_count"]
        or cpt_manifest.get("cpt_example_logical_fact_coverage_sha256")
        != record_metadata["logical_fact_coverage_sha256"]
    ):
        raise ValueError("source CPT manifest does not authenticate the CPT records")
    seen: Counter[tuple[str, str]] = Counter()
    for probe in probes:
        source = source_by_id.get(probe.get("source_cpt_record_id"))
        if source is None:
            raise ValueError("CPT-test probe references a record not seen during CPT")
        probe_type = probe.get("probe_type")
        if probe_type not in CPT_TEST_PROBE_TYPES:
            raise ValueError("CPT-test probe type is unsupported")
        prompt_field = (
            "canonical_prompt"
            if probe_type == "canonical_completion"
            else "held_out_prompt"
        )
        identity = {
            "method_version": CPT_TEST_METHOD_VERSION,
            "seed": manifest.get("seed"),
            "source_record_id": source["id"],
            "probe_type": probe_type,
        }
        if (
            probe.get("id") != f"cpt_probe_{hash_json_object(identity)[:32]}"
            or probe.get("entity") != source["entity"]
            or probe.get("attribute") != source["attribute"]
            or probe.get("prompt") != source[prompt_field]
            or probe.get("source_fact") != source["text"]
            or probe.get("source_fact_sha256") != hash_text(source["text"])
            or probe.get("gold_answer") != source["gold_answer"]
        ):
            raise ValueError("CPT-test probe source-fact provenance is inconsistent")
        _assert_isolated_probe(probe["prompt"], probe["source_fact"])
        seen[(source["id"], probe_type)] += 1
    expected_pairs = {
        (source_id, probe_type)
        for source_id in source_by_id
        for probe_type in CPT_TEST_PROBE_TYPES
    }
    if set(seen) != expected_pairs or any(count != 1 for count in seen.values()):
        raise ValueError(
            "CPT-test probes do not cover every CPT record once per probe type"
        )
    return {"manifest": manifest, "probes": probes}


def verify_exp2_cpt_test(
    *, training_data_dir: str | Path, cpt_test_dir: str | Path
) -> dict[str, Any]:
    return verify_cpt_test(
        training_data_dir=training_data_dir,
        cpt_test_dir=cpt_test_dir,
        expected_experiment="exp02_capacity_boundary",
    )

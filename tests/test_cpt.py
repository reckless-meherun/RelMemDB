import argparse
import math
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import scripts.generate_databases as database_script
import scripts.train as train_script
from config import load_config
from data.materialize import build_database_manifest, materialize_database
from data.serialize import (
    SERIALIZATION_FORMAT_VERSION,
    SERIALIZATION_STYLE,
    serialize_database_cpt,
)
from data.world import build_master_world
from training.cpt import (
    CPTArtifactError,
    build_cpt_training_plan,
    chunk_token_ids,
    enable_full_parameter_training,
    tokenize_cpt_corpus,
    verify_cpt_artifacts,
)
from utils.hashing import hash_file
from utils.io import read_json, read_text, write_json
from utils.paths import (
    EXP01_GENERATED_DATABASES_DIR,
    EXP01_RUNS_DIR,
    cpt_database_dir,
    cpt_run_dir,
    database_condition_dir,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


class FakeParameter:
    def __init__(self, size: int, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def requires_grad_(self, value: bool) -> "FakeParameter":
        self.requires_grad = value
        return self

    def numel(self) -> int:
        return self.size


class FakeModel:
    def __init__(self) -> None:
        self._parameters = [FakeParameter(3, False), FakeParameter(7, True)]

    def parameters(self) -> list[FakeParameter]:
        return self._parameters


@pytest.fixture()
def cpt_config() -> dict[str, Any]:
    config = deepcopy(load_config())
    config["data"]["reuse_t8_n10k"] = False
    config["data"]["t_sweep"]["fact_count"] = 200
    config["data"]["n_sweep"]["fact_counts"] = [200, 400]
    config["data"]["optional_n40k"]["fact_count"] = 800
    return config


def _temporary_condition(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    table_count: int = 12,
    fact_count: int = 200,
    serialize: bool = True,
) -> dict[str, Path]:
    condition_dir = tmp_path / "condition"
    condition_dir.mkdir()
    database_path = condition_dir / "database.sqlite"
    database_manifest_path = condition_dir / "manifest.json"
    world = build_master_world(config)
    materialization = materialize_database(
        world, table_count, fact_count, database_path
    )
    database_manifest = build_database_manifest(
        config,
        materialization,
        sweep="test_sweep",
        master_world_sha256="master-world-hash",
        configuration_sha256="configuration-hash",
        database_sha256=hash_file(database_path),
    )
    write_json(database_manifest_path, database_manifest)
    cpt_dir = condition_dir / "cpt"
    readable_book_path = cpt_dir / "book_readable.txt"
    train_text_path = cpt_dir / "train.txt"
    cpt_manifest_path = cpt_dir / "manifest.json"
    if serialize:
        cpt_manifest = serialize_database_cpt(
            config,
            database_path,
            database_manifest_path,
            train_text_path,
            readable_book_path=readable_book_path,
            expected_table_count=table_count,
            expected_logical_fact_count=fact_count,
        )
        write_json(cpt_manifest_path, cpt_manifest)
    return {
        "condition_dir": condition_dir,
        "database": database_path,
        "database_manifest": database_manifest_path,
        "readable_book": readable_book_path,
        "train_text": train_text_path,
        "cpt_manifest": cpt_manifest_path,
    }


def test_readable_book_is_natural_complete_and_repeated_exactly_x4(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config)
    readable_book = read_text(paths["readable_book"])
    train_text = read_text(paths["train_text"])
    manifest = read_json(paths["cpt_manifest"])

    assert readable_book.startswith("The Academic Database Book\n\nA continent")
    assert train_text == readable_book * 4
    assert manifest["format_version"] == SERIALIZATION_FORMAT_VERSION == 3
    assert manifest["serialization_style"] == SERIALIZATION_STYLE
    assert manifest["fact_exposure"] == 4
    assert manifest["readable_book_copy_count_in_train_text"] == 4
    assert manifest["logical_facts_per_exposure"] == 200
    assert manifest["attribute_facts_per_exposure"] == 145
    assert manifest["relation_facts_per_exposure"] == 55
    assert manifest["serialized_logical_fact_occurrences"] == 800
    assert manifest["identifiers_in_readable_book"] == 0
    assert manifest["source_identifier_count"] == 60
    assert manifest["readable_book_sha256"] == hash_file(paths["readable_book"])

    prohibited = (
        "BEGIN_DATABASE",
        "END_DATABASE",
        "SCHEMA",
        "DATA\n",
        "TABLE ",
        "COLUMNS ",
        "PRIMARY_KEY",
        "FOREIGN_KEY",
        "ROW ",
        "CREATE TABLE",
        "SELECT ",
    )
    assert not any(marker in readable_book for marker in prohibited)
    assert not re.search(r"\b[a-z][a-z0-9_]*=", readable_book)
    assert not re.search(
        r"\b(?:CTN|CTR|REG|CTY|CAM|SCH|DEP|SUB|CRS|OFF|ENR|STU)\d{6}\b",
        readable_book,
    )
    with sqlite3.connect(paths["database"]) as connection:
        raw_identifiers = [
            row[0]
            for table, id_field in (
                ("continent", "continent_id"),
                ("country", "country_id"),
                ("region", "region_id"),
                ("city", "city_id"),
                ("campus", "campus_id"),
                ("school", "school_id"),
                ("department", "department_id"),
                ("subject", "subject_id"),
                ("course", "course_id"),
                ("course_offering", "offering_id"),
                ("enrollment", "enrollment_id"),
                ("student", "student_id"),
            )
            for row in connection.execute(f'SELECT "{id_field}" FROM "{table}"')
        ]
    assert all(identifier not in readable_book for identifier in raw_identifiers)
    assert "Eastern Ancient Oak Isles is a continent with a polar climate band." in readable_book
    assert "Golden Glenovia Federation belongs to Eastern Ancient Oak Isles." in readable_book
    assert "Section G03 of Bright Markets and Institutions" in readable_book
    assert "Ravi K. Novak's primary enrollment is the Spring 2026 enrollment" in readable_book


def test_serialization_is_byte_deterministic(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config, serialize=False)
    outputs = []
    for suffix in ("first", "second"):
        book_path = tmp_path / suffix / "book_readable.txt"
        train_path = tmp_path / suffix / "train.txt"
        manifest = serialize_database_cpt(
            cpt_config,
            paths["database"],
            paths["database_manifest"],
            train_path,
            readable_book_path=book_path,
            expected_table_count=12,
            expected_logical_fact_count=200,
        )
        outputs.append((book_path.read_bytes(), train_path.read_bytes(), manifest))
    assert outputs[0] == outputs[1]


def test_physical_t_grouping_is_preserved_without_changing_logical_coverage(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    expected_groups = {
        4: [
            ["continent", "country", "region"],
            ["city", "campus", "school"],
            ["department", "subject", "course"],
            ["course_offering", "enrollment", "student"],
        ],
        8: [
            ["continent", "country"],
            ["region", "city"],
            ["campus", "school"],
            ["department", "subject"],
            ["course"],
            ["course_offering"],
            ["enrollment"],
            ["student"],
        ],
        12: [[spec] for spec in (
            "continent",
            "country",
            "region",
            "city",
            "campus",
            "school",
            "department",
            "subject",
            "course",
            "course_offering",
            "enrollment",
            "student",
        )],
    }
    expected_headings = {
        4: [
            "Continent, Country, and Region Records",
            "City, Campus, and School Records",
            "Department, Subject, and Course Records",
            "Course Offering, Enrollment, and Student Records",
        ],
        8: [
            "Continent and Country Records",
            "Region and City Records",
            "Campus and School Records",
            "Department and Subject Records",
            "Course Records",
            "Course Offering Records",
            "Enrollment Records",
            "Student Records",
        ],
        12: [
            f"{entity_type.replace('_', ' ').title()} Records"
            for entity_type in (
                "continent",
                "country",
                "region",
                "city",
                "campus",
                "school",
                "department",
                "subject",
                "course",
                "course_offering",
                "enrollment",
                "student",
            )
        ],
    }
    coverage_hashes = set()
    readable_books = set()
    for table_count in (4, 8, 12):
        fixture_root = tmp_path / f"t{table_count}"
        fixture_root.mkdir()
        paths = _temporary_condition(
            fixture_root, cpt_config, table_count=table_count
        )
        manifest = read_json(paths["cpt_manifest"])
        readable_book = read_text(paths["readable_book"])
        groups = manifest["physical_record_groups"]
        assert manifest["physical_record_group_count"] == table_count
        assert manifest["record_organization_sentence_count_per_exposure"] == 0
        assert [group["entity_types"] for group in groups] == expected_groups[
            table_count
        ]
        assert "physical record group" not in readable_book.lower()
        assert "This edition presents the database" not in readable_book
        assert [
            line for line in readable_book.splitlines() if line.endswith(" Records")
        ] == expected_headings[table_count]
        coverage_hashes.add(manifest["logical_fact_coverage_sha256"])
        readable_books.add(readable_book)
    assert len(coverage_hashes) == 1
    assert len(readable_books) == 3


def test_t12_n10k_temporary_book_accounts_for_all_canonical_facts(
    tmp_path: Path,
) -> None:
    paths = _temporary_condition(
        tmp_path, load_config(), table_count=12, fact_count=10_000
    )
    manifest = read_json(paths["cpt_manifest"])
    assert manifest["logical_facts_per_exposure"] == 10_000
    assert manifest["attribute_facts_per_exposure"] == 7_250
    assert manifest["relation_facts_per_exposure"] == 2_750
    assert manifest["logical_entities_per_exposure"] == 3_000
    assert manifest["source_identifier_count"] == 3_000
    assert manifest["identifiers_in_readable_book"] == 0
    assert all(
        group["row_count"] == 250
        for group in manifest["physical_record_groups"]
    )
    assert read_text(paths["train_text"]) == read_text(paths["readable_book"]) * 4


def test_serialization_only_mode_never_loads_world_or_rematerializes_database(
    tmp_path: Path, cpt_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config, serialize=False)
    database_sha256 = hash_file(paths["database"])
    monkeypatch.setattr(database_script, "load_config", lambda _: cpt_config)
    monkeypatch.setattr(
        database_script,
        "_parse_args",
        lambda: argparse.Namespace(
            config=tmp_path / "config.yaml",
            rebuild_master_world=False,
            master_world_only=False,
            serialize_cpt_only=True,
            table_count=12,
            fact_count=200,
        ),
    )
    monkeypatch.setattr(
        database_script,
        "_condition_destination",
        lambda *_: ("test_sweep", paths["condition_dir"]),
    )
    monkeypatch.setattr(
        database_script,
        "_load_verified_master_world",
        lambda *_: pytest.fail("serialization-only mode loaded the master world"),
    )
    monkeypatch.setattr(
        database_script,
        "_materialize_condition",
        lambda **_: pytest.fail("serialization-only mode rematerialized the database"),
    )

    database_script.main()

    assert hash_file(paths["database"]) == database_sha256
    assert paths["readable_book"].stat().st_size > 0
    assert paths["train_text"].stat().st_size > 0
    assert paths["cpt_manifest"].stat().st_size > 0


def test_cpt_provenance_verification_and_hash_failures(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config)
    provenance = verify_cpt_artifacts(
        cpt_config,
        table_count=12,
        fact_count=200,
        database_path=paths["database"],
        database_manifest_path=paths["database_manifest"],
        readable_book_path=paths["readable_book"],
        train_text_path=paths["train_text"],
        cpt_manifest_path=paths["cpt_manifest"],
    )
    assert provenance["T"] == 12
    assert provenance["N"] == 200
    assert provenance["fact_exposure"] == 4
    assert provenance["source_database_sha256"] == hash_file(paths["database"])
    assert provenance["readable_book_sha256"] == hash_file(
        paths["readable_book"]
    )

    paths["train_text"].write_text(
        read_text(paths["train_text"]) + "tampered", encoding="utf-8"
    )
    with pytest.raises(CPTArtifactError, match="train-text hash"):
        verify_cpt_artifacts(
            cpt_config,
            table_count=12,
            fact_count=200,
            database_path=paths["database"],
            database_manifest_path=paths["database_manifest"],
            readable_book_path=paths["readable_book"],
            train_text_path=paths["train_text"],
            cpt_manifest_path=paths["cpt_manifest"],
        )

    readable_tamper_root = tmp_path / "readable-tamper"
    readable_tamper_root.mkdir()
    readable_paths = _temporary_condition(readable_tamper_root, cpt_config)
    readable_paths["readable_book"].write_text(
        read_text(readable_paths["readable_book"]) + "tampered",
        encoding="utf-8",
    )
    with pytest.raises(CPTArtifactError, match="readable-book hash"):
        verify_cpt_artifacts(
            cpt_config,
            table_count=12,
            fact_count=200,
            database_path=readable_paths["database"],
            database_manifest_path=readable_paths["database_manifest"],
            readable_book_path=readable_paths["readable_book"],
            train_text_path=readable_paths["train_text"],
            cpt_manifest_path=readable_paths["cpt_manifest"],
        )


def test_inconsistent_t_n_and_empty_artifacts_fail_clearly(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config)
    manifest = read_json(paths["cpt_manifest"])
    manifest["requested_N"] = 201
    write_json(paths["cpt_manifest"], manifest)
    with pytest.raises(CPTArtifactError, match="N metadata"):
        verify_cpt_artifacts(
            cpt_config,
            table_count=12,
            fact_count=200,
            database_path=paths["database"],
            database_manifest_path=paths["database_manifest"],
            readable_book_path=paths["readable_book"],
            train_text_path=paths["train_text"],
            cpt_manifest_path=paths["cpt_manifest"],
        )

    paths["train_text"].write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="CPT train text is missing or empty"):
        verify_cpt_artifacts(
            cpt_config,
            table_count=12,
            fact_count=200,
            database_path=paths["database"],
            database_manifest_path=paths["database_manifest"],
            readable_book_path=paths["readable_book"],
            train_text_path=paths["train_text"],
            cpt_manifest_path=paths["cpt_manifest"],
        )


def test_chunking_uses_every_token_once_and_masks_only_padding() -> None:
    token_ids = list(range(1, 11))
    examples, statistics = chunk_token_ids(
        token_ids, context_length=4, pad_token_id=99
    )
    assert statistics == {
        "total_tokens": 10,
        "supervised_tokens": 10,
        "sequence_count": 3,
        "context_length": 4,
        "final_partial_sequence_size": 2,
        "final_sequence_real_token_count": 2,
        "padding_token_count": 2,
    }
    reconstructed = [
        token
        for example in examples
        for token, attention in zip(
            example["input_ids"], example["attention_mask"], strict=True
        )
        if attention
    ]
    assert reconstructed == token_ids
    for example in examples:
        for input_id, attention, label in zip(
            example["input_ids"],
            example["attention_mask"],
            example["labels"],
            strict=True,
        ):
            if attention:
                assert label == input_id
            else:
                assert label == -100


def test_complete_corpus_tokenization_has_no_global_truncation() -> None:
    text = "semantic database corpus"
    examples, statistics = tokenize_cpt_corpus(
        text, CharacterTokenizer(), context_length=8
    )
    assert statistics["total_tokens"] == len(text)
    assert statistics["supervised_tokens"] == len(text)
    assert statistics["sequence_count"] == math.ceil(len(text) / 8)
    assert statistics["final_partial_sequence_size"] == len(text) % 8
    assert sum(example["real_token_count"] for example in examples) == len(text)


def test_exact_multiple_has_no_partial_chunk() -> None:
    examples, statistics = chunk_token_ids(
        list(range(8)), context_length=4, pad_token_id=99
    )
    assert len(examples) == 2
    assert statistics["final_partial_sequence_size"] == 0
    assert statistics["final_sequence_real_token_count"] == 4
    assert statistics["padding_token_count"] == 0


def test_training_plan_is_exactly_one_pass(cpt_config: dict[str, Any]) -> None:
    plan = build_cpt_training_plan(
        cpt_config, table_count=12, fact_count=10_000, sequence_count=65
    )
    assert plan["epochs"] == plan["passes_over_serialized_corpus"] == 1
    assert plan["fact_exposure"] == 4
    assert plan["context_length"] == 256
    assert plan["batch_size"] == 32
    assert plan["optimizer_steps"] == 3
    assert plan["warmup_steps"] == 1
    assert plan["optimizer"] == "AdamW"
    assert plan["betas"] == [0.9, 0.999]
    assert plan["epsilon"] == 1e-8
    assert plan["weight_decay"] == 0.01
    assert plan["scheduler"] == "cosine"
    assert plan["shuffle"] is False


def test_cpt_enables_every_model_parameter() -> None:
    model = FakeModel()
    counts = enable_full_parameter_training(model)
    assert counts == {"total_parameters": 10, "trainable_parameters": 10}
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_canonical_cpt_and_run_path_resolution() -> None:
    condition = (
        EXP01_GENERATED_DATABASES_DIR / "t_sweep_N10K" / "T12"
    )
    assert database_condition_dir(12, 10_000) == condition
    assert cpt_database_dir(12, 10_000) == condition / "cpt"
    assert cpt_run_dir(12, 10_000) == (
        EXP01_RUNS_DIR / "t_sweep_N10K" / "T12" / "PLACEHOLDER_RUN"
    )
    paths = train_script.build_cpt_paths(
        load_config(), table_count=12, fact_count=10_000
    )
    assert paths["database"] == condition / "database.sqlite"
    assert paths["readable_book"] == condition / "cpt" / "book_readable.txt"
    assert paths["train_text"] == condition / "cpt" / "train.txt"
    assert paths["cpt_manifest"] == condition / "cpt" / "manifest.json"
    assert paths["output_checkpoint"].name == "gpt2_cpt_t12_n10k"


def test_source_checkpoint_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source checkpoint is missing"):
        train_script._resolve_source_checkpoint(tmp_path / "missing")

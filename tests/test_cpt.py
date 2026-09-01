import argparse
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import pytest

import scripts.generate_databases as database_script
import scripts.train as train_script
from config import load_config
from data.materialize import build_database_manifest, materialize_database
from data.serialize import build_database_serialization_block, serialize_database_cpt
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
    tmp_path: Path, config: dict[str, Any], *, serialize: bool = True
) -> dict[str, Path]:
    condition_dir = tmp_path / "condition"
    condition_dir.mkdir()
    database_path = condition_dir / "database.sqlite"
    database_manifest_path = condition_dir / "manifest.json"
    world = build_master_world(config)
    materialization = materialize_database(world, 12, 200, database_path)
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
    train_text_path = cpt_dir / "train.txt"
    cpt_manifest_path = cpt_dir / "manifest.json"
    if serialize:
        cpt_manifest = serialize_database_cpt(
            config,
            database_path,
            database_manifest_path,
            train_text_path,
            expected_table_count=12,
            expected_logical_fact_count=200,
        )
        write_json(cpt_manifest_path, cpt_manifest)
    return {
        "condition_dir": condition_dir,
        "database": database_path,
        "database_manifest": database_manifest_path,
        "train_text": train_text_path,
        "cpt_manifest": cpt_manifest_path,
    }


def test_x4_serialization_is_four_exact_complete_blocks(
    tmp_path: Path, cpt_config: dict[str, Any]
) -> None:
    paths = _temporary_condition(tmp_path, cpt_config)
    block, block_metadata = build_database_serialization_block(paths["database"])
    train_text = read_text(paths["train_text"])
    manifest = read_json(paths["cpt_manifest"])

    assert train_text == block * 4
    assert manifest["fact_exposure"] == manifest["database_block_count"] == 4
    assert manifest["logical_facts_per_exposure"] == 200
    assert manifest["serialized_logical_fact_occurrences"] == 800
    assert manifest["stored_value_occurrences_per_exposure"] == 260
    assert manifest["serialized_stored_value_occurrences"] == 1_040
    assert block_metadata["stored_value_occurrences"] == 260


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
        train_text_path=paths["train_text"],
        cpt_manifest_path=paths["cpt_manifest"],
    )
    assert provenance["T"] == 12
    assert provenance["N"] == 200
    assert provenance["fact_exposure"] == 4
    assert provenance["source_database_sha256"] == hash_file(paths["database"])

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
            train_text_path=paths["train_text"],
            cpt_manifest_path=paths["cpt_manifest"],
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
    assert paths["train_text"] == condition / "cpt" / "train.txt"
    assert paths["cpt_manifest"] == condition / "cpt" / "manifest.json"
    assert paths["output_checkpoint"].name == "gpt2_cpt_t12_n10k"


def test_source_checkpoint_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source checkpoint is missing"):
        train_script._resolve_source_checkpoint(tmp_path / "missing")

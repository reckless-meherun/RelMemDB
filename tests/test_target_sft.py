from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.train as train_script
from config import ConfigError, load_config, validate_config
from evaluation.inference import format_question_prompt
from training.target_sft import (
    build_target_sft_optimizer,
    build_target_sft_training_plan,
    collate_target_sft_examples,
    configure_gradient_checkpointing,
    encode_target_sft_example,
    ensure_target_sft_outputs_available,
    load_target_sft_records,
    seeded_dataloader_generator,
)
from utils.paths import PROJECT_ROOT, qa_condition_dir


class CharacterTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _record(**updates: Any) -> dict[str, Any]:
    record = {
        "id": "h2_example",
        "split": "validation",
        "hop": 2,
        "question": "What climate band does the containing continent have?",
        "gold_answer": "temperate",
        "source_entity_type": "region",
        "target_entity_type": "continent",
        "target_field": "climate_band",
        "support_fact_ids": ["private_h0_a", "private_h0_b", "private_h0_c"],
    }
    record.update(updates)
    return record


def test_target_sft_prompt_and_answer_only_labels_are_exact() -> None:
    tokenizer = CharacterTokenizer()
    record = _record()
    encoded = encode_target_sft_example(record, tokenizer, context_length=256)
    expected_prompt = format_question_prompt(record["question"])
    prompt_ids = tokenizer.encode(expected_prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(record["gold_answer"], add_special_tokens=False)

    assert encoded["prompt"] == (
        "Question:\n"
        "What climate band does the containing continent have?\n\n"
        "Answer:"
    )
    assert "private_h0" not in encoded["prompt"]
    assert "support_fact_ids" not in encoded["prompt"]
    assert encoded["input_ids"] == [*prompt_ids, *answer_ids, tokenizer.eos_token_id]
    assert encoded["labels"] == [
        *([-100] * len(prompt_ids)),
        *answer_ids,
        tokenizer.eos_token_id,
    ]
    assert all(label == -100 for label in encoded["labels"][: len(prompt_ids)])
    assert encoded["labels"][-1] == tokenizer.eos_token_id


def test_target_sft_padding_is_never_supervised() -> None:
    tokenizer = CharacterTokenizer()
    short = encode_target_sft_example(
        _record(id="short", question="Q?", gold_answer="A"),
        tokenizer,
        context_length=128,
    )
    long = encode_target_sft_example(
        _record(id="long", question="A somewhat longer question?", gold_answer="B"),
        tokenizer,
        context_length=128,
    )
    batch = collate_target_sft_examples([short, long], pad_token_id=0)
    short_length = short["sequence_length"]
    assert batch["attention_mask"][0, short_length:].tolist() == [0] * (
        long["sequence_length"] - short_length
    )
    assert batch["labels"][0, short_length:].tolist() == [-100] * (
        long["sequence_length"] - short_length
    )


def test_target_sft_overlength_fails_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_target_sft_example(
            _record(question="This question is too long?", gold_answer="answer"),
            CharacterTokenizer(),
            context_length=8,
        )


def test_canonical_target_sft_plan_has_exact_step_accounting() -> None:
    plan = build_target_sft_training_plan(
        load_config(), table_count=12, fact_count=10_000, example_count=2353
    )
    assert plan["batch_size"] == 32
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["effective_batch_size"] == 32
    assert plan["epochs"] == 10
    assert plan["context_length"] == 128
    assert plan["microbatches_per_epoch"] == 74
    assert plan["optimizer_steps_per_epoch"] == 74
    assert plan["total_optimizer_steps"] == 740
    assert plan["warmup_steps"] == 37
    assert plan["answer_only_loss"] is True
    assert plan["supervise_eos"] is True
    assert plan["test_split_used"] is False


def test_gradient_accumulation_includes_final_partial_group() -> None:
    config = deepcopy(load_config())
    config["target_sft"]["gradient_accumulation_steps"] = 4
    plan = build_target_sft_training_plan(
        config, table_count=12, fact_count=10_000, example_count=2353
    )
    assert plan["microbatches_per_epoch"] == 74
    assert plan["optimizer_steps_per_epoch"] == 19
    assert plan["total_optimizer_steps"] == 190
    assert plan["effective_batch_size"] == 128


def test_target_sft_seeded_shuffle_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    first = torch.randperm(
        40, generator=seeded_dataloader_generator(torch, 2025)
    ).tolist()
    second = torch.randperm(
        40, generator=seeded_dataloader_generator(torch, 2025)
    ).tolist()
    changed = torch.randperm(
        40, generator=seeded_dataloader_generator(torch, 2026)
    ).tolist()
    assert first == second
    assert first != changed


def test_target_sft_fused_optimizer_falls_back_safely() -> None:
    calls: list[dict[str, Any]] = []

    def adamw(parameters: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs.get("fused"):
            raise RuntimeError("fused unavailable")
        return SimpleNamespace(defaults={})

    fake_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=adamw))
    plan = build_target_sft_training_plan(
        load_config(), table_count=12, fact_count=10_000, example_count=2353
    )
    optimizer, fused_used, reason = build_target_sft_optimizer(
        fake_torch, [object()], plan
    )
    assert optimizer.defaults == {}
    assert fused_used is False
    assert reason == "RuntimeError: fused unavailable"
    assert calls[0]["fused"] is True
    assert "fused" not in calls[1]


def test_target_sft_gradient_checkpointing_is_explicit() -> None:
    class FakeModel:
        is_gradient_checkpointing = False

        def gradient_checkpointing_enable(self) -> None:
            self.is_gradient_checkpointing = True

        def gradient_checkpointing_disable(self) -> None:
            self.is_gradient_checkpointing = False

    model = FakeModel()
    configure_gradient_checkpointing(model, True)
    assert model.is_gradient_checkpointing is True
    configure_gradient_checkpointing(model, False)
    assert model.is_gradient_checkpointing is False


def test_target_sft_config_validation_rejects_unsafe_values() -> None:
    for key, value, message in (
        ("training_split", "test", "test is held out"),
        ("answer_only_loss", False, "answer_only_loss must be true"),
        ("supervise_eos", False, "supervise_eos must be true"),
        ("drop_last", True, "drop_last must be false"),
        ("precision", "fp32", "precision must be bf16"),
    ):
        config = deepcopy(load_config())
        config["target_sft"][key] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(config)


def test_canonical_validation_records_and_provenance_are_preserved() -> None:
    records, provenance = load_target_sft_records(
        qa_condition_dir(12, 10_000),
        training_split="validation",
        table_count=12,
        fact_count=10_000,
    )
    assert len(records) == 2353
    assert provenance["hop_counts"] == {
        "H0": 1286,
        "H1": 426,
        "H2": 341,
        "H3": 300,
    }
    assert set(provenance["input_file_sha256"]) == {"H0", "H1", "H2", "H3"}
    assert len(provenance["qa_manifest_sha256"]) == 64
    assert len(provenance["split_manifest_sha256"]) == 64
    assert provenance["zero_context"] is True
    assert provenance["test_split_used"] is False


def test_target_sft_rejects_test_before_resolving_any_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test split is held out and forbidden"):
        load_target_sft_records(
            tmp_path / "does_not_exist",
            training_split="test",
            table_count=12,
            fact_count=10_000,
        )


def test_target_sft_output_paths_are_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    run_config = tmp_path / "runs" / "config.yaml"
    train_log = tmp_path / "runs" / "train.jsonl"
    ensure_target_sft_outputs_available(
        source_checkpoint=source,
        output_checkpoint=output,
        run_config_path=run_config,
        train_log_path=train_log,
    )
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not an empty directory"):
        ensure_target_sft_outputs_available(
            source_checkpoint=source,
            output_checkpoint=output,
            run_config_path=run_config,
            train_log_path=train_log,
        )
    for artifact in (run_config, train_log):
        if output.exists():
            for child in output.iterdir():
                child.unlink()
            output.rmdir()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("occupied", encoding="utf-8")
        with pytest.raises(FileExistsError, match="run artifact already exists"):
            ensure_target_sft_outputs_available(
                source_checkpoint=source,
                output_checkpoint=output,
                run_config_path=run_config,
                train_log_path=train_log,
            )
        artifact.unlink()


def test_target_sft_path_derives_expected_checkpoint_from_source() -> None:
    paths = train_script.build_target_sft_paths(
        load_config(),
        table_count=12,
        fact_count=10_000,
        source_checkpoint=Path("models/trained_models/gpt2_cpt_t12_n10k_e20"),
    )
    assert paths["qa_condition_dir"] == qa_condition_dir(12, 10_000)
    assert paths["output_checkpoint"] == (
        PROJECT_ROOT
        / "models/trained_models/gpt2_cpt_t12_n10k_e20_sft_validation_e10"
    )
    assert "target_sft" in paths["run_config"].parts
    assert paths["run_config"] != paths["train_log"]


def test_train_cli_routes_target_sft_without_calling_cpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    expected_paths = {
        "qa_condition_dir": tmp_path / "qa",
        "run_config": tmp_path / "run.yaml",
        "train_log": tmp_path / "log.jsonl",
        "output_checkpoint": tmp_path / "output",
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        train_script,
        "_parse_args",
        lambda: argparse.Namespace(
            stage="target-sft",
            table_count=12,
            fact_count=10_000,
            source_checkpoint=source,
            config=Path("unused.yaml"),
        ),
    )
    monkeypatch.setattr(train_script, "load_config", lambda _: load_config())
    monkeypatch.setattr(
        train_script,
        "build_target_sft_paths",
        lambda *_, **__: expected_paths,
    )
    monkeypatch.setattr(
        train_script,
        "run_cpt_training",
        lambda *_, **__: pytest.fail("target-SFT routing called CPT"),
    )

    def fake_target(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "total_examples": 2353,
            "optimizer_steps": 740,
            "training_loss": 1.0,
            "output_checkpoint": str(expected_paths["output_checkpoint"]),
        }

    monkeypatch.setattr(train_script, "run_target_sft_training", fake_target)
    train_script.main()
    assert captured["qa_condition_dir"] == expected_paths["qa_condition_dir"]
    assert captured["source_checkpoint"] == source.resolve()


def test_train_cli_keeps_existing_cpt_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    paths = {
        key: tmp_path / key
        for key in (
            "database",
            "database_manifest",
            "readable_book",
            "train_text",
            "cpt_manifest",
            "run_config",
            "train_log",
            "output_checkpoint",
        )
    }
    called = {"cpt": False}
    monkeypatch.setattr(
        train_script,
        "_parse_args",
        lambda: argparse.Namespace(
            stage="cpt",
            table_count=12,
            fact_count=10_000,
            source_checkpoint=source,
            config=Path("unused.yaml"),
        ),
    )
    monkeypatch.setattr(train_script, "load_config", lambda _: load_config())
    monkeypatch.setattr(train_script, "build_cpt_paths", lambda *_, **__: paths)
    monkeypatch.setattr(
        train_script,
        "run_target_sft_training",
        lambda *_, **__: pytest.fail("CPT routing called target SFT"),
    )

    def fake_cpt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called["cpt"] = True
        return {
            "optimizer_steps": 1,
            "training_loss": 1.0,
            "final_checkpoint_path": "checkpoint",
        }

    monkeypatch.setattr(train_script, "run_cpt_training", fake_cpt)
    train_script.main()
    assert called["cpt"] is True

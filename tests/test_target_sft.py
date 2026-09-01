from __future__ import annotations

import argparse
import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.train as train_script
import training.target_sft as target_sft_module
from config import ConfigError, load_config, validate_config
from evaluation.inference import HOP_NAMES, format_question_prompt
from training.target_sft import (
    build_target_sft_optimizer,
    build_target_sft_training_plan,
    collate_target_sft_examples,
    configure_gradient_checkpointing,
    encode_target_sft_example,
    ensure_target_sft_outputs_available,
    evaluate_target_sft_dev,
    load_target_sft_dataset,
    seeded_dataloader_generator,
    summarize_target_sft_dev_metrics,
    target_sft_epoch_is_better,
    update_target_sft_selection,
)
from utils.hashing import hash_file
from utils.io import read_json, write_json
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
        "split": "train",
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
        load_config(), table_count=12, fact_count=10_000, example_count=6378
    )
    assert plan["batch_size"] == 32
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["effective_batch_size"] == 32
    assert plan["epochs"] == 10
    assert plan["context_length"] == 128
    assert plan["microbatches_per_epoch"] == 200
    assert plan["optimizer_steps_per_epoch"] == 200
    assert plan["maximum_optimizer_steps"] == 2000
    assert plan["total_optimizer_steps"] == 2000
    assert plan["warmup_steps"] == 100
    assert plan["early_stopping_patience"] == 3
    assert plan["answer_only_loss"] is True
    assert plan["supervise_eos"] is True
    assert plan["validation_split_used"] is False
    assert plan["test_split_used"] is False


def test_gradient_accumulation_includes_final_partial_group() -> None:
    config = deepcopy(load_config())
    config["target_sft"]["gradient_accumulation_steps"] = 4
    plan = build_target_sft_training_plan(
        config, table_count=12, fact_count=10_000, example_count=6378
    )
    assert plan["microbatches_per_epoch"] == 200
    assert plan["optimizer_steps_per_epoch"] == 50
    assert plan["total_optimizer_steps"] == 500
    assert plan["warmup_steps"] == 25
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
        load_config(), table_count=12, fact_count=10_000, example_count=6378
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
        ("dataset_dir", "validation", "dataset_dir must be target_sft"),
        ("training_split", "validation", "training_split must be train"),
        ("dev_split", "test", "dev_split must be dev"),
        ("early_stopping_patience", 2, "patience must be 3"),
        ("answer_only_loss", False, "answer_only_loss must be true"),
        ("supervise_eos", False, "supervise_eos must be true"),
        ("drop_last", True, "drop_last must be false"),
        ("precision", "fp32", "precision must be bf16"),
    ):
        config = deepcopy(load_config())
        config["target_sft"][key] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(config)


def test_canonical_target_sft_train_and_dev_are_authenticated() -> None:
    train_records, dev_records, provenance = load_target_sft_dataset(
        qa_condition_dir(12, 10_000),
        dataset_dir="target_sft",
        training_split="train",
        dev_split="dev",
        table_count=12,
        fact_count=10_000,
    )
    assert len(train_records) == 6378
    assert len(dev_records) == 709
    assert provenance["train"]["hop_counts"] == {
        "H0": 3485,
        "H1": 1168,
        "H2": 917,
        "H3": 808,
    }
    assert provenance["dev"]["hop_counts"] == {
        "H0": 388,
        "H1": 130,
        "H2": 102,
        "H3": 89,
    }
    assert provenance["train"]["chain_count"] == 135
    assert provenance["dev"]["chain_count"] == 15
    assert set(provenance["train"]["input_file_sha256"]) == set(HOP_NAMES)
    assert set(provenance["dev"]["input_file_sha256"]) == set(HOP_NAMES)
    assert len(provenance["target_sft_split_manifest_sha256"]) == 64
    assert len(provenance["train_manifest_sha256"]) == 64
    assert len(provenance["dev_manifest_sha256"]) == 64
    assert provenance["zero_context"] is True
    assert provenance["validation_split_used"] is False
    assert provenance["test_split_used"] is False


def test_target_sft_rejects_held_out_splits_before_resolving_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="validation and test are forbidden"):
        load_target_sft_dataset(
            tmp_path / "does_not_exist",
            dataset_dir="target_sft",
            training_split="test",
            dev_split="dev",
            table_count=12,
            fact_count=10_000,
        )


def test_target_sft_loader_never_reads_validation_or_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed: list[Path] = []
    original_read_json = target_sft_module.read_json
    original_read_jsonl = target_sft_module.read_jsonl
    original_hash_file = target_sft_module.hash_file

    def tracked_read_json(path: str | Path) -> Any:
        accessed.append(Path(path))
        return original_read_json(path)

    def tracked_read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        accessed.append(Path(path))
        return original_read_jsonl(path)

    def tracked_hash_file(path: str | Path) -> str:
        accessed.append(Path(path))
        return original_hash_file(path)

    monkeypatch.setattr(target_sft_module, "read_json", tracked_read_json)
    monkeypatch.setattr(target_sft_module, "read_jsonl", tracked_read_jsonl)
    monkeypatch.setattr(target_sft_module, "hash_file", tracked_hash_file)
    load_target_sft_dataset(
        qa_condition_dir(12, 10_000),
        dataset_dir="target_sft",
        training_split="train",
        dev_split="dev",
        table_count=12,
        fact_count=10_000,
    )
    assert accessed
    assert not any(
        part in {"validation", "test"} for path in accessed for part in path.parts
    )


def test_target_sft_loader_rejects_manifest_provenance_tampering(
    tmp_path: Path,
) -> None:
    condition_dir = tmp_path / "qa"
    shutil.copytree(qa_condition_dir(12, 10_000) / "target_sft", condition_dir / "target_sft")
    root_path = condition_dir / "target_sft" / "split_manifest.json"
    root = read_json(root_path)
    root["train_manifest_sha256"] = "0" * 64
    write_json(root_path, root)
    with pytest.raises(ValueError, match="train manifest hash"):
        load_target_sft_dataset(
            condition_dir,
            dataset_dir="target_sft",
            training_split="train",
            dev_split="dev",
            table_count=12,
            fact_count=10_000,
        )


def _scored_dev_record(
    record_id: str,
    hop: int,
    *,
    correct: bool,
    fact_type: str | None = None,
    support_fact_ids: list[str] | None = None,
) -> dict[str, Any]:
    record = {
        "id": record_id,
        "split": "dev",
        "hop": hop,
        "question": f"Question {record_id}?",
        "gold_answer": "correct",
        "prediction": "correct" if correct else "wrong",
        "raw_generation": "correct" if correct else "wrong",
        "strict_exact_match": correct,
        "normalized_exact_match": correct,
        "source_entity_type": "source",
        "target_entity_type": "target",
        "target_field": f"field_{hop}",
    }
    if hop == 0:
        record["fact_type"] = fact_type
    else:
        record["support_fact_ids"] = support_fact_ids
    return record


def test_target_sft_dev_metric_summary_tracks_selection_breakdowns() -> None:
    predictions = [
        _scored_dev_record("h0_attr", 0, correct=True, fact_type="attribute"),
        _scored_dev_record("h0_rel", 0, correct=False, fact_type="relation"),
        _scored_dev_record(
            "h1", 1, correct=True, support_fact_ids=["h0_attr", "h0_rel"]
        ),
        _scored_dev_record(
            "h2",
            2,
            correct=False,
            support_fact_ids=["h0_attr", "h0_rel", "h0_attr"],
        ),
        _scored_dev_record(
            "h3",
            3,
            correct=True,
            support_fact_ids=["h0_attr", "h0_rel", "h0_attr", "h0_rel"],
        ),
    ]
    metrics = summarize_target_sft_dev_metrics(
        predictions, dev_answer_only_loss=0.25
    )
    assert metrics == {
        "dev_answer_only_loss": 0.25,
        "dev_overall_normalized_exact_match": 0.6,
        "dev_H0_normalized_exact_match": 0.5,
        "dev_H1_normalized_exact_match": 1.0,
        "dev_H2_normalized_exact_match": 0.0,
        "dev_H3_normalized_exact_match": 1.0,
        "dev_H0_attribute_normalized_exact_match": 1.0,
        "dev_H0_relation_normalized_exact_match": 0.0,
    }


def test_target_sft_dev_evaluation_uses_only_supplied_dev_records_and_greedy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    predictions = [
        _scored_dev_record("h0_attr", 0, correct=True, fact_type="attribute"),
        _scored_dev_record("h0_rel", 0, correct=True, fact_type="relation"),
        _scored_dev_record(
            "h1", 1, correct=True, support_fact_ids=["h0_attr", "h0_rel"]
        ),
        _scored_dev_record(
            "h2",
            2,
            correct=True,
            support_fact_ids=["h0_attr", "h0_rel", "h0_attr"],
        ),
        _scored_dev_record(
            "h3",
            3,
            correct=True,
            support_fact_ids=["h0_attr", "h0_rel", "h0_attr", "h0_rel"],
        ),
    ]
    supplied_dev_records = [{"id": "dev-only", "split": "dev"}]
    captured: dict[str, Any] = {}

    def fake_generate(records: list[dict[str, Any]], **kwargs: Any):
        captured["records"] = records
        captured.update(kwargs)
        kwargs["tokenizer"].padding_side = "left"
        return predictions

    monkeypatch.setattr(
        target_sft_module, "generate_prediction_records", fake_generate
    )

    class FakeModel:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def __call__(self, **_: Any) -> Any:
            return SimpleNamespace(loss=torch.tensor(0.25))

    fake_torch = SimpleNamespace(
        bfloat16=object(),
        inference_mode=lambda: nullcontext(),
        autocast=lambda **_: nullcontext(),
    )
    tokenizer = SimpleNamespace(padding_side="right")
    dev_loader = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 2, 3]]),
        }
    ]
    model = FakeModel()
    metrics = evaluate_target_sft_dev(
        model=model,
        tokenizer=tokenizer,
        dev_records=supplied_dev_records,
        dev_loader=dev_loader,
        torch_module=fake_torch,
        device="cpu",
        generation_batch_size=32,
        generation_context_length=256,
        max_new_tokens=64,
    )
    assert captured["records"] is supplied_dev_records
    assert captured["batch_size"] == 32
    assert captured["context_length"] == 256
    assert captured["max_new_tokens"] == 64
    assert metrics["dev_answer_only_loss"] == 0.25
    assert metrics["dev_overall_normalized_exact_match"] == 1.0
    assert tokenizer.padding_side == "right"
    assert model.training is True


def _epoch(epoch: int, em: float, loss: float) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "dev_overall_normalized_exact_match": em,
        "dev_answer_only_loss": loss,
    }


def test_target_sft_best_epoch_selection_and_tie_breaks() -> None:
    assert target_sft_epoch_is_better(_epoch(1, 0.4, 0.8), None)
    assert target_sft_epoch_is_better(
        _epoch(2, 0.5, 0.9), _epoch(1, 0.4, 0.1)
    )
    assert target_sft_epoch_is_better(
        _epoch(2, 0.5, 0.7), _epoch(1, 0.5, 0.8)
    )
    assert not target_sft_epoch_is_better(
        _epoch(2, 0.5, 0.8), _epoch(1, 0.5, 0.8)
    )
    assert target_sft_epoch_is_better(
        _epoch(1, 0.5, 0.8), _epoch(2, 0.5, 0.8)
    )


def test_target_sft_patience_stops_after_three_completed_non_improving_epochs() -> None:
    best: dict[str, Any] | None = None
    without_improvement = 0
    stopped = False
    for candidate in (
        _epoch(1, 0.5, 0.5),
        _epoch(2, 0.4, 0.4),
        _epoch(3, 0.5, 0.6),
        _epoch(4, 0.5, 0.5),
    ):
        best, without_improvement, improved, stopped = update_target_sft_selection(
            candidate,
            incumbent=best,
            completed_epochs_without_improvement=without_improvement,
            patience=3,
        )
        if candidate["epoch"] == 1:
            assert improved is True
        elif candidate["epoch"] < 4:
            assert stopped is False
    assert best["epoch"] == 1
    assert without_improvement == 3
    assert stopped is True


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


def test_target_sft_path_derives_expected_checkpoint_from_source(
    tmp_path: Path,
) -> None:
    paths = train_script.build_target_sft_paths(
        load_config(),
        table_count=12,
        fact_count=10_000,
        source_checkpoint=Path("models/trained_models/gpt2_cpt_t12_n10k_e20"),
    )
    assert paths["qa_condition_dir"] == qa_condition_dir(12, 10_000)
    assert paths["output_checkpoint"] == (
        PROJECT_ROOT
        / "models/trained_models/gpt2_cpt_t12_n10k_e20_sft_target_e10"
    )
    previous_diagnostic = (
        PROJECT_ROOT
        / "models/trained_models/gpt2_cpt_t12_n10k_e20_sft_validation_e10"
    )
    previous_config_hash = hash_file(previous_diagnostic / "config.json")
    assert paths["output_checkpoint"] != previous_diagnostic
    with pytest.raises(FileExistsError, match="not an empty directory"):
        ensure_target_sft_outputs_available(
            source_checkpoint=PROJECT_ROOT
            / "models/trained_models/gpt2_cpt_t12_n10k_e20",
            output_checkpoint=previous_diagnostic,
            run_config_path=tmp_path / "config.yaml",
            train_log_path=tmp_path / "train.jsonl",
        )
    assert hash_file(previous_diagnostic / "config.json") == previous_config_hash
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
            "train_example_count": 6378,
            "dev_example_count": 709,
            "optimizer_steps": 2000,
            "selected_epoch": 10,
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

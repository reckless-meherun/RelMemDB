from __future__ import annotations

import json
import hashlib
import inspect
import re
from pathlib import Path

import pytest

from training.relational_qa import (
    PREFLIGHT_NAMESPACE,
    SKILL_TRAIN_NAMESPACE,
    SKILL_VALIDATION_NAMESPACE,
    answer_prefix_match,
    build_skill_example,
    classify_candidate_prediction,
    deterministic_json_bytes,
    encode_supervised_example,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    generate_skill_dataset,
    make_prediction_record,
    post_skill_decision,
    select_best_epoch,
    strict_exact_match,
    summarize_predictions,
    train_relational_skill,
    validate_prompt_lengths,
    visible_attribute_values,
    verify_skill_isolation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = PROJECT_ROOT / "models/base_models/gpt2"
TARGET_WORLD_PATH = (
    PROJECT_ROOT
    / "datasets/generated_databases/exp01_first_feasibility/master_world/world.json"
)


@pytest.fixture(scope="module")
def dataset() -> dict[int, list[dict]]:
    return generate_baseline_dataset(2025, 200)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(BASE_MODEL_DIR)


@pytest.fixture(scope="module")
def skill_dataset(gpt2_tokenizer) -> dict[str, dict[int, list[dict]]]:
    return generate_skill_dataset(2025, 5000, 1000, gpt2_tokenizer)


def test_baseline_generation_is_byte_deterministic(dataset) -> None:
    again = generate_baseline_dataset(2025, 200)
    for hop in (1, 2, 3):
        assert deterministic_json_bytes(
            dataset[hop], jsonl=True
        ) == deterministic_json_bytes(again[hop], jsonl=True)
        assert all(
            row["facts"] == repeated["facts"]
            for row, repeated in zip(dataset[hop], again[hop], strict=True)
        )


def test_exact_counts_and_fact_counts(dataset) -> None:
    assert {hop: len(rows) for hop, rows in dataset.items()} == {1: 200, 2: 200, 3: 200}
    assert all(len(row["facts"]) == 7 for rows in dataset.values() for row in rows)


def test_context_is_exact_chain_and_answer_is_correct(dataset) -> None:
    relation = re.compile(
        r"^Previous entity of (pf_ent_[0-9a-f]{12}) "
        r"is (pf_ent_[0-9a-f]{12})\.$"
    )
    attribute = re.compile(
        r"^attribute_0 of entity (pf_ent_[0-9a-f]{12}) "
        r"is (pf_val_[0-9a-f]{12})\.$"
    )
    for hop, rows in dataset.items():
        for row in rows:
            relation_matches = [
                match for fact in row["facts"] if (match := relation.fullmatch(fact))
            ]
            attribute_matches = [
                match for fact in row["facts"] if (match := attribute.fullmatch(fact))
            ]
            assert len(relation_matches) == 3
            assert len(attribute_matches) == 4
            assert len(relation_matches) + len(attribute_matches) == len(row["facts"])

            previous = {match.group(1): match.group(2) for match in relation_matches}
            attributes = {match.group(1): match.group(2) for match in attribute_matches}
            assert len(previous) == 3
            assert len(attributes) == 4
            assert len(set(attributes.values())) == 4

            source_match = re.match(r"^Starting from entity (pf_ent_[0-9a-f]{12}),", row["question"])
            assert source_match
            reached = source_match.group(1)
            assert reached in previous
            for _ in range(hop):
                reached = previous[reached]
            assert row["answer"] == attributes[reached]
            assert sum(value != row["answer"] for value in attributes.values()) == 3
            assert row["hop"] == hop


def test_answer_attribute_is_not_always_last(dataset) -> None:
    answer_positions = []
    for rows in dataset.values():
        for row in rows:
            answer_fact = next(
                fact
                for fact in row["facts"]
                if fact.endswith(f"is {row['answer']}.")
            )
            answer_positions.append(row["facts"].index(answer_fact))
    assert len(set(answer_positions)) > 1
    assert any(position != 6 for position in answer_positions)


def test_namespace_and_ids_are_distinct_unique_and_deterministic(dataset) -> None:
    assert PREFLIGHT_NAMESPACE == "preflight_baseline_v1"
    rows = [row for values in dataset.values() for row in values]
    assert len({row["id"] for row in rows}) == 600
    assert all(row["id"].startswith("pf_") for row in rows)
    serialized = deterministic_json_bytes(rows, jsonl=True).decode()
    assert "entity_" not in serialized.replace("pf_entity_", "")
    assert not any(token in serialized.lower() for token in ("t4", "t8", "t12", "n5k", "n10k", "n20k"))


def test_opaque_identifiers_do_not_collide(dataset) -> None:
    serialized_facts = "\n".join(
        fact
        for rows in dataset.values()
        for row in rows
        for fact in row["facts"]
    )
    identifiers = re.findall(r"\bpf_(?:ent|val)_[0-9a-f]{12}\b", serialized_facts)
    assert len(set(identifiers)) == 600 * 8


def test_prompt_format_is_exact_and_deterministic(dataset) -> None:
    example = dataset[2][0]
    expected = (
        "Facts:\n"
        + "\n".join(example["facts"])
        + "\n\nQuestion:\n"
        + example["question"]
        + "\n\nAnswer:"
    )
    assert format_relational_prompt(example) == expected
    assert format_relational_prompt(example) == format_relational_prompt(example.copy())
    assert format_copy_prompt(example["answer"]) == (
        f"Copy this token exactly: {example['answer']}\nAnswer:"
    )


def test_no_target_or_storage_metadata(dataset) -> None:
    text = json.dumps(dataset).lower()
    forbidden = (
        "select ", "insert ", "create table", "pragma ", "sqlite", "rowid",
        "master_world", "world.json", "datasets/", "cpt",
    )
    assert all(term not in text for term in forbidden)


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [("pf_val_a", "pf_val_a", True), (" pf_val_a", "pf_val_a", False), ("pf_val_a.", "pf_val_a", False)],
)
def test_strict_exact_match(prediction, gold, expected) -> None:
    assert strict_exact_match(prediction, gold) is expected


@pytest.mark.parametrize(
    ("continuation", "expected"),
    [(" pf_val_a", True), ("pf_val_a. explanation", True), ("pf_val_ab", False), ("other pf_val_a", False)],
)
def test_answer_prefix_match(continuation, expected) -> None:
    assert answer_prefix_match(continuation, "pf_val_a") is expected


def test_copy_control_scoring(dataset) -> None:
    example = dataset[1][0]
    record = make_prediction_record(example, "wrong", f" {example['answer']}\nextra")
    assert record["strict_exact_match"] is False
    assert record["copy_prediction"] == example["answer"]
    assert record["copy_strict_exact_match"] is True


def _metric_row(strict: bool = True, prefix: bool = True, copy: bool = True) -> dict:
    return {
        "strict_exact_match": strict,
        "answer_prefix_match": prefix,
        "copy_strict_exact_match": copy,
    }


def test_decision_rule_passes_at_thresholds() -> None:
    rows = {hop: [_metric_row()] * 9 + [_metric_row(False)] for hop in (1, 2, 3)}
    # 29/30 copy passes the 0.95 control threshold while each hop is exactly 0.90.
    rows[1][9] = _metric_row(False, False, False)
    result = summarize_predictions(rows, 0.90, 0.95)
    assert result["decision"] == "skip_skill_training"


@pytest.mark.parametrize("failure", ["H1", "H2", "H3", "copy"])
def test_decision_rule_fails_below_any_threshold(failure) -> None:
    rows = {hop: [_metric_row()] * 10 for hop in (1, 2, 3)}
    if failure == "copy":
        rows[1][0] = _metric_row(True, True, False)
        rows[1][1] = _metric_row(True, True, False)
    else:
        hop = int(failure[1])
        rows[hop][0] = _metric_row(False)
        rows[hop][1] = _metric_row(False)
    assert summarize_predictions(rows, 0.90, 0.95)["decision"] == "needs_skill_training"


def test_skill_dataset_counts_are_fixed_and_balanced(skill_dataset) -> None:
    assert {hop: len(rows) for hop, rows in skill_dataset["train"].items()} == {
        1: 5000, 2: 5000, 3: 5000,
    }
    assert {hop: len(rows) for hop, rows in skill_dataset["validation"].items()} == {
        1: 1000, 2: 1000, 3: 1000,
    }
    assert sum(map(len, skill_dataset["train"].values())) == 15_000
    assert sum(map(len, skill_dataset["validation"].values())) == 3_000


def test_skill_examples_have_complete_world_and_correct_answers(skill_dataset) -> None:
    relation = re.compile(
        r"^Previous entity of (pf_ent_[0-9]{12}) "
        r"is (pf_ent_[0-9]{12})\.$"
    )
    attribute = re.compile(
        r"^attribute_0 of entity (pf_ent_[0-9]{12}) "
        r"is (pf_val_[0-9]{12})\.$"
    )
    for splits in skill_dataset.values():
        for hop, rows in splits.items():
            for row in rows:
                relations = [m for fact in row["facts"] if (m := relation.fullmatch(fact))]
                attributes = [m for fact in row["facts"] if (m := attribute.fullmatch(fact))]
                assert len(row["facts"]) == 7
                assert len(relations) == 3
                assert len(attributes) == 4
                previous = {m.group(1): m.group(2) for m in relations}
                values = {m.group(1): m.group(2) for m in attributes}
                assert len(set(values.values())) == 4
                source = re.match(
                    r"^Starting from entity (pf_ent_[0-9]{12}),",
                    row["question"],
                ).group(1)
                reached = source
                for _ in range(hop):
                    reached = previous[reached]
                assert row["answer"] == values[reached]


def test_skill_generation_and_fact_order_are_deterministic(gpt2_tokenizer) -> None:
    first = build_skill_example(2025, "train", 2, 17, gpt2_tokenizer)
    second = build_skill_example(2025, "train", 2, 17, gpt2_tokenizer)
    assert first == second
    assert first["facts"] == second["facts"]
    assert deterministic_json_bytes(
        generate_skill_dataset(2025, 3, 2, gpt2_tokenizer)
    ) == (
        deterministic_json_bytes(
            generate_skill_dataset(2025, 3, 2, gpt2_tokenizer)
        )
    )


def test_skill_ids_are_independently_sha_derived_not_ordinal_affine(
    gpt2_tokenizer,
) -> None:
    def suffix(*parts: object) -> str:
        for nonce in range(10_000):
            material = "\x1f".join(
                str(part) for part in (*parts, "candidate", nonce)
            ).encode()
            candidate = (
                f"{int(hashlib.sha256(material).hexdigest(), 16) % 10**12:012d}"
            )
            if len(gpt2_tokenizer.encode(candidate, add_special_tokens=False)) <= 5:
                return candidate
        raise AssertionError("test rejection sampling failed")

    example = build_skill_example(2025, "train", 2, 17, gpt2_tokenizer)
    expected_source = (
        "pf_ent_"
        + suffix(2025, SKILL_TRAIN_NAMESPACE, 2, 17, "entity", 0)
    )
    expected_answer = (
        "pf_val_"
        + suffix(2025, SKILL_TRAIN_NAMESPACE, 2, 17, "value", 2)
    )
    assert example["question"].startswith(f"Starting from entity {expected_source},")
    assert example["answer"] == expected_answer
    source = inspect.getsource(build_skill_example)
    assert "_independent_skill_suffix" in source
    assert "ordinal" not in source


def test_skill_namespaces_are_fully_disjoint(skill_dataset, dataset) -> None:
    report = verify_skill_isolation(skill_dataset, dataset, TARGET_WORLD_PATH)
    assert report["train_validation_overlap"] == 0
    assert report["train_baseline_overlap"] == 0
    assert report["validation_baseline_overlap"] == 0
    assert report["skill_target_overlap"] == 0
    serialized = deterministic_json_bytes(
        [row for split in skill_dataset.values() for rows in split.values() for row in rows],
        jsonl=True,
    ).decode()
    assert "pf_ent_" in serialized and "pf_val_" in serialized
    assert "skilltr_" not in serialized and "skillval_" not in serialized
    assert not re.search(r'\b[ev]_[0-9a-f]{32}\b', serialized)


def test_hidden_v2_namespaces_never_appear_in_prompts(skill_dataset) -> None:
    assert SKILL_TRAIN_NAMESPACE == "relational_skill_train_v2"
    assert SKILL_VALIDATION_NAMESPACE == "relational_skill_validation_v2"
    prompts = [
        format_relational_prompt(row)
        for split in skill_dataset.values()
        for rows in split.values()
        for row in rows
    ]
    assert all(SKILL_TRAIN_NAMESPACE not in prompt for prompt in prompts)
    assert all(SKILL_VALIDATION_NAMESPACE not in prompt for prompt in prompts)
    assert all("skilltr_" not in prompt and "skillval_" not in prompt for prompt in prompts)


class _StubTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


def test_answer_only_supervised_labels(gpt2_tokenizer) -> None:
    example = build_skill_example(2025, "train", 1, 0, gpt2_tokenizer)
    tokenizer = _StubTokenizer()
    encoded = encode_supervised_example(example, tokenizer, 2000)
    prompt_length = encoded["prompt_length"]
    answer_ids = tokenizer.encode(example["answer"])
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded["labels"][prompt_length:-1] == answer_ids
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id


def test_all_skill_prompts_fit_actual_gpt2_without_truncation(skill_dataset) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    rows = [
        row
        for split in skill_dataset.values()
        for rows in split.values()
        for row in rows
    ]
    lengths = validate_prompt_lengths(
        [format_relational_prompt(row) for row in rows], tokenizer, 256
    )
    assert max(lengths) <= 256
    assert all(
        len(encode_supervised_example(row, tokenizer, 256)["input_ids"]) <= 256
        for row in rows
    )


def test_model_selection_uses_em_then_loss_then_earliest_epoch() -> None:
    rows = [
        {"epoch": 1, "validation_overall_strict_em": 0.8, "validation_loss": 0.2},
        {"epoch": 2, "validation_overall_strict_em": 0.9, "validation_loss": 0.3},
        {"epoch": 3, "validation_overall_strict_em": 0.9, "validation_loss": 0.2},
        {"epoch": 4, "validation_overall_strict_em": 0.9, "validation_loss": 0.2},
    ]
    assert select_best_epoch(rows)["epoch"] == 3


def test_step6a_baseline_is_not_a_training_or_selection_input() -> None:
    assert tuple(inspect.signature(select_best_epoch).parameters) == ("epoch_records",)
    parameters = inspect.signature(train_relational_skill).parameters
    assert "baseline_dataset" not in parameters
    assert "baseline_rows" not in parameters


def test_candidate_classification_diagnostics(gpt2_tokenizer) -> None:
    example = build_skill_example(2025, "validation", 2, 3, gpt2_tokenizer)
    candidates = visible_attribute_values(example)
    wrong_candidate = next(value for value in candidates if value != example["answer"])
    assert classify_candidate_prediction(example, example["answer"]) == "exact_correct"
    assert classify_candidate_prediction(example, wrong_candidate) == (
        "wrong_visible_candidate"
    )
    assert classify_candidate_prediction(example, "pf_val_999999999999") == (
        "non_candidate_generation"
    )
    records = {
        2: [
            make_prediction_record(example, example["answer"], "copy-failure"),
            make_prediction_record(example, wrong_candidate, "copy-failure"),
            make_prediction_record(example, "not-a-candidate", "copy-failure"),
        ]
    }
    metrics = summarize_predictions(records, 0.90, 0.95)["per_hop"]["H2"]
    assert metrics["exact_correct_rate"] == pytest.approx(1 / 3)
    assert metrics["wrong_visible_candidate_rate"] == pytest.approx(1 / 3)
    assert metrics["non_candidate_generation_rate"] == pytest.approx(1 / 3)


def test_post_skill_decision_ignores_copy_control() -> None:
    assert post_skill_decision({1: 0.90, 2: 0.90, 3: 0.90}, 0.90) == "relational_skill_ready"
    for hop in (1, 2, 3):
        scores = {1: 0.90, 2: 0.90, 3: 0.90}
        scores[hop] = 0.899
        assert post_skill_decision(scores, 0.90) == "stop_and_diagnose"

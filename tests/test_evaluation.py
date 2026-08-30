from __future__ import annotations

import json
import re

import pytest

from training.relational_qa import (
    PREFLIGHT_NAMESPACE,
    answer_prefix_match,
    deterministic_json_bytes,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    make_prediction_record,
    strict_exact_match,
    summarize_predictions,
)


@pytest.fixture(scope="module")
def dataset() -> dict[int, list[dict]]:
    return generate_baseline_dataset(2025, 200)


def test_baseline_generation_is_byte_deterministic(dataset) -> None:
    again = generate_baseline_dataset(2025, 200)
    assert deterministic_json_bytes(dataset[1], jsonl=True) == deterministic_json_bytes(
        again[1], jsonl=True
    )


def test_exact_counts_and_fact_counts(dataset) -> None:
    assert {hop: len(rows) for hop, rows in dataset.items()} == {1: 200, 2: 200, 3: 200}
    assert all(len(row["facts"]) == hop + 1 for hop, rows in dataset.items() for row in rows)


def test_context_is_exact_chain_and_answer_is_correct(dataset) -> None:
    relation = re.compile(
        r"^The entity immediately previous to entity (pf_ent_[0-9a-f]{20}) "
        r"is (pf_ent_[0-9a-f]{20})\.$"
    )
    attribute = re.compile(
        r"^attribute_0 of entity (pf_ent_[0-9a-f]{20}) "
        r"is (pf_val_[0-9a-f]{20})\.$"
    )
    for hop, rows in dataset.items():
        for row in rows:
            links = [relation.fullmatch(fact) for fact in row["facts"][:-1]]
            assert all(links)
            for left, right in zip(links, links[1:]):
                assert left.group(2) == right.group(1)
            final = attribute.fullmatch(row["facts"][-1])
            assert final
            assert links[-1].group(2) == final.group(1)
            assert row["answer"] == final.group(2)
            assert row["question"].startswith(
                f"Starting from entity {links[0].group(1)},"
            )
            assert row["hop"] == hop


def test_namespace_and_ids_are_distinct_unique_and_deterministic(dataset) -> None:
    assert PREFLIGHT_NAMESPACE == "preflight_baseline_v1"
    rows = [row for values in dataset.values() for row in values]
    assert len({row["id"] for row in rows}) == 600
    assert all(row["id"].startswith("pf_") for row in rows)
    serialized = deterministic_json_bytes(rows, jsonl=True).decode()
    assert "entity_" not in serialized.replace("pf_entity_", "")
    assert not any(token in serialized.lower() for token in ("t4", "t8", "t12", "n5k", "n10k", "n20k"))


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
        "master_world", "world.json", "database", "table", "rowid", "select ",
        " from ", "sqlite", "datasets/", "target", "cpt",
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
    else:
        hop = int(failure[1])
        rows[hop][0] = _metric_row(False)
        rows[hop][1] = _metric_row(False)
    assert summarize_predictions(rows, 0.90, 0.95)["decision"] == "needs_skill_training"

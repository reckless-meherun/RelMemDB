from __future__ import annotations

import hashlib
import inspect
import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.inference import (
    PROMPT_TEMPLATE,
    QAArtifactError,
    extract_first_nonempty_line,
    format_question_prompt,
    generate_prediction_records,
    load_verified_qa_split,
    prepare_result_directory,
)
from evaluation.metrics import (
    compute_evaluation_metrics,
    normalize_answer,
)
from evaluation.metrics import (
    normalized_exact_match as closed_book_normalized_exact_match,
)
from evaluation.metrics import (
    score_prediction as score_closed_book_prediction,
)
from evaluation.metrics import (
    strict_exact_match as closed_book_strict_exact_match,
)
from training.relational_qa import (
    PREFLIGHT_NAMESPACE,
    PRIMITIVE_TRAIN_NAMESPACE,
    PRIMITIVE_VALIDATION_NAMESPACE,
    SKILL_TRAIN_NAMESPACE,
    SKILL_VALIDATION_NAMESPACE,
    answer_prefix_match,
    build_skill_example,
    classify_candidate_prediction,
    curriculum_start_checkpoint,
    deterministic_json_bytes,
    encode_supervised_example,
    format_copy_prompt,
    format_relational_prompt,
    generate_baseline_dataset,
    generate_primitive_dataset,
    generate_skill_dataset,
    make_prediction_record,
    post_skill_decision,
    primitive_gate_decision,
    select_best_epoch,
    step6c_decision,
    strict_exact_match,
    summarize_predictions,
    train_primitive_skill,
    train_relational_skill,
    validate_prompt_lengths,
    verify_skill_isolation,
    verify_step6c_isolation,
    visible_attribute_values,
)
from utils.hashing import hash_file
from utils.io import write_json, write_jsonl

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


@pytest.fixture(scope="module")
def primitive_dataset(gpt2_tokenizer) -> dict[str, dict[str, list[dict]]]:
    return generate_primitive_dataset(2025, 5000, 1000, gpt2_tokenizer)


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


def test_primitive_dataset_counts_are_fixed_and_balanced(primitive_dataset) -> None:
    assert {name: len(rows) for name, rows in primitive_dataset["train"].items()} == {
        "relation": 5000,
        "attribute": 5000,
    }
    assert {
        name: len(rows) for name, rows in primitive_dataset["validation"].items()
    } == {"relation": 1000, "attribute": 1000}
    assert sum(map(len, primitive_dataset["train"].values())) == 10_000
    assert sum(map(len, primitive_dataset["validation"].values())) == 2_000


def test_primitive_relation_and_attribute_gold_are_visible_and_correct(
    primitive_dataset,
) -> None:
    relation_fact = re.compile(
        r"^Previous entity of (pf_ent_[0-9]{12}) is (pf_ent_[0-9]{12})\.$"
    )
    attribute_fact = re.compile(
        r"^attribute_0 of entity (pf_ent_[0-9]{12}) is (pf_val_[0-9]{12})\.$"
    )
    relation_question = re.compile(
        r"^Which entity is immediately previous to entity (pf_ent_[0-9]{12})\?$"
    )
    attribute_question = re.compile(
        r"^What is attribute_0 of entity (pf_ent_[0-9]{12})\?$"
    )
    for split in primitive_dataset.values():
        for qa_type, rows in split.items():
            for row in rows:
                previous = {
                    match.group(1): match.group(2)
                    for fact in row["facts"]
                    if (match := relation_fact.fullmatch(fact))
                }
                attributes = {
                    match.group(1): match.group(2)
                    for fact in row["facts"]
                    if (match := attribute_fact.fullmatch(fact))
                }
                assert len(row["facts"]) == 7
                assert len(previous) == 3
                assert len(attributes) == 4
                if qa_type == "relation":
                    source = relation_question.fullmatch(row["question"]).group(1)
                    assert row["answer"] == previous[source]
                else:
                    entity = attribute_question.fullmatch(row["question"]).group(1)
                    assert row["answer"] == attributes[entity]


def test_primitive_generation_namespaces_surface_and_determinism(
    primitive_dataset, gpt2_tokenizer
) -> None:
    assert PRIMITIVE_TRAIN_NAMESPACE == "primitive_visible_lookup_train_v1"
    assert PRIMITIVE_VALIDATION_NAMESPACE == "primitive_visible_lookup_validation_v1"
    first = generate_primitive_dataset(2025, 3, 2, gpt2_tokenizer)
    second = generate_primitive_dataset(2025, 3, 2, gpt2_tokenizer)
    assert deterministic_json_bytes(first) == deterministic_json_bytes(second)
    serialized = deterministic_json_bytes(
        [
            row
            for split in primitive_dataset.values()
            for rows in split.values()
            for row in rows
        ],
        jsonl=True,
    ).decode()
    identifier_sets = [
        set(re.findall(r"\bpf_(?:ent|val)_[0-9]{12}\b", "\n".join(row["facts"])))
        for split in primitive_dataset.values()
        for rows in split.values()
        for row in rows
    ]
    assert all(len(values) == 8 for values in identifier_sets)
    identifiers = set().union(*identifier_sets)
    assert len(identifiers) == (10_000 + 2_000) * 8
    assert PRIMITIVE_TRAIN_NAMESPACE not in serialized
    assert PRIMITIVE_VALIDATION_NAMESPACE not in serialized


def test_step6c_all_six_identifier_groups_are_isolated(
    primitive_dataset, skill_dataset, dataset
) -> None:
    relational_dataset = {
        split: [row for rows in skill_dataset[split].values() for row in rows]
        for split in ("train", "validation")
    }
    report = verify_step6c_isolation(
        primitive_dataset, relational_dataset, dataset, TARGET_WORLD_PATH
    )
    assert report["verified"] is True
    assert report["all_pairwise_overlaps_zero"] is True
    assert all(value == 0 for value in report["pairwise_overlap_counts"].values())
    assert set(report["group_identifier_counts"]) == {
        "primitive_train",
        "primitive_validation",
        "relational_train",
        "relational_validation",
        "step6a_baseline",
        "target_master_world",
    }


def test_primitive_prompts_fit_and_use_answer_only_eos_supervision(
    primitive_dataset, gpt2_tokenizer
) -> None:
    rows = [
        row
        for split in primitive_dataset.values()
        for rows in split.values()
        for row in rows
    ]
    lengths = validate_prompt_lengths(
        [format_relational_prompt(row) for row in rows], gpt2_tokenizer, 256
    )
    assert max(lengths) <= 256
    for row in rows:
        encoded = encode_supervised_example(row, gpt2_tokenizer, 256)
        prompt_length = encoded["prompt_length"]
        assert encoded["labels"][:prompt_length] == [-100] * prompt_length
        assert encoded["labels"][-1] == gpt2_tokenizer.eos_token_id
        assert encoded["input_ids"][-1] == gpt2_tokenizer.eos_token_id


def test_generation_configuration_is_deterministic() -> None:
    source = inspect.getsource(__import__("training.relational_qa", fromlist=["generate_continuations"]).generate_continuations)
    assert "do_sample=False" in source
    assert "num_beams=1" in source


def test_primitive_gate_and_curriculum_checkpoint_rule(tmp_path) -> None:
    checkpoint = tmp_path / "primitive"
    passing = {"relation": 0.90, "attribute": 0.91}
    assert primitive_gate_decision(passing) == "primitive_skill_ready"
    assert curriculum_start_checkpoint(passing, checkpoint) == checkpoint
    for failed in (
        {"relation": 0.899, "attribute": 1.0},
        {"relation": 1.0, "attribute": 0.899},
    ):
        assert primitive_gate_decision(failed) == "primitive_skill_failure"
        assert curriculum_start_checkpoint(failed, checkpoint) is None


def test_step6a_is_never_a_primitive_or_curriculum_training_selection_input() -> None:
    for function in (train_primitive_skill, train_relational_skill):
        parameters = inspect.signature(function).parameters
        assert "baseline_dataset" not in parameters
        assert "baseline_rows" not in parameters
    assert tuple(inspect.signature(select_best_epoch).parameters) == ("epoch_records",)


def test_step6c_three_way_decision_logic() -> None:
    ready_primitives = {"relation": 0.95, "attribute": 0.94}
    ready_relational = {1: 0.90, 2: 0.91, 3: 0.92}
    assert (
        step6c_decision({"relation": 0.89, "attribute": 1.0}, None)
        == "primitive_skill_failure"
    )
    assert (
        step6c_decision(ready_primitives, {1: 0.90, 2: 0.89, 3: 0.92})
        == "composition_skill_failure"
    )
    assert step6c_decision(ready_primitives, ready_relational) == "relational_skill_ready"


class _FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


class _FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]) if values else 0)

    def to(self, device: str):
        del device
        return self

    def clone(self):
        return _FakeTensor([list(row) for row in self.values])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            values = self.values[row][column]
            return _FakeVector(values if isinstance(values, list) else [values])
        return _FakeVector(self.values[key])


class _FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values, dtype=None):
        del dtype
        return _FakeTensor([list(row) for row in values])

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 1
        left, right = tensors
        return _FakeTensor(
            [
                [*left_row, *right_row]
                for left_row, right_row in zip(
                    left.values, right.values, strict=True
                )
            ]
        )

    @staticmethod
    @contextmanager
    def inference_mode():
        yield


class _ClosedBookTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    name_or_path = "stub-tokenizer"

    def __init__(self) -> None:
        self.pad_token_id = None
        self.padding_side = "right"
        self.seen_prompts: list[str] = []
        self.batch_calls: list[dict] = []

    @property
    def pad_token(self):
        return self.eos_token if self.pad_token_id == self.eos_token_id else None

    @pad_token.setter
    def pad_token(self, value) -> None:
        assert value == self.eos_token
        self.pad_token_id = self.eos_token_id

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) + 1 for character in text]

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors=None,
    ):
        assert add_special_tokens is False
        assert truncation is False
        if isinstance(text, str):
            return {"input_ids": self._encode(text)}
        self.seen_prompts.extend(text)
        self.batch_calls.append(
            {
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
                "padding_side": self.padding_side,
            }
        )
        encoded = [self._encode(value) for value in text]
        width = max(map(len, encoded))
        padded = [
            [self.pad_token_id] * (width - len(ids)) + ids for ids in encoded
        ]
        masks = [
            [0] * (width - len(ids)) + [1] * len(ids) for ids in encoded
        ]
        return {
            "input_ids": _FakeTorch.tensor(padded, dtype=_FakeTorch.long),
            "attention_mask": _FakeTorch.tensor(masks, dtype=_FakeTorch.long),
        }

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(chr(value - 1) for value in values if value != 0)


class _ClosedBookModel:
    def __init__(self, continuations: list[str]) -> None:
        self.continuations = list(continuations)
        self.eval_called = False
        self.generate_calls: list[dict] = []
        self.config = SimpleNamespace(_name_or_path="stub-model")

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        batch_continuations = [
            self.continuations.pop(0) for _ in range(batch_size)
        ]
        encoded = [_ClosedBookTokenizer._encode(text) for text in batch_continuations]
        width = max(map(len, encoded))
        continuation_tensor = _FakeTorch.tensor(
            [ids + [0] * (width - len(ids)) for ids in encoded],
            dtype=_FakeTorch.long,
        )
        self.generate_calls.append(
            {"attention_mask": attention_mask.clone(), **kwargs}
        )
        return _FakeTorch.cat((input_ids, continuation_tensor), dim=1)


def _closed_book_record(
    record_id: str,
    hop: int,
    gold_answer: str,
    *,
    fact_type: str | None = None,
    support_fact_ids: list[str] | None = None,
    target_field: str = "climate_band",
) -> dict:
    record = {
        "id": record_id,
        "split": "validation",
        "hop": hop,
        "question": f"Question for {record_id}?",
        "gold_answer": gold_answer,
        "source_entity_type": "country",
        "target_entity_type": "continent",
        "target_field": target_field,
    }
    if hop == 0:
        record["fact_type"] = fact_type or "attribute"
    else:
        record["support_fact_ids"] = support_fact_ids or []
    return record


def test_closed_book_question_only_prompt_is_exact() -> None:
    question = "What climate band does Eastern Oak Isles have?"
    assert format_question_prompt(question) == (
        "Question:\n"
        "What climate band does Eastern Oak Isles have?\n\n"
        "Answer:"
    )
    assert PROMPT_TEMPLATE == "Question:\n{question}\n\nAnswer:"


def test_closed_book_inference_has_no_database_or_context_source() -> None:
    import evaluation.inference as closed_book_inference

    source = inspect.getsource(closed_book_inference)
    for forbidden in (
        "sqlite3",
        "database.sqlite",
        "book_readable",
        "train.txt",
        "master_world",
    ):
        assert forbidden not in source
    loader_source = inspect.getsource(closed_book_inference.load_local_causal_lm)
    assert "AutoTokenizer" in loader_source
    assert "AutoModelForCausalLM" in loader_source
    assert loader_source.count("local_files_only=True") == 2
    assert 'model.to("cuda")' in loader_source
    assert "use_deterministic_algorithms(True)" in loader_source


def test_closed_book_batch_generation_is_question_only_and_continuation_only() -> None:
    records = [
        _closed_book_record("h0_a", 0, "Alpha"),
        _closed_book_record("h0_b", 0, "Beta"),
    ]
    records[0]["context"] = "must never reach the prompt"
    records[0]["support_fact_ids"] = ["private-support"]
    tokenizer = _ClosedBookTokenizer()
    model = _ClosedBookModel(["  Alpha\nexplanation", "Beta"])
    predictions = generate_prediction_records(
        records,
        tokenizer=tokenizer,
        model=model,
        torch_module=_FakeTorch,
        batch_size=2,
        context_length=256,
        device="cpu",
    )
    assert tokenizer.seen_prompts == [
        format_question_prompt(records[0]["question"]),
        format_question_prompt(records[1]["question"]),
    ]
    assert all("Alpha" not in prompt and "private-support" not in prompt for prompt in tokenizer.seen_prompts)
    assert all("must never reach" not in prompt for prompt in tokenizer.seen_prompts)
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.batch_calls == [
        {
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
            "padding_side": "left",
        }
    ]
    assert model.eval_called is True
    assert model.generate_calls[0]["do_sample"] is False
    assert model.generate_calls[0]["num_beams"] == 1
    assert model.generate_calls[0]["max_new_tokens"] == 64
    assert "temperature" not in model.generate_calls[0]
    assert predictions[0]["raw_generation"] == "  Alpha\nexplanation"
    assert predictions[0]["prediction"] == "Alpha"
    assert predictions[0]["strict_exact_match"] is True
    assert predictions[1]["raw_generation"] == "Beta"


def test_closed_book_prompt_overflow_fails_without_truncation_or_generation() -> None:
    tokenizer = _ClosedBookTokenizer()
    model = _ClosedBookModel(["unused"])
    with pytest.raises(ValueError, match="truncation is forbidden"):
        generate_prediction_records(
            [_closed_book_record("too_long", 0, "unused")],
            tokenizer=tokenizer,
            model=model,
            torch_module=_FakeTorch,
            batch_size=1,
            context_length=2,
            device="cpu",
        )
    assert model.generate_calls == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  WARM   Temperate.  ", "warm temperate"),
        ("A－B!", "a-b"),
        ("1-3 million", "1-3 million"),
        ("A, B", "a, b"),
        ("answer...", "answer.."),
    ],
)
def test_closed_book_conservative_normalization(value, expected) -> None:
    assert normalize_answer(value) == expected


def test_closed_book_strict_and_normalized_exact_match() -> None:
    assert closed_book_strict_exact_match(" A- ", "A-") is True
    assert closed_book_strict_exact_match("a-", "A-") is False
    assert closed_book_strict_exact_match("A-.", "A-") is False
    assert closed_book_normalized_exact_match(" WARM  Temperate!", "warm temperate")
    assert not closed_book_normalized_exact_match("1–3 million", "1-3 million")
    assert extract_first_nonempty_line(" \n  Answer A  \nsecond") == "Answer A"


def _scored_record(
    record_id: str,
    hop: int,
    gold: str,
    prediction: str,
    *,
    fact_type: str | None = None,
    supports: list[str] | None = None,
    target_field: str = "climate_band",
) -> dict:
    qa_record = _closed_book_record(
        record_id,
        hop,
        gold,
        fact_type=fact_type,
        support_fact_ids=supports,
        target_field=target_field,
    )
    return score_closed_book_prediction(qa_record, prediction, prediction)


def test_closed_book_metric_aggregation_baselines_and_conditional_accuracy() -> None:
    records = [
        _scored_record("a", 0, "Cold", "cold.", fact_type="attribute"),
        _scored_record(
            "b",
            0,
            "Eastern Isles",
            "Eastern Isles",
            fact_type="relation",
            target_field="continent_name",
        ),
        _scored_record("c", 0, "Warm", "wrong", fact_type="attribute"),
        _scored_record("d", 0, "Cold", "Cold", fact_type="attribute"),
        _scored_record("h1_good", 1, "Cold", "cold", supports=["a", "b"]),
        _scored_record("h1_ineligible", 1, "Warm", "Warm", supports=["c", "b"]),
        _scored_record("h2", 2, "Cold", "Cold", supports=["a", "b", "c"]),
        _scored_record(
            "h3",
            3,
            "Cold",
            "wrong",
            supports=["a", "b", "d", "a"],
        ),
    ]
    metrics = compute_evaluation_metrics(records)
    assert metrics["primary_metric"] == "normalized_exact_match"
    assert metrics["overall"]["count"] == 8
    assert metrics["overall"]["normalized_exact_match_correct"] == 6
    assert metrics["by_hop"]["H0"]["normalized_exact_match_accuracy"] == 0.75
    assert metrics["by_hop"]["H1"]["normalized_exact_match_accuracy"] == 1.0
    assert metrics["h0_by_fact_type"]["attribute"]["count"] == 3
    assert metrics["h0_by_fact_type"]["relation"]["count"] == 1
    assert "climate_band" in metrics["by_target_field"]
    assert metrics["target_field_macro_average"][
        "normalized_exact_match_accuracy"
    ] == pytest.approx((5 / 7 + 1.0) / 2)
    baseline = metrics["majority_gold_answer_baseline"]
    assert baseline["by_target_field"]["climate_band"][
        "majority_gold_answer"
    ] == "Cold"
    assert baseline["by_target_field"]["climate_band"]["accuracy"] == pytest.approx(5 / 7)
    conditional = metrics["conditional_relational_accuracy"]
    assert conditional["H1"] == {
        "total_count": 2,
        "support_correct_count": 1,
        "eligible_count": 1,
        "support_recall_rate": 0.5,
        "conditional_correct_count": 1,
        "conditional_accuracy": 1.0,
    }
    assert conditional["H2"]["eligible_count"] == 0
    assert conditional["H2"]["conditional_accuracy"] is None
    assert conditional["H3"]["eligible_count"] == 1
    assert conditional["H3"]["conditional_accuracy"] == 0.0


def _write_closed_book_qa_fixture(root: Path) -> Path:
    split_dir = root / "validation"
    records = {
        "H0": [
            _closed_book_record(f"h0_{index}", 0, f"answer {index}")
            for index in range(4)
        ],
        "H1": [
            _closed_book_record(
                "h1_0", 1, "answer 0", support_fact_ids=["h0_0", "h0_1"]
            )
        ],
        "H2": [
            _closed_book_record(
                "h2_0",
                2,
                "answer 0",
                support_fact_ids=["h0_0", "h0_1", "h0_2"],
            )
        ],
        "H3": [
            _closed_book_record(
                "h3_0",
                3,
                "answer 0",
                support_fact_ids=["h0_0", "h0_1", "h0_2", "h0_3"],
            )
        ],
    }
    paths = {}
    for hop_name, hop_records in records.items():
        path = split_dir / f"{hop_name}.jsonl"
        write_jsonl(path, hop_records)
        paths[hop_name] = path
    manifest = {
        "T": 12,
        "requested_N": 10_000,
        "split": "validation",
        "zero_context": True,
        "counts": {
            hop_name: {"final_retained_count": len(hop_records)}
            for hop_name, hop_records in records.items()
        },
        "final_retained_total": sum(map(len, records.values())),
        "output_file_hashes": {
            path.name: hash_file(path) for path in paths.values()
        },
    }
    manifest_path = split_dir / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        root / "split_manifest.json",
        {
            "T": 12,
            "requested_N": 10_000,
            "zero_context": True,
            "validation_manifest_sha256": hash_file(manifest_path),
        },
    )
    return split_dir


def test_closed_book_manifest_hash_count_and_support_verification(tmp_path) -> None:
    split_dir = _write_closed_book_qa_fixture(tmp_path / "qa")
    records, provenance = load_verified_qa_split(
        split_dir,
        split="validation",
        expected_table_count=12,
        expected_fact_count=10_000,
    )
    assert [record["hop"] for record in records] == [0, 0, 0, 0, 1, 2, 3]
    assert set(provenance["input_hashes"]) == {"H0", "H1", "H2", "H3"}
    with (split_dir / "H1.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(QAArtifactError, match="H1 hash"):
        load_verified_qa_split(
            split_dir,
            split="validation",
            expected_table_count=12,
            expected_fact_count=10_000,
        )


def test_closed_book_generation_is_deterministic_with_stubs() -> None:
    records = [_closed_book_record("h0_a", 0, "Alpha")]

    def run_once():
        return generate_prediction_records(
            records,
            tokenizer=_ClosedBookTokenizer(),
            model=_ClosedBookModel(["Alpha"]),
            torch_module=_FakeTorch,
            batch_size=1,
            context_length=256,
            device="cpu",
        )

    assert run_once() == run_once()


def test_closed_book_safe_output_directory_handling(tmp_path) -> None:
    output_dir = tmp_path / "result"
    assert prepare_result_directory(output_dir) == output_dir
    assert prepare_result_directory(output_dir) == output_dir
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_result_directory(output_dir)

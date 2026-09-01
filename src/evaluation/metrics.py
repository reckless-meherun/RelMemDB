import re
import unicodedata
from collections import Counter
from typing import Any

PRIMARY_METRIC = "normalized_exact_match"


def normalize_answer(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("answers must be strings")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.?!]$", "", normalized).rstrip()
    return normalized


def strict_exact_match(prediction: str, gold_answer: str) -> bool:
    if not isinstance(prediction, str) or not isinstance(gold_answer, str):
        raise TypeError("prediction and gold answer must be strings")
    return prediction.strip() == gold_answer.strip()


def normalized_exact_match(prediction: str, gold_answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold_answer)


def score_prediction(
    qa_record: dict[str, Any], raw_generation: str, prediction: str
) -> dict[str, Any]:
    gold_answer = qa_record.get("gold_answer")
    if not isinstance(gold_answer, str):
        raise TypeError("QA record gold_answer must be text")
    return {
        **qa_record,
        "raw_generation": raw_generation,
        "prediction": prediction,
        "strict_exact_match": strict_exact_match(prediction, gold_answer),
        "normalized_exact_match": normalized_exact_match(
            prediction, gold_answer
        ),
    }


def _accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    strict_correct = sum(bool(record["strict_exact_match"]) for record in records)
    normalized_correct = sum(
        bool(record["normalized_exact_match"]) for record in records
    )
    return {
        "count": count,
        "strict_exact_match_correct": strict_correct,
        "strict_exact_match_accuracy": strict_correct / count if count else None,
        "normalized_exact_match_correct": normalized_correct,
        "normalized_exact_match_accuracy": (
            normalized_correct / count if count else None
        ),
    }


def _majority_baseline(
    records_by_target_field: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_target_field: dict[str, dict[str, Any]] = {}
    for target_field, records in sorted(records_by_target_field.items()):
        counts = Counter(record["gold_answer"] for record in records)
        maximum = max(counts.values())
        majority_answer = min(
            (answer for answer, count in counts.items() if count == maximum),
            key=lambda answer: (normalize_answer(answer), answer),
        )
        correct = sum(
            normalized_exact_match(majority_answer, record["gold_answer"])
            for record in records
        )
        by_target_field[target_field] = {
            "count": len(records),
            "majority_gold_answer": majority_answer,
            "correct_count": correct,
            "accuracy": correct / len(records),
        }
    macro_accuracy = (
        sum(item["accuracy"] for item in by_target_field.values())
        / len(by_target_field)
        if by_target_field
        else None
    )
    return {
        "by_target_field": by_target_field,
        "macro_accuracy": macro_accuracy,
    }


def _conditional_relational_accuracy(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    h0_by_id = {
        record["id"]: record for record in records if record.get("hop") == 0
    }
    result: dict[str, dict[str, Any]] = {}
    for hop in (1, 2, 3):
        hop_records = [record for record in records if record.get("hop") == hop]
        eligible: list[dict[str, Any]] = []
        for record in hop_records:
            support_ids = record.get("support_fact_ids")
            if not isinstance(support_ids, list) or len(support_ids) != hop + 1:
                raise ValueError(f"H{hop} prediction has invalid support_fact_ids")
            try:
                supports = [h0_by_id[support_id] for support_id in support_ids]
            except KeyError as exc:
                raise ValueError(
                    f"H{hop} prediction references missing H0 support: {exc.args[0]}"
                ) from exc
            if all(support[PRIMARY_METRIC] for support in supports):
                eligible.append(record)
        conditional_correct = sum(
            bool(record[PRIMARY_METRIC]) for record in eligible
        )
        eligible_count = len(eligible)
        total_count = len(hop_records)
        result[f"H{hop}"] = {
            "total_count": total_count,
            "support_correct_count": eligible_count,
            "eligible_count": eligible_count,
            "support_recall_rate": (
                eligible_count / total_count if total_count else None
            ),
            "conditional_correct_count": conditional_correct,
            "conditional_accuracy": (
                conditional_correct / eligible_count if eligible_count else None
            ),
        }
    return result


def compute_evaluation_metrics(
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not prediction_records:
        raise ValueError("cannot compute metrics for an empty prediction set")
    ids = [record.get("id") for record in prediction_records]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("every prediction record must have a non-empty ID")
    if len(ids) != len(set(ids)):
        raise ValueError("prediction record IDs must be unique")
    for record in prediction_records:
        for key in (
            "gold_answer",
            "prediction",
            "strict_exact_match",
            "normalized_exact_match",
            "target_field",
            "hop",
        ):
            if key not in record:
                raise ValueError(f"prediction record is missing {key}")

    by_hop = {
        f"H{hop}": _accuracy(
            [record for record in prediction_records if record["hop"] == hop]
        )
        for hop in range(4)
    }
    h0_records = [record for record in prediction_records if record["hop"] == 0]
    h0_by_fact_type = {
        fact_type: _accuracy(
            [
                record
                for record in h0_records
                if record.get("fact_type") == fact_type
            ]
        )
        for fact_type in ("attribute", "relation")
    }
    records_by_target_field: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        target_field = record["target_field"]
        if not isinstance(target_field, str) or not target_field:
            raise ValueError("target_field must be non-empty text")
        records_by_target_field.setdefault(target_field, []).append(record)
    by_target_field = {
        target_field: _accuracy(records)
        for target_field, records in sorted(records_by_target_field.items())
    }
    target_field_macro_average = {
        metric: sum(field_metrics[metric] for field_metrics in by_target_field.values())
        / len(by_target_field)
        for metric in (
            "strict_exact_match_accuracy",
            "normalized_exact_match_accuracy",
        )
    }
    return {
        "primary_metric": PRIMARY_METRIC,
        "overall": _accuracy(prediction_records),
        "by_hop": by_hop,
        "h0_by_fact_type": h0_by_fact_type,
        "by_target_field": by_target_field,
        "target_field_macro_average": target_field_macro_average,
        "majority_gold_answer_baseline": _majority_baseline(
            records_by_target_field
        ),
        "conditional_relational_accuracy": _conditional_relational_accuracy(
            prediction_records
        ),
    }

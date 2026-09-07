from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.inference import EVALUATION_SEED, load_local_causal_lm


def _flatten_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return value


def _token_prefix_and_gold(
    tokenizer: Any, *, text: str, answer_start: int, answer_end: int
) -> tuple[list[int], list[int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = _flatten_ids(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    answer_token_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > answer_start and start < answer_end
    ]
    if not answer_token_indices:
        raise ValueError("CPT completion answer has no token span")
    first = answer_token_indices[0]
    last = answer_token_indices[-1] + 1
    if answer_token_indices != list(range(first, last)):
        raise ValueError("CPT completion answer token span is not contiguous")
    prefix_ids = token_ids[:first]
    gold_ids = token_ids[first:last]
    if not prefix_ids or not gold_ids:
        raise ValueError("CPT completion prompt or gold token sequence is empty")
    return prefix_ids, gold_ids


def evaluate_completion_probes_with_local_checkpoint(
    probes: list[dict[str, Any]],
    *,
    checkpoint: str | Path,
    context_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Greedily test the exact next-token span targeted by each CPT probe."""
    if not probes:
        raise ValueError("cannot evaluate an empty CPT completion probe set")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer")

    tokenizer, model, torch_module = load_local_causal_lm(checkpoint)
    predictions: list[dict[str, Any]] = []
    inference_context = getattr(torch_module, "inference_mode", None)

    for probe in probes:
        prompt = probe.get("prompt")
        gold_answer = probe.get("gold_answer")
        probe_type = probe.get("probe_type")
        source_fact = probe.get("source_fact")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("CPT completion probe prompt must be non-empty text")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("CPT completion probe gold answer must be non-empty text")

        if probe_type == "canonical_completion":
            if not isinstance(source_fact, str) or not source_fact:
                raise ValueError("canonical CPT probe must include its source fact")
            answer_start = source_fact.rfind(gold_answer)
            if answer_start < 0 or source_fact[:answer_start] != prompt:
                raise ValueError("canonical CPT probe is not an exact source-record prefix")
            scoring_text = source_fact
        elif probe_type == "heldout_declarative_completion":
            scoring_text = f"{prompt}{gold_answer}."
            answer_start = len(prompt)
        else:
            raise ValueError(f"unsupported CPT completion probe type: {probe_type!r}")
        answer_end = answer_start + len(gold_answer)
        prompt_ids, gold_ids = _token_prefix_and_gold(
            tokenizer,
            text=scoring_text,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        if len(prompt_ids) + len(gold_ids) > context_length:
            raise ValueError(
                f"CPT completion probe {probe.get('id')} exceeds context length "
                f"{context_length}"
            )

        input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device="cuda")
        attention_mask = torch_module.ones_like(input_ids)
        prompt_width = input_ids.shape[1]
        context_manager = (
            inference_context() if inference_context else torch_module.no_grad()
        )
        with context_manager:
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=len(gold_ids),
                min_new_tokens=len(gold_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        predicted_ids = _flatten_ids(generated[0, prompt_width:])
        raw_generation = tokenizer.decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gold_text = tokenizer.decode(
            gold_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        token_exact_match = predicted_ids == gold_ids
        predictions.append(
            {
                **probe,
                "prediction": raw_generation,
                "tokenized_gold": gold_text,
                "gold_token_count": len(gold_ids),
                "prediction_token_count": len(predicted_ids),
                "gold_token_ids": gold_ids,
                "prediction_token_ids": predicted_ids,
                "token_exact_match": token_exact_match,
            }
        )

    by_probe_type: dict[str, dict[str, Any]] = {}
    for probe_type in sorted({record["probe_type"] for record in predictions}):
        subset = [
            record for record in predictions if record["probe_type"] == probe_type
        ]
        correct = sum(record["token_exact_match"] for record in subset)
        by_probe_type[probe_type] = {
            "count": len(subset),
            "token_exact_match_correct": correct,
            "token_exact_match_accuracy": correct / len(subset),
        }

    correct = sum(record["token_exact_match"] for record in predictions)
    metrics = {
        "primary_metric": "token_exact_match",
        "overall": {
            "count": len(predictions),
            "token_exact_match_correct": correct,
            "token_exact_match_accuracy": correct / len(predictions),
        },
        "by_probe_type": by_probe_type,
    }
    model_identity = {
        "tokenizer_identity": getattr(tokenizer, "name_or_path", None)
        or str(Path(checkpoint).resolve()),
        "model_identity": getattr(getattr(model, "config", None), "_name_or_path", None)
        or str(Path(checkpoint).resolve()),
        "evaluation_seed": EVALUATION_SEED,
    }
    return predictions, metrics, model_identity

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


def evaluate_completion_probes_with_local_checkpoint(
    probes: list[dict[str, Any]],
    *,
    checkpoint: str | Path,
    context_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Greedily test whether a checkpoint emits each probe's gold token sequence."""
    if not probes:
        raise ValueError("cannot evaluate an empty CPT completion probe set")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer")

    tokenizer, model, torch_module = load_local_causal_lm(checkpoint)
    tokenizer.padding_side = "left"
    predictions: list[dict[str, Any]] = []
    inference_context = getattr(torch_module, "inference_mode", None)

    for probe in probes:
        prompt = probe.get("prompt")
        gold_answer = probe.get("gold_answer")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("CPT completion probe prompt must be non-empty text")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("CPT completion probe gold answer must be non-empty text")

        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_ids = _flatten_ids(encoded["input_ids"])
        gold_ids = _flatten_ids(
            tokenizer(
                gold_answer,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        if not gold_ids:
            raise ValueError("CPT completion probe gold answer tokenized to zero tokens")
        if len(prompt_ids) + len(gold_ids) > context_length:
            raise ValueError(
                f"CPT completion probe {probe.get('id')} exceeds context length "
                f"{context_length}"
            )

        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        context_manager = (
            inference_context() if inference_context else torch_module.no_grad()
        )
        with context_manager:
            generated = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                do_sample=False,
                num_beams=1,
                max_new_tokens=len(gold_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuation_ids = generated[0, prompt_width:]
        predicted_ids = _flatten_ids(continuation_ids)
        raw_generation = tokenizer.decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        token_exact_match = predicted_ids == gold_ids
        predictions.append(
            {
                **probe,
                "prediction": raw_generation,
                "gold_token_count": len(gold_ids),
                "prediction_token_count": len(predicted_ids),
                "gold_token_ids": gold_ids,
                "prediction_token_ids": predicted_ids,
                "token_exact_match": token_exact_match,
            }
        )

    by_probe_type: dict[str, dict[str, Any]] = {}
    probe_types = sorted({record["probe_type"] for record in predictions})
    for probe_type in probe_types:
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

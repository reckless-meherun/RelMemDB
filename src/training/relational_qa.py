"""Deterministic Step 6A data, prompts, scoring, and GPT-2 inference."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


PREFLIGHT_NAMESPACE = "preflight_baseline_v1"
MODEL_ID = "gpt2"
QUESTION_TEMPLATES = {
    1: (
        "Starting from entity {entity}, follow the previous-entity relation once. "
        "What is attribute_0 of the reached entity?"
    ),
    2: (
        "Starting from entity {entity}, follow the previous-entity relation two times. "
        "What is attribute_0 of the reached entity?"
    ),
    3: (
        "Starting from entity {entity}, follow the previous-entity relation three times. "
        "What is attribute_0 of the reached entity?"
    ),
}


def _digest(*parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _opaque_suffix(*parts: object) -> str:
    """Render 40 SHA-256-derived bits as 12 token-efficient hex characters."""
    # Decimal digits are a subset of hexadecimal characters and tokenize much
    # more compactly in GPT-2 than unconstrained a-f-heavy digest slices.
    return f"{int(_digest(*parts), 16) % 10**12:012d}"


def _semantic_id(record: dict[str, Any]) -> str:
    semantic = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return f"pf_{hashlib.sha256(semantic.encode('utf-8')).hexdigest()}"


def build_baseline_example(seed: int, hop: int, index: int) -> dict[str, Any]:
    """Build one independent example without consulting target-world artifacts."""
    if hop not in QUESTION_TEMPLATES:
        raise ValueError(f"unsupported preflight hop: {hop}")
    if index < 0:
        raise ValueError("example index must be non-negative")

    # Positions are ordered E3, E2, E1, E0 so following ``previous`` advances
    # one position. Every hop sees the same complete four-entity world shape.
    entities = [
        f"pf_ent_{_opaque_suffix(seed, PREFLIGHT_NAMESPACE, hop, index, 'entity', pos)}"
        for pos in range(4)
    ]
    values = [
        f"pf_val_{_opaque_suffix(seed, PREFLIGHT_NAMESPACE, hop, index, 'value', pos)}"
        for pos in range(4)
    ]
    labeled_facts = [
        *[
            (
                f"relation_{pos}",
                f"Previous entity of {source} is {target}.",
            )
            for pos, (source, target) in enumerate(zip(entities, entities[1:]))
        ],
        *[
            (f"attribute_{pos}", f"attribute_0 of entity {entity} is {value}.")
            for pos, (entity, value) in enumerate(zip(entities, values))
        ],
    ]
    # Fact-order keys deliberately omit the hop and fact content: the same
    # seed/index permutation applies to H1, H2, and H3 and cannot encode which
    # attribute is the answer.
    labeled_facts.sort(
        key=lambda item: (
            _digest(seed, PREFLIGHT_NAMESPACE, index, "fact_order", item[0]),
            item[0],
        )
    )
    facts = [fact for _, fact in labeled_facts]
    semantic = {
        "hop": hop,
        "facts": facts,
        "question": QUESTION_TEMPLATES[hop].format(entity=entities[0]),
        "answer": values[hop],
    }
    return {"id": _semantic_id(semantic), **semantic}


def generate_baseline_dataset(
    seed: int, examples_per_hop: int, hops: Iterable[int] = (1, 2, 3)
) -> dict[int, list[dict[str, Any]]]:
    if examples_per_hop <= 0:
        raise ValueError("examples_per_hop must be positive")
    dataset = {
        hop: [build_baseline_example(seed, hop, index) for index in range(examples_per_hop)]
        for hop in hops
    }
    seen_identifiers: set[str] = set()
    identifier_pattern = re.compile(r"\bpf_(?:ent|val)_[0-9a-f]{12}\b")
    for rows in dataset.values():
        for row in rows:
            identifiers = set(identifier_pattern.findall("\n".join(row["facts"])))
            if len(identifiers) != 8:
                raise ValueError(
                    f"opaque identifier collision in preflight example {row['id']}"
                )
            collisions = seen_identifiers.intersection(identifiers)
            if collisions:
                collision = min(collisions)
                raise ValueError(f"opaque identifier collision in baseline: {collision}")
            seen_identifiers.update(identifiers)
    return dataset


def format_relational_prompt(example: dict[str, Any]) -> str:
    facts = "\n".join(example["facts"])
    return f"Facts:\n{facts}\n\nQuestion:\n{example['question']}\n\nAnswer:"


def format_copy_prompt(gold_answer: str) -> str:
    return f"Copy this token exactly: {gold_answer}\nAnswer:"


def first_generated_line(continuation: str) -> str:
    stripped = continuation.strip()
    return stripped.splitlines()[0].strip() if stripped else ""


def strict_exact_match(prediction: str, gold_answer: str) -> bool:
    return prediction == gold_answer


def answer_prefix_match(continuation: str, gold_answer: str) -> bool:
    stripped = continuation.lstrip()
    if not stripped.startswith(gold_answer):
        return False
    remainder = stripped[len(gold_answer) :]
    return not remainder or not (remainder[0].isalnum() or remainder[0] == "_")


def make_prediction_record(
    example: dict[str, Any], continuation: str, copy_continuation: str
) -> dict[str, Any]:
    prediction = first_generated_line(continuation)
    copy_prediction = first_generated_line(copy_continuation)
    gold = example["answer"]
    return {
        "id": example["id"],
        "hop": example["hop"],
        "gold_answer": gold,
        "prediction": prediction,
        "strict_exact_match": strict_exact_match(prediction, gold),
        "answer_prefix_match": answer_prefix_match(continuation, gold),
        "copy_prediction": copy_prediction,
        "copy_strict_exact_match": strict_exact_match(copy_prediction, gold),
    }


def summarize_predictions(
    predictions_by_hop: dict[int, list[dict[str, Any]]],
    strict_threshold: float,
    copy_threshold: float,
) -> dict[str, Any]:
    def accuracy(rows: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0

    per_hop: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for hop in sorted(predictions_by_hop):
        rows = predictions_by_hop[hop]
        all_rows.extend(rows)
        per_hop[f"H{hop}"] = {
            "num_examples": len(rows),
            "strict_exact_match_accuracy": accuracy(rows, "strict_exact_match"),
            "prefix_match_accuracy": accuracy(rows, "answer_prefix_match"),
            "copy_control_exact_match_accuracy": accuracy(
                rows, "copy_strict_exact_match"
            ),
        }
    overall = {
        "num_examples": len(all_rows),
        "strict_exact_match_accuracy": accuracy(all_rows, "strict_exact_match"),
        "prefix_match_accuracy": accuracy(all_rows, "answer_prefix_match"),
        "copy_control_exact_match_accuracy": accuracy(
            all_rows, "copy_strict_exact_match"
        ),
    }
    adequate = all(
        metrics["strict_exact_match_accuracy"] >= strict_threshold
        for metrics in per_hop.values()
    ) and overall["copy_control_exact_match_accuracy"] >= copy_threshold
    return {
        "per_hop": per_hop,
        "overall": overall,
        "thresholds": {
            "baseline_strict_em_threshold": strict_threshold,
            "copy_control_threshold": copy_threshold,
        },
        "decision": "skip_skill_training" if adequate else "needs_skill_training",
    }


def deterministic_json_bytes(value: Any, *, jsonl: bool = False) -> bytes:
    if jsonl:
        return b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            for row in value
        )
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_baseline_dataset(
    output_dir: Path,
    dataset: dict[int, list[dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_entries: dict[str, Any] = {}
    for hop in sorted(dataset):
        name = f"H{hop}.jsonl"
        payload = deterministic_json_bytes(dataset[hop], jsonl=True)
        (output_dir / name).write_bytes(payload)
        file_entries[name] = {
            "examples": len(dataset[hop]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "dataset": "preflight_relational_qa_baseline",
        "namespace": PREFLIGHT_NAMESPACE,
        "seed": seed,
        "total_examples": sum(len(rows) for rows in dataset.values()),
        "files": file_entries,
    }
    (output_dir / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    return manifest


def load_gpt2(local_model_dir: Path) -> tuple[Any, Any, Any, str]:
    """Load local GPT-2, or cache the original HF checkpoint once."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("missing required dependency: torch") from exc
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("missing required dependency: transformers") from exc

    source = str(local_model_dir) if (local_model_dir / "config.json").is_file() else MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(source)
    except Exception as exc:
        raise RuntimeError(f"unable to load GPT-2 from {source}: {exc}") from exc
    if source == MODEL_ID:
        local_model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
        precision = "bfloat16"
    else:
        model = model.to(device=device, dtype=torch.float32)
        precision = "float32"
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, device, precision


def generate_continuations(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int = 16,
) -> list[str]:
    import torch

    validate_prompt_lengths(prompts, tokenizer, max_input_length)
    continuations: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        continuations.extend(
            tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)
        )
    return continuations


def validate_prompt_lengths(
    prompts: list[str], tokenizer: Any, max_input_length: int
) -> list[int]:
    """Return exact GPT-2 prompt lengths, rejecting rather than truncating."""
    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    maximum = max(lengths, default=0)
    if maximum > max_input_length:
        raise ValueError(
            f"preflight prompt length {maximum} exceeds maximum {max_input_length}"
        )
    return lengths

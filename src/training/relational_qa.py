"""Deterministic Step 6A/6B relational data, scoring, and GPT-2 training."""

from __future__ import annotations

import hashlib
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable


PREFLIGHT_NAMESPACE = "preflight_baseline_v1"
SKILL_TRAIN_NAMESPACE = "relational_skill_train_v2"
SKILL_VALIDATION_NAMESPACE = "relational_skill_validation_v2"
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
    """Render an independently SHA-256-derived value as 12 decimal digits."""
    return f"{int(_digest(*parts), 16) % 10**12:012d}"


def _semantic_id(record: dict[str, Any]) -> str:
    semantic = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return f"pf_{hashlib.sha256(semantic.encode('utf-8')).hexdigest()}"


def _independent_skill_suffix(
    tokenizer: Any, *semantic_parts: object, max_suffix_tokens: int = 5
) -> str:
    """Independently SHA-derive a compact 12-digit symbol by rejection sampling."""
    for nonce in range(10_000):
        suffix = _opaque_suffix(*semantic_parts, "candidate", nonce)
        if len(tokenizer.encode(suffix, add_special_tokens=False)) <= max_suffix_tokens:
            return suffix
    raise ValueError("unable to derive a token-efficient 12-digit skill identifier")


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


def build_skill_example(
    seed: int, split: str, hop: int, index: int, tokenizer: Any
) -> dict[str, Any]:
    if split not in ("train", "validation"):
        raise ValueError(f"unsupported skill split: {split}")
    if hop not in QUESTION_TEMPLATES:
        raise ValueError(f"unsupported skill hop: {hop}")
    if index < 0:
        raise ValueError("example index must be non-negative")
    namespace = (
        SKILL_TRAIN_NAMESPACE if split == "train" else SKILL_VALIDATION_NAMESPACE
    )
    entities = [
        "pf_ent_"
        + _independent_skill_suffix(
            tokenizer, seed, namespace, hop, index, "entity", pos
        )
        for pos in range(4)
    ]
    values = [
        "pf_val_"
        + _independent_skill_suffix(
            tokenizer, seed, namespace, hop, index, "value", pos
        )
        for pos in range(4)
    ]
    labeled_facts = [
        *[
            (f"relation_{pos}", f"Previous entity of {source} is {target}.")
            for pos, (source, target) in enumerate(zip(entities, entities[1:]))
        ],
        *[
            (f"attribute_{pos}", f"attribute_0 of entity {entity} is {value}.")
            for pos, (entity, value) in enumerate(zip(entities, values))
        ],
    ]
    labeled_facts.sort(
        key=lambda item: (
            _digest(seed, namespace, index, "fact_order", item[0]), item[0]
        )
    )
    semantic = {
        "split": split,
        "hop": hop,
        "facts": [fact for _, fact in labeled_facts],
        "question": QUESTION_TEMPLATES[hop].format(entity=entities[0]),
        "answer": values[hop],
    }
    record_id = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"id": f"pf_skill_{record_id}", **semantic}


def _example_identifiers(example: dict[str, Any]) -> set[str]:
    return set(
        re.findall(
            r"\bpf_(?:ent|val)_[0-9]{12}\b",
            "\n".join(example["facts"]),
        )
    )


def generate_skill_dataset(
    seed: int,
    train_examples_per_hop: int,
    validation_examples_per_hop: int,
    tokenizer: Any,
    hops: Iterable[int] = (1, 2, 3),
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    if train_examples_per_hop <= 0 or validation_examples_per_hop <= 0:
        raise ValueError("skill split sizes must be positive")
    counts = {
        "train": train_examples_per_hop,
        "validation": validation_examples_per_hop,
    }
    dataset = {
        split: {
            hop: [
                build_skill_example(seed, split, hop, i, tokenizer)
                for i in range(count)
            ]
            for hop in hops
        }
        for split, count in counts.items()
    }
    seen: set[str] = set()
    for split in ("train", "validation"):
        for rows in dataset[split].values():
            for row in rows:
                identifiers = _example_identifiers(row)
                if len(identifiers) != 8:
                    raise ValueError(f"identifier collision in skill example {row['id']}")
                collision = seen.intersection(identifiers)
                if collision:
                    raise ValueError(
                        f"identifier collision in skill dataset: {min(collision)}"
                    )
                seen.update(identifiers)
    return dataset


def verify_skill_isolation(
    skill_dataset: dict[str, dict[int, list[dict[str, Any]]]],
    baseline_dataset: dict[int, list[dict[str, Any]]],
    target_world_path: Path,
) -> dict[str, Any]:
    sets = {
        split: set().union(
            *(
                _example_identifiers(row)
                for rows in skill_dataset[split].values()
                for row in rows
            )
        )
        for split in ("train", "validation")
    }
    baseline = set().union(
        *(
            _example_identifiers(row)
            for rows in baseline_dataset.values()
            for row in rows
        )
    )
    target = set(
        re.findall(
            r"\b(?:e|v)_[0-9a-f]{32}\b",
            target_world_path.read_text(encoding="utf-8"),
        )
    )
    checks = {
        "train_validation_overlap": len(sets["train"] & sets["validation"]),
        "train_baseline_overlap": len(sets["train"] & baseline),
        "validation_baseline_overlap": len(sets["validation"] & baseline),
        "skill_target_overlap": len((sets["train"] | sets["validation"]) & target),
    }
    if any(checks.values()):
        raise ValueError(f"skill dataset isolation failure: {checks}")
    if not all(token.startswith(("pf_ent_", "pf_val_")) for token in sets["train"]):
        raise ValueError("training identifiers use an invalid surface syntax")
    if not all(
        token.startswith(("pf_ent_", "pf_val_")) for token in sets["validation"]
    ):
        raise ValueError("validation identifiers use an invalid surface syntax")
    checks.update(
        {
            "train_identifier_count": len(sets["train"]),
            "validation_identifier_count": len(sets["validation"]),
            "baseline_identifier_count": len(baseline),
            "target_identifier_count": len(target),
            "verified": True,
        }
    )
    return checks


def write_skill_dataset(
    output_dir: Path,
    dataset: dict[str, dict[int, list[dict[str, Any]]]],
    seed: int,
    isolation: dict[str, Any],
    maximum_prompt_tokens: int,
    maximum_sequence_tokens: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl")):
        rows = [row for hop in sorted(dataset[split]) for row in dataset[split][hop]]
        payload = deterministic_json_bytes(rows, jsonl=True)
        (output_dir / filename).write_bytes(payload)
        files[filename] = {
            "examples": len(rows),
            "per_hop": {
                f"H{hop}": len(dataset[split][hop]) for hop in sorted(dataset[split])
            },
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "dataset": "generic_relational_skill_v2",
        "seed": seed,
        "namespaces": {
            "train": SKILL_TRAIN_NAMESPACE,
            "validation": SKILL_VALIDATION_NAMESPACE,
        },
        "total_examples": sum(item["examples"] for item in files.values()),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_supervised_sequence_tokens": maximum_sequence_tokens,
        "identifier_surface_syntax": "pf_ent_<12 decimal digits> / pf_val_<12 decimal digits>",
        "identifier_generation": "independent SHA-256 rejection sampling",
        "files": files,
        "isolation": isolation,
        "step6a_baseline_used_for_training": False,
        "step6a_baseline_used_for_model_selection": False,
        "target_database_facts_used": False,
    }
    (output_dir / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    return manifest


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


def visible_attribute_values(example: dict[str, Any]) -> set[str]:
    pattern = re.compile(
        r"^attribute_0 of entity pf_ent_[0-9]{12} is (pf_val_[0-9]{12})\.$"
    )
    values = {
        match.group(1)
        for fact in example["facts"]
        if (match := pattern.fullmatch(fact))
    }
    if len(values) != 4:
        raise ValueError(f"expected four distinct visible values in {example['id']}")
    return values


def classify_candidate_prediction(example: dict[str, Any], prediction: str) -> str:
    if prediction == example["answer"]:
        return "exact_correct"
    if prediction in visible_attribute_values(example):
        return "wrong_visible_candidate"
    return "non_candidate_generation"


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
        "candidate_classification": classify_candidate_prediction(example, prediction),
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
        if rows and all("candidate_classification" in row for row in rows):
            for classification, metric_name in (
                ("exact_correct", "exact_correct_rate"),
                ("wrong_visible_candidate", "wrong_visible_candidate_rate"),
                ("non_candidate_generation", "non_candidate_generation_rate"),
            ):
                per_hop[f"H{hop}"][metric_name] = sum(
                    row["candidate_classification"] == classification for row in rows
                ) / len(rows)
    overall = {
        "num_examples": len(all_rows),
        "strict_exact_match_accuracy": accuracy(all_rows, "strict_exact_match"),
        "prefix_match_accuracy": accuracy(all_rows, "answer_prefix_match"),
        "copy_control_exact_match_accuracy": accuracy(
            all_rows, "copy_strict_exact_match"
        ),
    }
    if all_rows and all("candidate_classification" in row for row in all_rows):
        for classification, metric_name in (
            ("exact_correct", "exact_correct_rate"),
            ("wrong_visible_candidate", "wrong_visible_candidate_rate"),
            ("non_candidate_generation", "non_candidate_generation_rate"),
        ):
            overall[metric_name] = sum(
                row["candidate_classification"] == classification for row in all_rows
            ) / len(all_rows)
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
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
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


def encode_supervised_example(
    example: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Encode answer-only causal-LM supervision, including supervised EOS."""
    prompt_ids = tokenizer.encode(
        format_relational_prompt(example), add_special_tokens=False
    )
    answer_ids = tokenizer.encode(example["answer"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    input_ids = [*prompt_ids, *answer_ids, tokenizer.eos_token_id]
    if len(input_ids) > max_length:
        raise ValueError(
            f"supervised sequence length {len(input_ids)} exceeds maximum {max_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + [*answer_ids, tokenizer.eos_token_id],
        "prompt_length": len(prompt_ids),
    }


def collate_supervised(
    examples: list[dict[str, list[int]]], pad_token_id: int
) -> dict[str, Any]:
    import torch

    width = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    attention_masks = []
    labels = []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(example["attention_mask"] + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def select_best_epoch(epoch_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not epoch_records:
        raise ValueError("at least one epoch record is required")
    return max(
        epoch_records,
        key=lambda row: (
            row["validation_overall_strict_em"],
            -row["validation_loss"],
            -row["epoch"],
        ),
    )


def post_skill_decision(
    per_hop_strict_em: dict[int, float], threshold: float
) -> str:
    return (
        "relational_skill_ready"
        if all(per_hop_strict_em.get(hop, 0.0) >= threshold for hop in (1, 2, 3))
        else "stop_and_diagnose"
    )


def _validation_loss(
    model: Any,
    encoded_rows: list[dict[str, list[int]]],
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> float:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        encoded_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_supervised(batch, tokenizer.pad_token_id),
    )
    weighted_loss = 0.0
    supervised_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast:
                output = model(**batch)
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            weighted_loss += float(output.loss.item()) * token_count
            supervised_tokens += token_count
    return weighted_loss / supervised_tokens


def _validation_exact_match(
    model: Any,
    tokenizer: Any,
    device: Any,
    rows_by_hop: dict[int, list[dict[str, Any]]],
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
) -> dict[int, float]:
    previous_cache_setting = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    scores: dict[int, float] = {}
    for hop in sorted(rows_by_hop):
        rows = rows_by_hop[hop]
        continuations = generate_continuations(
            [format_relational_prompt(row) for row in rows],
            model,
            tokenizer,
            device,
            max_input_length,
            max_new_tokens,
            batch_size=batch_size,
        )
        scores[hop] = sum(
            first_generated_line(continuation) == row["answer"]
            for row, continuation in zip(rows, continuations, strict=True)
        ) / len(rows)
    model.config.use_cache = previous_cache_setting
    return scores


def train_relational_skill(
    model: Any,
    tokenizer: Any,
    device: Any,
    train_rows: list[dict[str, Any]],
    validation_rows_by_hop: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train once from base GPT-2 and save the validation-selected checkpoint."""
    import time

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        # Keep master parameters/AdamW state in FP32 while using BF16 forward
        # and backward computation through native autocast.
        model.float()

    train_encoded = [
        encode_supervised_example(row, tokenizer, config["max_length"])
        for row in train_rows
    ]
    validation_rows = [
        row
        for hop in sorted(validation_rows_by_hop)
        for row in validation_rows_by_hop[hop]
    ]
    validation_encoded = [
        encode_supervised_example(row, tokenizer, config["max_length"])
        for row in validation_rows
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    steps_per_epoch = (len(train_encoded) + config["batch_size"] - 1) // config[
        "batch_size"
    ]
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    model.config.use_cache = False
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            train_encoded,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: collate_supervised(batch, tokenizer.pad_token_id),
        )
        model.train()
        weighted_loss = 0.0
        supervised_tokens = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            weighted_loss += float(output.loss.item()) * token_count
            supervised_tokens += token_count
        validation_loss = _validation_loss(
            model,
            validation_encoded,
            tokenizer,
            device,
            config["batch_size"],
        )
        hop_scores = _validation_exact_match(
            model,
            tokenizer,
            device,
            validation_rows_by_hop,
            config["max_length"],
            max_new_tokens,
            config["batch_size"],
        )
        record = {
            "epoch": epoch,
            "train_loss": weighted_loss / supervised_tokens,
            "validation_loss": validation_loss,
            **{f"validation_H{hop}_strict_em": hop_scores[hop] for hop in (1, 2, 3)},
            "validation_overall_strict_em": sum(hop_scores.values()) / len(hop_scores),
            "elapsed_seconds": time.perf_counter() - started,
        }
        records.append(record)
        if best is None or select_best_epoch([best, record]) is record:
            best = record
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
        print(
            f"Epoch {epoch}: train_loss={record['train_loss']:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"H1={hop_scores[1]:.4f} H2={hop_scores[2]:.4f} "
            f"H3={hop_scores[3]:.4f} "
            f"overall={record['validation_overall_strict_em']:.4f}",
            flush=True,
        )
    assert best is not None
    summary = {
        "selected_epoch": best["epoch"],
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "precision": "bfloat16" if use_bf16 else "float32",
        "step6a_baseline_used_for_training": False,
        "step6a_baseline_used_for_model_selection": False,
        "target_database_facts_used": False,
    }
    (checkpoint_dir / "training_metadata.json").write_bytes(
        deterministic_json_bytes({"configuration": config, "epochs": records, **summary})
    )
    return records, summary

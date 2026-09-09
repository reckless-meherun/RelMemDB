#!/usr/bin/env python3
"""Run Experiment 3 using Experiment 2's CPT/SFT/evaluation behavior."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EXP03_NAME, load_config, validate_config
from data.cpt_test import CPT_TEST_PROBE_TYPES
from data.exp3 import (
    EXP3_INVERSE_SFT_DATASET_DIR,
    generate_exp3_qa,
    materialize_exp3_dataset,
    validate_exp3_fact_count,
    verify_exp3_dataset,
    verify_exp3_qa,
)
from evaluation.inference import (
    evaluate_with_local_checkpoint,
    generate_prediction_records,
    load_local_causal_lm,
    load_verified_qa_split,
)
from evaluation.metrics import (
    compute_evaluation_metrics,
    compute_unordered_exact_match_metrics,
)
from experiment import (
    apply_checkpoint_model_config,
    resolve_model_checkpoint,
    verify_checkpoint_layers,
)
from training.checkpoint_retention import retain_best_exp3_checkpoint
from training.cpt import run_cpt_training
from training.target_sft import run_target_sft_training
from utils.hashing import hash_file
from utils.io import write_json, write_jsonl, write_yaml
from utils.paths import safe_component

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp03_continent_inverse.yaml"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "generated_databases" / EXP03_NAME
QA_ROOT = PROJECT_ROOT / "datasets" / "qa" / EXP03_NAME
RUN_ROOT = PROJECT_ROOT / "runs" / EXP03_NAME
RESULT_ROOT = PROJECT_ROOT / "results" / EXP03_NAME
TRAINED_MODELS_ROOT = PROJECT_ROOT / "models" / "trained_models"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RelMemDB Experiment 3: standalone continent CPT, "
            "continent-to-climate SFT, and inverse aggregation evaluation."
        )
    )
    parser.add_argument("--fact-count", required=True, type=_positive_int)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layers", type=_positive_int)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cpt-epochs", type=_positive_int)
    parser.add_argument("--sft-epochs", type=_positive_int)
    parser.add_argument("--cpt-batch-size", type=_positive_int)
    parser.add_argument("--cpt-gradient-accumulation", type=_positive_int)
    parser.add_argument("--sft-batch-size", type=_positive_int)
    parser.add_argument("--sft-gradient-accumulation", type=_positive_int)
    parser.add_argument("--cpt-learning-rate", type=_positive_float)
    parser.add_argument("--sft-learning-rate", type=_positive_float)
    return parser.parse_args()


def _resolved_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(load_config(args.config.resolve()))
    if config["experiment"]["name"] != EXP03_NAME:
        raise ValueError(f"Experiment-3 runner requires experiment.name={EXP03_NAME}")
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        config["experiment"]["seed"] = args.seed
    base_model, native_layers = resolve_model_checkpoint(
        args.model, source_checkpoint=args.base_model
    )
    apply_checkpoint_model_config(config, model_name=args.model, checkpoint=base_model)
    layers = native_layers if args.layers is None else args.layers
    verify_checkpoint_layers(base_model, layers)
    overrides = {
        "cpt_epochs": ("training", "cpt_epochs", args.cpt_epochs),
        "sft_epochs": ("target_sft", "epochs", args.sft_epochs),
        "cpt_batch_size": ("training", "cpt_batch_size", args.cpt_batch_size),
        "cpt_gradient_accumulation": (
            "training",
            "gradient_accumulation_steps",
            args.cpt_gradient_accumulation,
        ),
        "sft_batch_size": ("target_sft", "batch_size", args.sft_batch_size),
        "sft_gradient_accumulation": (
            "target_sft",
            "gradient_accumulation_steps",
            args.sft_gradient_accumulation,
        ),
        "cpt_learning_rate": (
            "training",
            "learning_rate",
            args.cpt_learning_rate,
        ),
        "sft_learning_rate": (
            "target_sft",
            "learning_rate",
            args.sft_learning_rate,
        ),
    }
    for _, (section, key, value) in overrides.items():
        if value is not None:
            config[section][key] = value
    config["_runtime"] = {"run_timestamp": run_dir.name}
    validate_config(config)
    write_yaml(run_dir / "resolved_config.yaml", config)
    config["_base_model"] = base_model
    config["_layers"] = layers
    return config


def _find_dataset(fact_count: int, seed: int) -> dict[str, Any] | None:
    if not DATASET_ROOT.is_dir():
        return None
    for candidate in sorted(DATASET_ROOT.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        try:
            return verify_exp3_dataset(candidate, fact_count=fact_count, seed=seed)
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return None


def _find_qa(
    dataset: dict[str, Any], *, fact_count: int, seed: int
) -> dict[str, Any] | None:
    if not QA_ROOT.is_dir():
        return None
    for candidate in sorted(QA_ROOT.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        try:
            return verify_exp3_qa(
                candidate,
                dataset_dir=dataset["root"],
                fact_count=fact_count,
                seed=seed,
            )
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return None


def _evaluate(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    qa_root: Path,
    dataset_name: str,
    output_dir: Path,
    stage: str,
    fact_count: int,
    layers: int,
    include_unordered: bool = False,
) -> tuple[dict[str, Any], Path]:
    records, provenance = load_verified_qa_split(
        qa_root / dataset_name,
        split="test",
        expected_table_count=1,
        expected_fact_count=fact_count,
        manifest_hash_key=f"{dataset_name}_manifest_sha256",
    )
    evaluation = config["evaluation"]
    predictions, model_identity = evaluate_with_local_checkpoint(
        records,
        checkpoint=checkpoint,
        batch_size=evaluation["batch_size"],
        context_length=evaluation["context_length"],
        max_new_tokens=evaluation["max_new_tokens"],
    )
    metrics = compute_evaluation_metrics(predictions)
    if include_unordered:
        metrics["unordered_exact_match"] = compute_unordered_exact_match_metrics(
            predictions
        )
    output_dir.mkdir(parents=True)
    metadata_path = checkpoint / "training_metadata.json"
    evaluation_config = {
        "experiment_name": EXP03_NAME,
        "evaluation_stage": stage,
        "T": 1,
        "N": fact_count,
        "L": layers,
        "seed": config["experiment"]["seed"],
        "selected_tables": ["continent"],
        "split": "test",
        "test_dataset": dataset_name,
        "qa_data_dir": str(qa_root),
        "qa_record_count": len(records),
        "qa_manifest_sha256": provenance["qa_manifest_sha256"],
        "qa_split_manifest_sha256": provenance["qa_split_manifest_sha256"],
        "qa_input_hashes": provenance["input_hashes"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_training_metadata_sha256": (
            hash_file(metadata_path) if metadata_path.is_file() else None
        ),
        "decoding": {"strategy": "greedy", "do_sample": False, "temperature": None},
        "context_length": evaluation["context_length"],
        "max_new_tokens": evaluation["max_new_tokens"],
        "batch_size": evaluation["batch_size"],
        "primary_metric": "normalized_exact_match",
        **model_identity,
    }
    if include_unordered:
        evaluation_config["additional_metric"] = "unordered_normalized_exact_match"
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "evaluation_config.json", evaluation_config)
    return metrics, output_dir


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(
        isinstance(token, int) for token in encoded
    ):
        raise ValueError("CPT-test tokenization did not produce integer token IDs")
    return encoded


def _score_cpt_answer_spans(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    context_length: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    scored: list[dict[str, Any]] = []
    model.eval()
    for record in records:
        prompt = record["prompt"]
        if not isinstance(prompt, str) or prompt != prompt.rstrip():
            raise ValueError(
                "Exp03 CPT-test prompts must not end in whitespace"
            )
        expected_continuation = f" {record['gold_answer']}"
        if record.get("gold_continuation") != expected_continuation:
            raise ValueError("Exp03 CPT-test gold continuation is inconsistent")
        prompt_ids = _token_ids(tokenizer, prompt)
        target_ids = _token_ids(tokenizer, expected_continuation)
        combined_ids = _token_ids(tokenizer, prompt + expected_continuation)
        if not prompt_ids or not target_ids:
            raise ValueError("Exp03 CPT-test prompt and answer span must be non-empty")
        if combined_ids != prompt_ids + target_ids:
            raise ValueError("Exp03 CPT-test prompt and answer span are not token-aligned")
        sequence_ids = prompt_ids + target_ids
        if len(sequence_ids) > context_length:
            raise ValueError(
                f"CPT-test probe {record['id']} exceeds context length "
                f"{context_length}"
            )
        input_ids = torch_module.tensor(
            [sequence_ids], dtype=torch_module.long, device=device
        )
        with torch_module.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=torch_module.ones_like(input_ids),
            ).logits[0]
        answer_logits = logits[
            len(prompt_ids) - 1 : len(prompt_ids) + len(target_ids) - 1
        ].float()
        gold_tokens = torch_module.tensor(
            target_ids, dtype=torch_module.long, device=device
        )
        greedy_tokens = answer_logits.argmax(dim=-1)
        gold_log_probabilities = torch_module.log_softmax(
            answer_logits, dim=-1
        ).gather(1, gold_tokens.unsqueeze(1)).squeeze(1)
        first_logits = answer_logits[0]
        first_gold_logit = first_logits[gold_tokens[0]]
        first_token_rank = int(
            (first_logits > first_gold_logit).sum().item()
        ) + 1
        scored.append(
            {
                "id": record["id"],
                "answer_span_exact_match": bool(
                    torch_module.equal(greedy_tokens, gold_tokens)
                ),
                "first_token_correct": first_token_rank == 1,
                "gold_first_token_rank": first_token_rank,
                "gold_answer_log_probability": float(
                    gold_log_probabilities.sum().item()
                ),
                "gold_answer_nll": float(
                    -gold_log_probabilities.mean().item()
                ),
                "gold_answer_token_count": len(target_ids),
            }
        )
    return scored


def _summarize_cpt_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    count = len(predictions)
    if count == 0:
        raise ValueError("cannot summarize an empty CPT-test prediction set")
    answer_span_correct = sum(
        bool(record["answer_span_exact_match"]) for record in predictions
    )
    first_token_correct = sum(
        bool(record["first_token_correct"]) for record in predictions
    )
    strict_correct = sum(bool(record["strict_exact_match"]) for record in predictions)
    normalized_correct = sum(
        bool(record["normalized_exact_match"]) for record in predictions
    )
    log_probability_sum = sum(
        record["gold_answer_log_probability"] for record in predictions
    )
    return {
        "count": count,
        "answer_span_exact_match_correct": answer_span_correct,
        "answer_span_exact_match_accuracy": answer_span_correct / count,
        "first_token_correct": first_token_correct,
        "first_token_accuracy": first_token_correct / count,
        "gold_first_token_rank_mean": sum(
            record["gold_first_token_rank"] for record in predictions
        )
        / count,
        "gold_answer_log_probability_sum": log_probability_sum,
        "gold_answer_log_probability_mean": log_probability_sum / count,
        "gold_answer_nll_mean": sum(
            record["gold_answer_nll"] for record in predictions
        )
        / count,
        "gold_answer_token_count": sum(
            record["gold_answer_token_count"] for record in predictions
        ),
        "free_generation_strict_exact_match_correct": strict_correct,
        "free_generation_strict_exact_match_accuracy": strict_correct / count,
        "free_generation_normalized_exact_match_correct": normalized_correct,
        "free_generation_normalized_exact_match_accuracy": (
            normalized_correct / count
        ),
    }


def _build_cpt_metrics(
    predictions: list[dict[str, Any]], *, source_record_count: int
) -> dict[str, Any]:
    probe_counts = Counter(
        prediction.get("probe_type") for prediction in predictions
    )
    if set(probe_counts) != set(CPT_TEST_PROBE_TYPES) or any(
        probe_counts[probe_type] != source_record_count
        for probe_type in CPT_TEST_PROBE_TYPES
    ):
        raise RuntimeError(
            "CPT-test predictions do not contain one probe of each type per "
            "source record"
        )
    if len(predictions) != len(CPT_TEST_PROBE_TYPES) * source_record_count:
        raise RuntimeError("CPT-test prediction count is inconsistent")
    return {
        "primary_metric": "answer_span_exact_match",
        "overall": _summarize_cpt_predictions(predictions),
        "by_probe_type": {
            probe_type: _summarize_cpt_predictions(
                [
                    prediction
                    for prediction in predictions
                    if prediction["probe_type"] == probe_type
                ]
            )
            for probe_type in CPT_TEST_PROBE_TYPES
        },
    }


def _evaluate_cpt_test(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    qa: dict[str, Any],
    output_dir: Path,
    fact_count: int,
    layers: int,
) -> Path:
    probes = qa["cpt_test"]["probes"]
    cpt_test_manifest = qa["cpt_test"]["manifest"]
    records = [
        {
            **probe,
            "question": probe["prompt"],
            "hop": 0,
            "fact_type": "attribute",
            "source_entity_type": "continent",
            "target_entity_type": "continent",
            "target_field": probe["attribute"],
        }
        for probe in probes
    ]
    evaluation = config["evaluation"]
    tokenizer, model, torch_module = load_local_causal_lm(checkpoint)
    answer_span_scores = _score_cpt_answer_spans(
        records,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        context_length=evaluation["context_length"],
    )
    free_generation = generate_prediction_records(
        records,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        batch_size=evaluation["batch_size"],
        context_length=evaluation["context_length"],
        max_new_tokens=evaluation["max_new_tokens"],
        device="cuda",
        prompt_formatter=lambda prompt: prompt,
    )
    answer_span_by_id = {record["id"]: record for record in answer_span_scores}
    free_generation_ids = {record["id"] for record in free_generation}
    if (
        len(answer_span_by_id) != len(answer_span_scores)
        or len(free_generation_ids) != len(free_generation)
        or set(answer_span_by_id) != free_generation_ids
    ):
        raise RuntimeError("CPT-test answer-span and generation records differ")
    predictions = [
        {**record, **answer_span_by_id[record["id"]]}
        for record in free_generation
    ]
    source_record_count = cpt_test_manifest["source_cpt_record_count"]
    metrics = _build_cpt_metrics(
        predictions, source_record_count=source_record_count
    )
    tokenizer_identity = getattr(tokenizer, "name_or_path", None)
    model_identity = getattr(getattr(model, "config", None), "_name_or_path", None)
    output_dir.mkdir(parents=True)
    metadata_path = checkpoint / "training_metadata.json"
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "evaluation_config.json",
        {
            "experiment_name": EXP03_NAME,
            "evaluation_stage": "eval_cpt_test",
            "T": 1,
            "N": fact_count,
            "L": layers,
            "seed": config["experiment"]["seed"],
            "selected_tables": ["continent"],
            "test_dataset": "cpt_test",
            "cpt_test_method_version": cpt_test_manifest["method_version"],
            "cpt_test_manifest_sha256": hash_file(
                qa["root"] / "cpt_test" / "manifest.json"
            ),
            "cpt_test_probes_sha256": cpt_test_manifest["probes_sha256"],
            "cpt_test_probe_count": len(records),
            "checkpoint_path": str(checkpoint),
            "checkpoint_training_metadata_sha256": (
                hash_file(metadata_path) if metadata_path.is_file() else None
            ),
            "prompt_format": "raw_declarative_prefix",
            "natural_language_questions": False,
            "decoding": {"strategy": "greedy", "do_sample": False, "temperature": None},
            "context_length": evaluation["context_length"],
            "max_new_tokens": evaluation["max_new_tokens"],
            "batch_size": evaluation["batch_size"],
            "primary_metric": "answer_span_exact_match",
            "answer_span_evaluation": "teacher_forced_greedy_argmax",
            "answer_span_target": "space_prefixed_gold_answer",
            "secondary_diagnostics": [
                "free_generation_strict_exact_match",
                "free_generation_normalized_exact_match",
            ],
            "tokenizer_identity": tokenizer_identity or str(checkpoint.resolve()),
            "model_identity": model_identity or str(checkpoint.resolve()),
        },
    )
    return output_dir


def main() -> None:
    args = _parse_args()
    validate_exp3_fact_count(args.fact_count)
    timestamp = _timestamp()
    run_dir = RUN_ROOT / "pipeline_runs" / timestamp
    run_dir.mkdir(parents=True)
    state_path = run_dir / "pipeline_state.json"
    state: dict[str, Any] = {
        "experiment_name": EXP03_NAME,
        "status": "running",
        "created_at": _utc_iso(),
        "N": args.fact_count,
        "stage_order": [
            "cpt",
            "cpt_test",
            "attribute_sft",
            "inverse_sft",
            "attribute_test",
            "aggregation_test",
        ],
    }
    write_json(state_path, state)
    try:
        config = _resolved_config(args, run_dir)
        seed = config["experiment"]["seed"]
        layers = config.pop("_layers")
        base_model = config.pop("_base_model")

        dataset = _find_dataset(args.fact_count, seed)
        if dataset is None:
            dataset_dir = DATASET_ROOT / f"N{args.fact_count}_seed{seed}_{timestamp}"
            materialize_exp3_dataset(config, dataset_dir, fact_count=args.fact_count)
            dataset = verify_exp3_dataset(
                dataset_dir, fact_count=args.fact_count, seed=seed
            )
        state["dataset_path"] = str(dataset["root"])
        write_json(state_path, state)

        qa = _find_qa(dataset, fact_count=args.fact_count, seed=seed)
        if qa is None:
            qa_dir = QA_ROOT / dataset["root"].name
            generate_exp3_qa(
                dataset["root"], qa_dir, fact_count=args.fact_count, seed=seed
            )
            qa = verify_exp3_qa(
                qa_dir,
                dataset_dir=dataset["root"],
                fact_count=args.fact_count,
                seed=seed,
            )
        state["qa_path"] = str(qa["root"])
        state["inverse_sft_data_path"] = str(qa["inverse_sft_data_dir"])
        write_json(state_path, state)

        model_component = safe_component(args.model)
        stem = f"{model_component}_exp03_N{args.fact_count}_L{layers}_{timestamp}"
        cpt_checkpoint = TRAINED_MODELS_ROOT / stem
        cpt_summary = run_cpt_training(
            config,
            table_count=1,
            fact_count=args.fact_count,
            layers=layers,
            source_checkpoint=base_model,
            output_checkpoint=cpt_checkpoint,
            run_config_path=run_dir / "cpt" / "run_config.yaml",
            train_log_path=run_dir / "cpt" / "train_log.jsonl",
            database_path=dataset["database"],
            database_manifest_path=dataset["manifest_path"],
            readable_book_path=dataset["cpt_dir"] / "book_readable.txt",
            train_text_path=None,
            cpt_manifest_path=dataset["cpt_manifest"],
        )
        if cpt_summary.get("experiment") != EXP03_NAME:
            raise RuntimeError("CPT checkpoint has the wrong experiment identity")
        state["cpt_checkpoint_path"] = str(cpt_checkpoint)

        condition_results = RESULT_ROOT / f"N{args.fact_count}" / f"seed{seed}" / timestamp
        cpt_test_result = _evaluate_cpt_test(
            config,
            checkpoint=cpt_checkpoint,
            qa=qa,
            output_dir=condition_results / "cpt_test",
            fact_count=args.fact_count,
            layers=layers,
        )
        state["cpt_test_result_path"] = str(cpt_test_result)
        write_json(state_path, state)

        sft_checkpoint = TRAINED_MODELS_ROOT / f"{stem}_sft"
        sft_summary = run_target_sft_training(
            config,
            table_count=1,
            fact_count=args.fact_count,
            layers=layers,
            source_checkpoint=cpt_checkpoint,
            output_checkpoint=sft_checkpoint,
            run_config_path=run_dir / "target_sft" / "run_config.yaml",
            train_log_path=run_dir / "target_sft" / "train_log.jsonl",
            qa_condition_dir=qa["root"],
        )
        if sft_summary.get("experiment") != EXP03_NAME:
            raise RuntimeError("SFT checkpoint has the wrong experiment identity")
        state["sft_checkpoint_path"] = str(sft_checkpoint)
        state["attribute_sft_checkpoint_path"] = str(sft_checkpoint)
        write_json(state_path, state)

        inverse_sft_checkpoint = TRAINED_MODELS_ROOT / f"{stem}_inverse_sft"
        inverse_sft_summary = run_target_sft_training(
            config,
            table_count=1,
            fact_count=args.fact_count,
            layers=layers,
            source_checkpoint=sft_checkpoint,
            output_checkpoint=inverse_sft_checkpoint,
            run_config_path=run_dir / "inverse_sft" / "run_config.yaml",
            train_log_path=run_dir / "inverse_sft" / "train_log.jsonl",
            qa_condition_dir=qa["root"],
            dataset_dir=EXP3_INVERSE_SFT_DATASET_DIR,
        )
        if inverse_sft_summary.get("experiment") != EXP03_NAME:
            raise RuntimeError("inverse-SFT checkpoint has the wrong experiment identity")
        if Path(inverse_sft_summary.get("source_checkpoint", "")).resolve() != (
            sft_checkpoint.resolve()
        ):
            raise RuntimeError(
                "inverse SFT did not start from the attribute-SFT checkpoint"
            )
        state["inverse_sft_checkpoint_path"] = str(inverse_sft_checkpoint)
        state["final_checkpoint_path"] = str(inverse_sft_checkpoint)
        state["final_evaluation_checkpoint_path"] = str(inverse_sft_checkpoint)
        write_json(state_path, state)

        attribute_metrics, attribute_test_result = _evaluate(
            config,
            checkpoint=inverse_sft_checkpoint,
            qa_root=qa["root"],
            dataset_name="attribute_test",
            output_dir=condition_results / "sft_attribute_test",
            stage="eval_sft_attribute_test",
            fact_count=args.fact_count,
            layers=layers,
        )
        aggregation_metrics, aggregation_test_result = _evaluate(
            config,
            checkpoint=inverse_sft_checkpoint,
            qa_root=qa["root"],
            dataset_name="aggregation_test",
            output_dir=condition_results / "sft_aggregation_test",
            stage="eval_sft_aggregation_test",
            fact_count=args.fact_count,
            layers=layers,
            include_unordered=True,
        )
        attribute_em = attribute_metrics["overall"][
            "normalized_exact_match_accuracy"
        ]
        aggregation_em = aggregation_metrics["overall"][
            "normalized_exact_match_accuracy"
        ]
        aggregation_unordered_em = aggregation_metrics["unordered_exact_match"][
            "unordered_normalized_exact_match_accuracy"
        ]
        best_checkpoint = (
            TRAINED_MODELS_ROOT
            / "exp03_best"
            / model_component
            / f"N{args.fact_count}_L{layers}_seed{seed}"
            / "checkpoint"
        )
        best_metadata = {
            "experiment": EXP03_NAME,
            "N": args.fact_count,
            "model": args.model,
            "layers": layers,
            "epochs": config["target_sft"]["epochs"],
            "attribute_sft_epochs": config["target_sft"]["epochs"],
            "inverse_sft_epochs": config["target_sft"]["epochs"],
            "cpt_epochs": config["training"]["cpt_epochs"],
            "seed": seed,
            "EM": aggregation_em,
            "best_by_test_em": aggregation_em,
            "attribute_test_em": attribute_em,
            "aggregation_test_em": aggregation_em,
            "aggregation_unordered_em": aggregation_unordered_em,
            "checkpoint_source": str(inverse_sft_checkpoint),
            "attribute_sft_checkpoint_source": str(sft_checkpoint),
            "run": str(run_dir),
            "attribute_test_result": str(attribute_test_result),
            "aggregation_test_result": str(aggregation_test_result),
        }
        improved = retain_best_exp3_checkpoint(
            inverse_sft_checkpoint, best_checkpoint, best_metadata
        )
        state.update(
            {
                "sft_attribute_test_result_path": str(attribute_test_result),
                "sft_aggregation_test_result_path": str(aggregation_test_result),
                "best_checkpoint_path": str(best_checkpoint),
                "best_checkpoint_updated": improved,
                "attribute_test_normalized_exact_match": attribute_em,
                "aggregation_test_normalized_exact_match": aggregation_em,
                "aggregation_test_unordered_exact_match": aggregation_unordered_em,
                "status": "completed",
                "completed_at": _utc_iso(),
            }
        )
        write_json(state_path, state)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(state_path, state)
        raise

    print("Experiment 3 complete")
    print(f"Dataset: {state['dataset_path']}")
    print(f"QA: {state['qa_path']}")
    print(f"SFT checkpoint: {state['sft_checkpoint_path']}")
    print(f"Inverse SFT checkpoint: {state['inverse_sft_checkpoint_path']}")
    print(f"CPT test result: {state['cpt_test_result_path']}")
    print(f"Attribute test result: {state['sft_attribute_test_result_path']}")
    print(f"Aggregation test result: {state['sft_aggregation_test_result_path']}")
    print(f"Best checkpoint: {state['best_checkpoint_path']}")
    print(f"Best checkpoint updated: {state['best_checkpoint_updated']}")
    print(f"Run state: {state_path}")


if __name__ == "__main__":
    main()

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from evaluation.metrics import score_prediction
from utils.hashing import hash_file
from utils.io import read_json, read_jsonl

PROMPT_TEMPLATE = "Question:\n{question}\n\nAnswer:"
MAX_NEW_TOKENS = 64
EVALUATION_SEED = 2025
HOP_NAMES = ("H0", "H1", "H2", "H3")


class QAArtifactError(ValueError):
    """Raised when closed-book QA files fail provenance validation."""


def format_question_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return PROMPT_TEMPLATE.format(question=question)


def extract_first_nonempty_line(raw_generation: str) -> str:
    if not isinstance(raw_generation, str):
        raise TypeError("raw generation must be text")
    for line in raw_generation.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _require_nonempty_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _validate_qa_record(record: dict[str, Any], split: str, hop: int) -> None:
    required = {
        "id",
        "split",
        "hop",
        "question",
        "gold_answer",
        "source_entity_type",
        "target_entity_type",
        "target_field",
    }
    if hop == 0:
        required.add("fact_type")
    else:
        required.add("support_fact_ids")
    missing = required - record.keys()
    if missing:
        raise QAArtifactError(
            f"H{hop} QA record is missing fields: {', '.join(sorted(missing))}"
        )
    if record["split"] != split or record["hop"] != hop:
        raise QAArtifactError(f"H{hop} QA record split/hop metadata is inconsistent")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise QAArtifactError(f"H{hop} QA question must be non-empty text")
    if not isinstance(record["gold_answer"], str):
        raise QAArtifactError(f"H{hop} QA gold answer must be text")
    if "context" in record:
        raise QAArtifactError("closed-book QA must not contain a context field")
    if hop == 0 and record["fact_type"] not in {"attribute", "relation"}:
        raise QAArtifactError("H0 fact_type must be attribute or relation")
    if hop > 0:
        support_ids = record["support_fact_ids"]
        if not isinstance(support_ids, list) or len(support_ids) != hop + 1:
            raise QAArtifactError(f"H{hop} QA support count is invalid")


def load_verified_qa_split(
    split_dir: str | Path,
    *,
    split: str,
    expected_table_count: int,
    expected_fact_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    split_dir = Path(split_dir)
    manifest_path = _require_nonempty_file(
        split_dir / "manifest.json", "QA split manifest"
    )
    root_manifest_path = _require_nonempty_file(
        split_dir.parent / "split_manifest.json", "QA chain-split manifest"
    )
    manifest = read_json(manifest_path)
    root_manifest = read_json(root_manifest_path)
    manifest_sha256 = hash_file(manifest_path)
    if root_manifest.get(f"{split}_manifest_sha256") != manifest_sha256:
        raise QAArtifactError("QA split manifest hash does not match split_manifest.json")
    for artifact, label in ((manifest, "QA manifest"), (root_manifest, "split manifest")):
        if artifact.get("T") != expected_table_count:
            raise QAArtifactError(f"{label} T metadata does not match")
        if artifact.get("requested_N") != expected_fact_count:
            raise QAArtifactError(f"{label} N metadata does not match")
        if artifact.get("zero_context") is not True:
            raise QAArtifactError(f"{label} is not marked zero-context")
    if manifest.get("split") != split:
        raise QAArtifactError("QA manifest split metadata does not match")

    expected_hashes = manifest.get("output_file_hashes")
    counts = manifest.get("counts")
    if not isinstance(expected_hashes, dict) or not isinstance(counts, dict):
        raise QAArtifactError("QA manifest hashes or counts are missing")
    records: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    records_by_hop: dict[str, list[dict[str, Any]]] = {}
    for hop, hop_name in enumerate(HOP_NAMES):
        path = _require_nonempty_file(split_dir / f"{hop_name}.jsonl", hop_name)
        actual_hash = hash_file(path)
        if expected_hashes.get(path.name) != actual_hash:
            raise QAArtifactError(f"{hop_name} hash does not match the QA manifest")
        hop_records = read_jsonl(path)
        expected_count = counts.get(hop_name, {}).get("final_retained_count")
        if len(hop_records) != expected_count:
            raise QAArtifactError(f"{hop_name} count does not match the QA manifest")
        for record in hop_records:
            _validate_qa_record(record, split, hop)
            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise QAArtifactError("QA record IDs must be non-empty and unique")
            seen_ids.add(record_id)
        records_by_hop[hop_name] = hop_records
        records.extend(hop_records)
        input_hashes[hop_name] = actual_hash

    h0_ids = {record["id"] for record in records_by_hop["H0"]}
    for hop_name in HOP_NAMES[1:]:
        for record in records_by_hop[hop_name]:
            if any(
                support_id not in h0_ids
                for support_id in record["support_fact_ids"]
            ):
                raise QAArtifactError(
                    f"{hop_name} record references an unavailable H0 support"
                )
    if len(records) != manifest.get("final_retained_total"):
        raise QAArtifactError("QA total count does not match the manifest")
    return records, {
        "qa_manifest": manifest,
        "qa_manifest_sha256": manifest_sha256,
        "qa_split_manifest_sha256": hash_file(root_manifest_path),
        "input_hashes": input_hashes,
    }


def load_local_causal_lm(checkpoint: str | Path) -> tuple[Any, Any, Any]:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"local checkpoint is missing: {checkpoint}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("evaluation requires torch and transformers") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("closed-book evaluation requires CUDA")
    torch.manual_seed(EVALUATION_SEED)
    torch.cuda.manual_seed_all(EVALUATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return tokenizer, model, torch


def _token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def generate_prediction_records(
    qa_records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    batch_size: int,
    context_length: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    for value, name in (
        (batch_size, "batch_size"),
        (context_length, "context_length"),
        (max_new_tokens, "max_new_tokens"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not qa_records:
        raise ValueError("cannot evaluate an empty QA set")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = [format_question_prompt(record["question"]) for record in qa_records]
    prompt_lengths = [_token_count(tokenizer, prompt) for prompt in prompts]
    too_long = [
        (record["id"], length)
        for record, length in zip(qa_records, prompt_lengths, strict=True)
        if length > context_length
    ]
    if too_long:
        record_id, length = too_long[0]
        raise ValueError(
            f"prompt {record_id} has {length} tokens and exceeds context length "
            f"{context_length}; truncation is forbidden"
        )

    model.eval()
    prediction_records: list[dict[str, Any]] = []
    inference_context = getattr(torch_module, "inference_mode", None)
    for start in range(0, len(qa_records), batch_size):
        batch_records = qa_records[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device) for key, value in encoded.items()
        }
        prompt_width = encoded["input_ids"].shape[1]
        context_manager = inference_context() if inference_context else nullcontext()
        with context_manager:
            generated = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row_index, qa_record in enumerate(batch_records):
            continuation_ids = generated[row_index, prompt_width:]
            raw_generation = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            prediction = extract_first_nonempty_line(raw_generation)
            prediction_records.append(
                score_prediction(qa_record, raw_generation, prediction)
            )
    return prediction_records


def evaluate_with_local_checkpoint(
    qa_records: list[dict[str, Any]],
    *,
    checkpoint: str | Path,
    batch_size: int,
    context_length: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer, model, torch_module = load_local_causal_lm(checkpoint)
    predictions = generate_prediction_records(
        qa_records,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        batch_size=batch_size,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        device="cuda",
    )
    tokenizer_identity = getattr(tokenizer, "name_or_path", None)
    model_identity = getattr(getattr(model, "config", None), "_name_or_path", None)
    return predictions, {
        "tokenizer_identity": tokenizer_identity or str(Path(checkpoint).resolve()),
        "model_identity": model_identity or str(Path(checkpoint).resolve()),
    }


def prepare_result_directory(path: str | Path) -> Path:
    output_dir = Path(path)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"evaluation output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty evaluation directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    return output_dir

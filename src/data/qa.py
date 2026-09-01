import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from data.serialize import read_semantic_database
from data.world import NATURAL_IDENTIFIER_FIELDS, SEMANTIC_ENTITY_SPECS
from utils.hashing import hash_file, hash_json_object
from utils.io import read_json, write_json, write_jsonl

QA_FORMAT_VERSION = 2
QUESTION_TEMPLATE_VERSION = "semantic_academic_closed_book_v1"
SPLIT_METHOD_VERSION = "chain_order_3_reserved_1_validation_1_test_v1"
ID_DIGEST_LENGTH = 32
HOP_NAMES = ("H0", "H1", "H2", "H3")

RAW_ENTITY_IDENTIFIER = re.compile(
    r"\b(?:CTN|CTR|REG|CTY|CAM|SCH|DEP|SUB|CRS|OFF|ENR|STU)\d+\b"
)

H0_ATTRIBUTE_FIELDS: dict[str, tuple[str, ...]] = {
    "continent": ("climate_band",),
    "country": ("currency_name",),
    "region": ("administrative_type",),
    "city": ("population_band",),
    "campus": ("campus_type",),
    "school": ("founding_period",),
    "department": ("focus_area",),
    "subject": ("subject_level", "discipline_group"),
    "course": ("credit_hours", "delivery_mode"),
    "course_offering": ("meeting_period", "room_label"),
    "enrollment": ("final_grade", "enrollment_status"),
    "student": ("study_year", "scholarship_status"),
}

# Offering and enrollment relations are intentionally excluded because their
# composite natural anchors already name their targets.
H0_RELATION_SOURCE_POSITIONS = (1, 2, 3, 4, 5, 6, 7, 8, 11)
RELATIONAL_SOURCE_POSITIONS = {
    1: (1, 2, 3, 4, 5, 6, 7, 8, 11),
    2: (2, 3, 4, 5, 6, 7, 8),
    3: (3, 4, 5, 6, 7, 8),
}
TARGET_FIELD_BY_POSITION = {
    0: "climate_band",
    1: "currency_name",
    2: "administrative_type",
    3: "population_band",
    4: "campus_type",
    5: "founding_period",
    6: "focus_area",
    7: "discipline_group",
    10: "enrollment_status",
}

ATTRIBUTE_QUESTION_TEMPLATES = {
    ("continent", "climate_band"): "What climate band does {source} have?",
    ("country", "currency_name"): "Which currency does {source} use?",
    ("region", "administrative_type"): "What administrative type is {source}?",
    ("city", "population_band"): "What population band does {source} have?",
    ("campus", "campus_type"): "What type of campus is {source}?",
    ("school", "founding_period"): "During which period was {source} founded?",
    ("department", "focus_area"): "What focus area does {source} have?",
    ("subject", "subject_level"): "What subject level does {source} have?",
    ("subject", "discipline_group"): (
        "Which discipline group does {source} belong to?"
    ),
    ("course", "credit_hours"): "How many credit hours is {source} worth?",
    ("course", "delivery_mode"): "What delivery mode does {source} use?",
    ("course_offering", "meeting_period"): "When does {source} meet?",
    ("course_offering", "room_label"): "Where does {source} meet?",
    ("enrollment", "final_grade"): (
        "Which final grade was recorded for the {source}?"
    ),
    ("enrollment", "enrollment_status"): "What status does the {source} have?",
    ("student", "study_year"): "What study year is {source} in?",
    ("student", "scholarship_status"): (
        "What scholarship status does {source} have?"
    ),
}

RELATION_QUESTION_TEMPLATES = {
    "country": "Which continent does {source} belong to?",
    "region": "Which country contains {source}?",
    "city": "Which region is {source} located in?",
    "campus": "Which city is {source} located in?",
    "school": "Which campus is {source} located at?",
    "department": "Which school does {source} belong to?",
    "subject": "Which department does {source} belong to?",
    "course": "Which subject does {source} belong to?",
    "student": "What is {source_possessive} primary enrollment?",
}

RELATIONAL_QUESTION_TEMPLATES = {
    (1, "country"): "What climate band does the continent containing {source} have?",
    (1, "region"): "Which currency is used by the country containing {source}?",
    (1, "city"): (
        "What administrative type does the region containing {source} have?"
    ),
    (1, "campus"): (
        "What population band does the city where {source} is located have?"
    ),
    (1, "school"): "What type is the campus where {source} is located?",
    (1, "department"): (
        "During which period was the school containing {source} founded?"
    ),
    (1, "subject"): (
        "What focus area does the department containing {source} have?"
    ),
    (1, "course"): (
        "Which discipline group contains the subject to which {source} belongs?"
    ),
    (1, "student"): "What status does {source_possessive} primary enrollment have?",
    (2, "region"): (
        "What climate band does the continent containing the country that contains "
        "{source} have?"
    ),
    (2, "city"): (
        "Which currency is used by the country containing the region where {source} "
        "is located?"
    ),
    (2, "campus"): (
        "What administrative type does the region containing the city where {source} "
        "is located have?"
    ),
    (2, "school"): (
        "What population band does the city containing the campus where {source} "
        "is located have?"
    ),
    (2, "department"): (
        "What type is the campus hosting the school that contains {source}?"
    ),
    (2, "subject"): (
        "During which period was the school containing the department that contains "
        "{source} founded?"
    ),
    (2, "course"): (
        "What focus area does the department containing the subject to which {source} "
        "belongs have?"
    ),
    (3, "city"): (
        "What climate band does the continent containing the country that contains "
        "the region where {source} is located have?"
    ),
    (3, "campus"): (
        "Which currency is used by the country containing the region containing the "
        "city where {source} is located?"
    ),
    (3, "school"): (
        "What administrative type does the region containing the city containing the "
        "campus where {source} is located have?"
    ),
    (3, "department"): (
        "What population band does the city containing the campus hosting the school "
        "that contains {source} have?"
    ),
    (3, "subject"): (
        "What type is the campus hosting the school containing the department that "
        "contains {source}?"
    ),
    (3, "course"): (
        "During which period was the school containing the department containing the "
        "subject to which {source} belongs founded?"
    ),
}


def _stable_id(prefix: str, semantic_key: tuple[Any, ...]) -> str:
    digest = hash_json_object(list(semantic_key))[:ID_DIGEST_LENGTH]
    return f"{prefix}_{digest}"


def _possessive(value: str) -> str:
    return f"{value}'" if value.endswith("s") else f"{value}'s"


def normalize_for_leakage(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def answer_is_in_question(question: str, gold_answer: str) -> bool:
    normalized_question = normalize_for_leakage(question)
    normalized_answer = normalize_for_leakage(gold_answer)
    if not normalized_answer:
        raise ValueError("gold answer must remain non-empty after normalization")
    return f" {normalized_answer} " in f" {normalized_question} "


def assign_chain_splits(chain_count: int) -> dict[str, list[int]]:
    if isinstance(chain_count, bool) or not isinstance(chain_count, int) or chain_count <= 0:
        raise ValueError("chain_count must be a positive integer")
    if chain_count % 5:
        raise ValueError("chain_count must be divisible by five for the nested split")
    assignments = {"reserved": [], "validation": [], "test": []}
    for chain_index in range(chain_count):
        offset = chain_index % 5
        split = "reserved" if offset < 3 else "validation" if offset == 3 else "test"
        assignments[split].append(chain_index)
    return assignments


def _verify_database_manifest(
    database_path: Path,
    database_manifest_path: Path,
    *,
    expected_table_count: int,
    expected_logical_fact_count: int | None,
) -> dict[str, Any]:
    for path, label in (
        (database_path, "database"),
        (database_manifest_path, "database manifest"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
    manifest = read_json(database_manifest_path)
    database_sha256 = hash_file(database_path)
    if manifest.get("database_sha256") != database_sha256:
        raise ValueError("database does not match its manifest hash")
    table_count = manifest.get("table_count")
    requested_n = manifest.get("requested_N")
    if table_count != expected_table_count or manifest.get("T") != table_count:
        raise ValueError("database table count does not match the requested condition")
    if expected_logical_fact_count is not None and requested_n != expected_logical_fact_count:
        raise ValueError("database logical fact count does not match the condition")
    if requested_n != manifest.get("actual_logical_fact_count"):
        raise ValueError("database manifest logical fact counts are inconsistent")
    if manifest.get("latent_positions") != len(SEMANTIC_ENTITY_SPECS):
        raise ValueError("database manifest does not describe the academic chain")
    if manifest.get("attribute_facts_per_chain") != 29:
        raise ValueError("database manifest must contain 29 attributes per chain")
    if manifest.get("relation_facts_per_chain") != 11:
        raise ValueError("database manifest must contain 11 relations per chain")
    if manifest.get("experimental_facts_per_chain") != 40:
        raise ValueError("database manifest must contain 40 logical facts per chain")
    connection = sqlite3.connect(database_path)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SQLite integrity check failed")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"foreign-key violations: {violations}")
    finally:
        connection.close()
    return manifest


def load_verified_semantic_chains(
    database_path: str | Path,
    database_manifest_path: str | Path,
    *,
    expected_table_count: int,
    expected_logical_fact_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database_path = Path(database_path)
    database_manifest_path = Path(database_manifest_path)
    manifest = _verify_database_manifest(
        database_path,
        database_manifest_path,
        expected_table_count=expected_table_count,
        expected_logical_fact_count=expected_logical_fact_count,
    )
    _, entities_by_position, metadata = read_semantic_database(
        database_path, manifest
    )
    chain_count = manifest.get("selected_chain_count")
    if isinstance(chain_count, bool) or not isinstance(chain_count, int) or chain_count <= 0:
        raise ValueError("database manifest selected_chain_count is invalid")
    ordered_positions = [
        list(entities_by_position[position].values())
        for position in range(len(SEMANTIC_ENTITY_SPECS))
    ]
    if any(len(entities) != chain_count for entities in ordered_positions):
        raise ValueError("semantic entity counts do not match the chain count")
    chains: list[dict[str, Any]] = []
    for chain_index in range(chain_count):
        entities = [entities[chain_index] for entities in ordered_positions]
        for position in range(1, len(entities)):
            if entities[position]["relation_target_id"] != entities[position - 1]["entity_id"]:
                raise ValueError("semantic rows are not aligned in deterministic chain order")
        chains.append({"chain_index": chain_index, "entities": entities})
    if metadata["source_identifier_count"] != chain_count * len(SEMANTIC_ENTITY_SPECS):
        raise ValueError("database identifier count does not match complete chains")
    if chain_count * 40 != manifest["requested_N"]:
        raise ValueError("chain count does not match requested logical facts")
    return chains, manifest


def _attribute_question(entity: dict[str, Any], field: str) -> str:
    template = ATTRIBUTE_QUESTION_TEMPLATES[(entity["entity_type"], field)]
    return template.format(source=entity["natural_anchor"])


def _relation_question(entity: dict[str, Any]) -> str:
    template = RELATION_QUESTION_TEMPLATES[entity["entity_type"]]
    source = entity["natural_anchor"]
    return template.format(source=source, source_possessive=_possessive(source))


def _relational_question(hop: int, entity: dict[str, Any]) -> str:
    template = RELATIONAL_QUESTION_TEMPLATES[(hop, entity["entity_type"])]
    source = entity["natural_anchor"]
    return template.format(source=source, source_possessive=_possessive(source))


def _target_anchor_field(entity_type: str) -> str:
    return NATURAL_IDENTIFIER_FIELDS.get(entity_type, "natural_anchor")


def _assert_model_facing_text_is_safe(
    question: str,
    gold_answer: str,
    raw_identifiers: set[str],
) -> None:
    for value in (question, gold_answer):
        identifier = RAW_ENTITY_IDENTIFIER.search(value)
        if identifier or any(raw_id in value for raw_id in raw_identifiers):
            exposed = identifier.group(0) if identifier else "database identifier"
            raise ValueError(f"model-facing QA exposed raw identifier: {exposed}")


def _assert_relational_depth_is_not_shortened(
    question: str,
    path_entities: list[dict[str, Any]],
) -> None:
    normalized_question = f" {normalize_for_leakage(question)} "
    for entity in path_entities[1:]:
        anchor = normalize_for_leakage(entity["natural_anchor"])
        if f" {anchor} " in normalized_question:
            raise ValueError(
                "relational question exposes an intermediate or target entity anchor"
            )


def generate_qa_candidates(
    chains: list[dict[str, Any]],
    chain_indices: list[int],
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    records = {hop_name: [] for hop_name in HOP_NAMES}
    all_ids: set[str] = set()
    raw_identifiers = {
        entity["entity_id"]
        for chain_index in chain_indices
        for entity in chains[chain_index]["entities"]
    }

    def add_record(hop_name: str, record: dict[str, Any]) -> None:
        if record["id"] in all_ids:
            raise ValueError(f"duplicate or colliding QA ID: {record['id']}")
        _assert_model_facing_text_is_safe(
            record["question"], record["gold_answer"], raw_identifiers
        )
        all_ids.add(record["id"])
        records[hop_name].append(record)

    for chain_index in chain_indices:
        entities = chains[chain_index]["entities"]
        attribute_support_ids: dict[tuple[str, str], str] = {}
        relation_support_ids: dict[str, str] = {}
        for entity in entities:
            entity_type = entity["entity_type"]
            for field in H0_ATTRIBUTE_FIELDS[entity_type]:
                semantic_key = ("h0", "attribute", entity["entity_id"], field)
                record_id = _stable_id("h0_attr", semantic_key)
                attribute_support_ids[(entity["entity_id"], field)] = record_id
                add_record(
                    "H0",
                    {
                        "id": record_id,
                        "split": split,
                        "hop": 0,
                        "question": _attribute_question(entity, field),
                        "gold_answer": str(entity["attributes"][field]),
                        "fact_type": "attribute",
                        "source_entity_type": entity_type,
                        "target_entity_type": entity_type,
                        "target_field": field,
                    },
                )
        for source_position in H0_RELATION_SOURCE_POSITIONS:
            source = entities[source_position]
            target = entities[source_position - 1]
            semantic_key = ("h0", "relation", source["entity_id"])
            record_id = _stable_id("h0_rel", semantic_key)
            relation_support_ids[source["entity_id"]] = record_id
            add_record(
                "H0",
                {
                    "id": record_id,
                    "split": split,
                    "hop": 0,
                    "question": _relation_question(source),
                    "gold_answer": target["natural_anchor"],
                    "fact_type": "relation",
                    "source_entity_type": source["entity_type"],
                    "target_entity_type": target["entity_type"],
                    "target_field": _target_anchor_field(target["entity_type"]),
                },
            )

        for hop, source_positions in RELATIONAL_SOURCE_POSITIONS.items():
            for source_position in source_positions:
                target_position = 10 if source_position == 11 else source_position - hop
                target_field = TARGET_FIELD_BY_POSITION[target_position]
                source = entities[source_position]
                target = entities[target_position]
                path_positions = (
                    [11, 10]
                    if source_position == 11
                    else list(range(source_position, target_position - 1, -1))
                )
                path_entities = [entities[position] for position in path_positions]
                question = _relational_question(hop, source)
                _assert_relational_depth_is_not_shortened(question, path_entities)
                support_fact_ids = [
                    relation_support_ids[entities[position]["entity_id"]]
                    for position in path_positions[:-1]
                ]
                support_fact_ids.append(
                    attribute_support_ids[(target["entity_id"], target_field)]
                )
                semantic_key = (
                    f"h{hop}",
                    source["entity_id"],
                    target["entity_id"],
                    target_field,
                )
                add_record(
                    f"H{hop}",
                    {
                        "id": _stable_id(f"h{hop}", semantic_key),
                        "split": split,
                        "hop": hop,
                        "question": question,
                        "gold_answer": str(target["attributes"][target_field]),
                        "source_entity_type": source["entity_type"],
                        "target_entity_type": target["entity_type"],
                        "target_field": target_field,
                        "support_fact_ids": support_fact_ids,
                    },
                )
    return records


def filter_qa_candidates(
    candidates: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if set(candidates) != set(HOP_NAMES):
        raise ValueError("candidates must contain exactly H0, H1, H2, and H3")
    after_leakage: dict[str, list[dict[str, Any]]] = {
        hop_name: [] for hop_name in HOP_NAMES
    }
    exclusions: list[dict[str, Any]] = []
    leakage_counts: dict[str, int] = {}
    for hop_name in HOP_NAMES:
        for record in candidates[hop_name]:
            if answer_is_in_question(record["question"], record["gold_answer"]):
                exclusions.append(
                    {
                        "id": record["id"],
                        "hop": hop_name,
                        "reason": "gold_answer_contained_in_normalized_question",
                    }
                )
            else:
                after_leakage[hop_name].append(record)
        leakage_counts[hop_name] = len(candidates[hop_name]) - len(
            after_leakage[hop_name]
        )

    retained_h0_ids = {record["id"] for record in after_leakage["H0"]}
    retained: dict[str, list[dict[str, Any]]] = {
        "H0": after_leakage["H0"],
        "H1": [],
        "H2": [],
        "H3": [],
    }
    support_closure_counts = {hop_name: 0 for hop_name in HOP_NAMES}
    for hop_name in HOP_NAMES[1:]:
        for record in after_leakage[hop_name]:
            missing = [
                support_id
                for support_id in record["support_fact_ids"]
                if support_id not in retained_h0_ids
            ]
            if missing:
                support_closure_counts[hop_name] += 1
                exclusions.append(
                    {
                        "id": record["id"],
                        "hop": hop_name,
                        "reason": "required_h0_support_not_retained",
                        "missing_support_fact_ids": missing,
                    }
                )
            else:
                retained[hop_name].append(record)

    validate_retained_records(retained)
    counts = {}
    for hop_name in HOP_NAMES:
        counts[hop_name] = {
            "candidate_count": len(candidates[hop_name]),
            "leakage_filtered_count": leakage_counts[hop_name],
            "post_leakage_count": len(after_leakage[hop_name]),
            "support_closure_filtered_count": support_closure_counts[hop_name],
            "final_retained_count": len(retained[hop_name]),
        }
    return retained, {"counts": counts, "excluded_items": exclusions}


def validate_retained_records(
    records: dict[str, list[dict[str, Any]]],
) -> None:
    h0_ids = {record["id"] for record in records["H0"]}
    if len(h0_ids) != len(records["H0"]):
        raise ValueError("duplicate H0 IDs")
    all_ids = set(h0_ids)
    for record in records["H0"]:
        if answer_is_in_question(record["question"], record["gold_answer"]):
            raise ValueError("retained H0 record contains its normalized answer")
        if "support_fact_ids" in record:
            raise ValueError("H0 records must not contain support facts")
    for hop in (1, 2, 3):
        for record in records[f"H{hop}"]:
            if record["id"] in all_ids:
                raise ValueError("duplicate or colliding QA ID")
            all_ids.add(record["id"])
            support_ids = record.get("support_fact_ids")
            if not isinstance(support_ids, list) or len(support_ids) != hop + 1:
                raise ValueError(f"H{hop} record has an invalid support count")
            if any(support_id not in h0_ids for support_id in support_ids):
                raise ValueError(f"H{hop} record references an unknown H0 support")
            if answer_is_in_question(record["question"], record["gold_answer"]):
                raise ValueError(f"retained H{hop} record contains its normalized answer")
    for hop_name in HOP_NAMES:
        for record in records[hop_name]:
            if "context" in record:
                raise ValueError("closed-book QA records must not contain context")


def _split_manifest(
    *,
    config: dict[str, Any],
    database_path: Path,
    database_manifest_path: Path,
    database_manifest: dict[str, Any],
    assignments: dict[str, list[int]],
) -> dict[str, Any]:
    return {
        "format_version": QA_FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "T": database_manifest["T"],
        "requested_N": database_manifest["requested_N"],
        "source_database_sha256": hash_file(database_path),
        "source_database_manifest_sha256": hash_file(database_manifest_path),
        "split_method": "repeating chain-order blocks: reserved, reserved, reserved, validation, test",
        "split_method_version": SPLIT_METHOD_VERSION,
        "question_template_version": QUESTION_TEMPLATE_VERSION,
        "total_chain_count": database_manifest["selected_chain_count"],
        "reserved_chain_count": len(assignments["reserved"]),
        "validation_chain_count": len(assignments["validation"]),
        "test_chain_count": len(assignments["test"]),
        "reserved_chain_indices": assignments["reserved"],
        "validation_chain_indices": assignments["validation"],
        "test_chain_indices": assignments["test"],
        "chain_assignments_sha256": hash_json_object(assignments),
        "target_qa_training_generated": False,
        "zero_context": True,
    }


def _write_split(
    *,
    output_dir: Path,
    split: str,
    records: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    base_manifest: dict[str, Any],
    chain_indices: list[int],
) -> dict[str, Any]:
    split_dir = output_dir / split
    paths = {hop_name: split_dir / f"{hop_name}.jsonl" for hop_name in HOP_NAMES}
    for hop_name, path in paths.items():
        write_jsonl(path, records[hop_name])
    manifest = {
        "format_version": QA_FORMAT_VERSION,
        "experiment_name": base_manifest["experiment_name"],
        "T": base_manifest["T"],
        "requested_N": base_manifest["requested_N"],
        "split": split,
        "chain_count": len(chain_indices),
        "chain_indices": chain_indices,
        "reserved_chain_count": base_manifest["reserved_chain_count"],
        "validation_chain_count": base_manifest["validation_chain_count"],
        "test_chain_count": base_manifest["test_chain_count"],
        "source_database_sha256": base_manifest["source_database_sha256"],
        "source_database_manifest_sha256": base_manifest[
            "source_database_manifest_sha256"
        ],
        "split_method": base_manifest["split_method"],
        "split_method_version": SPLIT_METHOD_VERSION,
        "question_template_version": QUESTION_TEMPLATE_VERSION,
        "zero_context": True,
        "counts": audit["counts"],
        "candidate_total": sum(
            counts["candidate_count"] for counts in audit["counts"].values()
        ),
        "final_retained_total": sum(len(records[hop]) for hop in HOP_NAMES),
        "excluded_items": audit["excluded_items"],
        "output_file_hashes": {
            path.name: hash_file(path) for path in paths.values()
        },
    }
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def generate_condition_qa(
    config: dict[str, Any],
    database_path: str | Path,
    database_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    expected_table_count: int,
    expected_logical_fact_count: int | None = None,
) -> dict[str, Any]:
    database_path = Path(database_path)
    database_manifest_path = Path(database_manifest_path)
    output_dir = Path(output_dir)
    chains, database_manifest = load_verified_semantic_chains(
        database_path,
        database_manifest_path,
        expected_table_count=expected_table_count,
        expected_logical_fact_count=expected_logical_fact_count,
    )
    assignments = assign_chain_splits(len(chains))
    base_manifest = _split_manifest(
        config=config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        database_manifest=database_manifest,
        assignments=assignments,
    )
    split_manifests: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        candidates = generate_qa_candidates(
            chains, assignments[split], split
        )
        records, audit = filter_qa_candidates(candidates)
        split_manifests[split] = _write_split(
            output_dir=output_dir,
            split=split,
            records=records,
            audit=audit,
            base_manifest=base_manifest,
            chain_indices=assignments[split],
        )
    split_manifest = {
        **base_manifest,
        "validation_manifest_sha256": hash_file(
            output_dir / "validation" / "manifest.json"
        ),
        "test_manifest_sha256": hash_file(output_dir / "test" / "manifest.json"),
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    return {
        "split_manifest": split_manifest,
        "validation_manifest": split_manifests["validation"],
        "test_manifest": split_manifests["test"],
    }

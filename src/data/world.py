from math import gcd
from typing import Any

from config import validate_config
from utils.hashing import hash_text


FORMAT_VERSION = 2
IDENTIFIER_MODULUS = 1_000_000

# Logical semantic positions are ordered from the root of the chain to its leaf.
# These descriptors name logical entities and facts, not physical SQLite layouts.
SEMANTIC_ENTITY_SPECS: tuple[dict[str, Any], ...] = (
    {"entity_type": "continent", "id_field": "continent_id", "id_prefix": "CTN", "attributes": (("continent_name", "TEXT"), ("climate_band", "TEXT"))},
    {"entity_type": "country", "id_field": "country_id", "id_prefix": "CTR", "attributes": (("country_name", "TEXT"), ("currency_name", "TEXT")), "foreign_key": "continent_id"},
    {"entity_type": "region", "id_field": "region_id", "id_prefix": "REG", "attributes": (("region_name", "TEXT"), ("administrative_type", "TEXT")), "foreign_key": "country_id"},
    {"entity_type": "city", "id_field": "city_id", "id_prefix": "CTY", "attributes": (("city_name", "TEXT"), ("population_band", "TEXT")), "foreign_key": "region_id"},
    {"entity_type": "campus", "id_field": "campus_id", "id_prefix": "CAM", "attributes": (("campus_name", "TEXT"), ("campus_type", "TEXT")), "foreign_key": "city_id"},
    {"entity_type": "school", "id_field": "school_id", "id_prefix": "SCH", "attributes": (("school_name", "TEXT"), ("founding_period", "TEXT")), "foreign_key": "campus_id"},
    {"entity_type": "department", "id_field": "department_id", "id_prefix": "DEP", "attributes": (("department_name", "TEXT"), ("focus_area", "TEXT")), "foreign_key": "school_id"},
    {"entity_type": "subject", "id_field": "subject_id", "id_prefix": "SUB", "attributes": (("subject_name", "TEXT"), ("subject_level", "TEXT"), ("discipline_group", "TEXT")), "foreign_key": "department_id"},
    {"entity_type": "course", "id_field": "course_id", "id_prefix": "CRS", "attributes": (("course_title", "TEXT"), ("credit_hours", "INTEGER"), ("delivery_mode", "TEXT")), "foreign_key": "subject_id"},
    {"entity_type": "course_offering", "id_field": "offering_id", "id_prefix": "OFF", "attributes": (("section_label", "TEXT"), ("meeting_period", "TEXT"), ("room_label", "TEXT")), "foreign_key": "course_id"},
    {"entity_type": "enrollment", "id_field": "enrollment_id", "id_prefix": "ENR", "attributes": (("academic_term", "TEXT"), ("final_grade", "TEXT"), ("enrollment_status", "TEXT")), "foreign_key": "offering_id"},
    {"entity_type": "student", "id_field": "student_id", "id_prefix": "STU", "attributes": (("full_name", "TEXT"), ("study_year", "TEXT"), ("scholarship_status", "TEXT")), "foreign_key": "primary_enrollment_id"},
)


_STEMS = (
    "Alder", "Arden", "Briar", "Cedar", "Coral", "Dunmore", "Elm", "Falcon",
    "Glen", "Harbor", "Juniper", "Linden", "Maris", "Northwind", "Oak",
    "Pine", "Quartz", "Redwood", "Silver", "Willow",
)
_DIRECTIONS = ("Northern", "Southern", "Eastern", "Western", "Central", "Upper", "Lower")
_DISCIPLINES = (
    "Applied Mathematics", "Computing", "Economics", "Environmental Science",
    "History", "Linguistics", "Materials Science", "Public Policy",
    "Statistics", "Urban Studies", "Biology", "Digital Humanities",
)
_TOPICS = (
    "Data Systems", "Ecological Modeling", "Language and Society",
    "Markets and Institutions", "Modern Cultures", "Networks and Decisions",
    "Scientific Computing", "Sustainable Cities", "Public Reasoning",
    "Quantitative Methods", "Archives and Evidence", "Design Research",
)
_FIRST_NAMES = (
    "Amina", "Arun", "Clara", "Diego", "Elena", "Farah", "Grace", "Hana",
    "Isaac", "Jia", "Kofi", "Leila", "Mateo", "Nadia", "Omar", "Priya",
    "Ravi", "Sofia", "Tariq", "Yuna",
)
_LAST_NAMES = (
    "Ahmed", "Bennett", "Chen", "Das", "Evans", "Fernandez", "Gupta",
    "Hassan", "Ito", "Johnson", "Khan", "Lopez", "Mensah", "Novak",
    "Okafor", "Patel", "Rahman", "Silva", "Tran", "Williams",
)


def _stable_number(seed: int, *parts: object) -> int:
    payload = "|".join(
        ("relmemdb-semantic-world-v2", str(seed), *(str(part) for part in parts))
    )
    return int(hash_text(payload)[:16], 16)


def _choice(options: tuple[Any, ...], seed: int, *parts: object) -> Any:
    return options[_stable_number(seed, *parts) % len(options)]


def _identifier(seed: int, chain_index: int, position: int, prefix: str) -> str:
    multiplier = (_stable_number(seed, "id-multiplier", position) % IDENTIFIER_MODULUS) | 1
    while gcd(multiplier, IDENTIFIER_MODULUS) != 1:
        multiplier = (multiplier + 2) % IDENTIFIER_MODULUS
    offset = _stable_number(seed, "id-offset", position) % IDENTIFIER_MODULUS
    number = (multiplier * chain_index + offset) % IDENTIFIER_MODULUS
    return f"{prefix}{number:06d}"


def _attribute_value(field: str, seed: int, chain_index: int) -> str | int:
    stem = _choice(_STEMS, seed, field, chain_index, "stem")
    direction = _choice(_DIRECTIONS, seed, field, chain_index, "direction")
    discipline = _choice(_DISCIPLINES, seed, field, chain_index, "discipline")
    topic = _choice(_TOPICS, seed, field, chain_index, "topic")
    values: dict[str, Any] = {
        "continent_name": f"{stem} Reach",
        "climate_band": _choice(("polar", "cool temperate", "warm temperate", "subtropical", "tropical"), seed, field, chain_index),
        "country_name": f"Republic of {stem}{_choice(('ia', 'ara', 'ora', 'en', 'al'), seed, field, chain_index, 'suffix')}",
        "currency_name": f"{_choice(_STEMS, seed, field, chain_index, 'currency')} crown",
        "region_name": f"{direction} {stem}",
        "administrative_type": _choice(("province", "state", "territory", "prefecture", "district"), seed, field, chain_index),
        "city_name": f"{stem}{_choice(('ford', 'haven', 'bridge', 'port', 'field', 'view'), seed, field, chain_index, 'suffix')}",
        "population_band": _choice(("under 100,000", "100,000-499,999", "500,000-999,999", "1-3 million", "over 3 million"), seed, field, chain_index),
        "campus_name": f"{_choice(_STEMS, seed, field, chain_index, 'campus')} {_choice(('Central', 'Riverside', 'North', 'Innovation', 'Garden'), seed, field, chain_index, 'kind')} Campus",
        "campus_type": _choice(("urban", "suburban", "residential", "research park", "distributed"), seed, field, chain_index),
        "school_name": f"School of {discipline}",
        "founding_period": _choice(("before 1900", "1900-1949", "1950-1974", "1975-1999", "since 2000"), seed, field, chain_index),
        "department_name": f"Department of {discipline}",
        "focus_area": topic,
        "subject_name": topic,
        "subject_level": _choice(("introductory", "intermediate", "advanced", "graduate"), seed, field, chain_index),
        "discipline_group": discipline,
        "course_title": f"{topic}: {_choice(('Foundations', 'Methods', 'Applications', 'Contemporary Issues', 'Research Seminar'), seed, field, chain_index, 'subtitle')}",
        "credit_hours": _choice((2, 3, 4, 5), seed, field, chain_index),
        "delivery_mode": _choice(("in person", "hybrid", "online", "field based"), seed, field, chain_index),
        "section_label": f"Section {_choice(tuple('ABCDEFGH'), seed, field, chain_index)}{1 + _stable_number(seed, field, chain_index, 'section') % 20:02d}",
        "meeting_period": _choice(("Monday morning", "Tuesday afternoon", "Wednesday evening", "Thursday morning", "Friday afternoon"), seed, field, chain_index),
        "room_label": f"{_choice(('Alder Hall', 'Library Annex', 'Science Center', 'West Pavilion', 'Learning Commons'), seed, field, chain_index)} {100 + _stable_number(seed, field, chain_index, 'room') % 400}",
        "academic_term": _choice(("Autumn 2024", "Spring 2025", "Summer 2025", "Autumn 2025", "Spring 2026"), seed, field, chain_index),
        "final_grade": _choice(("A", "A-", "B+", "B", "B-", "C+", "Pass"), seed, field, chain_index),
        "enrollment_status": _choice(("completed", "in progress", "withdrawn", "deferred"), seed, field, chain_index),
        "full_name": f"{_choice(_FIRST_NAMES, seed, field, chain_index, 'first')} {_choice(_LAST_NAMES, seed, field, chain_index, 'last')}",
        "study_year": _choice(("first year", "second year", "third year", "fourth year", "postgraduate"), seed, field, chain_index),
        "scholarship_status": _choice(("no scholarship", "partial scholarship", "full scholarship", "research fellowship"), seed, field, chain_index),
    }
    return values[field]


def derive_master_world_counts(config: dict[str, Any]) -> dict[str, int]:
    data = config["data"]
    construction = data["master_world"]
    configured_fact_counts = [data["t_sweep"]["fact_count"], *data["n_sweep"]["fact_counts"]]
    if data["optional_n40k"]["enabled"]:
        configured_fact_counts.append(data["optional_n40k"]["fact_count"])
    maximum_configured_n = max(configured_fact_counts)
    facts_per_chain = construction["experimental_facts_per_chain"]
    total_chains = maximum_configured_n // facts_per_chain
    return {
        "latent_positions": construction["latent_positions"],
        "identifier_fields_per_chain": construction["identifier_fields_per_chain"],
        "attribute_facts_per_chain": construction["descriptive_facts_per_chain"],
        "relation_facts_per_chain": construction["relation_facts_per_chain"],
        "experimental_facts_per_chain": facts_per_chain,
        "total_chains": total_chains,
        "total_experimental_facts": total_chains * facts_per_chain,
        "maximum_supported_configured_n": maximum_configured_n,
    }


def _build_chain(seed: int, chain_index: int) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for position, spec in enumerate(SEMANTIC_ENTITY_SPECS):
        attributes = [
            {"name": name, "value": _attribute_value(name, seed, chain_index)}
            for name, _ in spec["attributes"]
        ]
        entities.append(
            {
                "position": position,
                "entity_type": spec["entity_type"],
                "entity_id": _identifier(seed, chain_index, position, spec["id_prefix"]),
                "attributes": attributes,
            }
        )
    relations = [
        {
            "relation_type": "references_parent",
            "source_position": position,
            "target_position": position - 1,
            "source_entity_id": entities[position]["entity_id"],
            "target_entity_id": entities[position - 1]["entity_id"],
        }
        for position in range(1, len(SEMANTIC_ENTITY_SPECS))
    ]
    return {"chain_index": chain_index, "entities": entities, "relations": relations}


def build_master_world(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    counts = derive_master_world_counts(config)
    seed = config["experiment"]["seed"]
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": seed,
        "construction": {"schema_topology": config["data"]["schema_topology"], **counts},
        "chains": [_build_chain(seed, index) for index in range(counts["total_chains"])],
    }


def build_master_world_manifest(
    config: dict[str, Any], *, configuration_sha256: str, world_sha256: str
) -> dict[str, Any]:
    counts = derive_master_world_counts(config)
    total_chains = counts["total_chains"]
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": config["experiment"]["seed"],
        "schema_topology": config["data"]["schema_topology"],
        **counts,
        "total_identifier_field_count": total_chains * counts["identifier_fields_per_chain"],
        "total_attribute_fact_count": total_chains * counts["attribute_facts_per_chain"],
        "total_relation_fact_count": total_chains * counts["relation_facts_per_chain"],
        "identifiers_counted_as_experimental_facts": False,
        "configuration_sha256": configuration_sha256,
        "world_json_sha256": world_sha256,
    }

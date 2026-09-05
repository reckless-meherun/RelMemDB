from typing import Any

from config import validate_config
from utils.hashing import hash_text


FORMAT_VERSION = 2
IDENTIFIER_MODULUS = 1_000_000
NATURAL_IDENTIFIER_FIELDS = {
    "continent": "continent_name",
    "country": "country_name",
    "region": "region_name",
    "city": "city_name",
    "campus": "campus_name",
    "school": "school_name",
    "department": "department_name",
    "subject": "subject_name",
    "course": "course_title",
    "student": "full_name",
}

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
_MODIFIERS = (
    "Ancient", "Blue", "Bright", "Emerald", "Golden", "Grand", "Highland",
    "Lake", "New", "Quiet", "Royal", "Sapphire", "Sunlit", "Verdant",
    "White", "Windward",
)
_LANDFORMS = (
    "Arc", "Basin", "Coast", "Highlands", "Isles", "Plateau", "Reach",
    "Rim", "Shores", "Uplands",
)
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
_COUNTRY_SUFFIXES = ("a", "ara", "en", "ia", "ora", "ovia", "stan", "une")
_GOVERNMENTS = ("Commonwealth", "Federation", "Kingdom", "Republic", "Union")
_CITY_SUFFIXES = ("bridge", "field", "ford", "haven", "port", "stead", "view", "wick")
_CAMPUS_SITES = (
    "Arts", "Garden", "Innovation", "Lakeside", "Research", "Riverside",
    "Technology", "Woodland",
)
_COURSE_SUBTITLES = (
    "Contemporary Issues", "Foundations", "Laboratory", "Methods",
    "Research Seminar", "Theory and Practice",
)

_NATURAL_NAME_NAMESPACE_CAPACITIES = {
    "continent_name": (
        len(_DIRECTIONS) * len(_MODIFIERS) * len(_STEMS) * len(_LANDFORMS)
    ),
    "country_name": (
        len(_MODIFIERS)
        * len(_STEMS)
        * len(_COUNTRY_SUFFIXES)
        * len(_GOVERNMENTS)
    ),
    "region_name": (
        len(_DIRECTIONS) * len(_MODIFIERS) * len(_STEMS) * len(_LANDFORMS)
    ),
    "city_name": len(_MODIFIERS) * len(_STEMS) * len(_CITY_SUFFIXES),
    "campus_name": len(_STEMS) * len(_MODIFIERS) * len(_CAMPUS_SITES),
    "school_name": len(_STEMS) * len(_DISCIPLINES) * len(_TOPICS),
    "department_name": len(_STEMS) * len(_DISCIPLINES) * len(_TOPICS),
    "subject_name": len(_MODIFIERS) * len(_TOPICS) * len(_DISCIPLINES),
    "course_title": (
        len(_MODIFIERS)
        * len(_TOPICS)
        * len(_COURSE_SUBTITLES)
        * len(_DISCIPLINES)
    ),
    "full_name": len(_FIRST_NAMES) * 26 * len(_LAST_NAMES),
}


def _stable_number(seed: int, *parts: object) -> int:
    payload = "|".join(
        ("relmemdb-semantic-world-v2", str(seed), *(str(part) for part in parts))
    )
    return int(hash_text(payload)[:16], 16)


def _choice(options: tuple[Any, ...], seed: int, *parts: object) -> Any:
    return options[_stable_number(seed, *parts) % len(options)]


def _identifier(
    seed: int,
    chain_index: int,
    position: int,
    prefix: str,
    used_identifiers: set[str],
    used_numeric_suffixes: set[str],
) -> str:
    if len(used_numeric_suffixes) < IDENTIFIER_MODULUS:
        for attempt in range(IDENTIFIER_MODULUS):
            number = _stable_number(
                seed, "entity-id", position, chain_index, attempt
            ) % IDENTIFIER_MODULUS
            suffix = f"{number:06d}"
            identifier = f"{prefix}{suffix}"
            if identifier in used_identifiers or suffix in used_numeric_suffixes:
                continue
            used_identifiers.add(identifier)
            used_numeric_suffixes.add(suffix)
            return identifier
    # The historical six-digit namespace is finite. Extend it only once every
    # six-digit suffix has genuinely been consumed; the chain/position ordinal
    # is deterministic, injective, and has no fixed upper bound.
    suffix = str(
        IDENTIFIER_MODULUS + chain_index * len(SEMANTIC_ENTITY_SPECS) + position
    )
    identifier = f"{prefix}{suffix}"
    if identifier in used_identifiers or suffix in used_numeric_suffixes:
        raise RuntimeError("deterministic extended identifier collision")
    used_identifiers.add(identifier)
    used_numeric_suffixes.add(suffix)
    return identifier


def _natural_name_candidate(
    field: str, seed: int, chain_index: int, attempt: int
) -> str:
    def choose(options: tuple[str, ...], salt: str) -> str:
        return _choice(options, seed, field, chain_index, attempt, salt)

    stem = choose(_STEMS, "stem")
    modifier = choose(_MODIFIERS, "modifier")
    discipline = choose(_DISCIPLINES, "discipline")
    topic = choose(_TOPICS, "topic")
    candidates = {
        "continent_name": (
            f"{choose(_DIRECTIONS, 'direction')} {modifier} {stem} "
            f"{choose(_LANDFORMS, 'landform')}"
        ),
        "country_name": (
            f"{modifier} {stem}{choose(_COUNTRY_SUFFIXES, 'suffix')} "
            f"{choose(_GOVERNMENTS, 'government')}"
        ),
        "region_name": (
            f"{choose(_DIRECTIONS, 'direction')} {modifier} {stem} "
            f"{choose(_LANDFORMS, 'landform')}"
        ),
        "city_name": f"{modifier} {stem}{choose(_CITY_SUFFIXES, 'suffix')}",
        "campus_name": (
            f"{stem} {modifier} {choose(_CAMPUS_SITES, 'site')} Campus"
        ),
        "school_name": f"{stem} School of {discipline} and {topic}",
        "department_name": f"{stem} Department of {discipline} and {topic}",
        "subject_name": f"{modifier} {topic} in {discipline}",
        "course_title": (
            f"{modifier} {topic}: {choose(_COURSE_SUBTITLES, 'subtitle')} "
            f"in {discipline}"
        ),
        "full_name": (
            f"{choose(_FIRST_NAMES, 'first')} "
            f"{chr(65 + _stable_number(seed, field, chain_index, attempt, 'middle') % 26)}. "
            f"{choose(_LAST_NAMES, 'last')}"
        ),
    }
    return candidates[field]


def _unique_natural_name(
    field: str, seed: int, chain_index: int, used_names: set[str]
) -> str:
    if len(used_names) < _NATURAL_NAME_NAMESPACE_CAPACITIES[field]:
        for attempt in range(IDENTIFIER_MODULUS):
            candidate = _natural_name_candidate(field, seed, chain_index, attempt)
            if candidate in used_names:
                continue
            used_names.add(candidate)
            return candidate
    # Readable combinatorial names are finite. Disambiguate only after that
    # field's complete base namespace has genuinely been consumed.
    candidate = (
        f"{_natural_name_candidate(field, seed, chain_index, 0)} "
        f"Record {chain_index + 1}"
    )
    if candidate in used_names:
        raise RuntimeError(
            f"deterministic extended natural-name collision for {field}"
        )
    used_names.add(candidate)
    return candidate


def canonical_table_names() -> tuple[str, ...]:
    return tuple(spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS)


def validate_selected_tables(tables: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(tables, (list, tuple)) or not tables:
        raise ValueError("at least one canonical table must be supplied with --tables")
    if any(not isinstance(table, str) or not table for table in tables):
        raise ValueError("selected table names must be non-empty strings")
    duplicates = sorted({table for table in tables if tables.count(table) > 1})
    if duplicates:
        raise ValueError(f"duplicate selected table names: {duplicates}")
    valid = canonical_table_names()
    invalid = [table for table in tables if table not in valid]
    if invalid:
        raise ValueError(
            f"unknown canonical table(s): {invalid}; valid tables are {list(valid)}"
        )
    if len(tables) > len(valid):
        raise ValueError(f"no more than {len(valid)} canonical tables may be selected")
    # Canonical order makes content and path identity independent of CLI ordering.
    selected = set(tables)
    return tuple(table for table in valid if table in selected)


def selected_table_positions(tables: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    selected = set(validate_selected_tables(tables))
    return tuple(
        position
        for position, spec in enumerate(SEMANTIC_ENTITY_SPECS)
        if spec["entity_type"] in selected
    )


def facts_per_selected_chain(tables: list[str] | tuple[str, ...]) -> int:
    positions = selected_table_positions(tables)
    return sum(
        len(SEMANTIC_ENTITY_SPECS[position]["attributes"]) + (position > 0)
        for position in positions
    )


def validate_exp2_fact_count(
    fact_count: int, tables: list[str] | tuple[str, ...]
) -> int:
    if isinstance(fact_count, bool) or not isinstance(fact_count, int) or fact_count <= 0:
        raise ValueError("N/--fact-count must be a positive integer")
    selected = validate_selected_tables(tables)
    per_chain = facts_per_selected_chain(selected)
    quotient, remainder = divmod(fact_count, per_chain)
    if remainder:
        lower = fact_count - remainder
        upper = lower + per_chain
        raise ValueError(
            f"Requested N={fact_count} cannot be represented exactly for selected "
            f"tables {list(selected)} because each generated chain contributes "
            f"{per_chain} exposed logical facts. Nearest valid values are "
            f"{lower if lower > 0 else per_chain} and {upper}."
        )
    return quotient


def build_world_for_chain_count(
    config: dict[str, Any], chain_count: int
) -> dict[str, Any]:
    """Build a prefix-stable world of an explicit size for Experiment 2."""
    if isinstance(chain_count, bool) or not isinstance(chain_count, int) or chain_count <= 0:
        raise ValueError("chain_count must be a positive integer")
    seed = config["experiment"]["seed"]
    used_identifiers: set[str] = set()
    used_numeric_suffixes: set[str] = set()
    used_natural_names = {field: set() for field in NATURAL_IDENTIFIER_FIELDS.values()}
    chains: list[dict[str, Any]] = []
    for index in range(chain_count):
        chain = _build_chain(
            seed, index, used_identifiers, used_numeric_suffixes, used_natural_names
        )
        chains.append(chain)
    _validate_world_uniqueness(chains)
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": seed,
        "construction": {
            "schema_topology": "chain",
            "latent_positions": 12,
            "identifier_fields_per_chain": 12,
            "attribute_facts_per_chain": 29,
            "relation_facts_per_chain": 11,
            "experimental_facts_per_chain": 40,
            "total_chains": chain_count,
            "total_experimental_facts": chain_count * 40,
            "maximum_supported_configured_n": chain_count * 40,
        },
        "chains": chains,
    }


def _attribute_value(field: str, seed: int, chain_index: int) -> str | int:
    discipline = _choice(_DISCIPLINES, seed, field, chain_index, "discipline")
    topic = _choice(_TOPICS, seed, field, chain_index, "topic")
    values: dict[str, Any] = {
        "climate_band": _choice(("polar", "cool temperate", "warm temperate", "subtropical", "tropical"), seed, field, chain_index),
        "currency_name": f"{_choice(_STEMS, seed, field, chain_index, 'currency')} crown",
        "administrative_type": _choice(("province", "state", "territory", "prefecture", "district"), seed, field, chain_index),
        "population_band": _choice(("under 100,000", "100,000-499,999", "500,000-999,999", "1-3 million", "over 3 million"), seed, field, chain_index),
        "campus_type": _choice(("urban", "suburban", "residential", "research park", "distributed"), seed, field, chain_index),
        "founding_period": _choice(("before 1900", "1900-1949", "1950-1974", "1975-1999", "since 2000"), seed, field, chain_index),
        "focus_area": topic,
        "subject_level": _choice(("introductory", "intermediate", "advanced", "graduate"), seed, field, chain_index),
        "discipline_group": discipline,
        "credit_hours": _choice((2, 3, 4, 5), seed, field, chain_index),
        "delivery_mode": _choice(("in person", "hybrid", "online", "field based"), seed, field, chain_index),
        "section_label": f"Section {_choice(tuple('ABCDEFGH'), seed, field, chain_index)}{1 + _stable_number(seed, field, chain_index, 'section') % 20:02d}",
        "meeting_period": _choice(("Monday morning", "Tuesday afternoon", "Wednesday evening", "Thursday morning", "Friday afternoon"), seed, field, chain_index),
        "room_label": f"{_choice(('Alder Hall', 'Library Annex', 'Science Center', 'West Pavilion', 'Learning Commons'), seed, field, chain_index)} {100 + _stable_number(seed, field, chain_index, 'room') % 400}",
        "academic_term": _choice(("Autumn 2024", "Spring 2025", "Summer 2025", "Autumn 2025", "Spring 2026"), seed, field, chain_index),
        "final_grade": _choice(("A", "A-", "B+", "B", "B-", "C+", "Pass"), seed, field, chain_index),
        "enrollment_status": _choice(("completed", "in progress", "withdrawn", "deferred"), seed, field, chain_index),
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


def _build_chain(
    seed: int,
    chain_index: int,
    used_identifiers: set[str],
    used_numeric_suffixes: set[str],
    used_natural_names: dict[str, set[str]],
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for position, spec in enumerate(SEMANTIC_ENTITY_SPECS):
        natural_identifier_field = NATURAL_IDENTIFIER_FIELDS.get(spec["entity_type"])
        attributes = []
        for name, _ in spec["attributes"]:
            if name == natural_identifier_field:
                value = _unique_natural_name(
                    name, seed, chain_index, used_natural_names[name]
                )
            else:
                value = _attribute_value(name, seed, chain_index)
            attributes.append({"name": name, "value": value})
        entities.append(
            {
                "position": position,
                "entity_type": spec["entity_type"],
                "entity_id": _identifier(
                    seed,
                    chain_index,
                    position,
                    spec["id_prefix"],
                    used_identifiers,
                    used_numeric_suffixes,
                ),
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


def _validate_world_uniqueness(chains: list[dict[str, Any]]) -> None:
    identifiers: set[str] = set()
    numeric_suffixes: set[str] = set()
    natural_names = {field: set() for field in NATURAL_IDENTIFIER_FIELDS.values()}
    for chain in chains:
        for entity in chain["entities"]:
            identifier = entity["entity_id"]
            if identifier in identifiers:
                raise RuntimeError(f"duplicate entity identifier: {identifier}")
            suffix = identifier[3:]
            if suffix in numeric_suffixes:
                raise RuntimeError(f"shared entity-ID numeric suffix: {suffix}")
            identifiers.add(identifier)
            numeric_suffixes.add(suffix)

            field = NATURAL_IDENTIFIER_FIELDS.get(entity["entity_type"])
            if field is None:
                continue
            attributes = {
                attribute["name"]: attribute["value"]
                for attribute in entity["attributes"]
            }
            value = attributes[field]
            if value in natural_names[field]:
                raise RuntimeError(f"duplicate {field}: {value}")
            natural_names[field].add(value)


def build_master_world(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    counts = derive_master_world_counts(config)
    seed = config["experiment"]["seed"]
    used_identifiers: set[str] = set()
    used_numeric_suffixes: set[str] = set()
    used_natural_names = {
        field: set() for field in NATURAL_IDENTIFIER_FIELDS.values()
    }
    chains = [
        _build_chain(
            seed,
            index,
            used_identifiers,
            used_numeric_suffixes,
            used_natural_names,
        )
        for index in range(counts["total_chains"])
    ]
    _validate_world_uniqueness(chains)
    return {
        "format_version": FORMAT_VERSION,
        "experiment_name": config["experiment"]["name"],
        "seed": seed,
        "construction": {"schema_topology": config["data"]["schema_topology"], **counts},
        "chains": chains,
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

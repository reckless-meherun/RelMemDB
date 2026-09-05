import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.materialize import (
    build_database_manifest,
    build_exp2_database_manifest,
    materialize_database,
    materialize_selected_tables_database,
)
from data.serialize import (
    build_selected_readable_database_book,
    database_schema_sha256,
    serialize_database_cpt,
)
from data.world import (
    SEMANTIC_ENTITY_SPECS,
    build_master_world,
    build_master_world_manifest,
    build_world_for_chain_count,
    facts_per_selected_chain,
    validate_exp2_fact_count,
    validate_selected_tables,
)
from experiment import ExperimentCondition
from utils.hashing import hash_file
from utils.io import read_json, write_json
from utils.paths import (
    EXP01_GENERATED_DATABASES_DIR,
    database_condition_dir,
    n_sweep_database_dir,
    t_sweep_database_dir,
    exp2_database_bundle_dir,
)


def _master_world_paths() -> tuple[Path, Path]:
    output_dir = EXP01_GENERATED_DATABASES_DIR / "master_world"
    return output_dir / "world.json", output_dir / "manifest.json"


def _rebuild_master_world(config: dict, config_path: Path) -> tuple[dict, str, str]:
    world_path, manifest_path = _master_world_paths()
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world = build_master_world(config)
    write_json(world_path, world)
    world_sha256 = hash_file(world_path)
    configuration_sha256 = hash_file(config_path)
    manifest = build_master_world_manifest(
        config,
        configuration_sha256=configuration_sha256,
        world_sha256=world_sha256,
    )
    write_json(manifest_path, manifest)
    print(
        "master_world: "
        f"chains={manifest['total_chains']}, "
        f"experimental_facts={manifest['total_experimental_facts']}, "
        f"world={world_path.relative_to(PROJECT_ROOT)}"
    )
    return world, world_sha256, configuration_sha256


def _load_verified_master_world(
    config: dict, config_path: Path
) -> tuple[dict, str, str]:
    world_path, manifest_path = _master_world_paths()
    for artifact_path in (world_path, manifest_path):
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"required master-world artifact is missing or empty: {artifact_path}; "
                "run with --rebuild-master-world first"
            )
    manifest = read_json(manifest_path)
    world_sha256 = hash_file(world_path)
    configuration_sha256 = hash_file(config_path)
    if manifest.get("world_json_sha256") != world_sha256:
        raise ValueError("master-world artifact does not match its manifest hash")
    world = read_json(world_path)
    if world != build_master_world(config):
        raise ValueError("master-world artifact is not reproducible from the config")
    return world, world_sha256, configuration_sha256


def _condition_destination(
    config: dict, table_count: int, logical_fact_count: int
) -> tuple[str, Path]:
    ExperimentCondition.from_config(
        config,
        table_count=table_count,
        fact_count=logical_fact_count,
        layers=config["model"]["layers"],
    )
    data = config["data"]
    if (
        logical_fact_count == data["t_sweep"]["fact_count"]
        and table_count in data["t_sweep"]["table_counts"]
    ):
        return "t_sweep", t_sweep_database_dir(table_count)
    if (
        table_count == data["n_sweep"]["table_count"]
        and logical_fact_count in data["n_sweep"]["fact_counts"]
    ):
        return "n_sweep", n_sweep_database_dir(logical_fact_count)
    return "condition", database_condition_dir(table_count, logical_fact_count)


def _materialize_condition(
    *,
    config: dict,
    world: dict,
    sweep: str,
    table_count: int,
    logical_fact_count: int,
    condition_dir: Path,
    master_world_sha256: str,
    configuration_sha256: str,
) -> dict:
    database_path = condition_dir / "database.sqlite"
    manifest_path = condition_dir / "manifest.json"
    occupied = [
        path
        for path in (database_path, manifest_path)
        if path.exists() and path.stat().st_size
    ]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite existing database artifacts: "
            + ", ".join(map(str, occupied))
        )
    condition_dir.mkdir(parents=True, exist_ok=True)
    materialization = materialize_database(
        world,
        table_count=table_count,
        logical_fact_count=logical_fact_count,
        output_path=database_path,
    )
    manifest = build_database_manifest(
        config,
        materialization,
        sweep=sweep,
        master_world_sha256=master_world_sha256,
        configuration_sha256=configuration_sha256,
        database_sha256=hash_file(database_path),
    )
    write_json(manifest_path, manifest)
    print(
        f"{sweep}: T={table_count}, N={logical_fact_count}, "
        f"chains={materialization['selected_chain_count']}, "
        f"database={database_path.relative_to(PROJECT_ROOT)}"
    )
    return manifest


def _serialize_condition_cpt(
    *,
    config: dict,
    condition_dir: Path,
    table_count: int,
    logical_fact_count: int,
) -> dict:
    database_path = condition_dir / "database.sqlite"
    database_manifest_path = condition_dir / "manifest.json"
    cpt_dir = condition_dir / "cpt"
    readable_book_path = cpt_dir / "book_readable.txt"
    train_text_path = cpt_dir / "train.txt"
    cpt_manifest_path = cpt_dir / "manifest.json"
    occupied = [
        path
        for path in (readable_book_path, train_text_path, cpt_manifest_path)
        if path.exists() and path.stat().st_size
    ]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite existing CPT serialization artifacts: "
            + ", ".join(map(str, occupied))
        )
    cpt_manifest = serialize_database_cpt(
        config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        train_text_path=train_text_path,
        readable_book_path=readable_book_path,
        expected_table_count=table_count,
        expected_logical_fact_count=logical_fact_count,
    )
    write_json(cpt_manifest_path, cpt_manifest)
    try:
        displayed_train_path = train_text_path.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_train_path = train_text_path
    print(
        f"cpt: T={table_count}, N={logical_fact_count}, "
        f"exposures={cpt_manifest['fact_exposure']}, "
        f"book={readable_book_path.name}, train={displayed_train_path}"
    )
    return cpt_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the semantic master world, materialize one database condition, "
            "or serialize CPT text from one existing condition."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--rebuild-master-world",
        action="store_true",
        help="deterministically replace master_world/world.json and its manifest",
    )
    parser.add_argument(
        "--master-world-only",
        action="store_true",
        help="stop after rebuilding the master world",
    )
    parser.add_argument(
        "--serialize-cpt-only",
        action="store_true",
        help="serialize one existing database without rebuilding or materializing it",
    )
    parser.add_argument("--table-count", type=int, help="physical table count T")
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Experiment-2 canonical tables to expose (T is inferred)",
    )
    parser.add_argument(
        "--fact-count", type=int, help="experimental logical fact count N"
    )
    args = parser.parse_args()
    if args.master_world_only and not args.rebuild_master_world:
        parser.error("--master-world-only requires --rebuild-master-world")
    if args.serialize_cpt_only and (
        args.rebuild_master_world or args.master_world_only
    ):
        parser.error(
            "--serialize-cpt-only cannot be combined with master-world generation"
        )
    # Exp1 retains its paired T/N CLI. Exp2 validation is performed after loading
    # the selected config so --fact-count may intentionally be omitted.
    if args.tables is None and (args.table_count is None) != (args.fact_count is None):
        parser.error("--table-count and --fact-count must be supplied together")
    if args.serialize_cpt_only and args.table_count is None:
        parser.error("--serialize-cpt-only requires --table-count and --fact-count")
    return args


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _assert_canonical_prefix(
    canonical_database: Path, generated_database: Path, prefix_rows: int
) -> None:
    """Prove every generated condition begins with the canonical logical rows."""
    with sqlite3.connect(canonical_database) as canonical, sqlite3.connect(
        generated_database
    ) as generated:
        table_names = [spec["entity_type"] for spec in SEMANTIC_ENTITY_SPECS]
        for table in table_names:
            expected = canonical.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid LIMIT ?', (prefix_rows,)
            ).fetchall()
            actual = generated.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid LIMIT ?', (prefix_rows,)
            ).fetchall()
            if len(expected) != prefix_rows or actual != expected:
                raise RuntimeError(
                    f"generated Experiment-2 rows do not preserve the canonical prefix for {table}"
                )


def _generate_exp2(config: dict, config_path: Path, args: argparse.Namespace) -> Path:
    if not args.tables:
        raise ValueError("Experiment 2 requires at least one table via --tables")
    if args.table_count is not None:
        raise ValueError("Experiment 2 infers T from --tables; do not pass --table-count")
    if args.rebuild_master_world or args.master_world_only or args.serialize_cpt_only:
        raise ValueError("Experiment 2 performs generation and serialization as one immutable bundle")
    selected = validate_selected_tables(args.tables)
    canonical_dir = _resolve_project_path(config["data"]["canonical_source"]["path"])
    canonical_database = canonical_dir / "database.sqlite"
    canonical_manifest_path = canonical_dir / "manifest.json"
    for path, label in (
        (canonical_database, "canonical database"),
        (canonical_manifest_path, "canonical database manifest"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
    canonical_manifest = read_json(canonical_manifest_path)
    canonical_database_hash_before = hash_file(canonical_database)
    canonical_manifest_hash_before = hash_file(canonical_manifest_path)
    if canonical_manifest.get("database_sha256") != canonical_database_hash_before:
        raise ValueError("canonical database does not match its manifest hash")
    canonical_schema_hash = database_schema_sha256(canonical_database)
    baseline_chains = canonical_manifest.get("selected_chain_count")
    if isinstance(baseline_chains, bool) or not isinstance(baseline_chains, int) or baseline_chains <= 0:
        raise ValueError("canonical manifest selected_chain_count is invalid")
    per_chain = facts_per_selected_chain(selected)
    fact_count = baseline_chains * per_chain if args.fact_count is None else args.fact_count
    chain_count = validate_exp2_fact_count(fact_count, selected)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    condition_dir = exp2_database_bundle_dir(selected, fact_count, timestamp)
    if condition_dir.exists():
        raise FileExistsError(f"refusing to overwrite Experiment-2 bundle: {condition_dir}")
    condition_dir.mkdir(parents=True)
    database_path = condition_dir / "database.sqlite"
    manifest_path = condition_dir / "manifest.json"
    world = build_world_for_chain_count(config, chain_count)
    materialization = materialize_selected_tables_database(
        world, selected, chain_count, database_path
    )
    _assert_canonical_prefix(
        canonical_database, database_path, min(chain_count, baseline_chains)
    )
    generated_schema_hash = database_schema_sha256(database_path)
    # Compute text statistics before sealing the database manifest. Serialization
    # repeats this deterministic render and authenticates the final manifest hash.
    readable_book, book_metadata = build_selected_readable_database_book(
        database_path, materialization
    )
    materialization.update(
        {
            "artifact_path": str(condition_dir.resolve()),
            "sentence_count": book_metadata["instance_sentence_count"]
            + book_metadata["schema_description_sentence_count"],
            "character_count": len(readable_book),
            "byte_count": len(readable_book.encode("utf-8")),
        }
    )
    manifest = build_exp2_database_manifest(
        config,
        materialization,
        generation_timestamp=timestamp,
        canonical_database_sha256=canonical_database_hash_before,
        canonical_database_manifest_sha256=canonical_manifest_hash_before,
        canonical_schema_sha256=canonical_schema_hash,
        generated_schema_sha256=generated_schema_hash,
        configuration_sha256=hash_file(config_path),
        database_sha256=hash_file(database_path),
    )
    write_json(manifest_path, manifest)
    _serialize_condition_cpt(
        config=config,
        condition_dir=condition_dir,
        table_count=len(selected),
        logical_fact_count=fact_count,
    )
    if (
        hash_file(canonical_database) != canonical_database_hash_before
        or hash_file(canonical_manifest_path) != canonical_manifest_hash_before
        or database_schema_sha256(canonical_database) != canonical_schema_hash
    ):
        raise RuntimeError("canonical database or schema changed during Experiment-2 generation")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Selected tables: {' '.join(selected)}")
    print(f"T: {len(selected)}")
    print(f"Facts per selected chain: {per_chain}")
    print(f"Rows/chains: {chain_count}")
    print(f"N: {fact_count}")
    print(f"Output: {condition_dir.relative_to(PROJECT_ROOT)}")
    return condition_dir


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if config["experiment"]["name"] == "exp02_capacity_boundary":
        _generate_exp2(config, config_path, args)
        return
    if args.serialize_cpt_only:
        _, condition_dir = _condition_destination(
            config, args.table_count, args.fact_count
        )
        _serialize_condition_cpt(
            config=config,
            condition_dir=condition_dir,
            table_count=args.table_count,
            logical_fact_count=args.fact_count,
        )
        return
    if args.rebuild_master_world:
        world, world_sha256, configuration_sha256 = _rebuild_master_world(
            config, config_path
        )
    else:
        world, world_sha256, configuration_sha256 = _load_verified_master_world(
            config, config_path
        )
    if args.master_world_only:
        return

    canonical = config["data"]["canonical_target"]
    table_count = args.table_count or canonical["table_count"]
    logical_fact_count = args.fact_count or canonical["fact_count"]
    sweep, condition_dir = _condition_destination(
        config, table_count, logical_fact_count
    )
    _materialize_condition(
        config=config,
        world=world,
        sweep=sweep,
        table_count=table_count,
        logical_fact_count=logical_fact_count,
        condition_dir=condition_dir,
        master_world_sha256=world_sha256,
        configuration_sha256=configuration_sha256,
    )


if __name__ == "__main__":
    main()

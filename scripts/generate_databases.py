import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.materialize import build_database_manifest, materialize_database
from data.serialize import serialize_database_cpt
from data.world import build_master_world, build_master_world_manifest
from utils.hashing import hash_file
from utils.io import read_json, write_json
from utils.paths import (
    EXP01_GENERATED_DATABASES_DIR,
    n_sweep_database_dir,
    t_sweep_database_dir,
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
    if manifest.get("configuration_sha256") != configuration_sha256:
        raise ValueError("configuration does not match the master-world manifest")
    world = read_json(world_path)
    if world != build_master_world(config):
        raise ValueError("master-world artifact is not reproducible from the config")
    return world, world_sha256, configuration_sha256


def _condition_destination(
    config: dict, table_count: int, logical_fact_count: int
) -> tuple[str, Path]:
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
    raise ValueError(
        f"T{table_count}/N{logical_fact_count} is not a configured t_sweep or n_sweep condition"
    )


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
    condition_dir.mkdir(parents=True, exist_ok=True)
    database_path = condition_dir / "database.sqlite"
    manifest_path = condition_dir / "manifest.json"
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
    train_text_path = cpt_dir / "train.txt"
    cpt_manifest_path = cpt_dir / "manifest.json"
    cpt_manifest = serialize_database_cpt(
        config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        train_text_path=train_text_path,
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
        f"train={displayed_train_path}"
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
    parser.add_argument("--fact-count", type=int, help="experimental logical fact count N")
    args = parser.parse_args()
    if args.master_world_only and not args.rebuild_master_world:
        parser.error("--master-world-only requires --rebuild-master-world")
    if args.serialize_cpt_only and (args.rebuild_master_world or args.master_world_only):
        parser.error(
            "--serialize-cpt-only cannot be combined with master-world generation"
        )
    if (args.table_count is None) != (args.fact_count is None):
        parser.error("--table-count and --fact-count must be supplied together")
    if args.serialize_cpt_only and args.table_count is None:
        parser.error(
            "--serialize-cpt-only requires --table-count and --fact-count"
        )
    return args


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
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

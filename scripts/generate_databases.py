import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.materialize import build_database_manifest, materialize_database
from data.world import build_master_world
from utils.hashing import hash_file
from utils.io import read_json, write_json
from utils.paths import (
    EXP01_GENERATED_DATABASES_DIR,
    n_sweep_database_dir,
    t_sweep_database_dir,
)


def _load_verified_master_world(config: dict) -> tuple[dict, str, str]:
    output_dir = EXP01_GENERATED_DATABASES_DIR / "master_world"
    world_path = output_dir / "world.json"
    manifest_path = output_dir / "manifest.json"
    for artifact_path in (world_path, manifest_path):
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"required master-world artifact does not exist: {artifact_path}"
            )

    manifest = read_json(manifest_path)
    world_sha256 = hash_file(world_path)
    configuration_sha256 = hash_file(DEFAULT_CONFIG_PATH)
    if manifest.get("world_json_sha256") != world_sha256:
        raise ValueError("master-world artifact does not match its manifest hash")
    if manifest.get("configuration_sha256") != configuration_sha256:
        raise ValueError("configuration does not match the master-world manifest")

    world = read_json(world_path)
    if world != build_master_world(config):
        raise ValueError("master-world artifact is not reproducible from the config")
    return world, world_sha256, configuration_sha256


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
    for output_path in (database_path, manifest_path):
        if not output_path.is_file():
            raise FileNotFoundError(
                f"required scaffold output file does not exist: {output_path}"
            )

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


def main() -> None:
    config = load_config()
    world, world_sha256, configuration_sha256 = _load_verified_master_world(config)
    data = config["data"]

    t_sweep_n = data["t_sweep"]["fact_count"]
    for table_count in data["t_sweep"]["table_counts"]:
        _materialize_condition(
            config=config,
            world=world,
            sweep="t_sweep",
            table_count=table_count,
            logical_fact_count=t_sweep_n,
            condition_dir=t_sweep_database_dir(table_count),
            master_world_sha256=world_sha256,
            configuration_sha256=configuration_sha256,
        )

    n_sweep_table_count = data["n_sweep"]["table_count"]
    shared_condition_dir = t_sweep_database_dir(n_sweep_table_count)
    for logical_fact_count in data["n_sweep"]["fact_counts"]:
        condition_dir = n_sweep_database_dir(logical_fact_count)
        if condition_dir == shared_condition_dir and logical_fact_count == t_sweep_n:
            continue
        _materialize_condition(
            config=config,
            world=world,
            sweep="n_sweep",
            table_count=n_sweep_table_count,
            logical_fact_count=logical_fact_count,
            condition_dir=condition_dir,
            master_world_sha256=world_sha256,
            configuration_sha256=configuration_sha256,
        )

    if data["optional_n40k"]["enabled"]:
        raise NotImplementedError("optional N40K materialization is not enabled in Step 3")


if __name__ == "__main__":
    main()

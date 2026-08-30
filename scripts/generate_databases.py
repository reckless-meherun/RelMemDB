import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG_PATH, load_config
from data.world import build_master_world, build_master_world_manifest
from utils.hashing import hash_file
from utils.io import write_json
from utils.paths import EXP01_GENERATED_DATABASES_DIR


def main() -> None:
    config = load_config()
    world = build_master_world(config)

    output_dir = EXP01_GENERATED_DATABASES_DIR / "master_world"
    world_path = output_dir / "world.json"
    manifest_path = output_dir / "manifest.json"
    for output_path in (world_path, manifest_path):
        if not output_path.is_file():
            raise FileNotFoundError(
                f"required scaffold output file does not exist: {output_path}"
            )

    write_json(world_path, world)
    world_sha256 = hash_file(world_path)
    manifest = build_master_world_manifest(
        config,
        configuration_sha256=hash_file(DEFAULT_CONFIG_PATH),
        world_sha256=world_sha256,
    )
    write_json(manifest_path, manifest)

    construction = world["construction"]
    print(f"chains: {construction['total_chains']}")
    print(
        "logical atomic facts: "
        f"{construction['total_logical_atomic_facts']}"
    )
    print(f"world: {world_path.relative_to(PROJECT_ROOT)}")
    print(f"world SHA-256: {world_sha256}")


if __name__ == "__main__":
    main()

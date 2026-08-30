import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config
from data.qa import generate_condition_qa
from utils.io import write_json
from utils.paths import (
    n_sweep_database_dir,
    n_sweep_qa_dir,
    t_sweep_database_dir,
    t_sweep_qa_dir,
)


HOP_NAMES = ("H0", "H1", "H2", "H3")


def _generate_condition(
    *,
    config: dict,
    table_count: int,
    logical_fact_count: int,
    database_dir: Path,
    qa_dir: Path,
) -> dict:
    database_path = database_dir / "database.sqlite"
    database_manifest_path = database_dir / "manifest.json"
    output_paths = {hop: qa_dir / f"{hop}.jsonl" for hop in HOP_NAMES}
    qa_manifest_path = qa_dir / "manifest.json"
    for required_path in (
        database_path,
        database_manifest_path,
        *output_paths.values(),
        qa_manifest_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"required scaffold file does not exist: {required_path}"
            )

    manifest = generate_condition_qa(
        config,
        database_path=database_path,
        database_manifest_path=database_manifest_path,
        output_paths=output_paths,
        expected_table_count=table_count,
        expected_logical_fact_count=logical_fact_count,
    )
    write_json(qa_manifest_path, manifest)
    print(
        f"T={table_count}, N={logical_fact_count}: "
        f"H0={manifest['H0_count']}, H1={manifest['H1_count']}, "
        f"H2={manifest['H2_count']}, H3={manifest['H3_count']}"
    )
    return manifest


def _verify_t_sweep_identity(qa_directories: list[Path]) -> None:
    for hop in HOP_NAMES:
        contents = [(qa_dir / f"{hop}.jsonl").read_bytes() for qa_dir in qa_directories]
        if any(content != contents[0] for content in contents[1:]):
            raise ValueError(f"T-sweep {hop} files are not byte-identical")


def _verify_n_sweep_nesting(
    n5k_qa_dir: Path, n10k_qa_dir: Path, n20k_qa_dir: Path
) -> None:
    for hop in HOP_NAMES:
        n5k = (n5k_qa_dir / f"{hop}.jsonl").read_bytes()
        n10k = (n10k_qa_dir / f"{hop}.jsonl").read_bytes()
        n20k = (n20k_qa_dir / f"{hop}.jsonl").read_bytes()
        if not len(n5k) < len(n10k) < len(n20k):
            raise ValueError(f"N-sweep {hop} files are not strictly increasing")
        if not n10k.startswith(n5k) or not n20k.startswith(n10k):
            raise ValueError(f"N-sweep {hop} files are not nested prefixes")


def main() -> None:
    config = load_config()
    data = config["data"]
    t_sweep_n = data["t_sweep"]["fact_count"]
    t_sweep_qa_directories: list[Path] = []

    for table_count in data["t_sweep"]["table_counts"]:
        qa_dir = t_sweep_qa_dir(table_count)
        _generate_condition(
            config=config,
            table_count=table_count,
            logical_fact_count=t_sweep_n,
            database_dir=t_sweep_database_dir(table_count),
            qa_dir=qa_dir,
        )
        t_sweep_qa_directories.append(qa_dir)

    n_sweep_table_count = data["n_sweep"]["table_count"]
    shared_database_dir = t_sweep_database_dir(n_sweep_table_count)
    shared_qa_dir = t_sweep_qa_dir(n_sweep_table_count)
    n_sweep_qa_directories: dict[int, Path] = {t_sweep_n: shared_qa_dir}
    for logical_fact_count in data["n_sweep"]["fact_counts"]:
        database_dir = n_sweep_database_dir(logical_fact_count)
        qa_dir = n_sweep_qa_dir(logical_fact_count)
        if database_dir == shared_database_dir and logical_fact_count == t_sweep_n:
            continue
        _generate_condition(
            config=config,
            table_count=n_sweep_table_count,
            logical_fact_count=logical_fact_count,
            database_dir=database_dir,
            qa_dir=qa_dir,
        )
        n_sweep_qa_directories[logical_fact_count] = qa_dir

    if data["optional_n40k"]["enabled"]:
        raise NotImplementedError("optional N40K QA generation is disabled in Step 5")

    _verify_t_sweep_identity(t_sweep_qa_directories)
    _verify_n_sweep_nesting(
        n_sweep_qa_directories[5_000],
        n_sweep_qa_directories[10_000],
        n_sweep_qa_directories[20_000],
    )


if __name__ == "__main__":
    main()

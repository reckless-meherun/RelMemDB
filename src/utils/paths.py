import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
GENERATED_DATABASES_DIR = DATASETS_DIR / "generated_databases"
QA_DIR = DATASETS_DIR / "qa"
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base_models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

EXP01_NAME = "exp01_first_feasibility"
EXP01_GENERATED_DATABASES_DIR = GENERATED_DATABASES_DIR / EXP01_NAME
EXP01_QA_DIR = QA_DIR / EXP01_NAME
EXP01_RUNS_DIR = RUNS_DIR / EXP01_NAME
EXP01_RESULTS_DIR = RESULTS_DIR / EXP01_NAME


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fact_count_label(fact_count: int) -> str:
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count % 1000 == 0:
        return f"N{fact_count // 1000}K"
    return f"N{fact_count}"


def t_sweep_database_dir(table_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    return EXP01_GENERATED_DATABASES_DIR / "t_sweep_N10K" / f"T{table_count}"


def n_sweep_database_dir(fact_count: int) -> Path:
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return t_sweep_database_dir(8)
    return EXP01_GENERATED_DATABASES_DIR / "n_sweep_T8" / _fact_count_label(fact_count)


def database_condition_dir(table_count: int, fact_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return t_sweep_database_dir(table_count)
    if table_count == 8:
        return n_sweep_database_dir(fact_count)
    raise ValueError(
        "non-N10K database conditions must use the n_sweep_T8 architecture"
    )


def cpt_database_dir(table_count: int, fact_count: int) -> Path:
    return database_condition_dir(table_count, fact_count) / "cpt"


def cpt_run_dir(table_count: int, fact_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return (
            EXP01_RUNS_DIR
            / "t_sweep_N10K"
            / f"T{table_count}"
            / "PLACEHOLDER_RUN"
        )
    if table_count == 8:
        return (
            EXP01_RUNS_DIR
            / "n_sweep_T8"
            / _fact_count_label(fact_count)
            / "PLACEHOLDER_RUN"
        )
    raise ValueError("non-N10K CPT runs must use the n_sweep_T8 architecture")


def t_sweep_qa_dir(table_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    return EXP01_QA_DIR / "t_sweep_N10K" / f"T{table_count}"


def n_sweep_qa_dir(fact_count: int) -> Path:
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return t_sweep_qa_dir(8)
    return EXP01_QA_DIR / "n_sweep_T8" / _fact_count_label(fact_count)


def qa_condition_dir(table_count: int, fact_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return t_sweep_qa_dir(table_count)
    if table_count == 8:
        return n_sweep_qa_dir(fact_count)
    raise ValueError("non-N10K QA conditions must use the n_sweep_T8 architecture")


def evaluation_result_dir(
    table_count: int, fact_count: int, split: str, run_name: str
) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if not isinstance(run_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name
    ):
        raise ValueError("run_name must be a filesystem-safe name")
    if fact_count == 10_000:
        condition_dir = EXP01_RESULTS_DIR / "t_sweep_N10K" / f"T{table_count}"
    elif table_count == 8:
        condition_dir = (
            EXP01_RESULTS_DIR / "n_sweep_T8" / _fact_count_label(fact_count)
        )
    else:
        raise ValueError(
            "non-N10K evaluation conditions must use the n_sweep_T8 architecture"
        )
    return condition_dir / split / run_name


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

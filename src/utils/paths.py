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


def t_sweep_qa_dir(table_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    return EXP01_QA_DIR / "t_sweep_N10K" / f"T{table_count}"


def n_sweep_qa_dir(fact_count: int) -> Path:
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return t_sweep_qa_dir(8)
    return EXP01_QA_DIR / "n_sweep_T8" / _fact_count_label(fact_count)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

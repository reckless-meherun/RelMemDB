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
EXP02_NAME = "exp02_capacity_boundary"
EXP02_GENERATED_DATABASES_DIR = GENERATED_DATABASES_DIR / EXP02_NAME
EXP02_QA_DIR = QA_DIR / EXP02_NAME
EXP02_RUNS_DIR = RUNS_DIR / EXP02_NAME
EXP02_RESULTS_DIR = RESULTS_DIR / EXP02_NAME


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
    return (
        EXP01_GENERATED_DATABASES_DIR
        / "conditions"
        / f"T{table_count}_{_fact_count_label(fact_count)}"
    )


def cpt_database_dir(table_count: int, fact_count: int) -> Path:
    return database_condition_dir(table_count, fact_count) / "cpt"


def _run_condition_dir(table_count: int, fact_count: int) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    if fact_count == 10_000:
        return EXP01_RUNS_DIR / "t_sweep_N10K" / f"T{table_count}"
    if table_count == 8:
        return EXP01_RUNS_DIR / "n_sweep_T8" / _fact_count_label(fact_count)
    return (
        EXP01_RUNS_DIR
        / "conditions"
        / f"T{table_count}_{_fact_count_label(fact_count)}"
    )


def cpt_run_dir(table_count: int, fact_count: int, layers: int) -> Path:
    layers = _positive_int(layers, "layers")
    return _run_condition_dir(table_count, fact_count) / f"L{layers}" / "cpt"


def target_sft_run_dir(table_count: int, fact_count: int, layers: int) -> Path:
    """Return the stage-specific run directory for closed-book target SFT."""
    layers = _positive_int(layers, "layers")
    return _run_condition_dir(table_count, fact_count) / f"L{layers}" / "target_sft"


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
    return (
        EXP01_QA_DIR / "conditions" / f"T{table_count}_{_fact_count_label(fact_count)}"
    )


def qa_reference_dir(config: dict) -> Path:
    reference = config.get("data", {}).get("qa_reference")
    if not isinstance(reference, dict):
        raise ValueError("data.qa_reference configuration is required")
    return qa_condition_dir(reference.get("table_count"), reference.get("fact_count"))


def evaluation_result_dir(
    table_count: int, fact_count: int, layers: int, split: str, run_name: str
) -> Path:
    table_count = _positive_int(table_count, "table_count")
    fact_count = _positive_int(fact_count, "fact_count")
    layers = _positive_int(layers, "layers")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if not isinstance(run_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name
    ):
        raise ValueError("run_name must be a filesystem-safe name")
    if fact_count == 10_000:
        condition_dir = EXP01_RESULTS_DIR / "t_sweep_N10K" / f"T{table_count}"
    elif table_count == 8:
        condition_dir = EXP01_RESULTS_DIR / "n_sweep_T8" / _fact_count_label(fact_count)
    else:
        condition_dir = (
            EXP01_RESULTS_DIR
            / "conditions"
            / f"T{table_count}_{_fact_count_label(fact_count)}"
        )
    return condition_dir / f"L{layers}" / split / run_name


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    if not normalized:
        raise ValueError("path component is empty after sanitization")
    return normalized


def exp2_condition_label(
    selected_tables: list[str] | tuple[str, ...], fact_count: int, timestamp: str
) -> str:
    if not selected_tables:
        raise ValueError("selected_tables must not be empty")
    if not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", timestamp):
        raise ValueError("timestamp must use YYYYMMDD_HHMMSS[_ffffff]")
    tables = "-".join(safe_component(table) for table in selected_tables)
    return f"T{len(selected_tables):02d}_N{_positive_int(fact_count, 'fact_count')}_{tables}_{timestamp}"


def exp2_database_bundle_dir(
    selected_tables: list[str] | tuple[str, ...], fact_count: int, timestamp: str
) -> Path:
    return EXP02_GENERATED_DATABASES_DIR / exp2_condition_label(
        selected_tables, fact_count, timestamp
    )


def exp2_qa_bundle_dir(
    selected_tables: list[str] | tuple[str, ...], fact_count: int, timestamp: str
) -> Path:
    return EXP02_QA_DIR / exp2_condition_label(selected_tables, fact_count, timestamp)


def exp2_artifact_stem(
    *, model: str, table_count: int, fact_count: int, layers: int, timestamp: str
) -> str:
    return (
        f"{safe_component(model)}_exp02_T{_positive_int(table_count, 'table_count'):02d}_"
        f"N{_positive_int(fact_count, 'fact_count')}_L{_positive_int(layers, 'layers')}_"
        f"{timestamp}"
    )

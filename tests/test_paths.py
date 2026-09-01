from config import DEFAULT_CONFIG_PATH, load_config
from utils.paths import (
    BASE_MODELS_DIR,
    CONFIGS_DIR,
    DATASETS_DIR,
    DOCS_DIR,
    EXP01_GENERATED_DATABASES_DIR,
    EXP01_QA_DIR,
    EXP01_RESULTS_DIR,
    EXP01_RUNS_DIR,
    GENERATED_DATABASES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    QA_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRAINED_MODELS_DIR,
    cpt_database_dir,
    cpt_run_dir,
    database_condition_dir,
    n_sweep_database_dir,
    n_sweep_qa_dir,
    t_sweep_database_dir,
    t_sweep_qa_dir,
)


def test_project_root_detection() -> None:
    assert PROJECT_ROOT == PROJECT_ROOT.resolve()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert DEFAULT_CONFIG_PATH == CONFIGS_DIR / "exp01_first_feasibility.yaml"


def test_main_directory_constants() -> None:
    assert CONFIGS_DIR == PROJECT_ROOT / "configs"
    assert DATASETS_DIR == PROJECT_ROOT / "datasets"
    assert GENERATED_DATABASES_DIR == DATASETS_DIR / "generated_databases"
    assert QA_DIR == DATASETS_DIR / "qa"
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert BASE_MODELS_DIR == PROJECT_ROOT / "models" / "base_models"
    assert TRAINED_MODELS_DIR == PROJECT_ROOT / "models" / "trained_models"
    assert RUNS_DIR == PROJECT_ROOT / "runs"
    assert RESULTS_DIR == PROJECT_ROOT / "results"
    assert DOCS_DIR == PROJECT_ROOT / "docs"
    assert EXP01_GENERATED_DATABASES_DIR == GENERATED_DATABASES_DIR / "exp01_first_feasibility"
    assert EXP01_QA_DIR == QA_DIR / "exp01_first_feasibility"
    assert EXP01_RUNS_DIR == RUNS_DIR / "exp01_first_feasibility"
    assert EXP01_RESULTS_DIR == RESULTS_DIR / "exp01_first_feasibility"


def test_t8_n10k_database_reuse() -> None:
    assert n_sweep_database_dir(10_000) == t_sweep_database_dir(8)
    assert "n_sweep_T8/N10K" not in n_sweep_database_dir(10_000).as_posix()


def test_t8_n10k_qa_reuse() -> None:
    assert n_sweep_qa_dir(10_000) == t_sweep_qa_dir(8)
    assert "n_sweep_T8/N10K" not in n_sweep_qa_dir(10_000).as_posix()


def test_non_reused_n_database_conditions() -> None:
    for fact_count, label in ((5_000, "N5K"), (20_000, "N20K")):
        path = n_sweep_database_dir(fact_count)
        assert path == EXP01_GENERATED_DATABASES_DIR / "n_sweep_T8" / label
        assert "t_sweep_N10K" not in path.parts


def test_non_reused_n_qa_conditions() -> None:
    for fact_count, label in ((5_000, "N5K"), (20_000, "N20K")):
        path = n_sweep_qa_dir(fact_count)
        assert path == EXP01_QA_DIR / "n_sweep_T8" / label
        assert "t_sweep_N10K" not in path.parts


def test_default_experiment_config() -> None:
    config = load_config()

    assert config["data"]["t_sweep"]["table_counts"] == [4, 8, 12]
    assert config["data"]["t_sweep"]["fact_count"] == 10_000
    assert config["data"]["n_sweep"]["fact_counts"] == [5_000, 10_000, 20_000]
    assert config["data"]["n_sweep"]["table_count"] == 8
    assert config["data"]["hops"] == [0, 1, 2, 3]
    assert config["data"]["canonical_target"] == {
        "table_count": 12,
        "fact_count": 10_000,
    }
    assert config["data"]["master_world"] == {
        "latent_positions": 12,
        "descriptive_facts_per_chain": 29,
        "relation_facts_per_chain": 11,
        "experimental_facts_per_chain": 40,
        "identifier_fields_per_chain": 12,
    }
    assert config["training"]["fact_exposure"] == 4
    assert config["training"]["cpt_batch_size"] == 32
    assert config["layer_study"]["enabled"] is False


def test_canonical_cpt_paths() -> None:
    condition = EXP01_GENERATED_DATABASES_DIR / "t_sweep_N10K" / "T12"
    assert database_condition_dir(12, 10_000) == condition
    assert cpt_database_dir(12, 10_000) == condition / "cpt"
    assert cpt_run_dir(12, 10_000) == (
        EXP01_RUNS_DIR / "t_sweep_N10K" / "T12" / "PLACEHOLDER_RUN"
    )

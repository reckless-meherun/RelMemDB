# RelMemDB

RelMemDB studies whether language models can **internalize a relational database** and answer natural-language questions directly from model parameters, without SQL generation, database execution, retrieval, or database access at inference time.

This repository contains two experiments:

- **Experiment 1** — the original fixed T/N/L feasibility study.
- **Experiment 2** — the current capacity-boundary study with flexible table selection, fact count, model, and layer depth.

---

# Experiment 2

Experiment 2 varies four main quantities:

- **T** — number of explicitly selected canonical tables.
- **N** — number of exposed logical database facts.
- **L** — transformer architecture depth.
- **M** — pretrained model identity.

The default config is:

```text
configs/exp02_capacity_boundary.yaml
```

The single-command runner is:

```text
scripts/run_exp02.py
```

## Run Experiment 2

From the repository root:

```bash
cd /home/hpc4090/meherun/relmemdb
```

A complete Experiment-2 run can now be launched with one command:

```bash
python3 scripts/run_exp02.py \
  --tables continent \
  --fact-count 500 \
  --model gpt2 \
  --layers 12 \
  --base-model models/base_models/gpt2 \
  --cpt-epochs 1000 \
  --sft-epochs 1000 \
  --cpt-batch-size 4 \
  --cpt-gradient-accumulation 8
```

This automatically runs:

```text
Database generation
        ↓
CPT
        ↓
QA + target-SFT generation
        ↓
CPT validation evaluation
        ↓
Target SFT
        ↓
SFT validation evaluation
```

The test split is **not evaluated by default**.

---

## Main arguments

```text
--tables
```

Canonical tables to expose.

Examples:

```bash
--tables continent
```

```bash
--tables continent country
```

```text
--fact-count
```

Exact exposed logical fact count N.

Example:

```bash
--fact-count 500
```

If omitted, Experiment 2 keeps the current baseline chain count and derives N automatically.

```text
--model
```

Model identity. Current default:

```bash
--model gpt2
```

```text
--layers
```

Actual transformer architecture depth.

For GPT-2:

```bash
--layers 12
```

```text
--base-model
```

Local pretrained checkpoint.

Example:

```bash
--base-model models/base_models/gpt2
```

---

## Training overrides

You can change training settings directly in the command:

```text
--cpt-epochs
--sft-epochs
--cpt-batch-size
--cpt-gradient-accumulation
--sft-batch-size
--sft-gradient-accumulation
--cpt-learning-rate
--sft-learning-rate
```

These overrides do **not** modify the committed Experiment-2 config. The runner creates a run-specific resolved config automatically.

---

## Resume an interrupted run

Existing artifacts can be reused explicitly with:

```text
--dataset-path
--qa-path
--cpt-checkpoint
--sft-checkpoint
```

Example:

```bash
python3 scripts/run_exp02.py \
  --tables continent \
  --fact-count 500 \
  --model gpt2 \
  --layers 12 \
  --dataset-path <EXACT_DATASET_PATH> \
  --cpt-checkpoint <EXACT_CPT_CHECKPOINT>
```

The runner verifies and reuses the supplied artifacts instead of regenerating them.

---

## Evaluate test explicitly

Validation is used by default.

To evaluate the final SFT checkpoint on test, add:

```bash
--evaluate-test
```

Do not repeatedly use the test split while selecting experimental settings.

---

# Experiment-2 outputs

## Generated databases

```text
datasets/generated_databases/exp02_capacity_boundary/
```

Example:

```text
T01_N500_continent_<timestamp>/
├── database.sqlite
├── manifest.json
└── cpt/
    ├── book_readable.txt
    ├── train.txt
    └── manifest.json
```

## QA

```text
datasets/qa/exp02_capacity_boundary/
```

Example:

```text
T01_N500_continent_<timestamp>/
├── split_manifest.json
├── validation/
├── test/
└── target_sft/
    ├── split_manifest.json
    ├── train/
    └── dev/
```

## Trained checkpoints

```text
models/trained_models/
```

## Training runs

```text
runs/exp02_capacity_boundary/
```

Each full pipeline run also stores:

```text
runs/exp02_capacity_boundary/pipeline_runs/<timestamp>/
├── resolved_config.yaml
└── pipeline_state.json
```

## Evaluation results

```text
results/exp02_capacity_boundary/
└── t_sweep/
    └── T{T}/
        └── n_sweep/
            └── N{N}/
                ├── validation/
                │   ├── eval_cpt/
                │   └── eval_sft/
                └── test/
                    └── eval_sft/
```

Each evaluation directory contains:

```text
evaluation_config.json
metrics.json
predictions.jsonl
```

---

# Fact counting in Experiment 2

N counts exposed logical database facts:

- each exposed non-ID attribute value = **1 fact**
- each exposed FK relation whose source table is selected = **1 fact**
- IDs do **not** count

For `continent`, each row contributes:

```text
continent_name
climate_band
```

So:

```text
250 continent rows × 2 facts = N500
```

For `continent + country`, each chain contributes five exposed facts, so N must be divisible by five.

---

# Experiment 1

Experiment 1 remains available and unchanged.

Example CPT training:

```bash
python3 scripts/train.py \
  --stage cpt \
  --table-count 12 \
  --fact-count 10000 \
  --layers 12 \
  --source-checkpoint models/base_models/gpt2 \
  --config configs/exp01_first_feasibility.yaml
```

Example validation evaluation:

```bash
python3 scripts/evaluate.py \
  --table-count 12 \
  --fact-count 10000 \
  --layers 12 \
  --split validation \
  --checkpoint <CHECKPOINT_PATH> \
  --run-name post_cpt \
  --config configs/exp01_first_feasibility.yaml
```

---

# Tests

Run the CPU test suite with:

```bash
python3 -m pytest -q
```

# RelMemDB

RelMemDB studies closed-book memory for deterministic relational databases. The
original Experiment 1 fixed sweeps remain available with their existing commands
and artifact paths. Experiment 2 adds independent, CLI-selected capacity
conditions without changing the canonical database or Experiment-1 semantics.

## Experiment 2: capacity boundary

The four experimental variables are:

- **T** — number of canonical tables explicitly selected with `--tables`.
  Hidden/supporting parent tables required for foreign-key integrity do not
  increase T.
- **N** — number of unique exposed logical facts. Each non-ID attribute-value
  instance counts as one fact, and each FK relation instance whose source table
  is selected counts as one fact. IDs do not count.
- **L** — actual transformer architecture depth. It is not the number of
  unfrozen layers.
- **M** — pretrained model identity. The current default is `gpt2`, with native
  depth L=12.

Experiment 2 has no fixed T/N sweep arrays. Every dataset, QA bundle, run,
checkpoint, and evaluation carries provenance linking it to its exact upstream
artifacts.

### Repository paths

Assuming the repository is located at:

```text
/home/hpc4090/meherun/relmemdb
```

the important Experiment-2 locations are:

```text
Config:
  /home/hpc4090/meherun/relmemdb/configs/exp02_capacity_boundary.yaml

Base GPT-2:
  /home/hpc4090/meherun/relmemdb/models/base_models/gpt2

Generated databases:
  /home/hpc4090/meherun/relmemdb/datasets/generated_databases/exp02_capacity_boundary/

Generated QA:
  /home/hpc4090/meherun/relmemdb/datasets/qa/exp02_capacity_boundary/

Training runs:
  /home/hpc4090/meherun/relmemdb/runs/exp02_capacity_boundary/

Trained checkpoints:
  /home/hpc4090/meherun/relmemdb/models/trained_models/

Evaluation results:
  /home/hpc4090/meherun/relmemdb/results/exp02_capacity_boundary/
```

## First Experiment-2 run: T=1, N=500, GPT-2 L12

Use `continent` as the single exposed table.

Start from the repository root:

```bash
cd /home/hpc4090/meherun/relmemdb
set -euo pipefail

ROOT=/home/hpc4090/meherun/relmemdb
CONFIG=$ROOT/configs/exp02_capacity_boundary.yaml
BASE_MODEL=$ROOT/models/base_models/gpt2
```

Verify the local GPT-2 checkpoint:

```bash
test -f "$BASE_MODEL/config.json" && echo "GPT-2 checkpoint ready: $BASE_MODEL"
```

### 1. Generate the T=1, N=500 database bundle

```bash
GEN_LOG=$(mktemp)

python3 "$ROOT/scripts/generate_databases.py" \
  --config "$CONFIG" \
  --tables continent \
  --fact-count 500 | tee "$GEN_LOG"

DATASET_REL=$(sed -n 's/^Output: //p' "$GEN_LOG" | tail -1)
DATASET_PATH="$ROOT/$DATASET_REL"
rm "$GEN_LOG"

echo "DATASET_PATH=$DATASET_PATH"
```

The generated bundle has the form:

```text
datasets/generated_databases/exp02_capacity_boundary/
└── T01_N500_continent_<timestamp>/
    ├── database.sqlite
    ├── manifest.json
    └── cpt/
        ├── book_readable.txt
        ├── train.txt
        └── manifest.json
```

For `continent`, each chain contributes two logical facts. Therefore N=500
corresponds to 250 chains.

Verify the bundle:

```bash
test -f "$DATASET_PATH/database.sqlite"
test -f "$DATASET_PATH/manifest.json"
test -f "$DATASET_PATH/cpt/book_readable.txt"
test -f "$DATASET_PATH/cpt/train.txt"
test -f "$DATASET_PATH/cpt/manifest.json"
```

### 2. CPT-train GPT-2 at L=12

```bash
CPT_LOG=$(mktemp)

python3 "$ROOT/scripts/train.py" \
  --config "$CONFIG" \
  --stage cpt \
  --training-data-dir "$DATASET_PATH" \
  --model gpt2 \
  --layers 12 \
  --source-checkpoint "$BASE_MODEL" 2>&1 | tee "$CPT_LOG"

CPT_CHECKPOINT=$(sed -n 's/.*checkpoint=\(.*\)$/\1/p' "$CPT_LOG" | tail -1)
rm "$CPT_LOG"

echo "CPT_CHECKPOINT=$CPT_CHECKPOINT"
```

The checkpoint is created under:

```text
models/trained_models/
└── gpt2_exp02_T01_N500_L12_<timestamp>/
```

Verify it:

```bash
test -f "$CPT_CHECKPOINT/config.json"
test -f "$CPT_CHECKPOINT/training_metadata.json"
```

### 3. Generate evaluation and target-SFT QA

```bash
QA_LOG=$(mktemp)

python3 "$ROOT/scripts/generate_target_sft_qa.py" \
  --config "$CONFIG" \
  --training-data-dir "$DATASET_PATH" | tee "$QA_LOG"

QA_REL=$(sed -n 's/^Target-SFT QA output: //p' "$QA_LOG" | tail -1)
QA_PATH="$ROOT/$QA_REL"
SFT_DATA_DIR="$QA_PATH/target_sft"
rm "$QA_LOG"

echo "QA_PATH=$QA_PATH"
echo "SFT_DATA_DIR=$SFT_DATA_DIR"
```

The generated QA structure is:

```text
datasets/qa/exp02_capacity_boundary/
└── T01_N500_continent_<timestamp>/
    ├── split_manifest.json
    ├── validation/
    ├── test/
    └── target_sft/
        ├── split_manifest.json
        ├── train/
        └── dev/
```

QA is generated only when all facts required to support the question are exposed.
Unavailable hop categories are valid empty outputs.

### 4. Evaluate CPT on validation

```bash
python3 "$ROOT/scripts/evaluate.py" \
  --config "$CONFIG" \
  --qa-data-dir "$QA_PATH" \
  --checkpoint "$CPT_CHECKPOINT" \
  --layers 12 \
  --split validation
```

Experiment-2 results are written to:

```text
results/exp02_capacity_boundary/
└── t_sweep/
    └── T01/
        └── n_sweep/
            └── N500/
                └── validation/
                    └── eval_cpt/
                        └── HH-MM-SS_DD-MM-YYYY/
                            ├── evaluation_config.json
                            ├── metrics.json
                            └── predictions.jsonl
```

### 5. Train target SFT

Use the exact `target_sft` directory generated above and the exact CPT
checkpoint:

```bash
SFT_LOG=$(mktemp)

python3 "$ROOT/scripts/train.py" \
  --config "$CONFIG" \
  --stage target-sft \
  --sft-data-dir "$SFT_DATA_DIR" \
  --source-checkpoint "$CPT_CHECKPOINT" \
  --model gpt2 \
  --layers 12 2>&1 | tee "$SFT_LOG"

SFT_CHECKPOINT=$(sed -n 's/.*checkpoint=\(.*\)$/\1/p' "$SFT_LOG" | tail -1)
rm "$SFT_LOG"

echo "SFT_CHECKPOINT=$SFT_CHECKPOINT"
```

The resulting checkpoint is created under:

```text
models/trained_models/
└── gpt2_exp02_T01_N500_L12_<timestamp>_sft/
```

### 6. Evaluate SFT on validation

```bash
python3 "$ROOT/scripts/evaluate.py" \
  --config "$CONFIG" \
  --qa-data-dir "$QA_PATH" \
  --checkpoint "$SFT_CHECKPOINT" \
  --layers 12 \
  --split validation
```

The output is written to:

```text
results/exp02_capacity_boundary/
└── t_sweep/
    └── T01/
        └── n_sweep/
            └── N500/
                └── validation/
                    └── eval_sft/
                        └── HH-MM-SS_DD-MM-YYYY/
                            ├── evaluation_config.json
                            ├── metrics.json
                            └── predictions.jsonl
```

Evaluation authenticates the checkpoint against the QA condition before
inference, including experiment identity, T, N, selected tables, database and
manifest provenance, architecture depth, model identity, and CPT/SFT stage.

### 7. Inspect the validation outputs

```bash
find "$ROOT/results/exp02_capacity_boundary/t_sweep/T01/n_sweep/N500/validation" \
  -type f \
  \( -name "metrics.json" -o -name "evaluation_config.json" \) \
  | sort
```

Do not use the test split repeatedly while selecting T/N/model/layer conditions.
Use validation for development and reserve test evaluation for the finalized
experimental setup.

## Generating other Experiment-2 conditions

A baseline run can omit N. The canonical chain count is retained and N is
computed from the selected tables:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent
```

For a chosen valid N:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent \
  --fact-count 1000
```

For two selected tables:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent country \
  --fact-count 5000
```

For `continent + country`, each chain contributes five exposed logical facts, so
N must be exactly divisible by five.

## Experiment-2 result organization

Results currently organize the T/N capacity study as:

```text
results/exp02_capacity_boundary/
└── t_sweep/
    ├── T01/
    │   └── n_sweep/
    ├── T02/
    │   └── n_sweep/
    ├── T03/
    │   └── n_sweep/
    └── T04/
        └── n_sweep/
```

Actual evaluation runs dynamically create:

```text
T{T:02d}/
└── n_sweep/
    └── N{exact_N}/
        ├── validation/
        │   ├── eval_cpt/
        │   │   └── HH-MM-SS_DD-MM-YYYY/
        │   └── eval_sft/
        │       └── HH-MM-SS_DD-MM-YYYY/
        └── test/
            ├── eval_cpt/
            └── eval_sft/
```

Each timestamped evaluation directory contains:

```text
evaluation_config.json
metrics.json
predictions.jsonl
```

Model identity, architecture depth, selected tables, QA provenance, checkpoint
provenance, and database hashes are stored inside `evaluation_config.json`
rather than encoded in the result path.

## Experiment 1

Existing Experiment-1 commands and path semantics remain unchanged.

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
  --checkpoint models/trained_models/gpt2_cpt_t12_n10k_e20 \
  --run-name post_cpt_e20 \
  --config configs/exp01_first_feasibility.yaml
```

## Tests

Run the CPU test suite with:

```bash
python3 -m pytest -q
```
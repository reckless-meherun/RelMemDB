# RelMemDB

RelMemDB studies closed-book memory for deterministic relational databases. The
original Experiment 1 fixed sweeps remain available with their existing commands
and artifact paths. Experiment 2 adds independent, CLI-selected capacity
conditions without changing the canonical database or Experiment-1 semantics.

## Experiment 2: capacity boundary

The four variables have precise meanings:

- **T** is the number of canonical tables explicitly selected with `--tables`.
  Hidden parent tables needed for SQLite foreign keys do not increase T.
- **N** is the number of unique exposed logical facts: every non-ID attribute
  instance plus every FK relation instance whose source table is selected. IDs
  never count. N is not a row, sentence, byte, character, or token count.
- **L** is the actual transformer architecture depth. It is not a count of
  unfrozen layers; CPT remains full-parameter training.
- **M** is the selected registered pretrained model. `gpt2` is the default and
  its native depth is L=12.

All commands are independent; Experiment 2 has no configured T/N sweep arrays.
Generated datasets, QA bundles, runs, checkpoints, and results are timestamped
and their manifests carry exact upstream paths and hashes.

Generate a baseline using the canonical chain count and observe N:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent
```

Generate a chosen valid N:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent \
  --fact-count 1000
```

Select two tables. Here each chain contributes five facts, so N must be divisible
by five:

```bash
python3 scripts/generate_databases.py \
  --config configs/exp02_capacity_boundary.yaml \
  --tables continent country \
  --fact-count 5000
```

The command prints the exact timestamped `<DATASET_PATH>`. CPT requires that
bundle explicitly and defaults to local `models/base_models/gpt2` at native L=12:

```bash
python3 scripts/train.py \
  --config configs/exp02_capacity_boundary.yaml \
  --stage cpt \
  --training-data-dir <DATASET_PATH>
```

Use `--model <registered-name>` and `--layers <depth>` to override the defaults.
Models resolve only from local checkpoints; the pipeline does not download a
model during an experiment. A depth override must match the checkpoint's actual
architecture.

Generate deterministic evaluation and target-SFT QA from the same exact bundle:

```bash
python3 scripts/generate_target_sft_qa.py \
  --config configs/exp02_capacity_boundary.yaml \
  --training-data-dir <DATASET_PATH>
```

Only selected attributes and selected-source relations can produce H0 support.
H1–H3 records are emitted only when every relation and target attribute on the
support path was exposed; unavailable hop files are valid empty JSONL files.

Train target SFT from an exact QA artifact and CPT checkpoint:

```bash
python3 scripts/train.py \
  --config configs/exp02_capacity_boundary.yaml \
  --stage target-sft \
  --sft-data-dir <QA_PATH> \
  --source-checkpoint <CPT_CHECKPOINT>
```

Evaluate with explicit QA and checkpoint paths:

```bash
python3 scripts/evaluate.py \
  --config configs/exp02_capacity_boundary.yaml \
  --qa-data-dir <QA_PATH> \
  --checkpoint <CHECKPOINT> \
  --split validation
```

N controls model-independent database knowledge content. Training metadata
separately records model/tokenizer identity, readable-book tokens, train tokens,
sequence count, epochs, optimizer steps, and effective fact exposure.

## Experiment 1 examples

Existing Experiment-1 arguments and T/N meanings are unchanged. For example:

```bash
python3 scripts/train.py \
  --stage cpt \
  --table-count 12 \
  --fact-count 10000 \
  --layers 12 \
  --source-checkpoint models/base_models/gpt2 \
  --config configs/exp01_first_feasibility.yaml
```

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

```bash
python3 -m pytest -q
```

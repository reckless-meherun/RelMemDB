# RelMemDB

## Training Base GPT-2 Model
```
python3 scripts/train.py \
  --stage cpt \
  --table-count 12 \
  --fact-count 10000 \
  --source-checkpoint models/base_models/gpt2 \
  --config configs/exp01_base_gpt2_cpt.yaml
```

## Validating Trained GPT-2 Model
```
python3 scripts/evaluate.py \
  --table-count 12 \
  --fact-count 10000 \
  --split validation \
  --checkpoint models/trained_models/gpt2_base_cpt_t12_n10k_e20 \
  --run-name base_gpt2_cpt_e20
  ```
## Test
```
python3 scripts/evaluate.py \
  --table-count 12 \
  --fact-count 10000 \
  --split test \
  --checkpoint models/trained_models/gpt2_cpt_t12_n10k_e20_sft_validation_e10 \
  --run-name cpt_e20_sft_validation_e10_test
  ```
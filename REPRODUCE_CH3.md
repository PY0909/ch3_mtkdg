# Chapter 3 Reproduction

This package contains the MTK-DG Chapter 3 implementation and three reference  
checkpoint artifacts. The raw datasets are not included. Set `DATA_ROOT` to a  
directory containing the C-MAPSS `CMAPSSData` files and  
`metropt+3+dataset/MetroPT3(AirCompressor).csv`.

The reference protocol is:

- FD002 and FD004: history length 50, engine-level split with split seed 2026,  
  per-condition train-only normalization, capped train/validation RUL at 125,  
  official raw test RUL, dynamic one-layer sensor graph with self-loops, gated  
  fusion, batch size 128, AdamW learning rate 2e-4, cosine schedule.
- MetroPT-3: first-month train/validation split, later months as test, history  
  168, prediction length 24, stride 60, train-only normalization, batch size  
  128, AdamW learning rate 2e-4, cosine schedule.

Use a new output directory for reproduction runs:

```bash
DATA_ROOT=/path/to/dataset
OUT_ROOT=/path/to/new/results

python scripts/run_experiment.py --dataset fd002 --model mtkdg --seed 2026 \
  --split-seed 2026 --device cuda --epochs 200 --patience 30 \
  --batch-size 128 --lr 2e-4 --lr-scheduler cosine \
  --norm-mode per_condition --evaluate-test \
  --run-id fd002_mtkdg_repro_seed2026 \
  --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT"

python scripts/run_experiment.py --dataset fd004 --model mtkdg --seed 2026 \
  --split-seed 2026 --device cuda --epochs 200 --patience 30 \
  --batch-size 128 --lr 2e-4 --lr-scheduler cosine \
  --norm-mode per_condition --evaluate-test \
  --run-id fd004_mtkdg_repro_seed2026 \
  --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT"

python scripts/run_experiment.py --dataset metropt3 --model mtkdg --seed 2026 \
  --device cuda --epochs 100 --patience 15 \
  --batch-size 128 --lr 2e-4 --lr-scheduler cosine --evaluate-test \
  --run-id metropt3_mtkdg_repro_seed2026 \
  --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT"
```

Run the data and model smoke tests with:

```bash
python -m pytest -q tests/test_realdata_smoke.py
python -m pytest -q
```

The `checkpoints/` directory contains validation-best reference artifacts for  
test-only comparison. Keep each checkpoint together with its `config.json` and  
`manifest.json`; the manifest records the split, normalizer, seed, and model  
provenance.

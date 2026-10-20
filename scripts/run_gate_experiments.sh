#!/bin/bash
# MC-Graph 门实验（评审指定的隔离消融，seed 2026，V1 协议）：
#   G1 corruption-only       : robust_train, availability none, consistency 0
#   G2 consistency-only      : robust_train, availability none, consistency 0.1
#   G3 mc-no-consistency     : availability both, robust_train, consistency 0
#   G4 full MC-Graph         : availability both, robust_train, consistency 0.1
# A0（frozen baseline）复用已有 V1 run，不重训。
# FD: 150ep/patience30/cosine/per-condition；MetroPT: 100ep/patience15/cosine。
set -u
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261014_gate
declare -A FLAGS
FLAGS[g1]="--robust-train"
FLAGS[g2]="--robust-train --consistency-lambda 0.1"
FLAGS[g3]="--availability-mode both --robust-train"
FLAGS[g4]="--availability-mode both --robust-train --consistency-lambda 0.1"

for g in g1 g2 g3 g4; do
  for ds in fd002 fd004; do
    rid="${ds}_mc${g}_seed2026_${TS}"
    if [ -f "results/$ds/$rid/test_metrics.json" ]; then echo "skip $rid"; continue; fi
    python scripts/run_experiment.py --model mtkdg --dataset $ds --seed 2026 --device cuda \
      --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine \
      --norm-mode per_condition --evaluate-test --run-id "$rid" \
      --data-root $DATA --output-dir results ${FLAGS[$g]} \
      > "logs/${rid}.log" 2>&1 && echo "done $rid"
  done
  rid="metropt3_mc${g}_seed2026_${TS}"
  if [ ! -f "results/metropt3/$rid/test_metrics.json" ]; then
    python scripts/run_experiment.py --model mtkdg --dataset metropt3 --seed 2026 --device cuda \
      --epochs 100 --patience 15 --batch-size 128 --lr-scheduler cosine \
      --evaluate-test --run-id "$rid" \
      --data-root $DATA --output-dir results ${FLAGS[$g]} \
      > "logs/${rid}.log" 2>&1 && echo "done $rid"
  else
    echo "skip $rid"
  fi
done

# 鲁棒性评估（新配置 + 冻结 A0 基线）
for ds in fd002 fd004; do
  python scripts/eval_robustness.py "results/$ds/${ds}_mtkdg_v1_percond_20261014_loop2" \
    --out "results/$ds/${ds}_mtkdg_v1_percond_20261014_loop2/robustness_eval.json"
  for g in g1 g2 g3 g4; do
    python scripts/eval_robustness.py "results/$ds/${ds}_mc${g}_seed2026_${TS}" \
      --out "results/$ds/${ds}_mc${g}_seed2026_${TS}/robustness_eval.json"
  done
done
python scripts/eval_robustness.py "results/metropt3/metropt3_mtkdg_seed2026_r2_20261013_211500" \
  --out "results/metropt3/metropt3_mtkdg_seed2026_r2_20261013_211500/robustness_eval.json"
for g in g1 g2 g3 g4; do
  python scripts/eval_robustness.py "results/metropt3/metropt3_mc${g}_seed2026_${TS}" \
    --out "results/metropt3/metropt3_mc${g}_seed2026_${TS}/robustness_eval.json"
done
echo "GATE EXPERIMENTS DONE"

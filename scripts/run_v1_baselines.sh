#!/bin/bash
# V1 统一重跑：5 baseline × FD002/FD004，per-condition 归一化
# 预算与 r2 baseline 一致（150/30 cosine）；MTK-DG V1 已有，跳过
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261014_loop2
for model in tcn gru patchtst timesnet itransformer; do
  for ds in fd002 fd004; do
    rid="${ds}_${model}_v1percond_${TS}"
    [ -f "results/$ds/$rid/test_metrics.json" ] && { echo "skip $rid"; continue; }
    python scripts/run_experiment.py --model $model --dataset $ds --seed 2026 --split-seed 2026 \
      --device cuda --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine \
      --norm-mode per_condition --evaluate-test \
      --run-id "$rid" --data-root $DATA --output-dir results \
      > "logs/${rid}.log" 2>&1 && echo "done $rid"
  done
done
echo "V1 BASELINES DONE"

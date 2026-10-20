#!/bin/bash
# 统一协议 baseline 矩阵：5 模型 × 3 数据集，seed 2026（与 MTK-DG r2 相同协议）
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261013_222400
for model in tcn gru patchtst timesnet itransformer; do
  python scripts/run_experiment.py --model $model --dataset fd002 --seed 2026 --device cuda \
    --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id fd002_${model}_seed2026_r2_${TS} --data-root $DATA --output-dir results \
    > logs/fd002_${model}_seed2026_r2.log 2>&1
  echo "done fd002/$model"
  python scripts/run_experiment.py --model $model --dataset fd004 --seed 2026 --device cuda \
    --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id fd004_${model}_seed2026_r2_${TS} --data-root $DATA --output-dir results \
    > logs/fd004_${model}_seed2026_r2.log 2>&1
  echo "done fd004/$model"
  python scripts/run_experiment.py --model $model --dataset metropt3 --seed 2026 --device cuda \
    --epochs 100 --patience 15 --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id metropt3_${model}_seed2026_r2_${TS} --data-root $DATA --output-dir results \
    > logs/metropt3_${model}_seed2026_r2.log 2>&1
  echo "done metropt3/$model"
done
echo "MATRIX DONE"

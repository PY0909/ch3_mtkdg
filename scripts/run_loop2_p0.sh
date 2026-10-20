#!/bin/bash
# P0: 多 seed 补齐 mtkdg + tcn（预注册矩阵的第一批）
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261014_loop2
run() { # dataset model epochs patience seed
  local out="results/$1/$1_$2_seed$5_${TS}/test_metrics.json"
  [ -f "$out" ] && { echo "skip $1/$2/s$5"; return; }
  python scripts/run_experiment.py --model "$2" --dataset "$1" --seed "$5" --split-seed 2026 \
    --device cuda --epochs "$3" --patience "$4" --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id "$1_$2_seed$5_${TS}" --data-root $DATA --output-dir results \
    > "logs/$1_$2_seed$5_${TS}.log" 2>&1 && echo "done $1/$2/s$5"
}
for seed in 2027 2028; do
  run fd002 mtkdg 200 30 $seed
  run fd004 mtkdg 200 30 $seed
  run metropt3 mtkdg 100 15 $seed
  run fd002 tcn 150 30 $seed
  run fd004 tcn 150 30 $seed
  run metropt3 tcn 100 15 $seed
done
echo "P0 DONE"

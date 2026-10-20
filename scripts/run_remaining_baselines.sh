#!/bin/bash
# 恢复矩阵：只跑缺失的 itransformer × 3（检测 test_metrics.json 跳过已完成 run）
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261013_222400
run() {  # run <dataset> <model> <epochs> <patience>
  local out="results/$1/$1_$2_seed2026_r2_${TS}/test_metrics.json"
  if [ -f "$out" ]; then echo "skip $1/$2 (done)"; return; fi
  python scripts/run_experiment.py --model "$2" --dataset "$1" --seed 2026 --device cuda \
    --epochs "$3" --patience "$4" --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id "$1_$2_seed2026_r2_${TS}" --data-root $DATA --output-dir results \
    > "logs/$1_$2_seed2026_r2.log" 2>&1 && echo "done $1/$2"
}
run fd002 itransformer 150 30
run fd004 itransformer 150 30
run metropt3 itransformer 100 15
echo "REMAINING DONE"

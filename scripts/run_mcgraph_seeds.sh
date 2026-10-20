#!/bin/bash
# MC-Graph 三 seed 配对矩阵：A0/G1/G2/G3/G4 × 3 数据集 × seeds 2027/2028
# （seed 2026 已有；FD A0 的 2027/2028 也需补）
set -u
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261014_seeds
declare -A FLAGS
FLAGS[a0]=""
FLAGS[g1]="--robust-train"
FLAGS[g2]="--robust-train --consistency-lambda 0.1"
FLAGS[g3]="--availability-mode both --robust-train"
FLAGS[g4]="--availability-mode both --robust-train --consistency-lambda 0.1"
run() { # run <ds> <cfg> <seed> <epochs> <patience>
  local rid="$1_mtk${2}_seed$3_${TS}"
  [ "$2" = "a0" ] && rid="$1_mtkdg_seed$3_${TS}"
  if [ -f "results/$1/$rid/test_metrics.json" ]; then echo "skip $rid"; return; fi
  python scripts/run_experiment.py --model mtkdg --dataset "$1" --seed "$3" --device cuda \
    --epochs "$4" --patience "$5" --batch-size 128 --lr-scheduler cosine \
    --norm-mode per_condition --evaluate-test --run-id "$rid" \
    --data-root $DATA --output-dir results ${FLAGS[$2]} \
    > "logs/${rid}.log" 2>&1 && echo "done $rid"
}
for seed in 2027 2028; do
  for ds in fd002 fd004; do
    run $ds a0 $seed 150 30
    for g in g1 g2 g3 g4; do run $ds $g $seed 150 30; done
  done
  for g in a0 g1 g2 g3 g4; do run metropt3 $g $seed 100 15; done
done
echo "SEED MATRIX DONE"

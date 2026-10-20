#!/bin/bash
# Loop-2 消融 batch：等 P0 完成后顺序执行 V1-V5（预注册见 MATRIX_ANALYSIS_PLAN §9）
source /root/anaconda3/etc/profile.d/conda.sh
conda activate torch23
export OMP_NUM_THREADS=2
DATA=/home/new_work/kaf_profiti/dataset
TS=20261014_loop2

# 等 P0 结束（同一 GPU，避免争用）
while ! grep -q "P0 DONE" logs/loop2_p0.log 2>/dev/null; do sleep 60; done

run() { # run_id_dataset model extra_args...
  local ds="$1"; shift
  local model="$1"; shift
  local out="results/$ds/${ds}_ablation_${model}_$@/test_metrics.json"
  local rid="ablation_$(echo "$@" | tr ' /' '__')"
  :
}
# 显式列出（可读性优先）
A() { # dataset model run_tag extra...
  local ds=$1 model=$2 tag=$3; shift 3
  local rid="${ds}_${model}_${tag}_${TS}"
  [ -f "results/$ds/$rid/test_metrics.json" ] && { echo "skip $rid"; return; }
  python scripts/run_experiment.py --model "$model" --dataset "$ds" --seed 2026 --split-seed 2026 \
    --device cuda --batch-size 128 --lr-scheduler cosine --evaluate-test \
    --run-id "$rid" --data-root $DATA --output-dir results "$@" \
    > "logs/${rid}.log" 2>&1 && echo "done $rid"
}
# V1 分工况归一化（选型仍 val_RMSE）
A fd002 mtkdg v1_percond --epochs 200 --patience 30 --norm-mode per_condition
A fd004 mtkdg v1_percond --epochs 200 --patience 30 --norm-mode per_condition
# V2a raw 单头对照（选型 val_RMSE_rawtarget）
A fd002 mtkdg v2a_rawsingle --epochs 200 --patience 30 --rul-head raw_single --checkpoint-metric val_RMSE_rawtarget
A fd004 mtkdg v2a_rawsingle --epochs 200 --patience 30 --rul-head raw_single --checkpoint-metric val_RMSE_rawtarget
# V2b 双头（capped+raw，test 用 raw 头）
A fd002 mtkdg v2b_dual --epochs 200 --patience 30 --rul-head dual --checkpoint-metric val_RMSE_rawtarget
A fd004 mtkdg v2b_dual --epochs 200 --patience 30 --rul-head dual --checkpoint-metric val_RMSE_rawtarget
# V3 退化通道剔除 / V4 persistence-skip（MetroPT）
A metropt3 mtkdg v3_excludedeg --epochs 100 --patience 15 --exclude-degenerate
A metropt3 mtkdg v4_delta --epochs 100 --patience 15 --state-head-mode delta
# V5 拆分单因子：epochs 300 / d128
A fd004 mtkdg v5a_ep300 --epochs 300 --patience 40
A fd004 mtkdg v5b_d128 --epochs 200 --patience 30 --d-model 128
echo "ABLATIONS DONE"

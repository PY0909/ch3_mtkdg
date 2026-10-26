#!/bin/bash
# FD 主表补齐批（2026-10-27，AutoDL 3090）：PatchTST/TimesNet/iTransformer × FD002/FD004 × 3 seeds。
# 目的：把 FD 主表从 3 模型扩到 6 模型，与 MetroPT 主表对称（单 seed 下三者落后 4.4-12 RMSE，
#       补 seed 不改变排名，纯呈现层对称性）。
# 命令模板复刻 run_v1_baselines.sh（V1 协议）；另含 1 个跨批对照 run：
#   fd002 mtkdg seed2026 在本卡重跑（xbatchctrl 变体，不入任何表），用于验证与 20261020 批逐位一致。
# 可恢复：已有 test_metrics.json 的 run 自动跳过。
set -u
cd /root/autodl-tmp/ch3_mtkdg
source /root/miniconda3/etc/profile.d/conda.sh
conda activate kaf_profiti
export OMP_NUM_THREADS=2
DATA=/root/autodl-tmp/kst_light_v2/dataset
TS=20261027_b3

# 跨批对照（~5 min）：与 results/fd002/fd002_mtkdg_a0ctrl_seed2026_20261020_abl 比对
rid="fd002_mtkdg_xbatchctrl_seed2026_${TS}"
if [ ! -f "results/fd002/$rid/test_metrics.json" ]; then
  python scripts/run_experiment.py --model mtkdg --dataset fd002 --seed 2026 --split-seed 2026 \
    --device cuda --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine \
    --norm-mode per_condition --evaluate-test \
    --run-id "$rid" --data-root "$DATA" --output-dir results \
    > "logs/${rid}.log" 2>&1 && echo "done $rid" || echo "FAIL $rid"
fi

# 18 个补齐 run
for model in patchtst timesnet itransformer; do
  for ds in fd002 fd004; do
    for seed in 2026 2027 2028; do
      rid="${ds}_${model}_seed${seed}_${TS}"
      if [ -f "results/$ds/$rid/test_metrics.json" ]; then echo "skip $rid"; continue; fi
      python scripts/run_experiment.py --model "$model" --dataset "$ds" --seed "$seed" --split-seed 2026 \
        --device cuda --epochs 150 --patience 30 --batch-size 128 --lr-scheduler cosine \
        --norm-mode per_condition --evaluate-test \
        --run-id "$rid" --data-root "$DATA" --output-dir results \
        > "logs/${rid}.log" 2>&1 && echo "done $rid" || echo "FAIL $rid"
    done
  done
done
echo "FD BASELINES EXTRA DONE"

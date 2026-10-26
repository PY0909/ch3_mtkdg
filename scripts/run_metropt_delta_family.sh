#!/bin/bash
# MetroPT delta 头补跑批次（2026-10-26，AutoDL 3090）：A0 + 结构消融 + G1-G4，全部 --state-head-mode delta。
# 动机：20261020 批的消融/G/稳健性全是 direct 头，与论文"direct 不放"口径冲突（见 refine-logs 记录）。
# 纪律：本批内部配对（dla0 为基准，8 变体 × 3 seeds），不与 20261020 direct 批混表；
#       变体名带 dl 前缀，避免与旧批在 aggregate_3seed.load_runs 的 (ds,model,variant,seed) key 上撞车。
# 可恢复：已存在 test_metrics.json / robustness_eval.json 的 run 自动跳过。
set -u
cd /root/autodl-tmp/ch3_mtkdg
source /root/miniconda3/etc/profile.d/conda.sh
conda activate kaf_profiti
export OMP_NUM_THREADS=2
DATA=/root/autodl-tmp/kst_light_v2/dataset
TS=20261026_dabl

declare -A FLAGS
FLAGS[dla0]=""
FLAGS[dlgraphnone]="--graph-mode none"
FLAGS[dlfusionadd]="--fusion-mode add"
FLAGS[dlfusionself_only]="--fusion-mode self_only"
FLAGS[dlg1]="--robust-train"
FLAGS[dlg2]="--robust-train --consistency-lambda 0.1"
FLAGS[dlg3]="--availability-mode both --robust-train"
FLAGS[dlg4]="--availability-mode both --robust-train --consistency-lambda 0.1"

# 训练：8 变体 × 3 seeds（MetroPT 冻结配置：100ep / patience15 / batch128 / cosine / val_RMSE 选型）
for variant in dla0 dlgraphnone dlfusionadd dlfusionself_only dlg1 dlg2 dlg3 dlg4; do
  for seed in 2026 2027 2028; do
    rid="metropt3_mtkdg_${variant}_seed${seed}_${TS}"
    if [ -f "results/metropt3/$rid/test_metrics.json" ]; then echo "skip $rid"; continue; fi
    python scripts/run_experiment.py --model mtkdg --dataset metropt3 --seed "$seed" --device cuda \
      --epochs 100 --patience 15 --batch-size 128 --lr-scheduler cosine \
      --state-head-mode delta --evaluate-test --run-id "$rid" \
      --data-root "$DATA" --output-dir results ${FLAGS[$variant]} \
      > "logs/${rid}.log" 2>&1 && echo "done $rid" || echo "FAIL $rid"
  done
done

# 稳健性评估：A0 + G1-G4（15 runs；与 20261020 批相同的 eval_robustness 协议）
for variant in dla0 dlg1 dlg2 dlg3 dlg4; do
  for seed in 2026 2027 2028; do
    rid="metropt3_mtkdg_${variant}_seed${seed}_${TS}"
    d="results/metropt3/$rid"
    [ -f "$d/test_metrics.json" ] || { echo "missing $rid"; continue; }
    [ -f "$d/robustness_eval.json" ] && { echo "skip-rob $rid"; continue; }
    python scripts/eval_robustness.py "$d" >> "logs/robustness_${TS}.log" 2>&1 \
      && echo "rob done $rid" || echo "rob FAIL $rid"
  done
done
echo "DELTA FAMILY DONE"

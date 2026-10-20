# Chapter 3: MTK-DG Point Prediction (from scratch)

独立实现第三章点预测实验：**MTK-DG Encoder**（Multi-Scale Time-Kernel Dynamic Sensor-Graph Encoder），  
覆盖三个任务：

| dataset       | task          | input                     | output                |
| ------------- | ------------- | ------------------------- | --------------------- |
| C-MAPSS FD002 | scalar RUL 回归 | `[B,H,17]`（3 工况 + 14 传感器） | `[B,1]` raw-cycle RUL |
| C-MAPSS FD004 | scalar RUL 回归 | `[B,H,17]`                | `[B,1]`               |
| MetroPT-3     | 未来多步多传感器状态预测  | `[B,168,15]`              | `[B,24,15]`           |

协议要点（详见 `plan` 参考）：

- MetroPT-3：第一自然月前 80% 时间 train、后 20% validation、第二个月及以后全部 test；  
  窗口不跨 split 边界；normalizer 仅由 train 拟合；test 只用 validation-best checkpoint 评估一次。
- FD002/FD004：engine-level train/validation 划分（禁止 engine 内窗口随机切分）；  
  官方 test engine 仅用于最终 test（每 engine 只保留末端窗口）；train/val 可用 capped RUL（默认 125），  
  test 使用官方 raw RUL；normalizer 仅由 train engines 拟合；FD002 与 FD004 分开训练、分开报告。
- 第三章只做 point loss；概率头/风险头在第四、五章。

## 目录

```
mtkdg/
  data/        # MetroPT / C-MAPSS adapter、split、normalizer
  models/      # MTK-DG encoder、任务头
  training/    # 训练 runner、checkpoint 管理
  evaluation/  # 指标、per-engine/per-sensor 分析、artifact 写出
scripts/
  run_experiment.py   # 统一 CLI 入口
tests/                 # pytest 单元与 smoke 测试
results/{dataset}/{run_id}/   # 运行产物（git 忽略）
```

## 环境与验证

conda env `torch23`（Python 3.9, PyTorch 2.5.1, CUDA RTX 4060 Ti 可用）。

已验证（CPU 真实数据 smoke，均为 2 epoch 小模型，仅管线验证、非实验结论）：

| dataset   | samples train/val/test | test 指标                               |
| --------- | ---------------------- | ------------------------------------- |
| FD002     | 32593 / 8426 / 259     | MAE=43.51, RMSE=52.01, NASA=116167.9  |
| FD004     | 39759 / 9289 / 248     | MAE=40.66, RMSE=48.31, NASA=60558.8   |
| MetroPT-3 | 2928 / 648 / 21699     | overall_MAE=1.058, overall_RMSE=2.794 |

## 常用命令

```bash
# 全部单元 + 真实数据测试
python -m pytest tests/ -q

# 真实数据 CPU smoke（FD002）
python scripts/run_experiment.py --dataset fd002 --epochs 2 --device cpu \
  --batch-size 256 --d-model 32 --evaluate-test --run-id smoke_fd002_cpu \
  --data-root /home/new_work/kaf_profiti/dataset --output-dir results

# MetroPT-3 smoke（换 --dataset metropt3）、FD004 smoke（--dataset fd004）同理

# 完整训练（示例，正式多 seed GPU 实验另行启动）
python scripts/run_experiment.py --dataset fd002 --seed 2026 --device cuda \
  --epochs 80 --batch-size 128 --evaluate-test \
  --data-root /home/new_work/kaf_profiti/dataset --output-dir results \
  --run-id fd002_seed2026
```

消融开关：`--graph-mode dynamic|static|none`、`--fusion-mode gated|add|self_only|no_graph`。

数据根目录默认指向 `/home/new_work/kaf_profiti/dataset`（只读）。

## 协议说明（与方案的偏差声明）

- MetroPT 的 `delta_t` / `log1p(|delta_t|)` 按方案 §5.1.2 计算，但当前主实验仅作  
  **审计统计**（manifest 的 gap_stats），不作为模型输入特征；若后续加入需重新预注册。
- MetroPT 特征/目标 scaler 启用方差下限 `min_std=0.01`（`METROPT_MIN_STD`），  
  处理 train 内常数通道（Oil_level 等）；FD002/FD004 通过删除常数通道处理，不用下限。

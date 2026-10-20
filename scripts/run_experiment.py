"""第三章统一实验入口（CLI）。

示例：
    python scripts/run_experiment.py --dataset fd002 --seed 2026 --device cpu \
        --epochs 2 --evaluate-test --run-id smoke_fd002 \
        --data-root /home/new_work/kaf_profiti/dataset --output-dir results
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtkdg.config import ExperimentConfig  # noqa: E402
from mtkdg.data.bundle import build_bundle  # noqa: E402
from mtkdg.evaluation.artifacts import save_run_artifacts  # noqa: E402
from mtkdg.training.runner import TrainingRunner, set_seed  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTK-DG chapter 3 point prediction")
    p.add_argument("--dataset", required=True, choices=["metropt3", "fd002", "fd004"])
    p.add_argument("--data-root", default="/home/new_work/kaf_profiti/dataset")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--run-id", default=None, help="缺省为 <dataset>_<seed>_<timestamp>")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--history-len", type=int, default=None)
    p.add_argument("--pred-len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--training-cap", type=int, default=125)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--loss", default="mse", choices=["mse", "huber"])
    p.add_argument("--lr-scheduler", default="none", choices=["none", "cosine"],
                   help="cosine 的 T_max=epochs（Round1-W1 收敛修复项）")
    # Loop-2 预注册消融开关（默认全关 = V0 主协议）
    p.add_argument("--norm-mode", default="global", choices=["global", "per_condition"],
                   help="V1: C-MAPSS 特征归一化（per_condition=6 工况分组）")
    p.add_argument("--rul-head", default="single", choices=["single", "raw_single", "dual"],
                   help="V2: single=capped(主协议) raw_single=P2a 对照 dual=P2b 双头")
    p.add_argument("--checkpoint-metric", default="val_RMSE",
                   choices=["val_RMSE", "val_RMSE_rawtarget"],
                   help="V2 变体按预注册用 val_RMSE_rawtarget 选型")
    p.add_argument("--state-head-mode", default="direct", choices=["direct", "delta"],
                   help="V4: delta=persistence-skip（y=x_last+f(z)）")
    p.add_argument("--exclude-degenerate", action="store_true",
                   help="V3: MetroPT 剔除 train std<0.05 通道的 encoder 输入")
    p.add_argument("--evaluate-test", action="store_true",
                   help="显式开启一次性 test 评估（默认关闭）")
    p.add_argument("--nrows", type=int, default=0,
                   help="仅 smoke：截断 MetroPT 读取行数")
    # 模型选择与超参
    p.add_argument("--model", default="mtkdg",
                   choices=["mtkdg", "tcn", "gru", "patchtst", "timesnet", "itransformer"],
                   help="baseline 与主模型共用数据协议/任务头/checkpoint 规则")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-kernels", type=int, default=4)
    p.add_argument("--graph-mode", default="dynamic", choices=["dynamic", "static", "none"])
    p.add_argument("--graph-layers", type=int, default=1)
    p.add_argument("--availability-mode", default="none",
                   choices=["none", "edge_bias", "source_suppress", "both"],
                   help="MC-Graph：传感器可用性条件化动态图（第四章机制）")
    p.add_argument("--robust-train", action="store_true",
                   help="训练期在线 corruption（双视图点损失）")
    p.add_argument("--corrupt-channel-rate", type=float, default=0.2)
    p.add_argument("--corrupt-block-ratio", type=float, default=0.2)
    p.add_argument("--consistency-lambda", type=float, default=0.0)
    p.add_argument("--fusion-mode", default="gated",
                   choices=["gated", "add", "self_only", "no_graph"])
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from datetime import datetime

    run_id = args.run_id or (
        f"{args.dataset}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    overrides = dict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_id=run_id,
        seed=args.seed,
        split_seed=args.split_seed,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        training_cap=args.training_cap,
        val_ratio=args.val_ratio,
        loss=args.loss,
        lr_scheduler=args.lr_scheduler,
        checkpoint_metric=args.checkpoint_metric,
        norm_mode=args.norm_mode,
        metropt_exclude_degenerate=args.exclude_degenerate,
        rul_head=args.rul_head,
        state_head_mode=args.state_head_mode,
        evaluate_test=args.evaluate_test,
        model_name=args.model,
        d_model=args.d_model,
        n_kernels=args.n_kernels,
        graph_mode=args.graph_mode,
        graph_layers=args.graph_layers,
        fusion_mode=args.fusion_mode,
        dropout=args.dropout,
        availability_mode=args.availability_mode,
        robust_train=args.robust_train,
        corrupt_channel_rate=args.corrupt_channel_rate,
        corrupt_block_ratio=args.corrupt_block_ratio,
        consistency_lambda=args.consistency_lambda,
    )
    for key in ("epochs", "history_len", "pred_len", "stride"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    cfg = ExperimentConfig.for_dataset(args.dataset, **overrides)
    set_seed(cfg.seed)

    print(f">>> dataset={cfg.dataset} task={cfg.task} run_id={run_id}")
    print(f">>> device={cfg.device} seed={cfg.seed} output={cfg.output_dir}")
    bundle = build_bundle(cfg, nrows=args.nrows)
    print(
        f">>> samples train={bundle.train.n_samples} "
        f"val={bundle.val.n_samples} test={bundle.test.n_samples} "
        f"channels={bundle.n_channels}"
    )

    runner = TrainingRunner(cfg, bundle)
    fit_info = runner.fit(verbose=True)

    test_result = None
    if cfg.evaluate_test:
        test_result = runner.run_test(verbose=True)

    save_run_artifacts(runner, bundle, fit_info, test_result)
    print(f">>> artifacts saved to {runner.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

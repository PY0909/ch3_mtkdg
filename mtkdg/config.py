"""实验配置：单一定义，供 runner / CLI / artifact 共享。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

VALID_DATASETS = ("metropt3", "fd002", "fd004")


@dataclass
class MTKDGConfig:
    """MTK-DG encoder 结构超参。"""

    d_model: int = 64
    n_kernels: int = 4
    scales: Tuple[float, ...] = (0.25, 0.5, 1.0)
    graph_mode: str = "dynamic"        # dynamic | static | none
    graph_layers: int = 1
    graph_self_loop: bool = True
    fusion_mode: str = "gated"         # gated | add | self_only | no_graph
    share_kernels_across_scales: bool = False
    dropout: float = 0.1
    # Loop-2 预注册消融（默认关闭 = V0 主协议）
    rul_head: str = "single"           # single | raw_single | dual
    aux_loss_weight: float = 1.0       # dual 模式 raw 头损失权重
    state_head_mode: str = "direct"    # direct | delta（persistence-skip）
    # MC-Graph（第四章）：传感器可用性条件化与鲁棒训练
    availability_mode: str = "none"    # none | edge_bias | source_suppress | both
    robust_train: bool = False         # 训练期在线 corruption（双视图）
    corrupt_channel_rate: float = 0.2  # 通道丢弃率（FD 自动保护工况通道）
    corrupt_block_ratio: float = 0.2   # 连续时间块占比
    corrupt_prob: float = 0.5          # 每个 batch 施加 corruption 的概率
    consistency_lambda: float = 0.0    # 双视图预测一致性权重（0=关闭）


@dataclass
class ExperimentConfig:
    """一次 run 的完整配置（全部可追溯字段）。"""

    # 模型选择
    model_name: str = "mtkdg"          # mtkdg | tcn | gru | patchtst | timesnet | itransformer
    # 数据
    dataset: str = "fd002"             # metropt3 | fd002 | fd004
    data_root: str = "/home/new_work/kaf_profiti/dataset"
    output_dir: str = "results"
    run_id: str = "run"
    # 划分协议
    history_len: int = 50
    pred_len: int = 24                 # metropt3 使用；RUL 任务忽略
    stride: int = 1                    # metropt3 默认 60，此处为 RUL 默认
    training_cap: int = 125            # 仅 C-MAPSS train/val
    training_target: str = "capped"    # raw | capped；仅新概率实验使用 raw
    val_ratio: float = 0.2             # 仅 C-MAPSS engine split
    split_seed: int = 2026             # C-MAPSS engine 划分种子
    # 训练
    seed: int = 2026
    device: str = "cpu"                # cpu | cuda
    epochs: int = 80
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 10
    loss: str = "mse"                  # mse | huber
    huber_beta: float = 1.0
    lr_scheduler: str = "none"         # none | cosine（cosine 的 T_max = epochs）
    checkpoint_metric: str = "val_RMSE"   # val_RMSE | val_RMSE_rawtarget（变体预注册用）
    # 归一化与通道（Loop-2 消融，默认 V0）
    norm_mode: str = "global"          # global | per_condition（仅 C-MAPSS）
    metropt_exclude_degenerate: bool = False   # 剔除 train std<0.05 通道的 encoder 输入
    # 评估
    evaluate_test: bool = False        # 只有显式开启才运行一次性 test
    # 模型
    model: MTKDGConfig = field(default_factory=MTKDGConfig)

    def __post_init__(self):
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {self.dataset}")
        if self.training_target not in ("raw", "capped"):
            raise ValueError("training_target must be 'raw' or 'capped'")
        if self.training_target == "raw" and self.dataset == "metropt3":
            raise ValueError("training_target='raw' is only valid for C-MAPSS datasets")
        if self.training_target == "capped" and self.training_cap <= 0:
            raise ValueError("training_cap must be positive when training_target='capped'")
        self.model = MTKDGConfig(**self.model) if isinstance(self.model, dict) else self.model

    # ---- 数据集默认值 ----
    @classmethod
    def for_dataset(cls, dataset: str, **overrides) -> "ExperimentConfig":
        defaults = dict(
            metropt3=dict(history_len=168, pred_len=24, stride=60, epochs=50),
            fd002=dict(history_len=50, pred_len=0, stride=1, epochs=80),
            fd004=dict(history_len=50, pred_len=0, stride=1, epochs=80),
        )[dataset]
        cfg = cls(dataset=dataset, **defaults)
        for key, value in overrides.items():
            if not hasattr(cfg, key) and not hasattr(cfg.model, key):
                raise ValueError(f"unknown config key: {key}")
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                setattr(cfg.model, key, value)
        return cfg

    @property
    def task(self) -> str:
        return "state_forecast" if self.dataset == "metropt3" else "rul_regression"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task
        if self.task == "rul_regression":
            # RUL 任务无 pred_len 概念，写 null 而不是占位 0
            d["pred_len"] = None
        return d

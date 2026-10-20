"""训练 runner：固定协议的逐 epoch 训练、validation-best checkpoint、一次性 test。

流程（每个 run）：
1. 固化 seed；train split 训练，每 epoch 打乱 batch；
2. 每 epoch 结束无梯度评估 validation（raw unit 指标）；
3. 以 validation RMSE（默认）选择 best checkpoint，patience 无改善 early stop；
4. 训练结束恢复 best 权重；只有 `evaluate_test=True` 才运行一次 test。
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..data.bundle import DatasetBundle
from ..data.common import SplitArrays
from ..evaluation.metrics import EVALUATOR_VERSION, rul_metrics, state_metrics
from ..models.baselines import BASELINE_NAMES, BaselinePointModel
from ..models.heads import MTKDGPointModel
from .checkpoint import load_checkpoint, save_checkpoint
from .corruption import corrupt_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 协议备注（r2 矩阵阶段记录）：不再强制 cudnn.deterministic=True。
    # 实测该标志使 conv1d 路径慢约 130x（baseline 卷积编码器受影响，
    # MTK-DG 的 linear/bmm 不受影响）。全部 RNG 流仍由 seed 固定，
    # 数据顺序与初始化完全可复现；卷积核在跨 GPU 架构时不保证逐位一致。
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss_fn(name: str, beta: float) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(f"unknown loss: {name}")


class TrainingRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        bundle: DatasetBundle,
        run_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.device = resolve_device(cfg.device)
        self.run_dir = run_dir or os.path.join(cfg.output_dir, cfg.dataset, cfg.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        pred_len = bundle.train.y.shape[1] if bundle.task == "state_forecast" else cfg.pred_len
        if cfg.model_name == "mtkdg":
            self.model = MTKDGPointModel(
                task=bundle.task,
                n_channels=bundle.n_channels,
                pred_len=pred_len,
                model_cfg=cfg.model,
                rul_mean=bundle.rul_mean,
                rul_std=bundle.rul_std,
                rul_raw_mean=bundle.rul_raw_mean,
                rul_raw_std=bundle.rul_raw_std,
            )
        elif cfg.model_name in BASELINE_NAMES:
            self.model = BaselinePointModel(
                task=bundle.task,
                encoder_name=cfg.model_name,
                n_channels=bundle.n_channels,
                pred_len=pred_len,
                d_model=cfg.model.d_model,
                history_len=cfg.history_len,
                rul_mean=bundle.rul_mean,
                rul_std=bundle.rul_std,
            )
        else:
            raise ValueError(f"unknown model: {cfg.model_name}")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        if cfg.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, cfg.epochs)
            )
        elif cfg.lr_scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"unknown lr_scheduler: {cfg.lr_scheduler}")
        self.loss_fn = _loss_fn(cfg.loss, cfg.huber_beta)

        self.history: Dict[str, list] = {
            "epoch": [], "train_loss": [], "lr": [],
            "val_MAE": [], "val_RMSE": [], "val_NASA": [],
            # 监控诊断（不参与 checkpoint 选择）：FD 任务 val 对未截断 raw 目标的 RMSE
            "val_RMSE_rawtarget": [],
        }
        self.best_epoch = 0
        self.best_metric = float("inf")
        self.epochs_ran = 0
        self.train_duration_s = 0.0
        self.inference_duration_s = 0.0
        self.test_evaluated = False

    # ---------------- batching ----------------
    def _iter_batches(self, split: SplitArrays, shuffle: bool):
        n = split.n_samples
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, self.cfg.batch_size):
            idx = order[start:start + self.cfg.batch_size].numpy()
            batch = {
                "x": torch.from_numpy(split.x[idx]).to(self.device),
                "t": torch.from_numpy(split.t[idx]).to(self.device),
                "mask": torch.from_numpy(split.mask[idx]).to(self.device),
                "y": torch.from_numpy(split.y[idx]).to(self.device),
            }
            if split.condition is not None:
                batch["condition"] = torch.from_numpy(split.condition[idx]).to(self.device)
            if split.y_raw_uncapped is not None:
                raw = (split.y_raw_uncapped[idx] - self.bundle.rul_raw_mean) / self.bundle.rul_raw_std
                batch["y_raw_std"] = torch.from_numpy(raw.astype(np.float32)).to(self.device)
            yield batch

    def _heads_loss(self, heads: Dict[str, torch.Tensor],
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """给定 forward_train 输出计算监督损失（clean/corrupt 视图共用）。"""
        if "state" in heads:                      # MetroPT：标准化空间 point loss
            return self.loss_fn(heads["state"], batch["y"])
        if self.bundle.task == "rul_regression":
            losses = []
            if "capped" in heads:
                pred_std = (heads["capped"] - self.bundle.rul_mean) / self.bundle.rul_std
                losses.append(self.loss_fn(pred_std, batch["y"]))
            if "raw" in heads:
                if "y_raw_std" not in batch:
                    raise RuntimeError("raw head requires y_raw_uncapped (train/val splits)")
                # raw 头输出 raw-cycle；用 raw 标签自身的统计标准化后算损失
                pred_std = (heads["raw"] - self.bundle.rul_raw_mean) / self.bundle.rul_raw_std
                losses.append(self.cfg.model.aux_loss_weight * self.loss_fn(pred_std, batch["y_raw_std"]))
            return sum(losses)
        raise RuntimeError("unsupported head outputs")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.model
        if hasattr(model, "forward_train"):
            heads = model.forward_train(
                batch["x"], batch["t"], batch["mask"], batch.get("condition")
            )
        else:  # baseline（无多模式）
            out = model(batch["x"], batch["t"], batch["mask"], batch.get("condition"))
            heads = {"state": out} if self.bundle.task == "state_forecast" else {"capped": out}
        return self._heads_loss(heads, batch)

    def _robust_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MC-Graph 鲁棒训练：clean 视图 + corruption 视图 + 一致性。

        corruption 只作用于训练 batch 的传感器通道（FD 工况通道受保护），
        validation/test 永远使用完整观测（冻结协议不变）。
        """
        cfg_model = self.cfg.model
        prot = self.bundle.condition_dim      # FD: 3（工况通道不损坏）；MetroPT: 0
        heads_c = self.model.forward_train(
            batch["x"], batch["t"], batch["mask"], batch.get("condition")
        )
        xc, mc = corrupt_batch(batch["x"], batch["mask"], cfg_model, prot)
        heads_r = self.model.forward_train(xc, batch["t"], mc, batch.get("condition"))
        loss = self._heads_loss(heads_c, batch)
        loss = loss + self._heads_loss(heads_r, batch)
        if cfg_model.consistency_lambda > 0:
            key = "state" if "state" in heads_c else "capped"
            loss = loss + cfg_model.consistency_lambda * F.smooth_l1_loss(
                heads_c[key], heads_r[key])
        return loss

    # ---------------- train / eval ----------------
    def train_one_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in self._iter_batches(self.bundle.train, shuffle=True):
            self.optimizer.zero_grad()
            loss = self._robust_loss(batch) if self.cfg.model.robust_train \
                else self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(1, count)

    @torch.no_grad()
    def predict_split(self, split: SplitArrays) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (preds_raw, targets_raw)，全部原始单位。"""
        self.model.eval()
        preds = []
        t0 = time.time()
        for batch in self._iter_batches(split, shuffle=False):
            out = self.model(batch["x"], batch["t"], batch["mask"], batch.get("condition"))
            if self.bundle.task == "rul_regression":
                preds.append(out.detach().cpu().numpy())          # [B,1] raw
            else:
                # 标准化输出反变换到 raw unit（逐通道）
                preds.append(
                    self.bundle.target_scaler.inverse_transform(
                        out.detach().cpu().numpy().reshape(-1, self.bundle.n_channels)
                    ).reshape(out.shape[0], -1, self.bundle.n_channels)
                )
        self.inference_duration_s += time.time() - t0
        pred_arr = np.concatenate(preds, axis=0)
        true_arr = split.y_raw                       # 原始单位真值
        return pred_arr, true_arr

    def evaluate_split(self, split: SplitArrays) -> Dict[str, float]:
        preds, trues = self.predict_split(split)
        if self.bundle.task == "rul_regression":
            m = rul_metrics(trues.reshape(-1), preds.reshape(-1))
            out = {"val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_NASA": m["NASA"]}
            if split.y_raw_uncapped is not None:
                # 监控诊断：对未截断 raw 目标的 RMSE（协议选择仍用 capped val_RMSE）
                m_raw = rul_metrics(split.y_raw_uncapped.reshape(-1), preds.reshape(-1))
                out["val_RMSE_rawtarget"] = m_raw["RMSE"]
            return out
        m = state_metrics(trues, preds)
        return {
            "val_MAE": m["overall_MAE"],
            "val_RMSE": m["overall_RMSE"],
            "val_NASA": float("nan"),
        }

    # ---------------- main loop ----------------
    def fit(self, verbose: bool = True) -> Dict[str, Any]:
        cfg = self.cfg
        t_start = time.time()
        stop_reason = "completed"
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self.train_one_epoch()
            val_metrics = self.evaluate_split(self.bundle.val)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            for k in ("val_MAE", "val_RMSE", "val_NASA"):
                self.history[k].append(val_metrics[k])
            self.history["val_RMSE_rawtarget"].append(
                val_metrics.get("val_RMSE_rawtarget", float("nan"))
            )
            self.epochs_ran = epoch

            metric_key = self.cfg.checkpoint_metric
            if metric_key not in val_metrics or val_metrics[metric_key] != val_metrics[metric_key]:
                raise RuntimeError(f"checkpoint metric '{metric_key}' unavailable in this run")
            improved = val_metrics[metric_key] < self.best_metric
            if improved:
                self.best_metric = val_metrics[metric_key]
                self.best_epoch = epoch
                save_checkpoint(
                    os.path.join(self.run_dir, "checkpoint_best.pt"),
                    self.model, self.optimizer, epoch, self.best_metric,
                    extra={"config": cfg.to_dict(), "evaluator_version": EVALUATOR_VERSION,
                           "checkpoint_metric": metric_key},
                )
            save_checkpoint(
                os.path.join(self.run_dir, "checkpoint_last.pt"),
                self.model, self.optimizer, epoch, self.best_metric,
                extra={"config": cfg.to_dict(), "evaluator_version": EVALUATOR_VERSION},
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose:
                self._print_epoch(epoch, train_loss, val_metrics)

            if epoch - self.best_epoch >= cfg.patience:
                stop_reason = f"early_stop@{epoch}(best={self.best_epoch})"
                if verbose:
                    print(f">>> Early stopping at epoch {epoch} (best epoch {self.best_epoch})")
                break

        self.train_duration_s = time.time() - t_start
        # 恢复 validation-best 权重
        best_path = os.path.join(self.run_dir, "checkpoint_best.pt")
        if os.path.exists(best_path):
            load_checkpoint(best_path, model=self.model)
            if verbose:
                print(
                    f"Best checkpoint: epoch={self.best_epoch} | "
                    f"val_RMSE={self.best_metric:.4f} (restored)"
                )
        return {"stop_reason": stop_reason, "best_epoch": self.best_epoch,
                "best_val_RMSE": self.best_metric}

    def _print_epoch(self, epoch: int, train_loss: float, m: Dict[str, float]) -> None:
        line = f"Epoch {epoch:03d}/{self.cfg.epochs:03d} | train_loss={train_loss:.4f}"
        if self.bundle.task == "rul_regression":
            line += (f" | val_MAE={m['val_MAE']:.4f} | val_RMSE={m['val_RMSE']:.4f}"
                     f" | val_NASA={m['val_NASA']:.2f}")
        else:
            line += (f" | val_overall_MAE={m['val_MAE']:.4f}"
                     f" | val_overall_RMSE={m['val_RMSE']:.4f}")
        print(line)

    @torch.no_grad()
    def run_test(self, verbose: bool = True) -> Dict[str, Any]:
        """一次性 test 评估（只允许在恢复 best checkpoint 后调用一次）。"""
        if self.test_evaluated:
            raise RuntimeError("test has already been evaluated once for this runner")
        self.test_evaluated = True
        preds, trues = self.predict_split(self.bundle.test)
        result: Dict[str, Any] = {"preds_raw": preds, "trues_raw": trues}
        if self.bundle.task == "rul_regression":
            m = rul_metrics(trues.reshape(-1), preds.reshape(-1))
            result["metrics"] = m
            if verbose:
                print(f"Test | MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | NASA={m['NASA']:.2f}")
        else:
            m = state_metrics(trues, preds)
            result["metrics"] = m
            if verbose:
                print(
                    f"Test | overall_MAE={m['overall_MAE']:.4f} | "
                    f"overall_RMSE={m['overall_RMSE']:.4f}"
                )
        return result

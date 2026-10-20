"""Fixed-budget frozen-encoder training for contextual quantile readouts."""

from __future__ import annotations

import numpy as np
import hashlib
import os
import pandas as pd
import torch

from ..data.calibration import make_calibration_view
from ..evaluation.calibration import ShrunkQuantileCalibrator
from ..models.context_adapter import ContextFeatureBuilder, ContextMetadata, ContextualResidualAdapter
from ..models.frozen_encoder import load_frozen_encoder
from ..models.prob_heads import (
    QUANTILE_LEVELS, RULQuantileHead, StateQuantileHead, pinball_loss,
)
from ..training.checkpoint import load_checkpoint
from ..models.heads import MTKDGPointModel, RULHead
from ..evaluation.probability_artifacts import save_probability_artifacts


class ContextualProbabilisticRunner:
    def __init__(self, bundle, checkpoint_path, dataset, seed=2026, mode="C1",
                 epochs=1, batch_size=128, lr=2e-4, device="cpu"):
        if mode not in ("C0", "C1", "C3", "Cpoint"):
            raise ValueError("mode must be C0, C1, C3, or Cpoint")
        if mode == "Cpoint" and bundle.task != "rul_regression":
            raise ValueError("Cpoint is only defined for C-MAPSS")
        self.bundle, self.dataset, self.mode = bundle, dataset.lower(), mode
        self.seed, self.epochs, self.batch_size = int(seed), int(epochs), int(batch_size)
        self.device = torch.device(device)
        self.quantile_levels = QUANTILE_LEVELS
        self.encoder = load_frozen_encoder(checkpoint_path, dataset).to(self.device)
        payload = load_checkpoint(checkpoint_path)
        cfg = payload["extra"]["config"]
        from ..config import MTKDGConfig
        model_cfg = MTKDGConfig(**cfg["model"])
        point_model = MTKDGPointModel(
            task=bundle.task, n_channels=bundle.n_channels,
            pred_len=bundle.train.y.shape[1] if bundle.task == "state_forecast" else 0,
            model_cfg=model_cfg, rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
            rul_raw_mean=bundle.rul_raw_mean, rul_raw_std=bundle.rul_raw_std,
        )
        point_model.load_state_dict(payload["state_dict"], strict=True)
        point_model.to(self.device).eval()
        self.point_model = point_model
        self.feature_builder = ContextFeatureBuilder(
            d_model=model_cfg.d_model, n_channels=bundle.n_channels,
            condition_dim=model_cfg.d_model if bundle.task == "rul_regression" else 0,
            n_scales=len(model_cfg.scales),
        )
        torch.manual_seed(self.seed)
        self.metadata, self.train_features, self.train_codes = self._fit_metadata()
        self.readout = (
            RULHead(model_cfg.d_model, bundle.rul_mean, bundle.rul_std).to(self.device)
            if self.mode == "Cpoint" else
            RULQuantileHead(model_cfg.d_model).to(self.device)
            if bundle.task == "rul_regression" else
            StateQuantileHead(model_cfg.d_model, bundle.n_channels, bundle.train.y.shape[1]).to(self.device)
        )
        if self.mode in ("C1", "C3"):
            self.adapter = ContextualResidualAdapter(
                d_model=model_cfg.d_model, feature_dim=self.feature_builder.feature_dim,
                n_channels=bundle.n_channels,
            ).to(self.device)
        else:
            self.adapter = None
        params = list(self.readout.parameters())
        if self.adapter is not None:
            params += list(self.adapter.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.calibrator = None
        self.history = []
        self.test_evaluated = False

    def _batch(self, split, indices):
        batch = {
            "x": torch.as_tensor(split.x[indices], device=self.device),
            "t": torch.as_tensor(split.t[indices], device=self.device),
            "mask": torch.as_tensor(split.mask[indices], device=self.device),
            "y": torch.as_tensor(
                split.y_raw[indices] if self.bundle.task == "rul_regression" else split.y[indices],
                dtype=torch.float32, device=self.device,
            ),
            "condition": None if split.condition is None else torch.as_tensor(split.condition[indices], device=self.device),
        }
        if self.bundle.task == "rul_regression":
            batch["y"] = batch["y"].reshape(-1)
        return batch

    def _encoder_output(self, batch):
        return self.encoder(batch["x"], batch["t"], batch["mask"], batch["condition"])

    def _codes(self, split, train=False):
        if self.bundle.task != "rul_regression":
            return np.zeros(split.n_samples, dtype=np.int64)
        settings = split.condition.mean(axis=1)
        rounded = np.round(settings, 2)
        if train:
            unique = sorted(set(map(tuple, rounded.tolist())))
            self.condition_lookup = {key: i for i, key in enumerate(unique)}
        return np.asarray([self.condition_lookup.get(tuple(row), -1) for row in rounded], dtype=np.int64)

    @torch.no_grad()
    def _representations(self, split):
        result = []
        for start in range(0, split.n_samples, self.batch_size):
            idx = np.arange(start, min(start + self.batch_size, split.n_samples))
            batch = self._batch(split, idx)
            result.append(self.feature_builder(self._encoder_output(batch)).cpu().numpy())
        return np.concatenate(result, axis=0)

    @torch.no_grad()
    def _point_errors(self, split):
        errors = []
        for start in range(0, split.n_samples, self.batch_size):
            idx = np.arange(start, min(start + self.batch_size, split.n_samples))
            batch = self._batch(split, idx)
            prediction = self.point_model(batch["x"], batch["t"], batch["mask"], batch["condition"])
            target = batch["y"]
            if self.bundle.task == "state_forecast":
                prediction = self._state_to_raw(prediction)
                target = self._state_to_raw(target)
            error = (prediction - target).abs()
            errors.append(error.reshape(error.size(0), -1).mean(dim=1).cpu().numpy())
        return np.concatenate(errors)

    def _fit_metadata(self):
        features = self._representations(self.bundle.train)
        codes = self._codes(self.bundle.train, train=True)
        errors = self._point_errors(self.bundle.train)
        if self.bundle.task == "rul_regression":
            clusters = np.asarray(self.bundle.train.meta["engine_ids"])
        else:
            timestamps = pd.to_datetime(self.bundle.train.meta["origin_timestamps"])
            clusters = timestamps.strftime("%G-%V").to_numpy()
        metadata = ContextMetadata.fit_train(
            features, errors, clusters, condition_code=codes, split="train", n_folds=5,
        )
        return metadata, features, codes

    def _probability(self, batch, codes, train=False):
        enc = self._encoder_output(batch)
        context = self.feature_builder(enc)
        if self.adapter is not None:
            assignment = self.metadata.assign(context.detach().cpu().numpy(), codes)
            weight = torch.as_tensor(assignment.support_weight, device=self.device)
            if self.mode == "C3" and train:
                permutation = torch.randperm(context.size(0), device=self.device)
                context, weight = context[permutation], weight[permutation]
            adapted = self.adapter(enc, context, weight)
            z = adapted.z_ctx
        else:
            z = enc["z"]
        if self.mode == "Cpoint":
            return self.readout(z, enc["cond_emb"])
        if self.bundle.task == "rul_regression":
            quantiles = self.readout(z, enc["cond_emb"])
        else:
            quantiles = self.readout(z)
        return quantiles

    def _state_to_raw(self, values):
        shape = values.shape
        flat = values.detach().cpu().numpy().reshape(-1, self.bundle.n_channels)
        raw = self.bundle.target_scaler.inverse_transform(flat)
        return torch.as_tensor(raw.reshape(shape), dtype=values.dtype, device=values.device)

    def _quantiles_to_raw(self, quantiles):
        if self.bundle.task == "rul_regression":
            return quantiles
        shape = quantiles.shape
        flat = quantiles.detach().cpu().numpy().transpose(0, 1, 3, 2).reshape(-1, self.bundle.n_channels)
        raw = self.bundle.target_scaler.inverse_transform(flat)
        restored = raw.reshape(shape[0], shape[1], shape[3], shape[2]).transpose(0, 1, 3, 2)
        return torch.as_tensor(restored, dtype=quantiles.dtype, device=quantiles.device)

    def fit(self):
        rng = np.random.RandomState(self.seed)
        n = self.bundle.train.n_samples
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            losses = []
            self.encoder.eval()
            self.readout.train()
            if self.adapter is not None:
                self.adapter.train()
            for start in range(0, n, self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = self._batch(self.bundle.train, indices)
                codes = self.train_codes[indices]
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self._probability(batch, codes, train=True)
                if self.mode == "Cpoint":
                    pred_std = (prediction.reshape(-1) - self.bundle.rul_mean) / self.bundle.rul_std
                    target_std = (batch["y"] - self.bundle.rul_mean) / self.bundle.rul_std
                    loss = torch.nn.functional.mse_loss(pred_std, target_std)
                else:
                    loss = pinball_loss(prediction, batch["y"])
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.history.append({"epoch": epoch + 1,
                                 "train_loss": float(np.mean(losses)),
                                 "loss_name": "mse_raw_target" if self.mode == "Cpoint" else "pinball"})
        return self.history

    @torch.no_grad()
    def predict(self, split):
        self.encoder.eval()
        self.readout.eval()
        if self.adapter is not None:
            self.adapter.eval()
        codes = self._codes(split)
        outputs = []
        for start in range(0, split.n_samples, self.batch_size):
            indices = np.arange(start, min(start + self.batch_size, split.n_samples))
            batch = self._batch(split, indices)
            outputs.append(self._probability(batch, codes[indices]).cpu().numpy())
        predictions = torch.as_tensor(np.concatenate(outputs, axis=0), dtype=torch.float32)
        return self._quantiles_to_raw(predictions).cpu().numpy()

    @torch.no_grad()
    def predict_point(self, split):
        if self.mode != "Cpoint":
            raise RuntimeError("predict_point is only available in Cpoint mode")
        codes = self._codes(split)
        outputs = []
        for start in range(0, split.n_samples, self.batch_size):
            indices = np.arange(start, min(start + self.batch_size, split.n_samples))
            batch = self._batch(split, indices)
            outputs.append(self._probability(batch, codes[indices]).reshape(-1).cpu().numpy())
        return np.concatenate(outputs)

    def evaluate_point_test_once(self):
        if self.mode != "Cpoint":
            raise RuntimeError("evaluate_point_test_once is only available in Cpoint mode")
        if self.test_evaluated:
            raise RuntimeError("test may only be evaluated once")
        self.test_evaluated = True
        return {"predictions": self.predict_point(self.bundle.test),
                "targets": self.bundle.test.y_raw.reshape(-1)}

    def calibrate(self):
        if self.mode == "Cpoint":
            raise RuntimeError("Cpoint is a point-regression comparator and has no calibrator")
        view = make_calibration_view(
            self.bundle.val, self.bundle.task,
            pred_len=self.bundle.val.y.shape[1] if self.bundle.task == "state_forecast" else 1,
        )
        quantiles = self.predict(view)
        targets = view.y_raw
        if self.bundle.task == "rul_regression":
            targets = targets.reshape(-1)
        codes = self._codes(view)
        features = self._representations(view)
        assignment = self.metadata.assign(features, codes)
        cluster_id = np.asarray(view.meta["cluster_id"])
        self.calibrator = ShrunkQuantileCalibrator.fit(
            quantiles, targets, assignment.group_keys, cluster_id,
            split="val_calibration", target_unit="raw cycle" if self.bundle.task == "rul_regression" else "raw sensor unit",
        )
        calibrated = self.calibrator.transform(quantiles, assignment.group_keys)
        return {"quantiles": calibrated, "targets": targets, "context_key": assignment.group_keys,
                "cluster_id": cluster_id, "raw_quantiles": quantiles}

    def evaluate_test_once(self):
        if self.mode == "Cpoint":
            raise RuntimeError("use evaluate_point_test_once for Cpoint")
        if self.test_evaluated:
            raise RuntimeError("test may only be evaluated once")
        if self.calibrator is None:
            raise RuntimeError("validation calibration must be fit before test evaluation")
        self.test_evaluated = True
        quantiles = self.predict(self.bundle.test)
        codes = self._codes(self.bundle.test)
        features = self._representations(self.bundle.test)
        assignment = self.metadata.assign(features, codes)
        calibrated = self.calibrator.transform(quantiles, assignment.group_keys)
        targets = self.bundle.test.y_raw
        if self.bundle.task == "rul_regression":
            targets = targets.reshape(-1)
        return {"quantiles": calibrated, "raw_quantiles": quantiles, "targets": targets}

    def save_artifacts(self, run_dir, validation, test=None):
        if self.mode == "Cpoint":
            raise RuntimeError("use save_point_artifacts for Cpoint")
        digest = hashlib.sha256()
        with open(self.encoder.metadata["checkpoint"], "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "dataset": self.dataset,
            "mode": self.mode,
            "seed": self.seed,
            "epochs": self.epochs,
            "training_target": self.bundle.meta.get("training_target", "not_applicable"),
            "encoder_checkpoint_sha256": digest.hexdigest(),
            "encoder_fingerprint": self.encoder.fingerprint(),
            "encoder_frozen": all(not p.requires_grad for p in self.encoder.encoder.parameters()),
            "split_identity": self.bundle.split_identity,
            "bundle_identity": self.bundle.identity,
            "scaler_identity": self.bundle.target_scaler.identity(),
            "feature_scaler_identity": self.bundle.feature_scaler.identity(),
            "context_metadata": self.metadata.state_dict(),
            "calibrator": self.calibrator.state_dict(),
            "history": self.history,
            "quality_gate_status": "pending",
            "evaluated_once": self.test_evaluated,
            "quantile_levels": list(self.quantile_levels),
            "target_unit": self.calibrator.target_unit,
        }
        arrays = {
            "val_targets_raw": validation["targets"],
            "val_quantiles_raw": validation["quantiles"],
            "val_quantiles_uncalibrated_raw": validation["raw_quantiles"],
            "val_context_key": validation["context_key"],
            "val_cluster_id": validation["cluster_id"].astype(str),
        }
        if test is not None:
            if not self.test_evaluated:
                raise RuntimeError("test artifact requires an evaluated test result")
            arrays.update({
                "test_targets_raw": test["targets"],
                "test_quantiles_raw": test["quantiles"],
                "test_quantiles_uncalibrated_raw": test["raw_quantiles"],
            })
        return save_probability_artifacts(run_dir, manifest, arrays)

    def save_point_artifacts(self, run_dir, validation, test=None):
        if self.mode != "Cpoint":
            raise RuntimeError("save_point_artifacts is only available in Cpoint mode")
        digest = hashlib.sha256()
        with open(self.encoder.metadata["checkpoint"], "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "dataset": self.dataset, "mode": self.mode, "seed": self.seed,
            "training_target": self.bundle.meta.get("training_target", "raw"),
            "encoder_checkpoint_sha256": digest.hexdigest(),
            "encoder_fingerprint": self.encoder.fingerprint(), "encoder_frozen": True,
            "split_identity": self.bundle.split_identity, "bundle_identity": self.bundle.identity,
            "scaler_identity": self.bundle.target_scaler.identity(),
            "quality_gate_status": "comparator_only", "evaluated_once": self.test_evaluated,
            "history": self.history, "prediction_type": "point",
        }
        arrays = {"val_targets_raw": validation["targets"], "val_predictions_raw": validation["predictions"]}
        if test is not None:
            arrays.update({"test_targets_raw": test["targets"], "test_predictions_raw": test["predictions"]})
        return save_probability_artifacts(run_dir, manifest, arrays)

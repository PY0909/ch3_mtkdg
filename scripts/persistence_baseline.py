"""MetroPT-3 persistence (y = x_last) 零参数基线，与 run_experiment 相同的首月协议 / 窗口 / raw-unit 指标。"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from mtkdg.data.metropt import load_metropt_frame, build_metropt_splits, metropt_csv_path

frame = load_metropt_frame(metropt_csv_path("dataset"))
splits = build_metropt_splits(frame)
out = {}
for name, arr in (("val", splits.val), ("test", splits.test)):
    x_last_raw = splits.scaler.inverse_transform(arr.x[:, -1, :])          # [N, C] raw unit
    pred = np.repeat(x_last_raw[:, None, :], arr.y_raw.shape[1], axis=1)   # [N, P, C]
    err = pred - arr.y_raw
    out[name] = {
        "n_windows": int(arr.y_raw.shape[0]),
        "overall_MAE": float(np.abs(err).mean()),
        "overall_RMSE": float(np.sqrt((err ** 2).mean())),
        "per_horizon_MAE_first_last": [float(np.abs(err[:, 0]).mean()), float(np.abs(err[:, -1]).mean())],
    }
# 也给出"预测 train 均值"的零参数基线作为第二参照
train_mean = splits.scaler.inverse_transform(splits.train.y.reshape(-1, splits.train.y.shape[-1])).mean(0)
err = train_mean[None, None, :] - splits.test.y_raw
out["test_train_mean_baseline"] = {"overall_MAE": float(np.abs(err).mean()), "overall_RMSE": float(np.sqrt((err**2).mean()))}
print(json.dumps(out, indent=2))
json.dump(out, open("results/metropt3_persistence_baseline.json", "w"), indent=2)

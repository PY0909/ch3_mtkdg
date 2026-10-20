"""Run one frozen-encoder contextual quantile experiment."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import build_bundle
from mtkdg.training.probabilistic_runner import ContextualProbabilisticRunner


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Frozen-encoder contextual probabilistic prediction")
    parser.add_argument("--dataset", required=True, choices=("fd002", "fd004", "metropt3"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, help="audited Chapter 3 checkpoint_best.pt")
    parser.add_argument("--mode", required=True, choices=("C0", "C1", "C3", "Cpoint"))
    parser.add_argument("--training-target", choices=("raw", "capped"), default="raw")
    parser.add_argument("--data-root", default="/home/new_work/kaf_profiti/dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    run_dir = os.path.join(args.output_dir, args.dataset, args.run_id)
    if os.path.exists(run_dir):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    overrides = {
        "data_root": args.data_root,
        "training_target": args.training_target if args.dataset != "metropt3" else "capped",
    }
    for key in ("history_len", "pred_len", "stride"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    config = ExperimentConfig.for_dataset(args.dataset, **overrides)
    bundle = build_bundle(config)
    runner = ContextualProbabilisticRunner(
        bundle=bundle, checkpoint_path=args.checkpoint, dataset=args.dataset,
        seed=args.seed, mode=args.mode, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.learning_rate, device=args.device,
    )
    history = runner.fit()
    if args.mode == "Cpoint":
        validation = {"predictions": runner.predict_point(bundle.val), "targets": bundle.val.y_raw.reshape(-1)}
        test = runner.evaluate_point_test_once() if args.evaluate_test else None
        paths = runner.save_point_artifacts(run_dir, validation, test)
    else:
        validation = runner.calibrate()
        test = runner.evaluate_test_once() if args.evaluate_test else None
        paths = runner.save_artifacts(run_dir, validation, test)
    print(f"run={run_dir} epochs={len(history)} test_evaluated={runner.test_evaluated}")
    print(f"manifest={paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

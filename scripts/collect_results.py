"""汇总 results/ 下全部 run 的关键指标为一张表。"""
import glob, json, os, sys

rows = []
for manifest_path in sorted(glob.glob("results/*/*/manifest.json")):
    d = os.path.dirname(manifest_path)
    run_id = os.path.basename(d)
    m = json.load(open(manifest_path))
    tm_path = os.path.join(d, "test_metrics.json")
    vm_path = os.path.join(d, "validation_metrics.json")
    tm = json.load(open(tm_path))["metrics"] if os.path.exists(tm_path) else {}
    vm = json.load(open(vm_path))["best"] if os.path.exists(vm_path) else {}
    model = m["model_version"].split("+")[0].replace("ch3_baselines_v1:", "")
    if model.startswith("mtkdg"):
        model = "mtkdg"
    rows.append({
        "run_id": run_id, "dataset": m["dataset"], "model": model,
        "params": m["parameter_count"], "best_epoch": vm.get("epoch"),
        "val_RMSE": round(vm["val_RMSE"], 4) if vm.get("val_RMSE") is not None else None,
        "MAE": round(tm["MAE"], 4) if tm.get("MAE") is not None else None,
        "RMSE": round(tm["RMSE"], 4) if tm.get("RMSE") is not None else None,
        "NASA": round(tm["NASA"], 1) if tm.get("NASA") is not None else None,
        "overall_MAE": round(tm["overall_MAE"], 4) if tm.get("overall_MAE") is not None else None,
        "overall_RMSE": round(tm["overall_RMSE"], 4) if tm.get("overall_RMSE") is not None else None,
    })

order = {"fd002": 0, "fd004": 1, "metropt3": 2}
rows.sort(key=lambda r: (order.get(r["dataset"], 9), r["model"], r["run_id"]))
print(f"{'dataset':9} {'model':14} {'params':>8} {'best_ep':>7} {'val_RMSE':>9} "
      f"{'MAE':>9} {'RMSE':>9} {'NASA':>10} {'oMAE':>8} {'oRMSE':>8}")
for r in rows:
    print(f"{r['dataset']:9} {r['model']:14} {r['params']:>8} {str(r['best_epoch']):>7} "
          f"{str(r['val_RMSE']):>9} {str(r['MAE']):>9} {str(r['RMSE']):>9} {str(r['NASA']):>10} "
          f"{str(r['overall_MAE']):>8} {str(r['overall_RMSE']):>8}")
if len(sys.argv) > 1:
    json.dump(rows, open(sys.argv[1], "w"), indent=2)
    print(f"\nsaved -> {sys.argv[1]}")

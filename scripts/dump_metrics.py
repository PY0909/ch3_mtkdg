import json, glob
rows = []
for f in sorted(glob.glob("results/*/*/test_metrics.json")):
    d = json.load(open(f)); m = d["metrics"]; ds = f.split("/")[1]; rid = f.split("/")[2]
    try:
        h = json.load(open(f.replace("test_metrics.json", "validation_metrics.json")))
        be = h.get("best_epoch", "?")
    except Exception:
        be = "?"
    if ds.startswith("fd"):
        rows.append("%-8s %-48s ep=%4s MAE=%6.2f RMSE=%6.2f NASA=%9.0f" % (ds, rid, be, m["MAE"], m["RMSE"], m["NASA"]))
    else:
        rows.append("%-8s %-48s ep=%4s oMAE=%.4f oRMSE=%.4f" % (ds, rid, be, m["overall_MAE"], m["overall_RMSE"]))
print("\n".join(rows))

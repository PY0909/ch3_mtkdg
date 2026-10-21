"""汇总第三章 3-seed 结果：主表 / MetroPT（direct+delta+persistence）/ 消融 / MC-Graph 稳健性。

用法：python scripts/aggregate_3seed.py [results_dir] [--md out.md] [--json out.json]
run_id 约定（见 run_seed_matrix.sh / run_ablations.sh / run_metropt_delta.sh / run_mcgraph_fix.sh）：
  {ds}_{model}_seed{S}_20261020_seeds              主表 a0（model: mtkdg/tcn/gru）
  {ds}_mtkdg_g{1..4}_seed{S}_20261020_seeds        MC-Graph G1-G4
  {ds}_mtkdg_a0ctrl_seed2026_20261020_abl          a0 seed2026 同硬件对照
  {ds}_mtkdg_{variant}_seed{S}_20261020_abl        结构消融 / V3 / V4
  metropt3_{model}_delta_seed{S}_20261020_abl      baseline delta 头
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SEEDS = (2026, 2027, 2028)
ABL_VARIANTS = ("graphstatic", "graphnone", "fusionadd", "fusionself_only", "fusionno_graph",
                "v3excludedgen", "v4deltaskip")
ABL_LABEL = {
    "a0": "A0 全模型（dynamic graph + gated fusion）",
    "graphstatic": "graph = static",
    "graphnone": "graph = none",
    "fusionadd": "fusion = add",
    "fusionself_only": "fusion = self_only（≡ no_graph）",
    "fusionno_graph": "fusion = no_graph",
    "v3excludedgen": "V3 剔除退化通道（MetroPT）",
    "v4deltaskip": "V4 persistence-skip 头（MetroPT）",
}
MC_LABEL = {"a0": "A0 基线", "g1": "G1 robust-train", "g2": "G2 robust + consistency",
            "g3": "G3 availability + robust", "g4": "G4 完整 MC-Graph"}
RID_RE = re.compile(
    r"^(?P<ds>fd002|fd004|metropt3)_(?P<model>mtkdg|tcn|gru|patchtst|timesnet|itransformer)"
    r"(?:_(?P<variant>[a-z0-9_]+?))?_seed(?P<seed>\d{4})_(?P<ts>\d{8}_[a-z]+)$")


def load_runs(root):
    runs = {}
    for f in glob.glob(os.path.join(root, "*", "*", "test_metrics.json")):
        rid = os.path.basename(os.path.dirname(f))
        m = RID_RE.match(rid)
        if not m:
            continue
        d = m.groupdict()
        variant = d["variant"] or "a0"
        if variant == "a0ctrl":
            variant = "a0"
        met = json.load(open(f))["metrics"]
        rob_path = os.path.join(os.path.dirname(f), "robustness_eval.json")
        rob = json.load(open(rob_path)) if os.path.exists(rob_path) else None
        key = (d["ds"], d["model"], variant, int(d["seed"]))
        if key in runs and runs[key]["rid"].endswith("_seeds"):
            continue  # 同配置重复时优先保留 seeds 命名
        runs[key] = {"rid": rid, "metrics": met, "robust": rob}
    return runs


def ms(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    a = np.asarray(vals, dtype=float)
    if len(a) == 1:
        return fmt.format(a[0]) + " (1 seed)"
    return (fmt + "±" + fmt).format(a.mean(), a.std(ddof=1))


def get(runs, ds, model, variant, metric):
    out = []
    for s in SEEDS:
        r = runs.get((ds, model, variant, s))
        out.append(r["metrics"].get(metric) if r else None)
    return out


def paired(a, b):
    """逐 seed 差值 a-b（仅两边都有的 seed）。返回 (mean, std, n, same_sign)。"""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if not d:
        return None
    d = np.asarray(d)
    same = bool(np.all(d < 0) or np.all(d > 0))
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0, len(d), same


def fmt_delta(p, fmt="{:+.2f}"):
    if p is None:
        return "—"
    mean, std, n, same = p
    return (fmt + "±{:.2f} (n={}{})").format(mean, std, n, "，同向" if same else "，不同向")


def main():
    args = sys.argv[1:]
    root = args[0] if args and not args[0].startswith("--") else "results"
    md_out = args[args.index("--md") + 1] if "--md" in args else os.path.join(root, "aggregate_3seed.md")
    js_out = args[args.index("--json") + 1] if "--json" in args else os.path.join(root, "aggregate_3seed.json")
    runs = load_runs(root)
    persist_path = os.path.join(root, "metropt3_persistence_baseline.json")
    persist = json.load(open(persist_path))["test"] if os.path.exists(persist_path) else None

    L = []
    L.append("# 第三章 3-seed 汇总（自动生成：scripts/aggregate_3seed.py）\n")
    L.append(f"run 总数（可解析）: {len(runs)}；seeds = {SEEDS}；mean±std 为样本标准差（ddof=1）。\n")

    # ---------- 1. FD 主表 ----------
    L.append("## 1. FD002 / FD004 主表（V1 协议，test raw cycle，3 seeds）\n")
    L.append("| 数据集 | 模型 | MAE | RMSE | NASA |")
    L.append("|---|---|---|---|---|")
    for ds in ("fd002", "fd004"):
        for model in ("mtkdg", "gru", "tcn", "patchtst", "timesnet", "itransformer"):
            mae = get(runs, ds, model, "a0", "MAE")
            if all(v is None for v in mae):
                continue
            L.append(f"| {ds.upper()} | {model} | {ms(mae)} | {ms(get(runs, ds, model, 'a0', 'RMSE'))} | "
                     f"{ms(get(runs, ds, model, 'a0', 'NASA'), '{:.0f}')} |")
    L.append("")
    L.append("配对差值 Δ(MTK-DG − baseline)，逐 seed 相减：\n")
    L.append("| 数据集 | 对比 | ΔMAE | ΔRMSE |")
    L.append("|---|---|---|---|")
    for ds in ("fd002", "fd004"):
        for b in ("gru", "tcn"):
            L.append(f"| {ds.upper()} | vs {b} | "
                     f"{fmt_delta(paired(get(runs, ds, 'mtkdg', 'a0', 'MAE'), get(runs, ds, b, 'a0', 'MAE')))} | "
                     f"{fmt_delta(paired(get(runs, ds, 'mtkdg', 'a0', 'RMSE'), get(runs, ds, b, 'a0', 'RMSE')))} |")
    L.append("")

    # ---------- 2. MetroPT ----------
    L.append("## 2. MetroPT-3（首月协议，test raw unit overall，3 seeds）\n")
    L.append("### 2.1 direct 头（原主协议）+ 零参数 persistence 基线\n")
    L.append("| 模型 | overall_MAE | overall_RMSE |")
    L.append("|---|---|---|")
    if persist:
        L.append(f"| **persistence（y = x_last，0 参数）** | {persist['overall_MAE']:.4f} | {persist['overall_RMSE']:.4f} |")
    for model in ("mtkdg", "tcn", "gru", "patchtst", "timesnet", "itransformer"):
        mae = get(runs, "metropt3", model, "a0", "overall_MAE")
        if all(v is None for v in mae):
            continue
        L.append(f"| {model} (direct) | {ms(mae, '{:.4f}')} | {ms(get(runs, 'metropt3', model, 'a0', 'overall_RMSE'), '{:.4f}')} |")
    L.append("")
    L.append("### 2.2 persistence-skip（delta）头，全模型同头\n")
    L.append("| 模型 | overall_MAE | overall_RMSE | ΔRMSE vs persistence |")
    L.append("|---|---|---|---|")
    for model in ("mtkdg", "tcn", "gru", "patchtst", "timesnet", "itransformer"):
        variant = "v4deltaskip" if model == "mtkdg" else "delta"
        mae = get(runs, "metropt3", model, variant, "overall_MAE")
        if all(v is None for v in mae):
            continue
        rmse = get(runs, "metropt3", model, variant, "overall_RMSE")
        dp = ""
        if persist:
            vals = [v - persist["overall_RMSE"] for v in rmse if v is not None]
            dp = f"{np.mean(vals):+.4f}" if vals else "—"
        L.append(f"| {model} (delta) | {ms(mae, '{:.4f}')} | {ms(rmse, '{:.4f}')} | {dp} |")
    L.append("")
    L.append("配对 Δ(MTK-DG − TCN)：direct 头 "
             f"ΔoRMSE = {fmt_delta(paired(get(runs, 'metropt3', 'mtkdg', 'a0', 'overall_RMSE'), get(runs, 'metropt3', 'tcn', 'a0', 'overall_RMSE')), '{:+.4f}')}；"
             f"delta 头 ΔoRMSE = {fmt_delta(paired(get(runs, 'metropt3', 'mtkdg', 'v4deltaskip', 'overall_RMSE'), get(runs, 'metropt3', 'tcn', 'delta', 'overall_RMSE')), '{:+.4f}')}\n")

    # ---------- 3. 消融 ----------
    L.append("## 3. 结构消融（MTK-DG，3 seeds，括号为逐 seed 配对 Δ vs A0）\n")
    for ds, mk_mae, mk_rmse in (("fd002", "MAE", "RMSE"), ("fd004", "MAE", "RMSE"),
                                ("metropt3", "overall_MAE", "overall_RMSE")):
        fmt = "{:.4f}" if ds == "metropt3" else "{:.2f}"
        dfmt = "{:+.4f}" if ds == "metropt3" else "{:+.2f}"
        L.append(f"### {ds.upper()}\n")
        L.append(f"| 变体 | {mk_mae} | {mk_rmse} | Δ{mk_rmse} vs A0 |")
        L.append("|---|---|---|---|")
        a0_rmse = get(runs, ds, "mtkdg", "a0", mk_rmse)
        L.append(f"| {ABL_LABEL['a0']} | {ms(get(runs, ds, 'mtkdg', 'a0', mk_mae), fmt)} | {ms(a0_rmse, fmt)} | — |")
        for v in ABL_VARIANTS:
            if v == "fusionno_graph":
                continue  # 与 self_only 输出一致，表中只保留 self_only
            r = get(runs, ds, "mtkdg", v, mk_rmse)
            if all(x is None for x in r):
                continue
            L.append(f"| {ABL_LABEL[v]} | {ms(get(runs, ds, 'mtkdg', v, mk_mae), fmt)} | {ms(r, fmt)} | "
                     f"{fmt_delta(paired(r, a0_rmse), dfmt)} |")
        L.append("")

    # ---------- 4. MC-Graph ----------
    L.append("## 4. MC-Graph 门实验（A0/G1–G4，3 seeds）：干净 test + 损坏输入稳健性\n")
    L.append("稳健性 = 相对 clean RMSE 的中位相对增幅（3 个 corruption seed），数值越小越稳健；"
             "worst@0.3 = 两类损坏在 rate 0.3 下的较差者。\n")
    for ds in ("fd002", "fd004", "metropt3"):
        mk = "overall_RMSE" if ds == "metropt3" else "RMSE"
        fmt = "{:.4f}" if ds == "metropt3" else "{:.2f}"
        L.append(f"### {ds.upper()}\n")
        L.append("| 配置 | clean RMSE | Δclean vs A0 | ch_drop@0.3 | time_block@0.3 | worst@0.3 |")
        L.append("|---|---|---|---|---|---|")
        a0 = get(runs, ds, "mtkdg", "a0", mk)
        for g in ("a0", "g1", "g2", "g3", "g4"):
            rm = get(runs, ds, "mtkdg", g, mk)
            if all(x is None for x in rm):
                continue
            cd, tb, worst = [], [], []
            for s in SEEDS:
                r = runs.get((ds, "mtkdg", g, s))
                if r and r["robust"]:
                    cd.append(r["robust"]["corruptions"]["channel_dropout_0.3"]["rel_RMSE_increase_median"])
                    tb.append(r["robust"]["corruptions"]["time_block_0.3"]["rel_RMSE_increase_median"])
                    worst.append(r["robust"]["worst_rel_increase_at_0.3"])
            pct = lambda v: ("{:+.1%}±{:.1%}".format(np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else "—")
            L.append(f"| {MC_LABEL[g]} | {ms(rm, fmt)} | "
                     f"{'—' if g == 'a0' else fmt_delta(paired(rm, a0), '{:+.4f}' if ds == 'metropt3' else '{:+.2f}')} | "
                     f"{pct(cd)} | {pct(tb)} | {pct(worst)} |")
        L.append("")

    md = "\n".join(L)
    open(md_out, "w").write(md)
    json.dump({"|".join(map(str, k)): v for k, v in runs.items()}, open(js_out, "w"), indent=1, ensure_ascii=False)
    print(md)
    print(f"\n[written] {md_out}\n[written] {js_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

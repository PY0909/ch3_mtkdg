"""聚合 MetroPT delta 头补跑批次（20261026_dabl）：结构消融 + MC-Graph 稳健性。

用法：PYTHONPATH=. python scripts/aggregate_delta_family.py [results_dir] [--md out.md]

与 aggregate_3seed.py 的分工：本脚本只处理 dl* 变体（2026-10-26 批，全 delta 头），
复用其 load_runs/paired 辅助函数但不改其输出——旧表（20261020 批，含 direct 消融/G）
保持逐位可复现。表内比较全部在本批内部完成（dla0 为配对基准）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_3seed import load_runs, ms, paired  # noqa: E402

SEEDS = (2026, 2027, 2028)
ABL = (
    ("dla0", "A0 全模型（dynamic graph + gated fusion）"),
    ("dlgraphnone", "graph = none"),
    ("dlfusionadd", "fusion = add"),
    ("dlfusionself_only", "fusion = self_only（≡ no_graph）"),
)
MC = (
    ("dla0", "A0 基线"),
    ("dlg1", "G1 robust-train"),
    ("dlg2", "G2 robust + consistency"),
    ("dlg3", "G3 availability + robust"),
    ("dlg4", "G4 完整 MC-Graph"),
)


def get(runs, variant, metric):
    return [runs[("metropt3", "mtkdg", variant, s)]["metrics"][metric] for s in SEEDS]


def rob_med(runs, variant, ctype):
    key = f"{ctype}_0.3"
    out = []
    for s in SEEDS:
        r = runs[("metropt3", "mtkdg", variant, s)].get("robust")
        out.append(r["corruptions"][key]["rel_RMSE_increase_median"] if r else float("nan"))
    return out


def fmt_pct(vals):
    return f"{np.mean(vals) * 100:+.1f}%±{np.std(vals, ddof=1) * 100:.1f}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    md_path = None
    if "--md" in sys.argv:
        md_path = sys.argv[sys.argv.index("--md") + 1]

    runs = load_runs(root)
    lines = []
    w = lines.append
    w("# MetroPT delta 头补跑批次汇总（20261026_dabl，3 seeds，批内配对）\n")
    w("> 全部 `--state-head-mode delta`；dla0 为本批 A0 基准。与 20261020 direct 批不混表；")
    w("> mean±std 为样本标准差（ddof=1），配对 Δ 为逐 seed 相减，括号注明同向性。\n")

    # ---- 结构消融 ----
    w("## 1. 结构消融（MetroPT，delta 头）\n")
    w("| 变体 | overall_MAE | overall_RMSE | ΔRMSE vs dla0 |")
    w("|---|---|---|---|")
    a0 = get(runs, "dla0", "overall_RMSE")
    for var, label in ABL:
        mae = get(runs, var, "overall_MAE")
        rmse = get(runs, var, "overall_RMSE")
        if var == "dla0":
            w(f"| {label} | {ms(mae, '{:.4f}')} | {ms(rmse, '{:.4f}')} | — |")
        else:
            pm, ps, pn, same = paired(rmse, a0)
            w(f"| {label} | {ms(mae, '{:.4f}')} | {ms(rmse, '{:.4f}')} | "
              f"{pm:+.4f}±{ps:.4f} (n={pn}，{'同向' if same else '不同向'}) |")
    w("")

    # ---- MC-Graph 门实验 + 稳健性 ----
    w("## 2. MC-Graph 门实验（MetroPT，delta 头）：clean + 损坏稳健性\n")
    w("| 配置 | clean RMSE | Δclean vs dla0 | ch_drop@0.3 | time_block@0.3 | worst@0.3 |")
    w("|---|---|---|---|---|---|")
    for var, label in MC:
        clean = get(runs, var, "overall_RMSE")
        ch = rob_med(runs, var, "channel_dropout")
        tb = rob_med(runs, var, "time_block")
        worst = [max(c, t) for c, t in zip(ch, tb)]
        if var == "dla0":
            w(f"| {label} | {ms(clean, '{:.4f}')} | — | {fmt_pct(ch)} | {fmt_pct(tb)} | {fmt_pct(worst)} |")
        else:
            pm, ps, pn, same = paired(clean, a0)
            w(f"| {label} | {ms(clean, '{:.4f}')} | {pm:+.4f}±{ps:.4f} (n={pn}，{'同向' if same else '不同向'}) "
              f"| {fmt_pct(ch)} | {fmt_pct(tb)} | {fmt_pct(worst)} |")
    w("")

    # ---- 逐 seed 明细 ----
    w("## 3. 逐 seed 明细（delta 头）\n")
    w("| 变体 | seed | oMAE | oRMSE | ch@0.3 | tb@0.3 |")
    w("|---|---|---|---|---|---|")
    for var, label in ABL + MC:
        for i, s in enumerate(SEEDS):
            r = runs[("metropt3", "mtkdg", var, s)]
            rob = r.get("robust")
            ch = f"{rob['corruptions']['channel_dropout_0.3']['rel_RMSE_increase_median']:+.1%}" if rob else "—"
            tb = f"{rob['corruptions']['time_block_0.3']['rel_RMSE_increase_median']:+.1%}" if rob else "—"
            w(f"| {var} | {s} | {r['metrics']['overall_MAE']:.4f} | "
              f"{r['metrics']['overall_RMSE']:.4f} | {ch} | {tb} |")
    w("")

    text = "\n".join(lines)
    print(text)
    if md_path:
        with open(md_path, "w") as f:
            f.write(text + "\n")
        print(f"-> {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

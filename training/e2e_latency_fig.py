#!/usr/bin/env python3
"""
Real-time bounds of the end-to-end campaign (reviewer comment 3.2).

(a) CDF of window end -> alert (system jitter), all windows, per W.
(b) CDF of packet observation -> alert for alerted flows, with the design bound
    W + G + T_flush + delta_max as a vertical line.

Inputs : results_e2e/raw/W{30,40,50,60}.jsonl, timeline.csv
Outputs: results_e2e/latency_bounds.{pdf,png}, latency_bounds.json, table_latency_bounds.tex
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "results_e2e" / "raw"
OUT = HERE / "results_e2e"
WINDOWS = [30, 40, 50, 60]
GRACE, FLUSH = 5.0, 1.0


def load(W: int, t0: float, t1: float):
    fast, diag = [], []
    for line in open(RAW / f"W{W}.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("stage") == "diagnostic" and "pred_to_explanation" in r.get("latency", {}):
            diag.append(r)
        if r.get("stage") != "fast_path" or "first_pkt_to_alert_s" not in r["latency"]:
            continue
        t_first = r["timestamps"]["server"]["epoch"] - r["latency"]["first_pkt_to_alert_s"]
        if t0 <= t_first <= t1:
            fast.append(r)
    return fast, diag


def stats(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def main() -> None:
    tl = {r["phase"]: (float(r["start"]), float(r["end"])) for r in csv.DictReader(open(RAW / "timeline.csv")) if r.get("start")}
    t0, t1 = tl["normal_only"][0], max(b for _, b in tl.values()) + 70

    res, data = {}, {}
    for W in WINDOWS:
        fast, diag = load(W, t0, t1)
        alerts = [r for r in fast if r["decision"]["pred_class"] > 0]
        d = {
            "window_end_to_alert_s": [r["latency"]["window_end_to_alert_s"] for r in fast],
            "emit_to_alert_ms": [1000 * r["latency"]["window_emit_to_alert_s"] for r in fast],
            "alert_to_shap_ms": [r["latency"]["pred_to_explanation"] for r in diag],
            "first_pkt_to_alert_s": [r["latency"]["first_pkt_to_alert_s"] for r in alerts],
            "last_pkt_to_alert_s": [r["latency"]["last_pkt_to_alert_s"] for r in alerts],
        }
        data[W] = d
        res[W] = {k: stats(v) for k, v in d.items()}
    delta_max = max(res[W]["emit_to_alert_ms"]["max"] for W in WINDOWS) / 1000
    for W in WINDOWS:
        res[W]["design_bound_s"] = W + GRACE + FLUSH + delta_max
    (OUT / "latency_bounds.json").write_text(json.dumps(res, indent=2))

    # ---- figure
    colors = dict(zip(WINDOWS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for W in WINDOWS:
        v = np.sort(data[W]["window_end_to_alert_s"])
        a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[W], label=f"W = {W} s")
        v = np.sort(data[W]["first_pkt_to_alert_s"])
        b.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[W], label=f"W = {W} s")
        b.axvline(res[W]["design_bound_s"], color=colors[W], ls="--", lw=1)
    a.axvline(GRACE, color="gray", ls=":", lw=1)
    a.axvline(GRACE + FLUSH + delta_max, color="gray", ls=":", lw=1)
    a.set_xlim(4.8, 6.8)
    a.set_xlabel("Window end $\\rightarrow$ alert [s]")
    a.set_ylabel("CDF")
    a.set_title("(a) System delay (all windows)", fontsize=9)
    a.legend(fontsize=7, loc="lower right")
    b.set_xlabel("First packet of the window $\\rightarrow$ alert [s]")
    b.set_title("(b) Detection latency (alerts) and bound", fontsize=9)
    b.set_xlim(0, 70)
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"latency_bounds.{ext}", dpi=300)

    # ---- table
    def cell(s, scale=1, fmt="{:.2f}"):
        return (fmt + " $\\pm$ " + fmt + " [" + fmt + ", " + fmt + "]").format(
            s["mean"] * scale, s["std"] * scale, s["min"] * scale, s["max"] * scale)
    rows = [("Window end $\\rightarrow$ alert [s]", "window_end_to_alert_s", "{:.2f}"),
            ("Feature emission $\\rightarrow$ alert [ms]", "emit_to_alert_ms", "{:.0f}"),
            ("Alert $\\rightarrow$ SHAP explanation [ms]", "alert_to_shap_ms", "{:.0f}"),
            ("First packet $\\rightarrow$ alert [s]", "first_pkt_to_alert_s", "{:.1f}")]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Interval} & \textbf{30 s} & \textbf{40 s} & \textbf{50 s} & \textbf{60 s} \\", r"\midrule"]
    for name, k, fmt in rows:
        L.append(name + " & " + " & ".join(cell(res[W][k], fmt=fmt) for W in WINDOWS) + r" \\")
    L.append(r"Design bound $W+G+T_f+\delta_{\max}$ [s] & " + " & ".join(f"{res[W]['design_bound_s']:.1f}" for W in WINDOWS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_latency_bounds.tex").write_text("\n".join(L) + "\n")
    print(json.dumps({W: {k: (round(v['mean'], 2), round(v['std'], 2), round(v['min'], 2), round(v['max'], 2), v['n'])
                          for k, v in res[W].items() if isinstance(v, dict)} for W in WINDOWS}, indent=1))
    print("delta_max", delta_max)


if __name__ == "__main__":
    main()

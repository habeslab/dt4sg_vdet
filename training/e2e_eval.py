#!/usr/bin/env python3
"""
End-to-end evaluation on the GNS3 testbed (reviewer comments 3.1 / 3.2).

Campaign: normal testbed traffic (master <-> 7 RTUs) for 10 min, then 10-min
segments of four UOWM attack captures whose classes were EXCLUDED from training
(c_ci_na_1 -> Integrity; c_rd_na_1_DoS, c_rp_na_1_DoS, m_sp_na_1_DoS -> Availability),
replayed in real time by the attacker node. Four net-agent instances observe the
same hub with windows W = 30/40/50/60 s and query the same disc-api.

Ground truth (flow level, from the UOWM per-flow labels of the iecserver2 captures):
  192.168.1.24 <-> 192.168.1.28   attack of the current phase (attacker .28)
  192.168.1.20 <-> 192.168.1.24   Normal (SCADA master), except during m_sp_na_1_DoS
  10.0.0.x (testbed)              Normal (rtuattacker .90 alters process values only)

Inputs : --raw DIR with W{30,40,50,60}.jsonl (net-agent /data/flows.jsonl) and timeline.csv
Outputs: --out DIR: summary.json, per_window.csv, latency.csv, table_e2e.tex, table_latency.tex
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["Normal", "Availability", "Integrity"]
PHASE_CLASS = {"c_ci_na_1": 2, "c_rd_na_1_DoS": 1, "c_rp_na_1_DoS": 1, "m_sp_na_1_DoS": 1}
WINDOWS = [30, 40, 50, 60]
GRACE = 5.0  # net-agent default GRACE_SEC


def load_timeline(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("start"):
            out[r["phase"]] = (float(r["start"]), float(r["end"]))
    return out


def phase_of(t: float, tl: dict[str, tuple[float, float]], W: int) -> str | None:
    for ph, (a, b) in tl.items():
        if ph != "normal_only" and a - W <= t <= b + W + GRACE:
            return ph
    return None


def label(flow_id: str, t: float, tl, W: int) -> tuple[int, str]:
    ips = {x.split(":")[0] for x in flow_id.split("|")[:2]}
    ph = phase_of(t, tl, W)
    if ips == {"192.168.1.24", "192.168.1.28"}:
        return (PHASE_CLASS[ph], ph) if ph else (-1, "unassigned")
    if ips == {"192.168.1.20", "192.168.1.24"}:
        if ph == "m_sp_na_1_DoS":
            return 1, ph
        return 0, "uowm_normal"
    return 0, "testbed_normal"


def prf(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(3), where=(p + r) > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_precision": float(p.mean()),
            "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
            **{f"recall_{c}": float(r[i]) for i, c in enumerate(CLASSES)},
            **{f"precision_{c}": float(p[i]) for i, c in enumerate(CLASSES)}}


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tl = load_timeline(args.raw / "timeline.csv")
    t_start = tl["normal_only"][0]
    t_end = max(b for _, b in tl.values()) + 60 + GRACE

    summary, per_window_rows, lat_rows = {}, [], []
    for W in WINDOWS:
        recs = [json.loads(l) for l in open(args.raw / f"W{W}.jsonl") if l.strip()]
        fast = [r for r in recs if r.get("stage") == "fast_path"]
        diag = {r["alert_id"]: r for r in recs if r.get("stage") == "diagnostic"}
        cm = np.zeros((3, 3), dtype=int)
        by_src = defaultdict(Counter)
        alerts_lat, diag_lat = defaultdict(list), defaultdict(list)
        first_alert, first_attack_pkt = {}, {}
        for r in fast:
            lat = r["latency"]
            t_alert = r["timestamps"]["server"]["epoch"]
            if "first_pkt_to_alert_s" not in lat:
                continue
            t_first = t_alert - lat["first_pkt_to_alert_s"]  # capture time of the first packet of the window
            if not (t_start <= t_first <= t_end):
                continue
            y, src = label(r["flow_id"], t_first, tl, W)
            if y < 0:
                continue
            yhat = int(r["decision"]["pred_class"])
            cm[y, yhat] += 1
            by_src[src][yhat] += 1
            if y > 0:
                first_attack_pkt[src] = min(first_attack_pkt.get(src, 1e18), t_first)
                if yhat > 0:
                    first_alert[src] = min(first_alert.get(src, 1e18), t_alert)
            if yhat > 0:
                for k in ("first_pkt_to_alert_s", "last_pkt_to_alert_s", "window_end_to_alert_s",
                          "window_emit_to_alert_s"):
                    if k in lat:
                        alerts_lat[k].append(lat[k])
                alerts_lat["http_rtt_ms"].append(lat["http_rtt_ms"])
                d = diag.get(r["decision"].get("alert_id"))
                if d:
                    for k in ("last_pkt_to_explanation_s", "pred_to_explanation", "compute"):
                        if k in d["latency"]:
                            diag_lat[k].append(d["latency"][k])
        m = prf(cm)
        # binary view: attack (Availability or Integrity) vs Normal
        tp = int(cm[1:, 1:].sum()); fn = int(cm[1:, 0].sum()); fp = int(cm[0, 1:].sum()); tn = int(cm[0, 0])
        bp = tp / (tp + fp) if tp + fp else 0.0
        br = tp / (tp + fn) if tp + fn else 0.0
        m.update({"binary_accuracy": (tp + tn) / cm.sum(), "binary_precision": bp, "binary_recall": br,
                  "binary_f1": 2 * bp * br / (bp + br) if bp + br else 0.0,
                  "fpr_testbed": sum(by_src["testbed_normal"][c] for c in (1, 2)) / max(1, sum(by_src["testbed_normal"].values())),
                  "fpr_all_normal": fp / max(1, fp + tn),
                  **{f"detection_{ph}": sum(by_src[ph][c] for c in (1, 2)) / max(1, sum(by_src[ph].values()))
                     for ph in PHASE_CLASS}})
        ttd = {ph: first_alert[ph] - first_attack_pkt[ph] for ph in first_attack_pkt if ph in first_alert}
        summary[W] = {
            "confusion": cm.tolist(), **m,
            "by_source": {k: {CLASSES[i]: v.get(i, 0) for i in range(3)} for k, v in by_src.items()},
            "time_to_first_alert_s": ttd,
            "n_alerts": int(sum(by_src[s][c] for s in by_src for c in (1, 2))),
            "n_diagnostic": len(diag_lat["pred_to_explanation"]),
        }
        lat_summary = {}
        for k, v in {**alerts_lat, **diag_lat}.items():
            lat_summary[k] = {"median": q(v, 50), "p95": q(v, 95), "n": len(v)}
            lat_rows.append({"W": W, "metric": k, "median": q(v, 50), "p95": q(v, 95), "n": len(v)})
        summary[W]["latency"] = lat_summary
        per_window_rows.append({"W": W, **{k: v for k, v in m.items()}})
        print(f"W={W}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"rec A/I={m['recall_Availability']:.3f}/{m['recall_Integrity']:.3f}  cm={cm.tolist()}")
        print("   by source:", {k: dict(v) for k, v in by_src.items()})
        print("   time-to-first-alert:", {k: round(v, 1) for k, v in ttd.items()})
        print("   latency:", {k: (round(v['median'], 3), round(v['p95'], 3)) for k, v in lat_summary.items()})

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.out / "per_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_window_rows[0].keys())); w.writeheader(); w.writerows(per_window_rows)
    with open(args.out / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["W", "metric", "median", "p95", "n"]); w.writeheader(); w.writerows(lat_rows)

    # LaTeX tables
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Metric} & \textbf{30 s} & \textbf{40 s} & \textbf{50 s} & \textbf{60 s} \\", r"\midrule"]
    rows = [("binary_accuracy", "Binary accuracy"), ("binary_precision", "Binary precision"),
            ("binary_recall", "Binary recall (detection rate)"), ("binary_f1", "Binary F1"),
            ("fpr_testbed", "False positive rate (testbed)"),
            *[(f"detection_{ph}", f"Detection {ph}") for ph in PHASE_CLASS],
            ("accuracy", "3-class accuracy"), ("macro_f1", "3-class macro-F1")]
    for key, name in rows:
        L.append(f"{name.replace('_', chr(92) + '_')} & " + " & ".join(f"{100 * summary[W][key]:.1f}\\%" for W in WINDOWS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "table_e2e.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Latency (median / p95)} & \textbf{30 s} & \textbf{40 s} & \textbf{50 s} & \textbf{60 s} \\", r"\midrule"]
    for key, name, scale, unit in [("last_pkt_to_alert_s", "Last packet $\\rightarrow$ alert", 1, "s"),
                                   ("window_end_to_alert_s", "Window end $\\rightarrow$ alert", 1, "s"),
                                   ("window_emit_to_alert_s", "Feature emission $\\rightarrow$ alert", 1000, "ms"),
                                   ("pred_to_explanation", "Alert $\\rightarrow$ explanation", 1, "ms"),
                                   ("last_pkt_to_explanation_s", "Last packet $\\rightarrow$ explanation", 1, "s")]:
        cells = []
        for W in WINDOWS:
            s = summary[W]["latency"].get(key)
            cells.append(f"{s['median'] * scale:.1f} / {s['p95'] * scale:.1f}" if s and s["n"] else "--")
        L.append(f"{name} [{unit}] & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "table_latency.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

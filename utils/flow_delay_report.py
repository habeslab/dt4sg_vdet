#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Delay Report — SOLO MALIGNI (multi-attacco + IP falsi-positivi)
====================================================================

Scelte:
- Detection timestamp: preferisci server.local, fallback client.local.
- Associazione detection→attacco: default 'start-only' (ultimo start ≤ t).
  Alternativa: --attack-match in-window (start ≤ t < end).
- IP falsi positivi (10.0.0.12–18): default --ignore-ips-mode drop (escludi).
- I BENIGNI NON SONO CONSIDERATI: vengono esclusi a monte.

Output (solo maligni):
- CSV arricchito (solo flussi maligni tenuti)
- attacks_summary.csv (per attacco: count, first, mean, median, p90)
- attacks_coverage.csv (coverage degli attacchi)
- Grafici: istogramma, CDF, bar mean per attacco, serie tempo/minuto
- REPORT.md minimale (solo maligni)

Uso:
  python3 flow_delay_report_maligni.py \
    --attack-file attacchi.txt \
    --input flows.jsonl \
    --outdir report_out \
    --ignore-ips-mode drop \
    --attack-match start-only
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import math
import re
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Timestamp utils ------------------------------

def parse_iso_local(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        raise ValueError("Timestamp deve includere timezone offset, es. +02:00")
    return dt

def pick_detection_ts(flow: Dict[str, Any]) -> Optional[datetime]:
    """Preferisci server.local; fallback client.local; None se assenti/invalidi."""
    ts = flow.get("timestamps", {}) or {}
    server_local = (ts.get("server") or {}).get("local")
    client_local = (ts.get("client") or {}).get("local")
    try:
        if server_local:
            return parse_iso_local(server_local)
        if client_local:
            return parse_iso_local(client_local)
    except Exception:
        return None
    return None

def first_any_local(rows: List[Dict[str, Any]]) -> Optional[datetime]:
    """Serve per dedurre data/tz degli attacchi."""
    for flow in rows:
        ts = flow.get("timestamps", {}) or {}
        v = (ts.get("server") or {}).get("local")
        if v:
            try:
                return parse_iso_local(v)
            except Exception:
                pass
    for flow in rows:
        ts = flow.get("timestamps", {}) or {}
        v = (ts.get("client") or {}).get("local")
        if v:
            try:
                return parse_iso_local(v)
            except Exception:
                pass
    return None

# ----------------------------- IP handling -----------------------------------

FALSE_POSITIVE_IPS = {
    "10.0.0.12","10.0.0.13","10.0.0.14","10.0.0.15","10.0.0.16","10.0.0.17","10.0.0.18"
}

def extract_ips(flow: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    src = flow.get("src_ip") or (flow.get("client") or {}).get("ip") or (flow.get("five_tuple") or {}).get("src_ip")
    dst = flow.get("dst_ip") or (flow.get("server") or {}).get("ip") or (flow.get("five_tuple") or {}).get("dst_ip")
    return src, dst

def is_false_positive_ip(flow: Dict[str, Any]) -> bool:
    src, dst = extract_ips(flow)
    return (src in FALSE_POSITIVE_IPS) or (dst in FALSE_POSITIVE_IPS)

# ----------------------------- Decision & latency ----------------------------

def is_synthetic(flow: Dict[str, Any]) -> bool:
    dec = flow.get("decision", {}) or {}
    return (dec.get("label") == "synthetic") and (dec.get("origin") == "synthetic")

def extract_end_to_end(flow: Dict[str, Any]) -> Optional[float]:
    lat = flow.get("latency", {}) or {}
    val = lat.get("end_to_end_s")
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

# ----------------------------- Attack file parsing ---------------------------

# Estrae HH:MM (evita i secondi "HH:MM:SS")
HHMM_RE_STRICT = re.compile(r"(\d{1,2})\s*:\s*(\d{2})(?!\s*:)")

def parse_attack_file(text_path: Path, base_date: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Ritorna lista di intervalli [(start, end)] tz-aware leggendo il file attacchi.
    Associa ogni riga "Ora lancio attacco X" con la successiva "Fine ... attacco".
    """
    content = text_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    tz = base_date.tzinfo or timezone.utc
    day0 = base_date.replace(hour=0, minute=0, second=0, microsecond=0)

    intervals: List[Tuple[datetime, datetime]] = []
    timestamps: List[datetime] = []

    prev_m = None
    day_cursor = day0

    # Funzione helper per creare datetime da hh:mm
    def make_dt(hh: int, mm: int, prev_m: Optional[int], day_cursor: datetime) -> Tuple[datetime, int, datetime]:
        m = hh * 60 + mm
        if prev_m is not None and m < prev_m:
            # rollover giorno successivo
            day_cursor = day_cursor + timedelta(days=1)
        ts = day_cursor.replace(hour=hh, minute=mm, tzinfo=tz)
        return ts, m, day_cursor

    for line in lines:
        matches = list(HHMM_RE_STRICT.finditer(line))
        if not matches:
            continue
        hh = int(matches[-1].group(1))
        mm = int(matches[-1].group(2))
        ts, prev_m, day_cursor = make_dt(hh, mm, prev_m, day_cursor)
        timestamps.append(ts)

    # Ora accoppia start/end a coppie consecutive
    for i in range(0, len(timestamps), 2):
        if i + 1 < len(timestamps):
            intervals.append((timestamps[i], timestamps[i+1]))

    return intervals


def assign_start_only(t: datetime, intervals: List[Tuple[datetime, datetime]]
                      ) -> Optional[Tuple[int, datetime, datetime]]:
    """Ultimo start ≤ t (ritorna anche end della stessa finestra)."""
    cand = None
    for idx, (st, en) in enumerate(intervals, start=1):
        if st <= t:
            cand = (idx, st, en)
        else:
            break
    return cand

def assign_in_window(t: datetime, intervals: List[Tuple[datetime, datetime]]
                     ) -> Optional[Tuple[int, datetime, datetime]]:
    """start ≤ t < end."""
    for idx, (st, en) in enumerate(intervals, start=1):
        if st <= t < en:
            return (idx, st, en)
    return None

# ----------------------------- Main ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Report SOLO MALIGNI da JSONL (multi-attacco, IP falsi-positivi).")
    ap.add_argument("--attack-file", required=True, help="File attacchi (HH:MM in coppie start/end).")
    ap.add_argument("--input", required=True, help="File JSONL dei flussi.")
    ap.add_argument("--outdir", default=None, help="Directory output (default: report_YYYYmmdd_HHMMSS)")
    ap.add_argument("--ignore-ips-mode", choices=["drop","keep"], default="drop",
                    help="Falsi positivi IP: 'drop' = escludi (default); 'keep' = tieni.")
    ap.add_argument("--attack-match", choices=["start-only","in-window"], default="start-only",
                    help="Associazione detection→attacco.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input non trovato: {in_path}")

    outdir = Path(args.outdir) if args.outdir else Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Carica JSONL
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                print(f"[WARN] Riga {ln}: JSON invalido ({e}). Salto.")
    if not rows:
        raise SystemExit("Nessun JSON valido trovato.")

    # Base data/tz per interpretare gli orari attacco
    base_dt = first_any_local(rows)
    if base_dt is None:
        for flow in rows:
            det = pick_detection_ts(flow)
            if det:
                base_dt = det
                break
    if base_dt is None:
        raise SystemExit("Impossibile dedurre data/timezone.")

    # Intervalli attacco
    attack_file = Path(args.attack_file)
    if not attack_file.exists():
        raise SystemExit(f"File attacchi non trovato: {attack_file}")
    attack_intervals = parse_attack_file(attack_file, base_date=base_dt)

    # ---------------- Enrichment (SOLO MALIGNI) ----------------
    recs = []
    for flow in rows:
        # scarta non-maligni
        if not is_synthetic(flow):
            continue

        # IP falsi positivi
        if args.ignore_ips_mode == "drop" and is_false_positive_ip(flow):
            continue

        det = pick_detection_ts(flow)
        if det is None:
            continue

        e2e = extract_end_to_end(flow)
        src_ip, dst_ip = extract_ips(flow)

        # associazione attacco
        attack_idx = None
        attack_start = None
        attack_end = None
        delay_from_attack = None
        if attack_intervals:
            assign = assign_start_only(det, attack_intervals) if args.attack_match == "start-only" \
                     else assign_in_window(det, attack_intervals)
            if assign:
                attack_idx, attack_start, attack_end = assign
                delay_from_attack = (det - attack_start).total_seconds()

        recs.append({
            "flow_id": flow.get("flow_id"),
            "detection_ts": det.isoformat(),
            "attack_index": attack_idx,
            "attack_start_iso": attack_start.isoformat() if attack_start else None,
            "attack_end_iso": attack_end.isoformat() if attack_end else None,
            "delay_from_attack_s": delay_from_attack,
            "src_ip": src_ip, "dst_ip": dst_ip,
            "end_to_end_s": e2e,
        })

    df = pd.DataFrame.from_records(recs)
    # solo righe con un delay valido
    df_synth = df[df["delay_from_attack_s"].notna()].copy()

    # ---------------- Coverage per attacco ----------------
    per_attack_rows = []
    for idx, (start, end) in enumerate(attack_intervals, start=1):
        sub = df_synth[df_synth["attack_index"] == idx]
        delays = sub["delay_from_attack_s"].dropna()
        count = int(sub.shape[0])
        first_det = float(delays.min()) if len(delays) else math.nan

        def q(s, p): return float(s.quantile(p)) if len(s) else float("nan")
        per_attack_rows.append({
            "attack_index": idx,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "count": count,
            "first_detection_s": first_det,
            "mean_delay_s": float(delays.mean()) if len(delays) else float("nan"),
            "median_delay_s": float(delays.median()) if len(delays) else float("nan"),
            "p90_delay_s": q(delays, 0.90),
        })
    df_per_attack = pd.DataFrame(per_attack_rows)

    total_attacks = len(attack_intervals)
    detected_attacks = int((df_per_attack["count"] > 0).sum()) if total_attacks else 0
    missed_attacks = total_attacks - detected_attacks
    coverage_pct = (100.0 * detected_attacks / total_attacks) if total_attacks else float("nan")
    df_coverage = pd.DataFrame([{
        "total_attacks": total_attacks,
        "detected_attacks": detected_attacks,
        "missed_attacks": missed_attacks,
        "coverage_pct": coverage_pct,
        "policy": args.attack_match,
        "ignore_ips_mode": args.ignore_ips_mode,
    }])

    # ---------------- Statistiche globali (solo maligni) ----------------
    delays_all = df_synth["delay_from_attack_s"].dropna()
    synth_count = int(delays_all.shape[0])
    synth_mean = float(delays_all.mean()) if synth_count else float("nan")
    synth_median = float(delays_all.median()) if synth_count else float("nan")
    synth_p90 = float(delays_all.quantile(0.90)) if synth_count else float("nan")

    # ---------------- Salvataggi CSV ----------------
    df_synth.to_csv(outdir / "flows_enriched.csv", index=False, encoding="utf-8")
    df_per_attack.to_csv(outdir / "attacks_summary.csv", index=False, encoding="utf-8")
    df_coverage.to_csv(outdir / "attacks_coverage.csv", index=False, encoding="utf-8")

    # ---------------- Grafici ----------------
    if synth_count > 0:
        # Istogramma
        plt.figure()
        delays_all.hist(bins=30)
        plt.xlabel("Reattività (s) da lancio attacco")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione reattività - Maligni")
        plt.savefig(outdir / "hist_delay_maligni.png", bbox_inches="tight")
        plt.close()

        # CDF
        plt.figure()
        xs = sorted(delays_all.values)
        ys = [i/len(xs) for i in range(1, len(xs)+1)]
        plt.plot(xs, ys)
        plt.xlabel("Reattività (s) da lancio attacco")
        plt.ylabel("Cumulativa")
        plt.title("CDF reattività - Maligni")
        plt.savefig(outdir / "cdf_delay_maligni.png", bbox_inches="tight")
        plt.close()

    # Bar per attacco
    if not df_per_attack.empty:
        plt.figure()
        plt.bar(df_per_attack["attack_index"].astype(str), df_per_attack["mean_delay_s"].fillna(0.0))
        plt.xlabel("Attacco"); plt.ylabel("Mean delay (s)")
        plt.title("Mean delay per attacco (maligni)")
        plt.savefig(outdir / "bar_mean_per_attack.png", bbox_inches="tight")
        plt.close()

    # Serie temporale conteggi (solo maligni tenuti)
    if df_synth["detection_ts"].notna().any():
        tmp = df_synth.dropna(subset=["detection_ts"]).copy()
        tmp["detection_ts"] = pd.to_datetime(tmp["detection_ts"])
        counts = tmp.set_index("detection_ts").resample("1min").size()
        plt.figure()
        counts.plot()
        plt.xlabel("Tempo (min)"); plt.ylabel("Conteggio flussi (maligni)")
        plt.title("Flussi maligni rilevati per minuto")
        plt.savefig(outdir / "timeseries_counts_per_min.png", bbox_inches="tight")
        plt.close()

    # ---------------- REPORT (solo maligni) ----------------
    md = []
    md.append(f"# Flow Delay Report — **SOLO MALIGNI**\n")
    md.append(f"- Generato: {datetime.now().isoformat(timespec='seconds')}\n")
    md.append(f"- Input flussi: `{in_path.name}`\n")
    md.append(f"- File attacchi: `{attack_file.name}`\n")
    md.append(f"- Associazione attacchi: `{args.attack_match}`\n")
    md.append(f"- IP falsi-positivi: `{args.ignore_ips_mode}`\n")

    md.append("\n## Finestre d'attacco\n")
    if attack_intervals:
        for idx, (st, en) in enumerate(attack_intervals, start=1):
            md.append(f"- Attacco {idx}: **{st.isoformat()} → {en.isoformat()}**\n")
    else:
        md.append("_Nessun intervallo d'attacco trovato._\n")

    md.append("\n## Coverage attacchi\n")
    md.append(f"- Attacchi totali: **{total_attacks}**\n")
    md.append(f"- Attacchi con ≥1 detection: **{detected_attacks}**\n")
    md.append(f"- Attacchi senza detection: **{missed_attacks}**\n")
    md.append(f"- Coverage: **{coverage_pct:.1f}%**\n")

    md.append("\n## Statistiche globali ritardi (maligni)\n")
    md.append(f"- Campioni (flussi maligni): **{synth_count}**\n")
    md.append(f"- Mean: **{synth_mean:.3f} s**\n" if not math.isnan(synth_mean) else "- Mean: n/d\n")
    md.append(f"- Median: **{synth_median:.3f} s**\n" if not math.isnan(synth_median) else "- Median: n/d\n")
    md.append(f"- P90: **{synth_p90:.3f} s**\n" if not math.isnan(synth_p90) else "- P90: n/d\n")

    md.append("\n## Statistiche per attacco\n")
    if not df_per_attack.empty:
        md.append("| Attacco | Start | End | # Flussi | First (s) | Mean (s) | Median (s) | P90 (s) |\n")
        md.append("|---:|---|---|---:|---:|---:|---:|---:|\n")
        for _, r in df_per_attack.iterrows():
            def fmt(x):
                return f"{x:.3f}" if (isinstance(x, (float,int)) and not math.isnan(x)) else "n/d"
            md.append(
                f"| {int(r['attack_index'])} | {r['start']} | {r['end']} | "
                f"{int(r['count'])} | {fmt(r['first_detection_s'])} | "
                f"{fmt(r['mean_delay_s'])} | {fmt(r['median_delay_s'])} | {fmt(r['p90_delay_s'])} |\n"
            )
    else:
        md.append("_Nessuna detection associabile agli attacchi._\n")

    md.append("\n## Grafici\n")
    if synth_count > 0:
        md.append("- Istogramma reattività maligni: `hist_delay_maligni.png`\n")
        md.append("- CDF reattività maligni: `cdf_delay_maligni.png`\n")
    if not df_per_attack.empty:
        md.append("- Mean per attacco: `bar_mean_per_attack.png`\n")
    if df_synth["detection_ts"].notna().any():
        md.append("- Flussi maligni per minuto: `timeseries_counts_per_min.png`\n")

    with (outdir / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"[OK] Report SOLO MALIGNI creato in: {outdir.resolve()}")
    print(f"- Markdown: {outdir / 'REPORT.md'}")
    print(f"- CSV maligni: {outdir / 'flows_enriched.csv'}")
    print(f"- Per-attacco: {outdir / 'attacks_summary.csv'}")
    print(f"- Coverage: {outdir / 'attacks_coverage.csv'}")
    print(f"- Grafici PNG nella stessa cartella.")

if __name__ == "__main__":
    main()

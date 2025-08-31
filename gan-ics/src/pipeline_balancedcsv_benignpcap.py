"""
Pipeline: Balanced CSV (maligni) + PCAP (benigni) -> train/val pronti (schema semplice)
- Scansione ricorsiva cartelle fornite
- Maligni: legge CSV già bilanciati dal sito (file che contengono 'train' e 'test' nel nome), proietta su schema minimo
- Benigni: converte PCAP in CSV numerici aggregando per (src,dst,sport,dport,proto, finestra) con contatori
- Etichette: benigni=0, maligni=1
- Merge: train/test fissi (maligni già separati); benigni splittati con ratio configurabile
- Oversampling: solo sul train (dopo deduplica)
- Scaler: fit su train, salva artifacts

Requisiti:
  - python3, pandas, numpy, scikit-learn
  - tshark (sudo apt install tshark)
"""

from __future__ import annotations
import argparse, os, sys, csv, json, pickle, shutil, subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ====== Schema numerico unificato ======
# (Campi salvati nei CSV intermedi/finali)
NUM_HEADER = ["sport", "dport", "proto", "pkt_count", "bytes_total"]

FEATURES_JSON = "features.json"
SCALER_PKL    = "scaler.pkl"
TRAIN_RAW     = "gan_train_raw.csv"
VAL_RAW       = "gan_val_raw.csv"
MAL_TRAIN_CAT = "mal_train_concat.csv"
MAL_TEST_CAT  = "mal_test_concat.csv"
BEN_CAT       = "benign_concat.csv"

# -------------------- util --------------------
def echo(s=""): print(s, flush=True)

def check_tshark():
    try:
        subprocess.check_output(["tshark", "-v"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        echo("[ERRORE] tshark non trovato. Installa con: sudo apt install tshark")
        sys.exit(1)

def is_csv(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".csv"

def is_pcap(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in (".pcap", ".pcapng")

def list_files_recursive(roots: List[Path], pred) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            echo(f"[WARN] cartella non trovata: {r}")
            continue
        for current, _dirs, files in os.walk(r):
            base = Path(current)
            for fn in files:
                p = base / fn
                try:
                    if pred(p):
                        out.append(p)
                except Exception:
                    pass
    return sorted(out)

# -------------------- PCAP -> CSV numerico (schema con finestra) --------------------
def run_tshark(pcap: Path, final_filter: Optional[str], timeout_sec: int = 90) -> List[str]:
    """
    Estrae i campi minimi per contatori:
    - no name resolution (-n)
    - niente TCP reassembly
    - timeout per evitare blocchi su file problematici
    Nota: final_filter è inteso come *display filter* (es. 'tcp.port==2404'), passato a -Y.
    """
    cmd = [
        "tshark", "-n",
        "-r", str(pcap),
        "-T", "fields",
        "-o", "tcp.desegment_tcp_streams:false",
        # campi necessari per aggregazione a finestra e 5-tuple
        "-e", "frame.time_epoch",
        "-e", "ip.src",
        "-e", "ip.dst",
        # campi numerici
        "-e", "frame.len",
        "-e", "ip.proto",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"
    ]
    if final_filter:
        cmd += ["-Y", final_filter]

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=timeout_sec)
        return proc.stdout.splitlines()
    except subprocess.TimeoutExpired:
        echo(f"[WARN] tshark TIMEOUT su {pcap.name} ({timeout_sec}s) → skip")
        return []
    except subprocess.CalledProcessError as e:
        echo(f"[WARN] tshark fallito su {pcap.name} (rc={e.returncode})")
        return []

def pcap_to_numeric_df(pcap: Path, user_filter: Optional[str], window_sec: int = 60) -> pd.DataFrame:
    """
    Aggrega per finestra temporale e 5-tuple (src,dst,sport,dport,proto).
    Manteniamo in output solo le 5 feature numeriche NUM_HEADER.
    Filtra: solo TCP (proto=6) e porte > 0.
    """
    rows = run_tshark(pcap, user_filter)
    if not rows:
        return pd.DataFrame(columns=NUM_HEADER)

    reader = csv.DictReader(rows)
    buckets: Dict[Tuple[str, str, int, int, int, int], dict] = {}
    t0: Optional[float] = None

    for r in reader:
        try:
            t = float(r.get("frame.time_epoch") or 0.0)
            src = (r.get("ip.src") or "").strip()
            dst = (r.get("ip.dst") or "").strip()
            proto  = int(r.get("ip.proto") or 0)
            sport  = int(r.get("tcp.srcport") or 0)
            dport  = int(r.get("tcp.dstport") or 0)
            length = int(r.get("frame.len") or 0)
        except Exception:
            continue

        # tieni solo TCP IEC-104 e porte valide
        if proto != 6:
            continue
        if not (sport > 0 and dport > 0):
            continue

        if t0 is None:
            t0 = t
        win = int((t - t0) // window_sec)

        key = (src, dst, sport, dport, proto, win)
        agg = buckets.setdefault(key, {"pkt_count": 0, "bytes_total": 0})
        agg["pkt_count"]  += 1
        agg["bytes_total"] += length

    if not buckets:
        return pd.DataFrame(columns=NUM_HEADER)

    recs = []
    for (_src, _dst, sport, dport, proto, _win), a in buckets.items():
        recs.append({
            "sport": sport,
            "dport": dport,
            "proto": proto,
            "pkt_count": a["pkt_count"],
            "bytes_total": a["bytes_total"],
        })
    out = pd.DataFrame.from_records(recs, columns=NUM_HEADER)
    return out

def convert_benign_pcaps(pcaps: List[Path], out_dir: Path, bpf: Optional[str], window_sec: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, p in enumerate(pcaps, 1):
        try:
            df = pcap_to_numeric_df(p, user_filter=bpf, window_sec=window_sec)
            frames.append(df)
            echo(f"[OK] {p.name}: {len(df)} righe")
        except Exception as e:
            echo(f"[WARN] errore su {p}: {type(e).__name__}: {e}")

    df_all = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=NUM_HEADER)
    cat_path = out_dir / BEN_CAT
    df_all.to_csv(cat_path, index=False)
    echo(f"[BENIGN] concatenato -> {cat_path} (righe={len(df_all)})")
    return cat_path

# -------------------- CSV maligni (già bilanciati) -> proiezione semplice --------------------
def _project_malicious_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proietta un CSV (CICFlowMeter / custom) sullo schema minimo NUM_HEADER,
    usando SEMPRE Serie (mai scalari) come fallback per evitare errori di astype.
    In più: elimina le righe completamente nulle e filtra solo TCP/porte>0.
    """
    n = len(df)
    zeros_i = pd.Series(np.zeros(n, dtype=int))

    cols = {c.strip(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # ports
    c_sport = col("Src Port", "src_port", "Sport", "tcp.srcport", "sport")
    c_dport = col("Dst Port", "dst_port", "Dport", "tcp.dstport", "dport")
    # protocol
    c_proto = col("Protocol", "protocol", "ip.proto", "proto")
    # packets
    c_fwdpk = col("Tot Fwd Pkts", "total_fwd_packets", "Fwd Pkts")
    c_bwdpk = col("Tot Bwd Pkts", "total_bwd_packets", "Bwd Pkts")
    c_pk    = None
    if c_fwdpk is None and c_bwdpk is None:
        c_pk = col("Tot Pkts", "total_packets", "Pkt Count")
    # bytes
    c_fwdby = col("TotLen Fwd Pkts", "total_fwd_bytes", "Fwd Bytes", "Tot Fwd Bytes")
    c_bwdby = col("TotLen Bwd Pkts", "total_bwd_bytes", "Bwd Bytes", "Tot Bwd Bytes")
    c_by    = None
    if c_fwdby is None and c_bwdby is None:
        c_by = col("Tot Bytes", "total_bytes", "Bytes Total")

    # sport/dport
    sport = (pd.to_numeric(df[c_sport], errors="coerce").fillna(0).astype(int)) if c_sport else zeros_i.copy()
    dport = (pd.to_numeric(df[c_dport], errors="coerce").fillna(0).astype(int)) if c_dport else zeros_i.copy()

    # proto
    if c_proto:
        if df[c_proto].dtype == object:
            proto_map = {"TCP": 6, "tcp": 6, "UDP": 17, "udp": 17, "ICMP": 1, "icmp": 1}
            proto = df[c_proto].map(proto_map)
            proto = proto.fillna(pd.to_numeric(df[c_proto], errors="coerce"))
            proto = proto.fillna(0).astype(int)
        else:
            proto = pd.to_numeric(df[c_proto], errors="coerce").fillna(0).astype(int)
    else:
        proto = zeros_i.copy()

    # pkt_count
    if c_pk:
        pkt_count = pd.to_numeric(df[c_pk], errors="coerce").fillna(0).astype(int)
    else:
        pk_f = (pd.to_numeric(df[c_fwdpk], errors="coerce").fillna(0).astype(int)) if c_fwdpk else zeros_i.copy()
        pk_b = (pd.to_numeric(df[c_bwdpk], errors="coerce").fillna(0).astype(int)) if c_bwdpk else zeros_i.copy()
        pkt_count = (pk_f + pk_b).astype(int)

    # bytes_total
    if c_by:
        bytes_total = pd.to_numeric(df[c_by], errors="coerce").fillna(0).astype(int)
    else:
        by_f = (pd.to_numeric(df[c_fwdby], errors="coerce").fillna(0).astype(int)) if c_fwdby else zeros_i.copy()
        by_b = (pd.to_numeric(df[c_bwdby], errors="coerce").fillna(0).astype(int)) if c_bwdby else zeros_i.copy()
        bytes_total = (by_f + by_b).astype(int)

    out = pd.DataFrame({
        "sport": sport,
        "dport": dport,
        "proto": proto,
        "pkt_count": pkt_count,
        "bytes_total": bytes_total
    }, columns=NUM_HEADER)

    # --- drop righe nulle ---
    mask_valid = (out[["sport", "dport", "proto", "pkt_count", "bytes_total"]] != 0).any(axis=1)
    out = out[mask_valid].reset_index(drop=True)

    # --- tieni solo TCP e porte > 0 ---
    out = out[(out["proto"] == 6) & (out["sport"] > 0) & (out["dport"] > 0)].reset_index(drop=True)

    return out

def concat_malicious_csvs(mal_root: Path, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csvs = list_files_recursive([mal_root], is_csv)
    if not all_csvs:
        echo(f"[ERRORE] nessun CSV maligno trovato in {mal_root}")
        sys.exit(2)

    train_csvs = [p for p in all_csvs if "train" in p.name.lower()]
    test_csvs  = [p for p in all_csvs if "test"  in p.name.lower()]
    if not train_csvs or not test_csvs:
        echo("[ERRORE] non ho trovato entrambe le partizioni train/test nei CSV maligni.")
        echo(f" - train trovati: {len(train_csvs)}")
        echo(f" - test  trovati: {len(test_csvs)}")
        sys.exit(2)

    def _load_and_project(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, low_memory=False)
        out = _project_malicious_simple(df)
        if out.empty:
            raise RuntimeError("tutte le righe proiettate risultano nulle dopo la proiezione")
        return out

    tr_frames, te_frames = [], []
    for p in train_csvs:
        try:
            tr_frames.append(_load_and_project(p))
        except Exception as e:
            echo(f"[WARN] skip {p.name}: {e}")
    for p in test_csvs:
        try:
            te_frames.append(_load_and_project(p))
        except Exception as e:
            echo(f"[WARN] skip {p.name}: {e}")

    df_tr = pd.concat(tr_frames, ignore_index=True, sort=False) if tr_frames else pd.DataFrame(columns=NUM_HEADER)
    df_te = pd.concat(te_frames, ignore_index=True, sort=False) if te_frames else pd.DataFrame(columns=NUM_HEADER)

    tr_out = out_dir / MAL_TRAIN_CAT
    te_out = out_dir / MAL_TEST_CAT
    df_tr.to_csv(tr_out, index=False)
    df_te.to_csv(te_out, index=False)
    echo(f"[MAL] train concat -> {tr_out} (righe={len(df_tr)})")
    echo(f"[MAL] test  concat -> {te_out} (righe={len(df_te)})")
    return tr_out, te_out

# -------------------- Merge + Preprocess --------------------
def save_feature_names(feat_cols: List[str], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"features": feat_cols}, f, indent=2)

def merge_and_prepare(benign_concat_csv: Path,
                      mal_train_concat_csv: Path,
                      mal_test_concat_csv: Path,
                      out_root: Path,
                      benign_val_size: float,
                      random_state: int = 42):
    out_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # carica
    df_b    = pd.read_csv(benign_concat_csv) if benign_concat_csv.exists() else pd.DataFrame(columns=NUM_HEADER)
    df_m_tr = pd.read_csv(mal_train_concat_csv) if mal_train_concat_csv.exists() else pd.DataFrame(columns=NUM_HEADER)
    df_m_te = pd.read_csv(mal_test_concat_csv)  if mal_test_concat_csv.exists()  else pd.DataFrame(columns=NUM_HEADER)

    # etichette
    df_m_tr["label"] = 1
    df_m_te["label"] = 1

    # split benigni in train/val (val = test/eval)
    if len(df_b):
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(df_b))
        rng.shuffle(idx)
        n_val = int(np.floor(len(idx) * benign_val_size))
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]
        df_b_tr = df_b.iloc[tr_idx].copy()
        df_b_te = df_b.iloc[val_idx].copy()
    else:
        df_b_tr = pd.DataFrame(columns=NUM_HEADER)
        df_b_te = pd.DataFrame(columns=NUM_HEADER)
    df_b_tr["label"] = 0
    df_b_te["label"] = 0

    # merge preliminare
    df_tr = pd.concat([df_b_tr, df_m_tr], ignore_index=True)
    df_te = pd.concat([df_b_te, df_m_te], ignore_index=True)

    # --- deduplica PRIMA dell'oversampling (per ridurre overfit da repliche)
    dedup_key = ["sport", "dport", "proto", "pkt_count", "bytes_total"]
    df_tr = df_tr.drop_duplicates(subset=dedup_key + ["label"]).reset_index(drop=True)
    df_te = df_te.drop_duplicates(subset=dedup_key + ["label"]).reset_index(drop=True)

    # --- opzionale: bilancia la VAL 1:1 se possibile
    if "label" in df_te.columns:
        n0 = (df_te["label"] == 0).sum()
        n1 = (df_te["label"] == 1).sum()
        if n0 > 0 and n1 > 0:
            n_keep = min(n0, n1)
            df_te = pd.concat([
                df_te[df_te["label"] == 0].sample(n=n_keep, random_state=random_state),
                df_te[df_te["label"] == 1].sample(n=n_keep, random_state=random_state),
            ], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # oversampling SOLO sul train
    label_col = "label"
    feat_cols = [c for c in df_tr.columns if c != label_col]

    cnts_before = df_tr[label_col].value_counts().sort_index()
    echo(f"[INFO] class counts (train, before): {cnts_before.to_dict()}")
    if len(cnts_before) == 2:
        max_n = cnts_before.max()
        rng = np.random.default_rng(random_state)
        parts = []
        for cls, n in cnts_before.items():
            df_c = df_tr[df_tr[label_col] == cls]
            if n < max_n and len(df_c) > 0:
                need = max_n - n
                idx_add = rng.choice(df_c.index.values, size=need, replace=True)
                df_c_aug = pd.concat([df_c, df_tr.loc[idx_add]], ignore_index=False)
                parts.append(df_c_aug)
            else:
                parts.append(df_c)
        df_tr = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    cnts_after = df_tr[label_col].value_counts().sort_index()
    echo(f"[INFO] class counts (train, after):  {cnts_after.to_dict()}")

    # salva raw
    train_raw = out_root / TRAIN_RAW
    val_raw   = out_root / VAL_RAW
    df_tr.to_csv(train_raw, index=False)
    df_te.to_csv(val_raw, index=False)
    echo(f"[OK] RAW salvati: {train_raw} (rows={len(df_tr)}), {val_raw} (rows={len(df_te)})")

    # scaler
    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[label_col].to_numpy(dtype=np.int32)
    X_te = df_te[feat_cols].to_numpy(dtype=np.float32)
    y_te = df_te[label_col].to_numpy(dtype=np.int32)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    with open(artifacts_dir / SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    save_feature_names(feat_cols, artifacts_dir / FEATURES_JSON)
    echo(f"[OK] Artifacts: {artifacts_dir / SCALER_PKL}, {artifacts_dir / FEATURES_JSON}")
    echo(f"[INFO] Shapes: Xtr={X_tr_s.shape} Xte={X_te_s.shape}")

    # mini report
    with open(out_root / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_rows": int(len(df_tr)),
            "val_rows": int(len(df_te)),
            "train_class_counts": cnts_after.to_dict(),
            "val_class_counts": df_te[label_col].value_counts().sort_index().to_dict(),
            "features": feat_cols
        }, f, indent=2)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Balanced CSV (maligni) + Benign PCAP -> train/val con oversampling (schema semplice, finestra temporale)")
    ap.add_argument("--benign-pcaps", type=str, required=True,
                    help="Cartella PCAP benigni (scansione ricorsiva)")
    ap.add_argument("--malicious-balanced-root", type=str, required=True,
                    help="Cartella root dei CSV maligni bilanciati (scansione ricorsiva; separazione train/test per nome)")
    ap.add_argument("--tmp-out", type=str, required=True,
                    help="Cartella per output intermedi (CSV concatenati, benigni convertiti)")
    ap.add_argument("--out-root", type=str, required=True,
                    help="Cartella output finale (train/val + artifacts)")
    ap.add_argument("--bpf", type=str, default="tcp.port==2404",
                    help="Display filter Wireshark da passare a -Y (es. 'tcp.port==2404')")
    ap.add_argument("--window", type=int, default=60,
                    help="Larghezza finestra (secondi) per l'aggregazione dei benigni")
    ap.add_argument("--benign-val-size", type=float, default=0.2,
                    help="Quota dei benigni da inviare a VAL/TEST (0..1) prima dell'eventuale bilanciamento 1:1")
    ap.add_argument("--clean-out", action="store_true",
                    help="Svuota tmp-out e out-root prima di scrivere")
    args = ap.parse_args()

    benign_root = Path(args.benign_pcaps).expanduser().resolve()
    mal_root    = Path(args.malicious_balanced_root).expanduser().resolve()
    tmp_out     = Path(args.tmp_out).expanduser().resolve()
    out_root    = Path(args.out_root).expanduser().resolve()

    if args.clean_out:
        for d in (tmp_out, out_root):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    (tmp_out / "benign_csv").mkdir(parents=True, exist_ok=True)
    (tmp_out / "mal_csv").mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) MALIGNI: concat train/test dai CSV bilanciati (proiezione semplice + filtri TCP)
    mal_tr_cat, mal_te_cat = concat_malicious_csvs(mal_root, tmp_out / "mal_csv")

    # 2) BENIGNI: PCAP -> CSV numerici + concat (schema a finestra)
    check_tshark()
    benign_pcaps = list_files_recursive([benign_root], is_pcap)
    echo(f"[INFO] trovati {len(benign_pcaps)} PCAP benigni")
    ben_cat = convert_benign_pcaps(benign_pcaps, tmp_out / "benign_csv", args.bpf, window_sec=args.window)

    # 3) MERGE + dedup + bilanciamento val + oversampling + scaler
    merge_and_prepare(ben_cat, mal_tr_cat, mal_te_cat, out_root, benign_val_size=args.benign_val_size)

    echo("[DONE] Pipeline completata.")

if __name__ == "__main__":
    main()

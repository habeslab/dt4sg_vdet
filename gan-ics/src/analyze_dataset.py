"""
EDA per dataset GAN-ICS (schema semplice)
- Legge da OUT_ROOT: gan_train_raw.csv, gan_val_raw.csv, artifacts/features.json
- Controlli qualità: NaN, duplicati, range, outlier (IQR), class balance
- Confronto train/val: PSI per feature, KS approx (binning), media/std diff
- Grafici: istogrammi per feature (overall e per classe), boxplot per classe, heatmap correlazioni
- Output: OUT_ROOT/eda/ con PNG, CSV, report.md, eda_report.json

Requisiti:
  - python3
  - pandas, numpy, matplotlib
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- util ----------
def echo(s=""): print(s, flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_features(features_json: Path) -> List[str]:
    with open(features_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    feats = j.get("features", [])
    # fallback: in caso di file vuoto o assente, prova a dedurre
    return list(feats)

def psi(baseline: np.ndarray, target: np.ndarray, bins: int = 20) -> float:
    """
    Population Stability Index tra due distribuzioni 1D.
    - Binning uniforme su baseline (min/max).
    - Smoothing con epsilon per evitare log(0).
    """
    if baseline.size == 0 or target.size == 0:
        return np.nan
    bmin, bmax = np.min(baseline), np.max(baseline)
    if bmin == bmax:
        # degenerate
        return 0.0
    edges = np.linspace(bmin, bmax, bins + 1)
    b_hist, _ = np.histogram(baseline, bins=edges)
    t_hist, _ = np.histogram(target,  bins=edges)
    b_prob = b_hist / max(1, np.sum(b_hist))
    t_prob = t_hist / max(1, np.sum(t_hist))
    eps = 1e-8
    b_prob = np.clip(b_prob, eps, 1.0)
    t_prob = np.clip(t_prob, eps, 1.0)
    return float(np.sum((t_prob - b_prob) * np.log(t_prob / b_prob)))

def iqr_outlier_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (x < lo) | (x > hi)
    return float(np.mean(mask))

def ks_approx_pvalue_via_bins(a: np.ndarray, b: np.ndarray, bins: int = 50) -> Tuple[float, float]:
    """
    Approssimazione 'poor man' del KS:
    - Discretizza su bins comuni, calcola CDF cumulate e D=max|F1-F2|.
    - Stima p-value con formula asintotica (senza SciPy).
    Nota: è solo indicativa per spotting di grossi shift.
    """
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if lo == hi:
        return 0.0, 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    ca = np.cumsum(ha) / max(1, np.sum(ha))
    cb = np.cumsum(hb) / max(1, np.sum(hb))
    D = float(np.max(np.abs(ca - cb)))
    na, nb = len(a), len(b)
    n_eff = (na * nb) / (na + nb + 1e-9)
    t = 2.0 * (np.exp(-2 * (D**2) * n_eff) - np.exp(-8 * (D**2) * n_eff) + np.exp(-18 * (D**2) * n_eff))
    p = max(0.0, min(1.0, float(t)))
    return D, p

def save_dataframe(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

# ---------- plotting helpers ----------
def plot_histograms(df: pd.DataFrame, col: str, out_dir: Path, by_label: bool = True):
    # overall
    plt.figure()
    plt.hist(df[col].dropna().values, bins=50)
    plt.title(f"Histogram - {col} (overall)")
    plt.xlabel(col); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_{col}_overall.png", dpi=120)
    plt.close()

    if by_label and "label" in df.columns:
        for lab in sorted(df["label"].dropna().unique()):
            sub = df[df["label"] == lab][col].dropna().values
            if sub.size == 0: 
                continue
            plt.figure()
            plt.hist(sub, bins=50)
            plt.title(f"Histogram - {col} (label={lab})")
            plt.xlabel(col); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / f"hist_{col}_label{lab}.png", dpi=120)
            plt.close()

def plot_box_by_label(df: pd.DataFrame, col: str, out_dir: Path):
    if "label" not in df.columns: 
        return
    groups = []
    labels = []
    for lab in sorted(df["label"].dropna().unique()):
        arr = df[df["label"] == lab][col].dropna().values
        if arr.size:
            groups.append(arr)
            labels.append(str(lab))
    if len(groups) >= 2:
        plt.figure()
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title(f"Boxplot - {col} by label")
        plt.xlabel("label"); plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out_dir / f"box_{col}_by_label.png", dpi=120)
        plt.close()

def plot_corr_heatmap(df_num: pd.DataFrame, out_dir: Path, fname: str = "corr_heatmap.png"):
    if df_num.shape[1] < 2:
        return
    corr = df_num.corr(numeric_only=True)
    plt.figure(figsize=(max(6, 0.6 * corr.shape[1]), max(5, 0.6 * corr.shape[1])))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title("Correlation heatmap (train)")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=130)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="EDA su dataset GAN-ICS (schema semplice)")
    ap.add_argument("--out-root", type=str, required=True, help="Cartella OUT della pipeline (contiene gan_train_raw.csv, gan_val_raw.csv, artifacts/)")
    ap.add_argument("--max-plots", type=int, default=1000, help="Limite massimo grafici per non esplodere con molte feature")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    train_csv = out_root / "gan_train_raw.csv"
    val_csv   = out_root / "gan_val_raw.csv"
    feats_j   = out_root / "artifacts" / "features.json"

    if not train_csv.exists() or not val_csv.exists():
        echo("[ERRORE] Non trovo gan_train_raw.csv / gan_val_raw.csv in --out-root")
        return 2

    features = load_features(feats_j) if feats_j.exists() else None

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    eda_dir = out_root / "eda"
    ensure_dir(eda_dir)
    ensure_dir(eda_dir / "plots")

    # ----- colonne, label e features -----
    if "label" not in df_tr.columns or "label" not in df_va.columns:
        echo("[ERRORE] Manca la colonna 'label' nei CSV.")
        return 3

    if not features:
        # deduci tutto tranne label
        features = [c for c in df_tr.columns if c != "label"]

    # ----- class balance -----
    cls_tr = df_tr["label"].value_counts().sort_index()
    cls_va = df_va["label"].value_counts().sort_index()
    echo(f"[INFO] class balance train: {cls_tr.to_dict()}")
    echo(f"[INFO] class balance val:   {cls_va.to_dict()}")

    # ----- qualità dati -----
    # NaN
    nan_tr = df_tr[features].isna().sum()
    nan_va = df_va[features].isna().sum()
    # Duplicati
    dup_tr = int(df_tr.duplicated().sum())
    dup_va = int(df_va.duplicated().sum())

    # Stats base
    desc_tr = df_tr[features].describe(include="all").T
    desc_tr["missing"] = nan_tr
    desc_tr.reset_index().rename(columns={"index": "feature"})
    save_dataframe(desc_tr.reset_index().rename(columns={"index": "feature"}), eda_dir / "train_describe.csv")

    desc_va = df_va[features].describe(include="all").T
    desc_va["missing"] = nan_va
    save_dataframe(desc_va.reset_index().rename(columns={"index": "feature"}), eda_dir / "val_describe.csv")

    # Outlier fraction (IQR) per feature su train
    outlier_info = []
    for col in features:
        frac = iqr_outlier_fraction(df_tr[col].dropna().values.astype(float))
        outlier_info.append({"feature": col, "iqr_outlier_frac_train": frac})
    df_out = pd.DataFrame(outlier_info)
    save_dataframe(df_out, eda_dir / "train_outlier_iqr.csv")

    # ----- confronto train vs val -----
    shift_rows = []
    for col in features:
        a = df_tr[col].dropna().values.astype(float)
        b = df_va[col].dropna().values.astype(float)
        if a.size == 0 or b.size == 0:
            shift_rows.append({"feature": col, "psi": np.nan, "ks_D": np.nan, "ks_p": np.nan,
                               "mean_tr": np.nan, "mean_va": np.nan, "std_tr": np.nan, "std_va": np.nan})
            continue
        s_psi = psi(a, b, bins=20)
        ksD, ksP = ks_approx_pvalue_via_bins(a, b, bins=50)
        shift_rows.append({
            "feature": col,
            "psi": s_psi,
            "ks_D": ksD,
            "ks_p": ksP,
            "mean_tr": float(np.mean(a)),
            "mean_va": float(np.mean(b)),
            "std_tr": float(np.std(a)),
            "std_va": float(np.std(b)),
        })
    df_shift = pd.DataFrame(shift_rows).sort_values("psi", ascending=False)
    save_dataframe(df_shift, eda_dir / "train_val_shift.csv")

    # ----- correlazioni su train -----
    df_num_tr = df_tr[features].select_dtypes(include=[np.number])
    if not df_num_tr.empty:
        plot_corr_heatmap(df_num_tr, eda_dir / "plots", fname="corr_heatmap_train.png")

    # ----- grafici per feature -----
    plotted = 0
    for col in features:
        try:
            plot_histograms(df_tr, col, eda_dir / "plots", by_label=True)
            plot_box_by_label(df_tr, col, eda_dir / "plots")
            plotted += 3
            if plotted >= args.max_plots:
                break
        except Exception:
            pass

    # ----- report sintetico -----
    report = {
        "rows_train": int(len(df_tr)),
        "rows_val": int(len(df_va)),
        "class_balance": {
            "train": {int(k): int(v) for k, v in cls_tr.to_dict().items()},
            "val":   {int(k): int(v) for k, v in cls_va.to_dict().items()},
        },
        "duplicates": {"train": dup_tr, "val": dup_va},
        "nan_counts_train": nan_tr.to_dict(),
        "nan_counts_val": nan_va.to_dict(),
        "top_psi_features": df_shift[["feature", "psi"]].head(10).to_dict(orient="records"),
        "iqr_outlier_top": df_out.sort_values("iqr_outlier_frac_train", ascending=False).head(10).to_dict(orient="records"),
        "artifacts": {
            "train_describe_csv": str((eda_dir / "train_describe.csv").name),
            "val_describe_csv": str((eda_dir / "val_describe.csv").name),
            "train_outlier_iqr_csv": str((eda_dir / "train_outlier_iqr.csv").name),
            "train_val_shift_csv": str((eda_dir / "train_val_shift.csv").name),
            "plots_dir": "eda/plots/",
        }
    }
    with open(eda_dir / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown rapido
    md = []
    md.append("# EDA Report\n")
    md.append(f"- **Train rows**: {report['rows_train']}")
    md.append(f"- **Val rows**: {report['rows_val']}")
    md.append(f"- **Class balance (train)**: {report['class_balance']['train']}")
    md.append(f"- **Class balance (val)**: {report['class_balance']['val']}")
    md.append(f"- **Duplicati**: train={dup_tr}, val={dup_va}")
    md.append("\n## Top PSI (train vs val)\n")
    for r in report["top_psi_features"]:
        md.append(f"- {r['feature']}: PSI={r['psi']:.4f}")
    md.append("\n## Outlier (IQR) - Top\n")
    for r in report["iqr_outlier_top"]:
        md.append(f"- {r['feature']}: frac={r['iqr_outlier_frac_train']:.4f}")
    md.append("\n## File generati\n")
    for k, v in report["artifacts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("\n## Correlazioni\n")
    md.append("![corr](plots/corr_heatmap_train.png)\n")
    with open(eda_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    echo(f"[OK] EDA completata → {eda_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

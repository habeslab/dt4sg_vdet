#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- CPU first: se l'utente ha passato --cpu, spegniamo la GPU PRIMA di importare TF ---
CPU_MODE = ("--cpu" in sys.argv)
if CPU_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silenzioso ma non invasivo

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import robust_load_model, logits_to_proba  # softmax robusta

# --- loader tollerante al MinibatchStdDev (come nei tuoi script) ---
def _load_disc_tolerant(path: str) -> tf.keras.Model:
    try:
        return robust_load_model(path, custom_objects=None, compile=False)
    except Exception as e:
        msg = str(e)
        if "MinibatchStdDev" in msg or "mbstd" in msg:
            class MinibatchStdDev(tf.keras.layers.Layer):
                def __init__(self, epsilon: float = 1e-8, **kwargs):
                    super().__init__(**kwargs); self.epsilon = float(epsilon)
                def call(self, x):
                    mean = tf.reduce_mean(x, axis=0, keepdims=True)
                    var  = tf.reduce_mean(tf.square(x - mean), axis=0, keepdims=True)
                    std  = tf.sqrt(var + self.epsilon)
                    mean_std = tf.reduce_mean(std)
                    bsz = tf.shape(x)[0]
                    return tf.concat([x, tf.fill([bsz, 1], mean_std)], axis=1)
                def get_config(self):
                    return {**super().get_config(), "epsilon": self.epsilon}
            return robust_load_model(path, custom_objects={"MinibatchStdDev": MinibatchStdDev}, compile=False)
        raise

# --- plotting ---
def _cm_and_hists_real(y_true_real: np.ndarray, prob_real: np.ndarray, out_path: Path):
    # Rinormalizza sulle colonne {0,1} e soglia a 0.5
    p0, p1 = prob_real[:,0], prob_real[:,1]
    s = p0 + p1 + 1e-9
    p1_2c = p1 / s
    pred2 = (p1_2c >= 0.5).astype(np.int32)

    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(y_true_real, pred2):
        cm[t, p] += 1

    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1)
    im = ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i,j], ha="center", va="center", color="black")
    ax1.set_title("Confusion Matrix (0/1) sui SOLI reali")
    ax1.set_xlabel("Pred"); ax1.set_ylabel("True")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2,2,2)
    ax2.hist(prob_real[:,2], bins=50)
    ax2.set_title("Histogram p(fake) sui reali"); ax2.set_xlabel("p(fake)"); ax2.set_ylabel("count"); ax2.grid(True)

    ax3 = plt.subplot(2,2,3)
    ax3.hist(p1_2c, bins=50)
    ax3.set_title("Histogram p(1 | {0,1}) sui reali"); ax3.set_xlabel("p1_2c"); ax3.set_ylabel("count"); ax3.grid(True)

    ax4 = plt.subplot(2,2,4)
    counts = np.bincount(pred2, minlength=2)
    ax4.bar([0,1], counts)
    ax4.set_title("Pred 0/1 (threshold 0.5) sui reali"); ax4.set_xticks([0,1]); ax4.grid(True, axis="y")

    fig.suptitle("Discriminator evaluation — REAL ONLY", fontsize=12)
    fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig(out_path, dpi=140); plt.close(fig)

def _plots_gen(prob: np.ndarray, out_path: Path):
    pred = np.argmax(prob, axis=-1)
    counts = np.bincount(pred, minlength=3)
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,2,1)
    ax1.bar([0,1,2], counts)
    ax1.set_title("Pred class counts su sintetici"); ax1.set_xticks([0,1,2]); ax1.grid(True, axis="y")
    ax2 = plt.subplot(1,2,2)
    ax2.hist(prob[:,2], bins=50)
    ax2.set_title("Histogram p(fake) su sintetici"); ax2.set_xlabel("p(fake)"); ax2.set_ylabel("count"); ax2.grid(True)
    fig.suptitle("Generator evaluation via Discriminator", fontsize=12)
    fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig(out_path, dpi=140); plt.close(fig)

# --- core ---
def evaluate_discriminator(model_path: Path, data_root: Path, csv_path: Path|None):
    ART = data_root / "artifacts"
    feats = json.loads((ART / "features.json").read_text(encoding="utf-8"))["features"]
    import pickle
    with open(ART / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    if csv_path is None:
        for cand in (data_root/"gan_val_raw.csv", data_root/"gan_train_raw.csv"):
            if cand.exists(): csv_path = cand; break
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError("CSV non trovato per la valutazione del discriminatore.")

    df = pd.read_csv(csv_path)
    X  = df[feats].to_numpy(np.float32)
    y  = df["label"].to_numpy(np.int32)
    Xs = scaler.transform(X).astype(np.float32)

    D  = _load_disc_tolerant(str(model_path))
    logits = D.predict(Xs, verbose=0)
    prob   = logits_to_proba(logits)

    pred = np.argmax(prob, axis=-1)
    mask_real = pred != 2  # soli "reali"
    acc_real_only = float(np.mean(pred[mask_real] == y[mask_real])) if np.any(mask_real) else 0.0
    rfp = float(np.mean(pred == 2))

    # --- metriche per-classe sui soli reali ---
    if np.any(mask_real):
        y_real = y[mask_real]
        prob_real = prob[mask_real]
        p0, p1 = prob_real[:,0], prob_real[:,1]
        s = p0 + p1 + 1e-9
        p1_2c = p1 / s
        pred2 = (p1_2c >= 0.5).astype(np.int32)
        cm = np.zeros((2,2), dtype=int)
        for t,pv in zip(y_real, pred2):
            cm[t,pv] += 1
        TN, FP, FN, TP = map(int, [cm[0,0], cm[0,1], cm[1,0], cm[1,1]])
        prec1 = TP / (TP + FP + 1e-9)
        rec1  = TP / (TP + FN + 1e-9)
        fpr0  = FP / (FP + TN + 1e-9)
    else:
        TN=FP=FN=TP=0
        prec1=rec1=fpr0=0.0

    # --- sweep soglia su p(1|{0,1}) mantenendo la stessa policy di astensione ---
    targets = [0.50, 0.60, 0.70, 0.80, 0.90]
    sweep = []
    if np.any(mask_real):
        p0_all, p1_all = prob[:,0], prob[:,1]
        s_all = p0_all + p1_all + 1e-9
        p1_2c_all = p1_all / s_all
        for th in targets:
            pred2_th = (p1_2c_all >= th).astype(np.int32)
            y_r = y[mask_real]; pr = pred2_th[mask_real]
            cm_th = np.zeros((2,2), dtype=int)
            for t,pv in zip(y_r, pr): cm_th[t,pv]+=1
            TNt, FPt, FNt, TPt = cm_th[0,0], cm_th[0,1], cm_th[1,0], cm_th[1,1]
            fpr_t = FPt / (FPt + TNt + 1e-9)
            acc_t = (TNt + TPt) / max(len(y_r),1)
            sweep.append({"th": float(th), "acc_real_only": float(acc_t), "fpr_benigno": float(fpr_t)})

    # --- plot sui reali ---
    plots_dir = ART / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    if np.any(mask_real):
        _cm_and_hists_real(y_true_real=y[mask_real], prob_real=prob[mask_real], out_path=plots_dir / "disc_eval_all.png")

    ret = {
        "csv": str(csv_path.name),
        "acc_real_only": float(acc_real_only),
        "rFP": float(rfp),
        "n_samples": int(len(df)),
        "cm_real_only": {"TN": TN, "FP": FP, "FN": FN, "TP": TP},
        "precision_maligno": float(prec1),
        "recall_maligno": float(rec1),
        "fpr_benigno": float(fpr0),
        "threshold_sweep": sweep
    }
    return ret

def evaluate_generator(gen_path: Path, disc_path: Path, data_root: Path, csv_for_N: Path|None, latent_dim: int):
    ART = data_root / "artifacts"
    feats = json.loads((ART / "features.json").read_text(encoding="utf-8"))["features"]

    if csv_for_N is None:
        for cand in (data_root/"gan_val_raw.csv", data_root/"gan_train_raw.csv"):
            if cand.exists(): csv_for_N = cand; break
    if csv_for_N is None or not csv_for_N.exists():
        raise FileNotFoundError("CSV non trovato per determinare N sintetici da generare.")

    n = len(pd.read_csv(csv_for_N))
    # Carica modelli
    G = robust_load_model(str(gen_path), compile=False)
    D = _load_disc_tolerant(str(disc_path))

    # check dimensioni
    out_dim = int(G.output_shape[-1])
    if out_dim != len(feats):
        raise ValueError(f"Mismatch dimensioni: G.output={out_dim}, features.json={len(feats)}")

    # genera e valuta
    bsz, remaining, syn = 1024, n, []
    while remaining > 0:
        cur = min(bsz, remaining)
        z = tf.random.normal([cur, latent_dim])
        syn.append(G.predict(z, verbose=0))
        remaining -= cur
    Xg = np.concatenate(syn, axis=0)

    logits = D.predict(Xg, verbose=0)
    prob   = logits_to_proba(logits)
    pred   = np.argmax(prob, axis=-1)
    fr     = float(np.mean(pred != 2))
    counts = np.bincount(pred, minlength=3).tolist()

    plots_dir = ART / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    _plots_gen(prob, plots_dir / "gen_eval_all.png")

    return {
        "n_synth": int(len(Xg)),
        "fooling_rate": fr,
        "pred_counts": {"benigno": counts[0], "maligno": counts[1], "fake": counts[2]},
    }

def main():
    ap = argparse.ArgumentParser(description="Valutazione unificata di Discriminatore e Generatore.")
    ap.add_argument("--data-root", required=True, help="Cartella OUT della pipeline (gan_*_raw + artifacts/)")
    ap.add_argument("--disc", required=True, help="Path .keras del discriminatore")
    ap.add_argument("--gen", required=False, help="Path .keras del generatore (opzionale)")
    ap.add_argument("--csv", required=False, help="CSV RAW da usare (opzionale). Se omesso -> gan_val_raw.csv o train.")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    DATA_ROOT = Path(args.data_root).expanduser().resolve()
    ART = DATA_ROOT / "artifacts"
    if not (ART / "features.json").exists() or not (ART / "scaler.pkl").exists():
        raise FileNotFoundError("features.json/scaler.pkl mancanti in artifacts/ della pipeline.")

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None

    report = {"discriminator": None, "generator": None}

    # --- D su reali ---
    report["discriminator"] = evaluate_discriminator(
        model_path=Path(args.disc).expanduser().resolve(),
        data_root=DATA_ROOT,
        csv_path=csv_path
    )

    # --- G vs D (se fornito) ---
    if args.gen:
        report["generator"] = evaluate_generator(
            gen_path=Path(args.gen).expanduser().resolve(),
            disc_path=Path(args.disc).expanduser().resolve(),
            data_root=DATA_ROOT,
            csv_for_N=csv_path,
            latent_dim=int(args.latent_dim)
        )

    # salva JSON
    reports_dir = ART / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "eval_all.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[OK] Report salvato ->", out_json)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

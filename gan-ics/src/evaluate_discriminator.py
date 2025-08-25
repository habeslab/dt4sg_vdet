# evaluate_discriminator.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import project_dirs, robust_load_model, logits_to_proba, autodetect_csv

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

def _save_eval_plots(y_true: np.ndarray, prob: np.ndarray, plots_dir: Path, tag: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    # 2-class normalization (ignora fake)
    p0, p1 = prob[:,0], prob[:,1]
    s = p0 + p1 + 1e-9
    p1_2c = p1 / s

    # Confusion matrix (pred tra 0/1)
    pred2 = (p1_2c >= 0.5).astype(np.int32)
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(y_true, pred2):
        cm[t, p] += 1

    # Figure 2x2: CM, hist p(fake), hist p1_2c, barre counts pred2
    fig = plt.figure(figsize=(12,8))

    ax1 = plt.subplot(2,2,1)
    im = ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i,j], ha="center", va="center", color="black")
    ax1.set_title("Confusion Matrix (0/1)")
    ax1.set_xlabel("Pred"); ax1.set_ylabel("True")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2,2,2)
    ax2.hist(prob[:,2], bins=50)
    ax2.set_title("Histogram p(fake) su REAL")
    ax2.set_xlabel("p(fake)"); ax2.set_ylabel("count"); ax2.grid(True)

    ax3 = plt.subplot(2,2,3)
    ax3.hist(p1_2c, bins=50)
    ax3.set_title("Histogram p(1 | {0,1}) su REAL")
    ax3.set_xlabel("p1_2c"); ax3.set_ylabel("count"); ax3.grid(True)

    ax4 = plt.subplot(2,2,4)
    counts = np.bincount(pred2, minlength=2)
    ax4.bar([0,1], counts)
    ax4.set_title("Pred 0/1 (threshold 0.5)"); ax4.set_xticks([0,1]); ax4.grid(True, axis="y")

    fig.suptitle("Discriminator evaluation (real CSV)", fontsize=12)
    out = plots_dir / f"disc_eval_{tag}.png"
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[PLOT] Salvato {out.name} -> {plots_dir}")

def main():
    ap = argparse.ArgumentParser(description="Valuta il discriminatore su CSV RAW.")
    ap.add_argument("--model", required=True, help="Path .keras del discriminatore")
    ap.add_argument("--csv", required=False, help="CSV RAW (default: autodetect processed/gan_val_raw.csv->gan_train_raw.csv)")
    ap.add_argument("--cpu", action="store_true", help="Forza CPU")
    ap.add_argument("--out-tag", default=None, help="Tag per naming plot (default: timestamp)")
    args = ap.parse_args()

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    from datetime import datetime
    tag = args.out_tag or datetime.now().strftime("%Y%m%d-%H%M%S")

    root, art, proc = project_dirs()
    plots_dir = art / "plots"

    csv_path = Path(args.csv) if args.csv else autodetect_csv(proc)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError("CSV non trovato (passa --csv oppure lascia l'autodetect funzionare).")

    # Carica features e scaler
    feat_names = art / "features.json"
    if not feat_names.exists():
        raise FileNotFoundError("features.json mancante in artifacts.")
    import json, pickle
    feats = json.loads(feat_names.read_text(encoding="utf-8"))["features"]
    with open(art / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Carica dati e normalizza
    df = pd.read_csv(csv_path)
    X = df[feats].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)
    Xs = scaler.transform(X).astype(np.float32)

    # Carica modello (tollerante ad mbstd)
    D = _load_disc_tolerant(args.model)

    # Predici e metriche
    logits = D.predict(Xs, verbose=0)
    prob = logits_to_proba(logits)
    pred = np.argmax(prob, axis=-1)

    mask_real = pred != 2
    acc = float(np.mean((pred[mask_real] == y[mask_real]))) if np.any(mask_real) else 0.0
    rfp = float(np.mean(pred == 2))

    print(f"[OK] Valutazione D su {csv_path.name}: acc(real-only)={acc:.4f}  rFP={rfp:.4f}")

    # Plot valutazione
    _save_eval_plots(y_true=y, prob=prob, plots_dir=plots_dir, tag=tag)

if __name__ == "__main__":
    main()

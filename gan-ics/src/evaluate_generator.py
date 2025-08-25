# evaluate_generator.py
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

from utils import project_dirs, robust_load_model, autodetect_csv, logits_to_proba

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

def _save_gen_eval_plots(prob: np.ndarray, plots_dir: Path, tag: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    pred = np.argmax(prob, axis=-1)
    counts = np.bincount(pred, minlength=3)

    fig = plt.figure(figsize=(12,4))
    # barre per classi 0/1/2
    ax1 = plt.subplot(1,2,1)
    ax1.bar([0,1,2], counts)
    ax1.set_title("Pred class counts su sintetici")
    ax1.set_xticks([0,1,2]); ax1.grid(True, axis="y")

    # istogramma p(fake)
    ax2 = plt.subplot(1,2,2)
    ax2.hist(prob[:,2], bins=50)
    ax2.set_title("Histogram p(fake) su sintetici")
    ax2.set_xlabel("p(fake)"); ax2.set_ylabel("count"); ax2.grid(True)

    fig.suptitle("Generator evaluation via Discriminator", fontsize=12)
    out = plots_dir / f"gen_eval_{tag}.png"
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[PLOT] Salvato {out.name} -> {plots_dir}")

def main():
    import os, json, pickle
    ap = argparse.ArgumentParser(description="Valuta il generatore usando il discriminatore.")
    ap.add_argument("--gen", required=True, help="Path .keras del generatore")
    ap.add_argument("--disc", required=True, help="Path .keras del discriminatore")
    ap.add_argument("--csv", required=False, help="CSV RAW reale (default: autodetect processed/gan_val_raw.csv->gan_train_raw.csv)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "64")))
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
    feats = json.loads(feat_names.read_text(encoding="utf-8"))["features"]
    with open(art / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Dati reali (solo per definire dimensione; non usiamo y)
    df = pd.read_csv(csv_path)
    X = df[feats].to_numpy(np.float32)
    Xs = scaler.transform(X).astype(np.float32)  # non usato, solo per coerenza dimensionale
    n = len(Xs)

    # Carica modelli
    G = robust_load_model(args.gen, compile=False)
    D = _load_disc_tolerant(args.disc)

    # Genera come nell'eval originale (a blocchi per memoria)
    bsz = 1024
    syn = []
    remaining = n
    while remaining > 0:
        cur = min(bsz, remaining)
        z = tf.random.normal([cur, args.latent_dim])
        syn.append(G.predict(z, verbose=0))
        remaining -= cur
    Xg = np.concatenate(syn, axis=0)

    # Valuta con D
    logits = D.predict(Xg, verbose=0)
    prob = logits_to_proba(logits)
    fr = float(np.mean(np.argmax(prob, axis=-1) != 2))
    print(f"[OK] Valutazione G vs D: fooling_rate={fr:.4f}  su {len(Xg)} campioni sintetici.")

    # plot
    _save_gen_eval_plots(prob, plots_dir, tag)

if __name__ == "__main__":
    main()

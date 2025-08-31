from __future__ import annotations
import os, time, json, pickle, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf

# plotting headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_feature_names
from generator import build_generator
from discriminator import build_discriminator

# ----------------- Parametri base (env override) -----------------
EPOCHS      = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "128"))
LATENT_DIM  = int(os.getenv("LATENT_DIM", "64"))
LR_G        = float(os.getenv("LR_G", "2e-4"))
LR_D        = float(os.getenv("LR_D", "2e-4"))

LABEL_SMOOTH = float(os.getenv("LABEL_SMOOTH", "0.0"))
NOISE_STD    = float(os.getenv("NOISE_STD", "0.0"))
CLIP_NORM    = float(os.getenv("CLIP_NORM", "0.0"))

USE_MBSTD    = os.getenv("USE_MBSTD", "1") not in ("0", "false", "False")
HIDDEN_G     = tuple(int(x) for x in os.getenv("HIDDEN_G", "256,256,128").split(",") if x)
HIDDEN_D     = tuple(int(x) for x in os.getenv("HIDDEN_D", "256,128,64").split(",") if x)

G_STEPS = int(os.getenv("G_STEPS", "1"))
D_STEPS = int(os.getenv("D_STEPS", "1"))
FM_W    = float(os.getenv("FM_W", "0.0"))

LOG_CKPT   = os.getenv("LOG_CKPT", "0") not in ("0", "false", "False")  # ← FIX: tolta parentesi in più
PLOT_LIVE  = os.getenv("PLOT_LIVE", "0") not in ("0", "false", "False")
PLOT_EVERY = int(os.getenv("PLOT_EVERY", "0"))  # 0=off

AUTOTUNE = tf.data.AUTOTUNE

def one_hot_3(y):
    return tf.one_hot(tf.cast(y, tf.int32), depth=3, dtype=tf.float32)

def make_feat_extractor(D: tf.keras.Model) -> tf.keras.Model:
    """
    Ritorna un estrattore di feature robusto:
    1) prova l'ultimo layer 'dense_<k>' se esiste
    2) altrimenti usa il penultimo layer
    3) fallback: usa direttamente l'output (non cambia la logica perché FM_W=0 di default)
    """
    try:
        last_idx = -1
        for l in D.layers:
            if l.name.startswith("dense_"):
                try:
                    idx = int(l.name.split("_")[1])
                    last_idx = max(last_idx, idx)
                except Exception:
                    pass
        if last_idx >= 0:
            feat_layer = D.get_layer(f"dense_{last_idx}")
            return tf.keras.Model(D.input, feat_layer.output, name="disc_feat_extractor")
    except Exception:
        pass
    # penultimo layer
    try:
        if len(D.layers) >= 2:
            return tf.keras.Model(D.input, D.layers[-2].output, name="disc_feat_extractor")
    except Exception:
        pass
    # fallback sicuro
    return tf.keras.Model(D.input, D.output, name="disc_feat_extractor")

@tf.function
def disc_step(D, G, disc_opt, real_x, real_y, latent_dim, label_smooth, noise_std):
    bsz = tf.shape(real_x)[0]
    if noise_std > 0:
        real_x = real_x + tf.random.normal(tf.shape(real_x), stddev=noise_std)
    with tf.GradientTape() as tape_d:
        logits_real = D(real_x, training=True)
        y3 = one_hot_3(real_y)
        if label_smooth > 0:
            y3 = y3 * (1.0 - label_smooth) + label_smooth / 3.0
        loss_sup = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y3, logits_real, from_logits=True)
        )
        z = tf.random.normal([bsz, latent_dim])
        fake_x = G(z, training=True)
        if noise_std > 0:
            fake_x = fake_x + tf.random.normal(tf.shape(fake_x), stddev=noise_std)
        logits_fake = D(fake_x, training=True)
        y_fake = tf.one_hot(tf.fill([bsz], 2), depth=3, dtype=tf.float32)
        loss_unsup = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_fake, logits_fake, from_logits=True)
        )
        d_loss = loss_sup + loss_unsup
        if D.losses:
            d_loss += tf.add_n(D.losses)
    d_grads = tape_d.gradient(d_loss, D.trainable_variables)
    disc_opt.apply_gradients(zip(d_grads, D.trainable_variables))
    pred_real = tf.argmax(logits_real, axis=-1, output_type=tf.int32)
    acc_real  = tf.reduce_mean(tf.cast(tf.equal(pred_real, tf.cast(real_y, tf.int32)), tf.float32))
    prob_real = tf.nn.softmax(logits_real, axis=-1)
    rfp_real  = tf.reduce_mean(tf.cast(prob_real[:, 2] > 0.5, tf.float32))
    return d_loss, acc_real, rfp_real

@tf.function
def gen_step(D, G, F, gen_opt, real_x, latent_dim, noise_std, fm_w):
    bsz = tf.shape(real_x)[0]
    with tf.GradientTape() as tape_g:
        z2 = tf.random.normal([bsz, latent_dim])
        fake_x2 = G(z2, training=True)
        if noise_std > 0:
            fake_x2 = fake_x2 + tf.random.normal(tf.shape(fake_x2), stddev=noise_std)
        logits_fake2 = D(fake_x2, training=False)
        logit_fake_col = logits_fake2[:, 2]
        g_loss = tf.reduce_mean(tf.nn.softplus(logit_fake_col))
        if fm_w > 0.0:
            real_feat = F(real_x, training=False)
            fake_feat = F(fake_x2, training=False)
            fm_loss = tf.reduce_mean(
                tf.square(tf.reduce_mean(real_feat, axis=0) - tf.reduce_mean(fake_feat, axis=0))
            )
            g_loss = g_loss + fm_w * fm_loss
        if G.losses:
            g_loss += tf.add_n(G.losses)
    g_grads = tape_g.gradient(g_loss, G.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, G.trainable_variables))
    p_fake2 = tf.nn.softmax(logits_fake2, axis=-1)
    fooled = tf.reduce_mean(tf.cast(tf.argmax(p_fake2, axis=-1) != 2, tf.float32))
    return g_loss, fooled

def make_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool=True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch, drop_remainder=False).prefetch(AUTOTUNE)
    return ds

def eval_disc_on(ds: tf.data.Dataset, D: tf.keras.Model) -> tuple[float, float]:
    accs, rfps = [], []
    for xb, yb in ds:
        logits = D(xb, training=False)
        prob   = tf.nn.softmax(logits, axis=-1)
        pred   = tf.argmax(prob, axis=-1, output_type=tf.int32)
        accs.append(tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(yb, tf.int32)), tf.float32)).numpy())
        rfps.append(tf.reduce_mean(tf.cast(prob[:, 2] > 0.5, tf.float32)).numpy())
    return float(np.mean(accs)) if accs else 0.0, float(np.mean(rfps)) if rfps else 0.0

def _save_training_plots(csv_log: Path, plots_dir: Path, tag: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not csv_log.exists():
        return
    df = pd.read_csv(csv_log)
    if df.empty:
        return
    epochs = df["epoch"].to_numpy()
    # Loss
    fig1 = plt.figure(figsize=(8,5))
    plt.plot(epochs, df["d_loss"], label="D_loss")
    plt.plot(epochs, df["g_loss"], label="G_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per epoca")
    plt.grid(True); plt.legend()
    out1 = plots_dir / f"loss_curves_{tag}.png"
    fig1.tight_layout(); fig1.savefig(out1, dpi=140); plt.close(fig1)
    # ACC & rFP
    fig2 = plt.figure(figsize=(8,5))
    plt.plot(epochs, df["acc_tr"], label="ACC(train)")
    plt.plot(epochs, df["acc_val"], label="ACC(val)")
    plt.plot(epochs, df["rfp_tr"], label="rFP(train)")
    plt.plot(epochs, df["rfp_val"], label="rFP(val)")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title("ACC & rFP")
    plt.grid(True); plt.legend()
    out2 = plots_dir / f"acc_rfp_{tag}.png"
    fig2.tight_layout(); fig2.savefig(out2, dpi=140); plt.close(fig2)
    # FR
    fig3 = plt.figure(figsize=(8,5))
    plt.plot(epochs, df["fr_tr"], label="FR(train)")
    plt.plot(epochs, df["fr_val"], label="FR(val)")
    plt.xlabel("Epoch"); plt.ylabel("Fooling Rate"); plt.title("Fooling rate")
    plt.grid(True); plt.legend()
    out3 = plots_dir / f"fooling_{tag}.png"
    fig3.tight_layout(); fig3.savefig(out3, dpi=140); plt.close(fig3)
    # Tempo
    fig4 = plt.figure(figsize=(8,5))
    plt.plot(epochs, df["secs"], label="seconds/epoch")
    plt.xlabel("Epoch"); plt.ylabel("Seconds"); plt.title("Tempo per epoca")
    plt.grid(True); plt.legend()
    out4 = plots_dir / f"time_{tag}.png"
    fig4.tight_layout(); fig4.savefig(out4, dpi=140); plt.close(fig4)

def main():
    ap = argparse.ArgumentParser(description="Train semi-supervised GAN su dati tabellari")
    ap.add_argument("--data-root", required=True, help="Cartella OUT della pipeline (contiene gan_train_raw.csv, gan_val_raw.csv, artifacts/)")
    ap.add_argument("--val-csv", default="gan_val_raw.csv", help="CSV di validazione relativo alla root (default: gan_val_raw.csv)")
    ap.add_argument("--cpu", action="store_true", help="Forza esecuzione su CPU")
    args = ap.parse_args()

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    DATA_ROOT = Path(args.data_root).expanduser().resolve()
    ART = DATA_ROOT / "artifacts"
    TR_RAW = DATA_ROOT / "gan_train_raw.csv"
    VA_RAW = (DATA_ROOT / "csv" / args.val_csv) if (DATA_ROOT / "csv" / args.val_csv).exists() else (DATA_ROOT / args.val_csv)

    # Check file pipeline
    for p in (ART / "features.json", ART / "scaler.pkl", TR_RAW):
        if not p.exists():
            raise FileNotFoundError(f"File mancante: {p}. Generali prima con la pipeline.")
    if not VA_RAW.exists():
        # fallback al vecchio default
        VA_RAW = DATA_ROOT / "gan_val_raw.csv"
    if not VA_RAW.exists():
        raise FileNotFoundError(f"CSV di validazione non trovato: {args.val_csv}")

    # Carica split RAW e scaler
    df_tr = pd.read_csv(TR_RAW)
    df_va = pd.read_csv(VA_RAW)

    feats = load_feature_names(ART / "features.json")
    with open(ART / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    Xtr = scaler.transform(df_tr[feats].to_numpy(dtype=np.float32)).astype(np.float32)
    ytr = df_tr["label"].to_numpy(dtype=np.int32)
    Xva = scaler.transform(df_va[feats].to_numpy(dtype=np.float32)).astype(np.float32)
    yva = df_va["label"].to_numpy(dtype=np.int32)

    input_dim = Xtr.shape[1]

    # Modelli
    G = build_generator(latent_dim=LATENT_DIM, output_dim=input_dim, hidden_dims=HIDDEN_G)
    D = build_discriminator(input_dim=input_dim, hidden_dims=HIDDEN_D, use_mbstd=USE_MBSTD)
    F = make_feat_extractor(D)

    # Optimizer
    opt_kw = {}
    if CLIP_NORM > 0:
        opt_kw["global_clipnorm"] = CLIP_NORM
    gen_opt  = tf.keras.optimizers.Adam(LR_G, beta_1=0.5, beta_2=0.999, **opt_kw)
    disc_opt = tf.keras.optimizers.Adam(LR_D, beta_1=0.5, beta_2=0.999, **opt_kw)

    # Datasets
    ds_train = make_dataset(Xtr, ytr, batch=BATCH_SIZE, shuffle=True)
    ds_val   = make_dataset(Xva, yva, batch=BATCH_SIZE, shuffle=False)

    # Log & checkpoints (tutto dentro ARTifacts della pipeline)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    (ART / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ART / "models").mkdir(parents=True, exist_ok=True)
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    (ART / "plots").mkdir(parents=True, exist_ok=True)

    gen_path  = ART / "checkpoints" / f"generator_{ts}.keras"
    disc_path = ART / "checkpoints" / f"discriminator_{ts}.keras"
    csv_log   = ART / "logs" / f"train_metrics_{ts}.csv"
    plots_dir = ART / "plots"

    with open(csv_log, "w", encoding="utf-8") as f:
        f.write("epoch,d_loss,g_loss,acc_tr,rfp_tr,fr_tr,acc_val,rfp_val,fr_val,steps,secs\n")

    # --- BEST tracking su rFP(val) + salvataggi fissi ---
    best_rfp = float("inf")
    best_fixed_disc = ART / "models" / "discriminator_best.keras"
    best_fixed_gen  = ART / "models" / "generator_best.keras"
    best_fixed_json = ART / "models" / "best_val_metrics.json"

    best_val = -1.0  # <— vecchia logica: best su FR(val) per i checkpoint timestampati
    steps_per_epoch = int(np.ceil(len(Xtr) / BATCH_SIZE))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        dl_acc, gl_acc, fr_acc, a_tr_acc, rfp_tr_acc = [], [], [], [], []

        for xb, yb in ds_train:
            for _ in range(D_STEPS):
                d_loss, acc_real, rfp_real = disc_step(
                    D, G, disc_opt, xb, yb, LATENT_DIM,
                    tf.constant(LABEL_SMOOTH, tf.float32),
                    tf.constant(NOISE_STD, tf.float32),
                )
            g_losses, fools = [], []
            for _ in range(G_STEPS):
                g_loss, fr = gen_step(
                    D, G, F, gen_opt, xb, LATENT_DIM,
                    tf.constant(NOISE_STD, tf.float32),
                    tf.constant(FM_W, tf.float32),
                )
                g_losses.append(g_loss.numpy()); fools.append(fr.numpy())
            dl_acc.append(d_loss.numpy())
            gl_acc.append(np.mean(g_losses))
            fr_acc.append(np.mean(fools))
            a_tr_acc.append(acc_real.numpy())
            rfp_tr_acc.append(rfp_real.numpy())

        acc_val, rfp_val = eval_disc_on(ds_val, D)

        val_fr = []
        for xb, _ in ds_val:
            z = tf.random.normal([tf.shape(xb)[0], LATENT_DIM])
            fx = G(z, training=False)
            logits = D(fx, training=False)
            p = tf.nn.softmax(logits, axis=-1)
            val_fr.append(tf.reduce_mean(tf.cast(tf.argmax(p, axis=-1) != 2, tf.float32)).numpy())
        fr_val = float(np.mean(val_fr)) if val_fr else 0.0

        secs = time.time() - t0
        print(
            f"[E{epoch:03d}] D_loss={np.mean(dl_acc):.4f}  G_loss={np.mean(gl_acc):.4f}  "
            f"ACC(tr)={np.mean(a_tr_acc):.3f}  ACC(val)={acc_val:.3f}  "
            f"rFP(tr)={np.mean(rfp_tr_acc):.3f}  rFP(val)={rfp_val:.3f}  "
            f"FR(tr)={np.mean(fr_acc):.3f}  FR(val)={fr_val:.3f}  steps={steps_per_epoch}  time={secs:.1f}s"
        )
        with open(csv_log, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{np.mean(dl_acc):.6f},{np.mean(gl_acc):.6f},"
                f"{np.mean(a_tr_acc):.6f},{np.mean(rfp_tr_acc):.6f},{np.mean(fr_acc):.6f},"
                f"{acc_val:.6f},{rfp_val:.6f},{fr_val:.6f},{steps_per_epoch},{secs:.3f}\n"
            )

        if fr_val > best_val:
            best_val = fr_val
            G.save(gen_path)
            D.save(disc_path)
            if LOG_CKPT:
                print(f"  [CKPT] Salvati G -> {gen_path.name}  D -> {disc_path.name}  (best val_FR={best_val:.3f})")

        if rfp_val < best_rfp:
            best_rfp = float(rfp_val)
            D.save(best_fixed_disc)
            G.save(best_fixed_gen)
            with open(best_fixed_json, "w", encoding="utf-8") as f:
                json.dump({"epoch": int(epoch),
                           "rFP_val": float(rfp_val),
                           "acc_real_only_val": float(acc_val),
                           "FR_val": float(fr_val)}, f, indent=2)
            print(f"  [BEST] Nuovo best rFP(val)={best_rfp:.6f} @ epoch {epoch} -> "
                  f"{best_fixed_disc.name}, {best_fixed_gen.name}")

        if PLOT_LIVE and PLOT_EVERY >= 1 and (epoch % PLOT_EVERY == 0):
            _save_training_plots(csv_log, plots_dir, tag=ts)

    # salvataggi finali 
    G.save(ART / "models" / "generator_model.keras")
    D.save(ART / "models" / "discriminator_model.keras")
    print("[OK] Training completato. Log CSV:", csv_log)
    _save_training_plots(csv_log, plots_dir, tag=ts)

if __name__ == "__main__":
    main()

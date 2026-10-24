#!/usr/bin/env python3
"""
Quantitative evaluation of the SHAP explanations of the deployed detector
(reviewer comment 3.4): fidelity, stability and cost.

Model:  artifacts/rf_model_three.pkl (RF on the 15 SHAP-selected raw features)
Data:   same protocol and split as MainProject.ipynb / train_eval.py (seed 42)
Explainer: shap.TreeExplainer, interventional, background = 500 training flows
           (same setting as MainProject.ipynb). Explanations refer to the
           predicted class, as in the runtime XAI stage.

Metrics
  Fidelity  - deletion test on flows predicted as attacks: features are replaced
              one at a time by the training median, in order of local |SHAP|,
              of global SHAP importance, or at random. Reported: normalised
              area under the predicted-class probability curve (AOPC-like,
              lower = more faithful) and label-flip rate after k deletions.
            - additivity error |E[f] + sum(phi) - f(x)|.
  Stability - background: K different background samples;
            - model: RF retrained with M different random seeds;
            - input: small Gaussian perturbations (eps * training std).
              Reported: Spearman / Kendall correlation of global importance and
              mean top-k Jaccard overlap of per-flow explanations.
  Cost      - per-flow explanation latency (single-flow calls).

Outputs (in --out): xai_metrics.json, deletion_curves.csv, deletion_curves.{png,pdf},
                    shap_summary_class{0,1,2}.{png,pdf}, table_xai.tex

Usage:
  .venv/bin/python xai_eval.py
  .venv/bin/python xai_eval.py --n-explain 500 --model-seeds 2 --bg-seeds 2   # quick check
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train_eval import CLASS_NAMES, TEST_SIZE, load_dataset

SPLIT_SEED = 42      # split used to train the deployed model
BG_SIZE = 500
TOP_K = 3


# --- helpers -------------------------------------------------------------------
def explain(model, background: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """SHAP values as array (n_samples, n_features, n_classes)."""
    ex = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    sv = ex.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return sv, ex


def pred_class_phi(sv: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return sv[np.arange(len(pred)), :, pred]


def global_importance(phi: np.ndarray) -> np.ndarray:
    return np.abs(phi).mean(axis=0)


def topk_sets(phi: np.ndarray, k: int = TOP_K) -> np.ndarray:
    return np.argsort(-np.abs(phi), axis=1)[:, :k]


def mean_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    vals = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]
    return float(np.mean(vals))


def pairwise_stats(globals_: list[np.ndarray], locals_: list[np.ndarray]) -> dict[str, float]:
    sp, kt, jac = [], [], []
    for i, j in itertools.combinations(range(len(globals_)), 2):
        sp.append(spearmanr(globals_[i], globals_[j]).statistic)
        kt.append(kendalltau(globals_[i], globals_[j]).statistic)
        jac.append(mean_jaccard(locals_[i], locals_[j]))
    return {"spearman_mean": float(np.mean(sp)), "spearman_min": float(np.min(sp)),
            "kendall_mean": float(np.mean(kt)), "kendall_min": float(np.min(kt)),
            f"top{TOP_K}_jaccard_mean": float(np.mean(jac)), "n_pairs": len(sp)}


def deletion_curve(model, X: pd.DataFrame, order: np.ndarray, baseline: np.ndarray,
                   cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted-class probability and flip rate after deleting 0..F features."""
    rows = np.arange(len(X))
    Xc = X.to_numpy(copy=True)
    cols = X.columns
    proba = model.predict_proba(pd.DataFrame(Xc, columns=cols))
    p_curve, flip_curve = [proba[rows, cls].mean()], [0.0]
    for k in range(order.shape[1]):
        idx = order[:, k]
        Xc[rows, idx] = baseline[idx]
        proba = model.predict_proba(pd.DataFrame(Xc, columns=cols))
        p_curve.append(proba[rows, cls].mean())
        flip_curve.append(float((proba.argmax(axis=1) != cls).mean()))
    return np.array(p_curve), np.array(flip_curve)


def norm_auc(curve: np.ndarray) -> float:
    # trapezoidal area normalised to [0, 1] over the deletion steps
    return float(np.trapz(curve, dx=1.0) / (len(curve) - 1))


# --- main --------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "merged_network_flow_leayer.csv")
    ap.add_argument("--model", type=Path, default=here / "artifacts" / "rf_model_three.pkl")
    ap.add_argument("--features", type=Path, default=here / "artifacts" / "features_three.json")
    ap.add_argument("--n-explain", type=int, default=3000, help="stratified test subsample to explain")
    ap.add_argument("--bg-seeds", type=int, default=5)
    ap.add_argument("--model-seeds", type=int, default=10)
    ap.add_argument("--noise-eps", type=float, default=0.01)
    ap.add_argument("--noise-reps", type=int, default=5)
    ap.add_argument("--random-orders", type=int, default=5)
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=here / "results_xai")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    feats = json.loads(args.features.read_text(encoding="utf-8"))
    X, y = load_dataset(args.data)
    X = X[feats]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    if args.n_explain < len(Xte):
        Xte, _, yte, _ = train_test_split(Xte, yte, train_size=args.n_explain, random_state=0, stratify=yte)
    model = joblib.load(args.model)
    pred = model.predict(Xte)
    print(f"explaining {len(Xte)} test flows | test acc of deployed model = {(pred == yte).mean():.3f}")

    metrics: dict[str, object] = {"n_explained": len(Xte), "features": feats}

    # --- reference explanation (same setting as the notebook) ---
    bg_ref = Xtr.sample(n=BG_SIZE, random_state=1)
    t0 = time.time()
    sv, explainer = explain(model, bg_ref, Xte)
    metrics["shap_batch_seconds"] = round(time.time() - t0, 2)
    phi = pred_class_phi(sv, pred)

    # additivity: E[f]_c + sum phi = f_c(x)
    proba = model.predict_proba(Xte)
    expected = np.asarray(explainer.expected_value)[pred]
    add_err = np.abs(expected + phi.sum(axis=1) - proba[np.arange(len(pred)), pred])
    metrics["additivity_error"] = {"mean": float(add_err.mean()), "max": float(add_err.max())}

    # --- fidelity: deletion test on flows predicted as attacks ---
    mal = pred != 0
    Xm, phim, clsm = Xte[mal], phi[mal], pred[mal]
    baseline = Xtr.median().to_numpy()
    n_f = len(feats)
    orders = {
        "SHAP (local)": np.argsort(-np.abs(phim), axis=1),
        "SHAP (global)": np.tile(np.argsort(-global_importance(phi)), (len(Xm), 1)),
    }
    curves = {}
    for name, order in orders.items():
        curves[name] = deletion_curve(model, Xm, order, baseline, clsm)
    rand_p, rand_f = [], []
    for r in range(args.random_orders):
        order = np.array([rng.permutation(n_f) for _ in range(len(Xm))])
        p, f = deletion_curve(model, Xm, order, baseline, clsm)
        rand_p.append(p)
        rand_f.append(f)
    curves["Random"] = (np.mean(rand_p, axis=0), np.mean(rand_f, axis=0))

    fid = {}
    for name, (p, f) in curves.items():
        fid[name] = {"aopc_norm_auc": norm_auc(p), "p_start": float(p[0]),
                     "flip@1": float(f[1]), "flip@3": float(f[3]), "flip@5": float(f[5])}
    metrics["fidelity_deletion"] = {"n_attack_flows": int(mal.sum()), "baseline": "training median", **fid}
    pd.DataFrame({f"{n} {k}": v for n, (p, f) in curves.items() for k, v in (("p", p), ("flip", f))}
                 ).rename_axis("k_deleted").to_csv(args.out / "deletion_curves.csv")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, (p, _) in curves.items():
        ax.plot(range(n_f + 1), p, marker="o", ms=3, label=f"{name} (AUC={norm_auc(p):.2f})")
    ax.set_xlabel("Number of deleted features")
    ax.set_ylabel("Predicted-class probability")
    ax.grid(alpha=.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"deletion_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- stability w.r.t. the background sample ---
    g_list, l_list = [global_importance(phi)], [topk_sets(phi)]
    for s in range(2, 2 + args.bg_seeds - 1):
        sv_b, _ = explain(model, Xtr.sample(n=BG_SIZE, random_state=s), Xte)
        ph = pred_class_phi(sv_b, pred)
        g_list.append(global_importance(ph))
        l_list.append(topk_sets(ph))
    metrics["stability_background"] = pairwise_stats(g_list, l_list)

    # --- stability w.r.t. the model training seed (same split, same config) ---
    g_list, l_list = [], []
    params = {k: v for k, v in model.get_params().items() if k in ("n_estimators", "max_depth", "min_samples_leaf")}
    for s in range(args.model_seeds):
        m = RandomForestClassifier(**params, random_state=s, n_jobs=-1).fit(Xtr, ytr)
        sv_m, _ = explain(m, bg_ref, Xte)
        ph = pred_class_phi(sv_m, pred)   # explain the deployed model's decision class
        g_list.append(global_importance(ph))
        l_list.append(topk_sets(ph))
        print(f"  model seed {s} done", flush=True)
    metrics["stability_model_seed"] = {**pairwise_stats(g_list, l_list), "rf_params": params}

    # --- stability w.r.t. small input perturbations (prediction unchanged) ---
    std = Xtr.std().to_numpy()
    ref_top = topk_sets(phi)
    jac, kept = [], []
    for r in range(args.noise_reps):
        Xn = np.clip(Xte.to_numpy() + rng.normal(0, args.noise_eps * std, size=Xte.shape), 0, None)
        Xn = pd.DataFrame(Xn, columns=feats, index=Xte.index)
        same = model.predict(Xn) == pred
        sv_n = explainer.shap_values(Xn[same], check_additivity=False)
        if isinstance(sv_n, list):
            sv_n = np.stack(sv_n, axis=-1)
        ph = pred_class_phi(sv_n, pred[same])
        jac.append(mean_jaccard(ref_top[same], topk_sets(ph)))
        kept.append(float(same.mean()))
    metrics["stability_input_noise"] = {"eps_times_std": args.noise_eps,
                                        f"top{TOP_K}_jaccard_mean": float(np.mean(jac)),
                                        "share_prediction_unchanged": float(np.mean(kept))}

    # --- cost: single-flow explanation latency ---
    lat = []
    for i in rng.choice(len(Xte), size=min(args.latency_n, len(Xte)), replace=False):
        t0 = time.perf_counter()
        explainer.shap_values(Xte.iloc[[i]], check_additivity=False)
        lat.append((time.perf_counter() - t0) * 1000)
    metrics["latency_ms_per_flow"] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                                      "background_size": BG_SIZE}

    # --- Fig. 8 regenerated on the deployed model (one plot per class) ---
    for c, cname in enumerate(CLASS_NAMES):
        shap.summary_plot(sv[:, :, c], Xte, feature_names=feats, show=False, max_display=n_f)
        plt.title(f"{cname} traffic")
        for ext in ("png", "pdf"):
            plt.savefig(args.out / f"shap_summary_class{c}.{ext}", dpi=300, bbox_inches="tight")
        plt.close()

    (args.out / "xai_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- LaTeX table ---
    sb, sm, sn = metrics["stability_background"], metrics["stability_model_seed"], metrics["stability_input_noise"]
    lines = [r"\begin{tabular}{llc}", r"\toprule", r"\textbf{Property} & \textbf{Metric} & \textbf{Value} \\", r"\midrule"]
    for name in ("SHAP (local)", "SHAP (global)", "Random"):
        lines.append(f"Fidelity ({name}) & AOPC / flip@3 & {fid[name]['aopc_norm_auc']:.3f} / {fid[name]['flip@3']:.3f} \\\\")
    lines += [
        f"Stability (background, {args.bg_seeds} samples) & Spearman / top-{TOP_K} Jaccard & "
        f"{sb['spearman_mean']:.3f} / {sb[f'top{TOP_K}_jaccard_mean']:.3f} \\\\",
        f"Stability (model, {args.model_seeds} seeds) & Spearman / top-{TOP_K} Jaccard & "
        f"{sm['spearman_mean']:.3f} / {sm[f'top{TOP_K}_jaccard_mean']:.3f} \\\\",
        f"Stability (input noise, $\\epsilon={args.noise_eps}\\sigma$) & top-{TOP_K} Jaccard & "
        f"{sn[f'top{TOP_K}_jaccard_mean']:.3f} \\\\",
        f"Additivity & mean abs. error & {metrics['additivity_error']['mean']:.1e} \\\\",
        f"Cost & latency per flow (median / p95) & {metrics['latency_ms_per_flow']['median']:.1f} / "
        f"{metrics['latency_ms_per_flow']['p95']:.1f} ms \\\\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (args.out / "table_xai.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()

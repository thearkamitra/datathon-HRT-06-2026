#!/usr/bin/env python3
"""
Report P(R>0), margin |p-0.5|, and binary entropy for ``prob_sign`` (same fit as production).

Usage::

  PYTHONPATH=src python Tools/prob_sign_margin_entropy.py --data-dir data
  PYTHONPATH=src python Tools/prob_sign_margin_entropy.py -o Tools/out/prob_sign_scores.csv
  PYTHONPATH=src python Tools/prob_sign_margin_entropy.py --data-dir data --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from datathon_baseline.io import BARS_SEEN_PRIVATE_TEST, BARS_SEEN_PUBLIC_TEST, read_bars
from datathon_baseline.paths import data_dir as default_data_dir
from datathon_sharpe.distributional_mono import (
    binary_entropy_nats,
    fit_distributional_mono,
)
from datathon_sharpe.sentiment_features import FEATURE_COLUMNS_SHARPE, build_sharpe_session_features, load_sentiments_seen_test
from datathon_sharpe.training_table import load_training_feature_matrices


def _save_distribution_plot(
    p_tr: np.ndarray,
    p_te: np.ndarray,
    margin_tr: np.ndarray,
    margin_te: np.ndarray,
    h_tr: np.ndarray,
    h_te: np.ndarray,
    out_path: Path,
    *,
    ridge_alpha: float,
    augment: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bins_p = 55
    bins_m = 45
    bins_h = 45

    panels = [
        (axes[0], p_tr, p_te, r"$p=\mathbb{P}(R>0\mid X)$", r"$p$", 0.0, 1.0, bins_p),
        (axes[1], margin_tr, margin_te, r"margin $=|p-\frac{1}{2}|$", "margin", 0.0, 0.5, bins_m),
        (axes[2], h_tr, h_te, r"entropy $H(p)$ (nats)", r"$H(p)$", 0.0, float(np.log(2.0) * 1.02), bins_h),
    ]
    for ax, a_tr, a_te, title, xlab, xl, xr, bins in panels:
        ax.hist(
            a_tr,
            bins=bins,
            range=(xl, xr),
            alpha=0.65,
            label=f"train (n={len(a_tr):,})",
            density=True,
            color="C0",
        )
        ax.hist(
            a_te,
            bins=bins,
            range=(xl, xr),
            alpha=0.5,
            label=f"test (n={len(a_te):,})",
            density=True,
            color="C1",
        )
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper right")

    axes[0].axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axvline(np.log(2.0), color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    fig.suptitle(
        f"prob_sign full distributions (ridge_alpha={ridge_alpha}, augment={augment})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ascii_hist(x: np.ndarray, bins: int, lo: float, hi: float) -> str:
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    mx = int(counts.max()) if counts.size else 1
    lines = []
    for i, c in enumerate(counts):
        a, b = edges[i], edges[i + 1]
        bar = "#" * max(1, int(40 * c / mx)) if c > 0 else ""
        lines.append(f"  [{a:5.3f}, {b:5.3f})  {int(c):5d}  {bar}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="prob_sign: p, margin, entropy (train + test)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-augment-test-proxy",
        action="store_true",
        help="Match --no-augment-test-proxy training (train rows only in fit).",
    )
    p.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for per-session CSV (train+test; can be large).",
    )
    p.add_argument(
        "--plot",
        type=Path,
        nargs="?",
        const=_REPO / "Tools" / "prob_sign_distribution.png",
        default=None,
        help="Save PNG with overlaid train/test histograms (density). "
        "Default path if flag given with no path: Tools/prob_sign_distribution.png",
    )
    args = p.parse_args()

    dd = args.data_dir or default_data_dir()
    dd = dd.resolve()
    if not dd.is_dir():
        raise SystemExit(f"Data directory not found: {dd}")

    augment = not args.no_augment_test_proxy
    feat_main, feat_fit = load_training_feature_matrices(
        dd,
        within_session_split=False,
        augment_test_with_proxy=augment,
    )
    R_fit = feat_fit["R"].to_numpy(dtype=np.float64)
    X_fit = feat_fit[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)

    pred = fit_distributional_mono(
        X_fit,
        R_fit,
        policy="prob_sign",
        ridge_reg=args.ridge_alpha,
        random_state=args.seed,
    )

    X_main = feat_main[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    p_tr = pred.predict_prob_positive(X_main)
    margin_tr = np.abs(p_tr - 0.5)
    H_tr = binary_entropy_nats(p_tr)

    bars_pub = read_bars(dd, BARS_SEEN_PUBLIC_TEST)
    bars_priv = read_bars(dd, BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    try:
        h_pub = pd.read_parquet(dd / "headlines_seen_public_test.parquet")
        h_priv = pd.read_parquet(dd / "headlines_seen_private_test.parquet")
        headlines_te = pd.concat([h_pub, h_priv], ignore_index=True)
    except Exception:
        headlines_te = None
    sen_te = load_sentiments_seen_test(dd)
    feat_te = build_sharpe_session_features(bars_te, headlines_te, sen_te, first_half=False)
    X_te = feat_te[FEATURE_COLUMNS_SHARPE].to_numpy(dtype=np.float64)
    p_te = pred.predict_prob_positive(X_te)
    margin_te = np.abs(p_te - 0.5)
    H_te = binary_entropy_nats(p_te)

    print("prob_sign diagnostics (logistic P(R>0), same fit as distributional_mono)")
    print(f"  data_dir={dd}")
    print(f"  ridge_alpha={args.ridge_alpha}  (logistic C=1/alpha)")
    print(f"  augment_test_with_proxy={augment}")
    print(f"  fit rows={len(feat_fit)}  train_sessions={len(feat_main)}  test_sessions={len(feat_te)}")
    print()
    print("Binary entropy H(p) in nats; max = ln(2) ≈ 0.693 at p=0.5; min = 0 at p in {0,1}.")
    print()

    def block(name: str, p_: np.ndarray, m_: np.ndarray, h_: np.ndarray) -> None:
        qs = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
        pct = np.percentile(p_, qs)
        print(f"=== {name} (n={len(p_)}) ===")
        print(f"  p       : min={p_.min():.4f}  max={p_.max():.4f}  mean={p_.mean():.4f}  std={p_.std():.4f}")
        print("  p percentiles:", dict(zip(qs, np.round(pct, 4))))
        print(f"  margin  : mean={m_.mean():.4f}  median={np.median(m_):.4f}")
        print(f"  entropy : mean={h_.mean():.4f}  median={np.median(h_):.4f}")
        print("  p histogram (10 bins on [0,1]):")
        print(_ascii_hist(p_, 10, 0.0, 1.0))
        print()

    block("train (1000 competition-R sessions)", p_tr, margin_tr, H_tr)
    block("test (submission sessions)", p_te, margin_te, H_te)

    plot_path = args.plot
    if plot_path is not None:
        plot_path = plot_path.resolve()
        _save_distribution_plot(
            p_tr,
            p_te,
            margin_tr,
            margin_te,
            H_tr,
            H_te,
            plot_path,
            ridge_alpha=args.ridge_alpha,
            augment=augment,
        )
        print(f"Wrote figure {plot_path}")

    out = args.output_csv
    if out is not None:
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df_tr = pd.DataFrame(
            {
                "session": feat_main["session"].to_numpy(),
                "split": "train",
                "p": p_tr,
                "margin": margin_tr,
                "entropy_nats": H_tr,
                "f_raw_2p_minus_1": 2.0 * p_tr - 1.0,
            }
        )
        df_te = pd.DataFrame(
            {
                "session": feat_te["session"].to_numpy(),
                "split": "test",
                "p": p_te,
                "margin": margin_te,
                "entropy_nats": H_te,
                "f_raw_2p_minus_1": 2.0 * p_te - 1.0,
            }
        )
        pd.concat([df_tr, df_te], ignore_index=True).to_csv(out, index=False)
        print(f"Wrote {out} ({len(df_tr) + len(df_te)} rows)")


if __name__ == "__main__":
    main()

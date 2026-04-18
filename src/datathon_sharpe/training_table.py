"""Load the same feature + label table used for fitting in ``train_model``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datathon_baseline.io import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
    read_bars,
)
from datathon_baseline.labels import train_realized_returns
from datathon_sharpe.path_features import build_session_features_with_path
from datathon_sharpe.labels_seen_split import (
    proxy_returns_second_seen_half_from_bars,
    train_proxy_returns_second_seen_half,
)


def load_training_feature_matrices(
    data_dir: Path,
    *,
    within_session_split: bool = False,
    augment_test_with_proxy: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns ``(feat_main, feat_fit)`` aligned with ``fit_full_train_and_submission``.

    ``feat_main``: train sessions only (competition R).
    ``feat_fit``: rows used in the optimizer (by default includes augmented test
    rows when ``augment_test_with_proxy`` is True).
    """
    if within_session_split and augment_test_with_proxy:
        raise ValueError("Choose either within_session_split or augment_test_with_proxy, not both.")

    bars_tr = read_bars(data_dir, BARS_SEEN_TRAIN)
    try:
        headlines_tr = pd.read_parquet(data_dir / "headlines_seen_train.parquet")
    except Exception:
        headlines_tr = None

    bars_pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST)
    bars_priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    try:
        h_pub = pd.read_parquet(data_dir / "headlines_seen_public_test.parquet")
        h_priv = pd.read_parquet(data_dir / "headlines_seen_private_test.parquet")
        headlines_te = pd.concat([h_pub, h_priv], ignore_index=True)
    except Exception:
        headlines_te = None

    if augment_test_with_proxy:
        labels_main = train_realized_returns(data_dir)
        feat_main = build_session_features_with_path(bars_tr, headlines_tr, first_half=False)
        feat_main = feat_main.merge(labels_main[["session", "R"]], on="session", how="inner")
        if len(feat_main) != len(labels_main):
            raise RuntimeError("Feature / label session alignment failed (train).")

        labels_aug = proxy_returns_second_seen_half_from_bars(bars_te)
        feat_aug = build_session_features_with_path(bars_te, headlines_te, first_half=True)
        feat_aug = feat_aug.merge(labels_aug[["session", "R"]], on="session", how="inner")
        if len(feat_aug) != len(labels_aug):
            raise RuntimeError("Feature / label session alignment failed (test augment).")

        feat_fit = pd.concat([feat_main, feat_aug], ignore_index=True)
    elif within_session_split:
        labels = train_proxy_returns_second_seen_half(data_dir)
        feat_tr = build_session_features_with_path(bars_tr, headlines_tr, first_half=True)
        feat_tr = feat_tr.merge(labels[["session", "R"]], on="session", how="inner")
        if len(feat_tr) != len(labels):
            raise RuntimeError("Feature / label session alignment failed.")
        feat_main = feat_tr
        feat_fit = feat_tr
    else:
        labels = train_realized_returns(data_dir)
        feat_tr = build_session_features_with_path(bars_tr, headlines_tr, first_half=False)
        feat_tr = feat_tr.merge(labels[["session", "R"]], on="session", how="inner")
        if len(feat_tr) != len(labels):
            raise RuntimeError("Feature / label session alignment failed.")
        feat_main = feat_tr
        feat_fit = feat_tr

    feat_fit = feat_fit.sort_values("session").reset_index(drop=True)
    feat_main = feat_main.sort_values("session").reset_index(drop=True)
    return feat_main, feat_fit


__all__ = ["load_training_feature_matrices"]

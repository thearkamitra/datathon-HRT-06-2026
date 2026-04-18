"""End-to-end probabilistic-regression pipeline.

Stages (mirroring ``plans/probablisticReg.txt``):

1.  Build per-session OHLC features from the first-half bars
    (reuses :func:`features.build_session_features` from the existing
    ``src/tailored-modeler`` module so we do not duplicate the carefully
    tuned feature family).
2.  Fit the news featurizer on the training headlines + sentiments and
    transform both train and test. News features act as *session identity*
    in the design matrix: entity footprint, topic fingerprint (TF-IDF +
    SVD), and directional sentiment aggregates. Reuses
    :class:`news.NewsFeaturizer` from the existing ``src/regime`` module.
3.  Compute the per-session realised return ``R`` on train.
4.  Run the two-pass heteroskedastic OOF protocol in :mod:`heads`: OOF
    mean-head predictions -> OOF squared residuals -> OOF variance-head
    predictions.
5.  Project ``(mu, sigma)`` into the ``(mu, p_up, q10/50/90, u)`` contract
    used by the existing Sharpe-aware sizer and grid-tune the sizer on
    the OOF sizing frame.
6.  Refit both heads on the full train set with honest OOF-residual
    variance targets and emit test-time ``(mu, sigma)`` predictions.
7.  Apply the tuned sizer, format the competition-format submission.

The module deliberately loads the sibling utilities via :mod:`importlib`
instead of copy-pasting them:

* ``src/tailored-modeler/features.py`` -> ``build_session_features`` /
  ``feature_columns``
* ``src/regime/news.py`` -> ``NewsConfig`` / ``NewsFeaturizer`` /
  ``load_train_news`` / ``load_test_news``
* ``src/tailored-modeler/sizing.py`` -> ``SizingConfig`` / ``apply_sizing``
  / ``tune_sizing`` / ``build_ranking`` / ``sharpe``

The :mod:`paths` shim is pre-registered in :mod:`sys.modules` so the
regime ``news.py`` can resolve its ``from paths import ...`` import.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# importlib bridges to sibling modules
# ---------------------------------------------------------------------------


_SRC_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register ``paths`` under the canonical name so regime/news.py resolves it.
# Our own ``paths.py`` has the exact same constants so the regime featurizer
# is happy either way; here we prefer our own for auditability.
_paths_mod = _load_module("paths", Path(__file__).resolve().parent / "paths.py")

# Reuse OHLC session-feature builder from tailored-modeler.
_tm_features = _load_module(
    "prob_reg_tm_features", _SRC_ROOT / "tailored-modeler" / "features.py"
)
build_session_features = _tm_features.build_session_features
feature_columns = _tm_features.feature_columns

# Reuse first-half news featurizer from regime/.
_regime_news = _load_module(
    "prob_reg_regime_news", _SRC_ROOT / "regime" / "news.py"
)
NewsConfig = _regime_news.NewsConfig
NewsFeaturizer = _regime_news.NewsFeaturizer
load_train_news = _regime_news.load_train_news
load_test_news = _regime_news.load_test_news

# Reuse Sharpe-aware sizer from tailored-modeler.
_tm_sizing = _load_module(
    "prob_reg_tm_sizing", _SRC_ROOT / "tailored-modeler" / "sizing.py"
)
SizingConfig = _tm_sizing.SizingConfig
apply_sizing = _tm_sizing.apply_sizing
build_ranking = _tm_sizing.build_ranking
sharpe = _tm_sizing.sharpe
size_with_fallback = _tm_sizing.size_with_fallback
tune_sizing = _tm_sizing.tune_sizing


from heads import (  # noqa: E402
    FittedHeads,
    HeadsConfig,
    OOFResult,
    fit_heads,
    predict_heads,
    run_heteroskedastic_cv,
    to_sizing_frame,
)
from labels import train_realized_returns  # noqa: E402
from paths import (  # noqa: E402
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
)


# ---------------------------------------------------------------------------
# Configuration / result types
# ---------------------------------------------------------------------------


@dataclass
class ProbRegConfig:
    random_state: int = 0
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    news: NewsConfig = field(default_factory=lambda: NewsConfig(enabled=True))
    target_scale: float = 1.0
    clip_quantile: float = 0.999


@dataclass
class ProbRegResult:
    submission: pd.DataFrame
    rankings: pd.DataFrame
    diagnostics: dict
    tuned_sizing: SizingConfig
    config: ProbRegConfig
    oof: OOFResult


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _build_feature_frame(
    bars: pd.DataFrame,
    news_featurizer: Optional[NewsFeaturizer],
    headlines: Optional[pd.DataFrame],
    sentiments: Optional[pd.DataFrame],
    sessions_order: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    ohlc_feats = build_session_features(bars).sort_values("session").reset_index(drop=True)
    if news_featurizer is not None:
        req_sessions = ohlc_feats["session"].tolist()
        news_feats = news_featurizer.transform(
            headlines if headlines is not None else pd.DataFrame(columns=["session", "headline", "bar_ix"]),
            sentiments if sentiments is not None else pd.DataFrame(
                columns=["session", "bar_ix", "company", "sentiment", "sentiment_score", "confidence"]
            ),
            req_sessions,
        )
        ohlc_feats = ohlc_feats.merge(news_feats, on="session", how="left").fillna(0.0)
    if sessions_order is not None:
        ohlc_feats = (
            ohlc_feats.set_index("session")
            .reindex(sessions_order)
            .reset_index()
            .fillna(0.0)
        )
    return ohlc_feats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: Path,
    config: ProbRegConfig = ProbRegConfig(),
) -> ProbRegResult:
    data_dir = Path(data_dir)

    # ---- Load training raw data ------------------------------------------
    bars_tr = pd.read_parquet(data_dir / BARS_SEEN_TRAIN)
    labels = train_realized_returns(data_dir)

    headlines_tr = sentiments_tr = None
    featurizer: Optional[NewsFeaturizer] = None
    if config.news.enabled:
        headlines_tr, sentiments_tr = load_train_news(data_dir)
        featurizer = NewsFeaturizer(config.news).fit(headlines_tr, sentiments_tr)

    # ---- Build aligned training design matrix ----------------------------
    feats_tr = _build_feature_frame(bars_tr, featurizer, headlines_tr, sentiments_tr)
    feats_tr = feats_tr.merge(labels[["session", "R"]], on="session", how="inner")
    if feats_tr.empty:
        raise RuntimeError("Training feature / label join produced 0 rows.")
    feats_tr = feats_tr.sort_values("session").reset_index(drop=True)
    feat_cols = feature_columns(feats_tr)
    X_tr = feats_tr[feat_cols].astype(np.float64)
    y_tr = feats_tr["R"].to_numpy(dtype=np.float64)
    sessions_tr = feats_tr["session"].to_numpy(dtype=np.int64)

    # ---- Heteroskedastic OOF (Pass 1 mean + Pass 2 variance) -------------
    oof = run_heteroskedastic_cv(X_tr, y_tr, sessions_tr, cfg=config.heads)

    # ---- Tune the Sharpe-aware sizer on the OOF sizing frame -------------
    tuned_cfg, tune_info = tune_sizing(
        oof.oof_sizing_frame,
        y_tr,
        target_scale=config.target_scale,
        clip_quantile=config.clip_quantile,
    )
    oof_positions, _ = apply_sizing(oof.oof_sizing_frame, tuned_cfg)
    oof_sharpe = sharpe(oof_positions * y_tr)
    flat_sharpe = sharpe(y_tr * config.target_scale)
    sign_sharpe = sharpe(np.sign(oof.oof_preds["mu"].to_numpy()) * y_tr)
    p_sign_sharpe = sharpe(
        np.where(oof.oof_sizing_frame["p_up"].to_numpy() > 0.5, 1.0, -1.0) * y_tr
    )
    mu_over_var_pnl = (
        oof.oof_preds["mu"].to_numpy() / np.maximum(oof.oof_preds["sigma2"].to_numpy(), 1e-9)
    ) * y_tr
    mu_over_var_sharpe = sharpe(mu_over_var_pnl)

    # ---- Refit heads on FULL training data with honest OOF variance targets
    fitted = fit_heads(
        X_tr,
        y_tr,
        oof_residuals_sq=oof.residuals * oof.residuals,
        cfg=config.heads,
    )

    # ---- Build test design matrix + predict ------------------------------
    bars_pub = pd.read_parquet(data_dir / BARS_SEEN_PUBLIC_TEST)
    bars_priv = pd.read_parquet(data_dir / BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)

    headlines_te = sentiments_te = None
    if config.news.enabled:
        headlines_te, sentiments_te = load_test_news(data_dir)

    feats_te = _build_feature_frame(bars_te, featurizer, headlines_te, sentiments_te)
    feats_te = feats_te.sort_values("session").reset_index(drop=True)
    # Make sure the test design matrix has exactly the same feature columns
    # (zeros for any column that the train set had but test didn't).
    for col in feat_cols:
        if col not in feats_te.columns:
            feats_te[col] = 0.0
    X_te = feats_te[feat_cols].astype(np.float64)

    preds_te_raw = predict_heads(fitted, X_te)
    preds_te = to_sizing_frame(
        feats_te["session"].to_numpy(dtype=np.int64),
        preds_te_raw,
        cfg=config.heads,
    )
    positions_te = size_with_fallback(preds_te, tuned_cfg)
    rankings = build_ranking(
        feats_te["session"].to_numpy(dtype=np.int64), preds_te, positions_te
    )
    rankings["sigma"] = preds_te["sigma"].to_numpy(dtype=np.float64)
    rankings["sigma2"] = preds_te["sigma2"].to_numpy(dtype=np.float64)
    submission = rankings[["session", "target_position"]].copy()

    diagnostics = {
        "n_train_sessions": int(len(feats_tr)),
        "n_test_sessions": int(len(feats_te)),
        "n_features": int(len(feat_cols)),
        "news_enabled": bool(config.news.enabled),
        "train_R_mean": float(np.mean(y_tr)),
        "train_R_std": float(np.std(y_tr, ddof=0)),
        "oof_mse_mean_head": float(oof.oof_mse),
        "oof_mean_mu": float(np.mean(oof.oof_preds["mu"].to_numpy())),
        "oof_std_mu": float(np.std(oof.oof_preds["mu"].to_numpy(), ddof=0)),
        "oof_mean_sigma": float(np.mean(oof.oof_preds["sigma"].to_numpy())),
        "oof_sharpe_tuned": float(oof_sharpe),
        "oof_sharpe_flat_long": float(flat_sharpe),
        "oof_sharpe_sign_mu": float(sign_sharpe),
        "oof_sharpe_sign_p": float(p_sign_sharpe),
        "oof_sharpe_mu_over_var": float(mu_over_var_sharpe),
        "tuned_mode": str(tuned_cfg.mode),
        "tuned_alpha": float(tuned_cfg.alpha),
        "tuned_baseline": float(tuned_cfg.baseline),
        "tuned_lambda": float(tuned_cfg.lam),
        "tuned_theta": float(tuned_cfg.theta),
        "tuned_tau_quantile": float(tuned_cfg.tau_quantile),
        "tuned_allow_short": bool(tuned_cfg.allow_short),
        "tune_info": tune_info,
        "heads": {
            "mean_regularizer": config.heads.mean_regularizer,
            "mean_alpha": getattr(fitted.mu_model, "alpha_", None),
            "variance_alpha": float(fitted.var_model.alpha),
            "sigma2_floor": float(fitted.sigma2_floor),
            "use_gaussian_quantiles": bool(config.heads.use_gaussian_quantiles),
            "cv_splits": int(config.heads.cv_splits),
        },
    }

    return ProbRegResult(
        submission=submission,
        rankings=rankings,
        diagnostics=diagnostics,
        tuned_sizing=tuned_cfg,
        config=config,
        oof=oof,
    )

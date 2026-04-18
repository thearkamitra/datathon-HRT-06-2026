"""End-to-end Phase-1 tailored modeler pipeline.

The pipeline implements the plan's Stages 1-10 for the OHLC-only branch
(news stays gated behind the toggle in :mod:`news`):

1. Build rich session-level OHLC features (:mod:`features`).
2. Compute the R target from train seen+unseen parquets (:mod:`labels`).
3. Train three LightGBM heads (mean / sign / quantiles) with early stopping
   via repeated K-fold (:mod:`models`).
4. Auto-tune the Sharpe-aware sizing layer on the OOF predictions
   (:mod:`sizing`).
5. Retrain each head on the full training set with the best iteration counts
   and predict on the 20000 public+private test sessions.
6. Run an adversarial-validation guardrail (:mod:`adversarial_validation`).
7. Emit a ranking frame with mu / p / q10 / q50 / q90 / u / edge / z /
   target_position and a competition-format submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from adversarial_validation import AdversarialReport, run_adversarial
from features import build_session_features, feature_columns
from labels import train_realized_returns
from models import BoosterHyper, SampleWeightConfig, TabularHeads, backend_name
from news import NewsConfig, build_news_features
from paths import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
)
from sizing import (
    SizingConfig,
    apply_sizing,
    build_ranking,
    sharpe,
    size_with_fallback,
    tune_sizing,
)


@dataclass
class TailoredConfig:
    random_state: int = 0
    cv_splits: int = 5
    cv_repeats: int = 2
    quantiles: tuple = (0.1, 0.5, 0.9)
    hyper: BoosterHyper = field(default_factory=BoosterHyper)
    sample_weights: SampleWeightConfig = field(default_factory=SampleWeightConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    target_scale: float = 1.0
    clip_quantile: float = 0.999
    run_adversarial: bool = True


@dataclass
class TailoredResult:
    submission: pd.DataFrame
    rankings: pd.DataFrame
    train_diagnostics: dict
    feature_importances: Optional[pd.DataFrame]
    adversarial: Optional[AdversarialReport]
    tuned_sizing: SizingConfig
    config: TailoredConfig


def _merge_news(
    features: pd.DataFrame,
    config: NewsConfig,
    data_dir: Path,
    splits: tuple[str, ...],
) -> pd.DataFrame:
    news_feats = build_news_features(
        features["session"], config, data_dir=data_dir, splits=splits,
    )
    # When disabled this is a zero-filled frame; we still merge to keep the
    # columns present so feature_columns() sees a consistent schema.
    return features.merge(news_feats, on="session", how="left")


def run_pipeline(
    data_dir: Path,
    config: TailoredConfig = TailoredConfig(),
) -> TailoredResult:
    data_dir = Path(data_dir)

    # ---- Train features + labels ------------------------------------------
    bars_tr = pd.read_parquet(data_dir / BARS_SEEN_TRAIN)
    feats_tr = build_session_features(bars_tr)
    if config.news.enabled:
        feats_tr = _merge_news(feats_tr, config.news, data_dir, ("train_seen",))

    labels = train_realized_returns(data_dir)
    feats_tr = feats_tr.merge(labels[["session", "R"]], on="session", how="inner")
    if feats_tr.empty:
        raise RuntimeError("Training feature / label join produced 0 rows.")
    feats_tr = feats_tr.sort_values("session").reset_index(drop=True)

    feat_cols = feature_columns(feats_tr)
    X_tr = feats_tr[feat_cols].astype(np.float64)
    y_tr = feats_tr["R"].to_numpy(dtype=np.float64)

    # ---- OOF predictions via repeated KFold -------------------------------
    heads_proto = TabularHeads(
        random_state=config.random_state,
        quantiles=config.quantiles,
        hyper=config.hyper,
        sample_weights=config.sample_weights,
    )
    oof_preds, fold_groups = heads_proto.cross_val_predict(
        X_tr,
        y_tr,
        n_splits=config.cv_splits,
        n_repeats=config.cv_repeats,
        return_folds=True,
    )

    # ---- Auto-tune the Sharpe-aware sizer on OOF --------------------------
    tuned_cfg, tune_info = tune_sizing(
        oof_preds,
        y_tr,
        fold_groups=fold_groups,
        target_scale=config.target_scale,
        clip_quantile=config.clip_quantile,
    )

    # Compute supplementary OOF diagnostics.
    oof_positions, _ = apply_sizing(oof_preds, tuned_cfg)
    oof_kelly_sharpe = sharpe(oof_positions * y_tr)
    always_long_sharpe = sharpe(y_tr)
    sign_only_sharpe = sharpe(np.sign(oof_preds["mu"].to_numpy()) * y_tr)
    p_sign_sharpe = sharpe(np.where(oof_preds["p_up"].to_numpy() > 0.5, 1.0, -1.0) * y_tr)
    edge_raw = oof_preds["mu"].to_numpy() * (2.0 * oof_preds["p_up"].to_numpy() - 1.0)
    edge_only_sharpe = sharpe(edge_raw * y_tr)

    # ---- Final retrain on ALL 1000 train sessions using CV-median best-iter
    # (Previous version trained on only 80% of the data via a single KFold
    # split, wasting 200 labelled sessions and biasing the final model.)
    cv_best_iters = dict(heads_proto.best_iters_)
    final_heads = TabularHeads(
        random_state=config.random_state,
        quantiles=config.quantiles,
        hyper=config.hyper,
        sample_weights=config.sample_weights,
    )
    final_heads.fit(
        X_tr, y_tr,
        X_val=None, y_val=None,
        n_iters_override=cv_best_iters if cv_best_iters else None,
    )

    feat_importance = final_heads.feature_importance()

    # ---- Test features + predictions --------------------------------------
    bars_pub = pd.read_parquet(data_dir / BARS_SEEN_PUBLIC_TEST)
    bars_priv = pd.read_parquet(data_dir / BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    feats_te = build_session_features(bars_te)
    if config.news.enabled:
        feats_te = _merge_news(
            feats_te, config.news, data_dir, ("public_test", "private_test"),
        )
    feats_te = feats_te.sort_values("session").reset_index(drop=True)

    X_te = feats_te[feat_cols].astype(np.float64)
    preds_te = final_heads.predict(X_te)

    positions_te = size_with_fallback(preds_te, tuned_cfg)

    rankings = build_ranking(
        feats_te["session"].to_numpy(), preds_te, positions_te
    )
    submission = rankings[["session", "target_position"]].copy()

    # ---- Adversarial validation guardrail ---------------------------------
    adversarial: Optional[AdversarialReport] = None
    if config.run_adversarial:
        try:
            adversarial = run_adversarial(
                train_features=X_tr.assign(session=feats_tr["session"].to_numpy()),
                test_features=X_te.assign(session=feats_te["session"].to_numpy()),
                n_splits=5,
                random_state=config.random_state,
            )
        except Exception as exc:  # pragma: no cover - diagnostic is optional
            adversarial = None
            print(f"[warn] adversarial validation failed: {exc}")

    diagnostics = {
        "backend": backend_name(),
        "n_train_sessions": int(len(feats_tr)),
        "n_test_sessions": int(len(feats_te)),
        "n_features": int(len(feat_cols)),
        "cv_splits": int(config.cv_splits),
        "cv_repeats": int(config.cv_repeats),
        "sample_weight_enabled": bool(config.sample_weights.enabled),
        "use_news": bool(config.news.enabled),
        "train_R_mean": float(np.mean(y_tr)),
        "train_R_std": float(np.std(y_tr, ddof=0)),
        "oof_sharpe_tuned": float(oof_kelly_sharpe),
        "oof_sharpe_always_long": float(always_long_sharpe),
        "oof_sharpe_sign_mu": float(sign_only_sharpe),
        "oof_sharpe_sign_p": float(p_sign_sharpe),
        "oof_sharpe_edge_raw": float(edge_only_sharpe),
        "tuned_mode": str(tuned_cfg.mode),
        "tuned_alpha": float(tuned_cfg.alpha),
        "tuned_lambda": float(tuned_cfg.lam),
        "tuned_theta": float(tuned_cfg.theta),
        "tuned_tau_quantile": float(tuned_cfg.tau_quantile),
        "tune_info": tune_info,
        "cv_median_best_iters": cv_best_iters,
        "final_train_size": int(len(X_tr)),
    }
    if adversarial is not None:
        diagnostics["adversarial_auc"] = float(adversarial.overall_auc)
        diagnostics["adversarial_fold_aucs"] = [float(x) for x in adversarial.fold_aucs]

    return TailoredResult(
        submission=submission,
        rankings=rankings,
        train_diagnostics=diagnostics,
        feature_importances=feat_importance,
        adversarial=adversarial,
        tuned_sizing=tuned_cfg,
        config=config,
    )

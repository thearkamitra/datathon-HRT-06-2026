"""Adversarial validation diagnostic (Stage 9 of the plan).

Trains a LightGBM / fallback binary classifier to distinguish *train* rows
from *test* rows on the engineered feature table. The classifier's OOF ROC-AUC
is a standard practitioner proxy for covariate shift:

* AUC close to 0.5  -> distributions are similar (no obvious shift).
* AUC close to 1.0  -> obvious shift - the model is seeing wildly different
  feature distributions at inference time.

This is a *guardrail*, not a split mechanism. The plan explicitly recommends
session-level KFold for the primary CV and uses adversarial validation only to
flag features whose importance in this classifier is disproportionate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:  # pragma: no cover
        return float("nan")


@dataclass
class AdversarialReport:
    overall_auc: float
    fold_aucs: List[float]
    top_features: pd.DataFrame  # feature, importance


def _fit_binary(X: np.ndarray, y: np.ndarray, seed: int):
    if _HAS_LGB:
        params = dict(
            objective="binary",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            verbosity=-1,
            seed=seed,
            deterministic=True,
        )
        dtr = lgb.Dataset(X, label=y, free_raw_data=False)
        return lgb.train(
            params=params,
            train_set=dtr,
            num_boost_round=400,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    if _HAS_XGB:  # pragma: no cover
        m = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            objective="binary:logistic",
        )
        m.fit(X, y)
        return m
    raise RuntimeError("No boosting backend available for adversarial validation.")


def _predict_binary(model, X: np.ndarray) -> np.ndarray:
    if _HAS_LGB and isinstance(model, lgb.Booster):
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def run_adversarial(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 0,
    top_k: int = 15,
) -> AdversarialReport:
    """Return AUC + top-``top_k`` most shift-contributing features."""
    common = [c for c in train_features.columns if c in test_features.columns and c != "session"]
    Xa = train_features[common].to_numpy(dtype=np.float64)
    Xb = test_features[common].to_numpy(dtype=np.float64)
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X), dtype=np.float64)
    aucs: List[float] = []
    for fold_idx, (tr, va) in enumerate(kf.split(X)):
        model = _fit_binary(X[tr], y[tr], random_state + fold_idx)
        oof[va] = _predict_binary(model, X[va])
        aucs.append(_safe_auc(y[va], oof[va]))

    overall_auc = _safe_auc(y, oof)

    # Fit one final model on everything to rank shift-driving features.
    final = _fit_binary(X, y, random_state + 999)
    if _HAS_LGB and isinstance(final, lgb.Booster):
        imp = final.feature_importance(importance_type="gain")
    else:
        imp = getattr(final, "feature_importances_", np.zeros(len(common)))
    top = (
        pd.DataFrame({"feature": common, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return AdversarialReport(overall_auc=overall_auc, fold_aucs=aucs, top_features=top)

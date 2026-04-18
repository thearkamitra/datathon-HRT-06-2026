"""Three-head tabular booster for the tailored modeler (Stage 5 of the plan).

The Phase-1 blueprint in ``plans/Brainstorming-models.pdf`` prescribes three
parallel heads to feed the Sharpe-aware sizing layer:

* **M1 - mean head**: LightGBM regression (Huber loss) -> ``mu_hat``.
* **M2 - sign head**: LightGBM binary classification -> ``p_hat = P(R > 0)``.
* **M3 - uncertainty head**: three LightGBM quantile models
  (``q10``, ``q50``, ``q90``) -> uncertainty ``u = q90 - q10``.

Why three heads rather than ``mu / sigma^2`` alone? With only 1000 labelled
sessions, point-estimate regression fed directly into Kelly sizing flips sign
whenever ``mu_hat`` is noisy, wiping out the positive drift the leaderboard
sessions inherit. The dedicated sign classifier and quantile spread let the
sizer in ``sizing.py`` apply a confidence gate (``edge = mu * (2p - 1)`` and
``tanh`` / indicator squashing) that keeps the model long by default and only
risks capital when all three heads concur.

Design choices mirroring the plan:

* Small trees (``num_leaves <= 31``, ``max_depth <= 6``) to protect against
  overfitting the 1000-row training set.
* Deterministic seeds per fold / per head.
* Optional clipped economic sample weights ``w_i = min(c, a + b * |y_i|)``
  (Stage 6 of the plan) to emphasise economically meaningful sessions.
* Clean XGBoost fallback so the code runs even if LightGBM isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold

try:  # LightGBM is the primary backend recommended by the plan.
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover - fallback only
    _HAS_LGB = False

try:  # Secondary backend used only when LightGBM is unavailable.
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_XGB = True
except Exception:  # pragma: no cover - fallback only
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Sample-weighting (Stage 6 of the plan).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleWeightConfig:
    """Clipped economic sample weights ``w_i = min(cap, a + b * |y_i|)``."""

    enabled: bool = True
    a: float = 1.0
    b: float = 0.5
    cap: float = 4.0

    def compute(self, y: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        y = np.asarray(y, dtype=np.float64)
        scale = float(np.std(y, ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        w = self.a + self.b * (np.abs(y) / scale)
        w = np.minimum(w, self.cap)
        w = np.maximum(w, 0.0)
        return w.astype(np.float64)


# ---------------------------------------------------------------------------
# Booster configuration and factory.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoosterHyper:
    """Default LightGBM hyper-parameters for the three heads.

    These are the defaults that produced the public-leaderboard 2.24 run.
    """

    n_estimators: int = 1500
    learning_rate: float = 0.02
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 30
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.0
    lambda_l2: float = 1.0
    early_stopping_rounds: int = 100


_DEFAULT_HYPER = BoosterHyper()


def _lgb_params(
    objective: str,
    alpha: Optional[float],
    random_state: int,
    hyper: BoosterHyper,
) -> Dict:
    params = dict(
        objective=objective,
        learning_rate=hyper.learning_rate,
        num_leaves=hyper.num_leaves,
        max_depth=hyper.max_depth,
        min_data_in_leaf=hyper.min_data_in_leaf,
        feature_fraction=hyper.feature_fraction,
        bagging_fraction=hyper.bagging_fraction,
        bagging_freq=hyper.bagging_freq,
        lambda_l1=hyper.lambda_l1,
        lambda_l2=hyper.lambda_l2,
        verbosity=-1,
        seed=random_state,
        deterministic=True,
    )
    if alpha is not None:
        params["alpha"] = float(alpha)
    return params


def _fit_lgb(
    params: Dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    n_estimators: int,
    early_stopping_rounds: int,
    sample_weight: Optional[np.ndarray] = None,
):
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight, free_raw_data=False)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: List = []
    if X_va is not None and y_va is not None and early_stopping_rounds > 0:
        valid_set = lgb.Dataset(
            X_va, label=y_va, reference=train_set, free_raw_data=False
        )
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )
    callbacks.append(lgb.log_evaluation(period=0))
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


# ---------------------------------------------------------------------------
# XGBoost fallback stubs (keeps the module importable without LightGBM).
# ---------------------------------------------------------------------------


def _xgb_regressor(objective: str, alpha: Optional[float], random_state: int, hyper: BoosterHyper):
    kwargs = dict(
        n_estimators=hyper.n_estimators,
        learning_rate=hyper.learning_rate,
        max_depth=hyper.max_depth,
        min_child_weight=max(1.0, hyper.min_data_in_leaf / 10.0),
        subsample=hyper.bagging_fraction,
        colsample_bytree=hyper.feature_fraction,
        reg_alpha=hyper.lambda_l1,
        reg_lambda=hyper.lambda_l2,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
    )
    if objective == "quantile":
        kwargs["objective"] = "reg:quantileerror"
        kwargs["quantile_alpha"] = float(alpha)
    elif objective == "huber":
        kwargs["objective"] = "reg:pseudohubererror"
    elif objective == "binary":
        kwargs["objective"] = "binary:logistic"
        return XGBClassifier(**kwargs)
    else:
        kwargs["objective"] = "reg:squarederror"
    return XGBRegressor(**kwargs)


# ---------------------------------------------------------------------------
# Three-head bundle.
# ---------------------------------------------------------------------------


@dataclass
class TabularHeads:
    random_state: int = 0
    quantiles: tuple = (0.1, 0.5, 0.9)
    hyper: BoosterHyper = field(default_factory=lambda: _DEFAULT_HYPER)
    sample_weights: SampleWeightConfig = field(default_factory=SampleWeightConfig)
    min_u: float = 1e-4

    mean_model_: object = field(default=None, init=False, repr=False)
    sign_model_: object = field(default=None, init=False, repr=False)
    quantile_models_: Dict[float, object] = field(
        default_factory=dict, init=False, repr=False
    )
    feature_names_: List[str] = field(default_factory=list, init=False, repr=False)
    best_iters_: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabularHeads":
        self.feature_names_ = list(X.columns)
        X_ = X.to_numpy(dtype=np.float64)
        y_ = np.asarray(y, dtype=np.float64)
        sign_y = (y_ > 0).astype(np.int32)
        sw = self.sample_weights.compute(y_)

        X_va_np = X_val[self.feature_names_].to_numpy(dtype=np.float64) if X_val is not None else None
        y_va_np = np.asarray(y_val, dtype=np.float64) if y_val is not None else None
        sign_va = (y_va_np > 0).astype(np.int32) if y_va_np is not None else None

        if _HAS_LGB:
            mean_params = _lgb_params("huber", None, self.random_state, self.hyper)
            self.mean_model_ = _fit_lgb(
                mean_params, X_, y_, X_va_np, y_va_np,
                self.hyper.n_estimators, self.hyper.early_stopping_rounds, sw,
            )
            self.best_iters_["mean"] = int(
                self.mean_model_.best_iteration or self.mean_model_.current_iteration()
            )

            sign_params = _lgb_params("binary", None, self.random_state + 11, self.hyper)
            self.sign_model_ = _fit_lgb(
                sign_params, X_, sign_y.astype(np.float64), X_va_np,
                sign_va.astype(np.float64) if sign_va is not None else None,
                self.hyper.n_estimators, self.hyper.early_stopping_rounds, sw,
            )
            self.best_iters_["sign"] = int(
                self.sign_model_.best_iteration or self.sign_model_.current_iteration()
            )

            self.quantile_models_ = {}
            for qi, q in enumerate(self.quantiles):
                qp = _lgb_params("quantile", q, self.random_state + 101 + qi, self.hyper)
                booster = _fit_lgb(
                    qp, X_, y_, X_va_np, y_va_np,
                    self.hyper.n_estimators, self.hyper.early_stopping_rounds, sw,
                )
                self.quantile_models_[float(q)] = booster
                self.best_iters_[f"q{int(q * 100):02d}"] = int(
                    booster.best_iteration or booster.current_iteration()
                )
        elif _HAS_XGB:  # pragma: no cover - fallback
            mean = _xgb_regressor("huber", None, self.random_state, self.hyper)
            mean.fit(X_, y_, sample_weight=sw)
            self.mean_model_ = mean

            sign = _xgb_regressor("binary", None, self.random_state + 11, self.hyper)
            sign.fit(X_, sign_y, sample_weight=sw)
            self.sign_model_ = sign

            self.quantile_models_ = {}
            for qi, q in enumerate(self.quantiles):
                m = _xgb_regressor("quantile", q, self.random_state + 101 + qi, self.hyper)
                m.fit(X_, y_, sample_weight=sw)
                self.quantile_models_[float(q)] = m
        else:
            raise RuntimeError(
                "Neither LightGBM nor XGBoost is available. Install `lightgbm`."
            )

        return self

    # -------------------------------------------------------------- predict
    def _lgb_predict(self, booster, X: np.ndarray) -> np.ndarray:
        return booster.predict(X, num_iteration=booster.best_iteration)

    def _xgb_predict(self, model, X: np.ndarray, proba: bool = False) -> np.ndarray:
        if proba:
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_model_ is None:
            raise RuntimeError("TabularHeads.fit must be called before predict.")
        X_ = X[self.feature_names_].to_numpy(dtype=np.float64)
        if _HAS_LGB and isinstance(self.mean_model_, lgb.Booster):
            mu = self._lgb_predict(self.mean_model_, X_)
            p = self._lgb_predict(self.sign_model_, X_)
            qs = {q: self._lgb_predict(m, X_) for q, m in self.quantile_models_.items()}
        else:  # pragma: no cover - fallback only
            mu = self._xgb_predict(self.mean_model_, X_)
            p = self._xgb_predict(self.sign_model_, X_, proba=True)
            qs = {q: self._xgb_predict(m, X_) for q, m in self.quantile_models_.items()}

        q_lo = qs[float(self.quantiles[0])]
        q_med = qs[float(self.quantiles[1])]
        q_hi = qs[float(self.quantiles[-1])]
        q_lo_c = np.minimum(q_lo, q_hi)
        q_hi_c = np.maximum(q_lo, q_hi)
        u = np.maximum(q_hi_c - q_lo_c, self.min_u)
        return pd.DataFrame(
            {
                "mu": mu,
                "p_up": np.clip(p, 0.0, 1.0),
                "q_lower": q_lo_c,
                "q_median": q_med,
                "q_upper": q_hi_c,
                "u": u,
            }
        )

    # -------------------------------------------------------- cross-val OOF
    def cross_val_predict(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 1,
        return_folds: bool = False,
    ):
        """Out-of-fold mu / p / quantiles averaged across repeats.

        If ``return_folds`` is True the method also returns a ``fold_groups``
        array marking each row with its (first-repeat) validation fold, which
        the sizing-layer tuner consumes to compute per-fold Sharpe and pick
        robust configurations.
        """
        self.feature_names_ = list(X.columns)
        X_ = X.to_numpy(dtype=np.float64)
        y_ = np.asarray(y, dtype=np.float64)
        n = y_.size

        mu_oof = np.zeros(n, dtype=np.float64)
        p_oof = np.zeros(n, dtype=np.float64)
        q_oof = {float(q): np.zeros(n, dtype=np.float64) for q in self.quantiles}
        counts = np.zeros(n, dtype=np.float64)
        fold_groups = np.full(n, -1, dtype=np.int64)

        splitter = (
            RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
            if n_repeats > 1
            else KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        )

        for repeat_fold_idx, (tr, va) in enumerate(splitter.split(X_)):
            fold_idx = repeat_fold_idx % n_splits
            if fold_groups[va[0]] == -1:
                fold_groups[va] = fold_idx  # record the first-repeat labelling
            heads = TabularHeads(
                random_state=self.random_state + repeat_fold_idx,
                quantiles=self.quantiles,
                hyper=self.hyper,
                sample_weights=self.sample_weights,
                min_u=self.min_u,
            )
            heads.fit(
                X.iloc[tr],
                y_[tr],
                X_val=X.iloc[va],
                y_val=y_[va],
            )
            preds = heads.predict(X.iloc[va])
            mu_oof[va] += preds["mu"].to_numpy()
            p_oof[va] += preds["p_up"].to_numpy()
            for q in self.quantiles:
                col = {0.1: "q_lower", 0.5: "q_median", 0.9: "q_upper"}.get(float(q))
                if col is not None:
                    q_oof[float(q)][va] += preds[col].to_numpy()
            counts[va] += 1.0

        counts = np.maximum(counts, 1.0)
        mu_oof /= counts
        p_oof /= counts
        for q in self.quantiles:
            q_oof[float(q)] /= counts

        q_lo = q_oof[float(self.quantiles[0])]
        q_med = q_oof[float(self.quantiles[1])]
        q_hi = q_oof[float(self.quantiles[-1])]
        q_lo_c = np.minimum(q_lo, q_hi)
        q_hi_c = np.maximum(q_lo, q_hi)
        u = np.maximum(q_hi_c - q_lo_c, self.min_u)
        preds_df = pd.DataFrame(
            {
                "mu": mu_oof,
                "p_up": np.clip(p_oof, 0.0, 1.0),
                "q_lower": q_lo_c,
                "q_median": q_med,
                "q_upper": q_hi_c,
                "u": u,
            }
        )
        if return_folds:
            return preds_df, fold_groups
        return preds_df

    # --------------------------------------------------- feature importance
    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.mean_model_ is None:
            return None
        if _HAS_LGB and isinstance(self.mean_model_, lgb.Booster):
            imp = self.mean_model_.feature_importance(importance_type="gain")
        else:
            imp = getattr(self.mean_model_, "feature_importances_", None)
            if imp is None:
                return None
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def backend_name() -> str:
    if _HAS_LGB:
        return "lightgbm"
    if _HAS_XGB:
        return "xgboost"
    return "none"

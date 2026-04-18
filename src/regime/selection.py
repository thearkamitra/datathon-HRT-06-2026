"""Model selection for the pooled Gaussian HMM.

The plan calls out a two-level rule:

* **Level 1 - statistical fit**: track log-likelihood, AIC, and BIC across the
  candidate state counts (2, 3, 4, 5) and covariance structures.
* **Level 2 - task fit**: for each fitted HMM, run session-level cross
  validation that (a) fits the HMM on the training fold only, (b) forecasts
  the held-out sessions' second-half return distributions, and (c) scores the
  sized positions against the competition Sharpe metric.

Only Level-2 winners that are also statistically plausible (BIC within a
small tolerance of the best BIC) are promoted. Ties break toward the smaller
state count for parsimony.

This module is orchestration only: it reuses
:func:`hmm_model.fit_pooled_gaussian_hmm` and
:func:`forecast.forecast_sessions_mc` so there is a single code path for
training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from emissions import EmissionBundle, EmissionConfig
from forecast import MCConfig, forecast_sessions_mc
from hmm_model import HMMBundle, HMMHyper, fit_pooled_gaussian_hmm
from progress import log as _log, tick as _tick
from sizing import SizingConfig, apply_sizing, sharpe, tune_sizing


@dataclass(frozen=True)
class SelectionGrid:
    """Search grid for state counts and covariance structures.

    The defaults follow the plan's recommendation (2-5 states, diagonal
    covariance as the primary entry because the emission dimensionality is
    small and diagonal covariance is the most stable choice on 50 bars of
    simulated OHLC).
    """

    n_components: Tuple[int, ...] = (2, 3, 4)
    covariance_types: Tuple[str, ...] = ("diag",)
    n_starts: int = 4


@dataclass(frozen=True)
class SelectionConfig:
    grid: SelectionGrid = field(default_factory=SelectionGrid)
    cv_splits: int = 4
    cv_random_state: int = 0
    # Absolute BIC tolerance: candidates within ``best_bic + bic_tol`` are
    # eligible to win based on CV Sharpe. Set very large to disable.
    bic_tol: float = 50.0
    # Sharpe improvement below this threshold does *not* justify a larger
    # state count (parsimony bias).
    sharpe_tiebreak_tol: float = 0.01


@dataclass
class SelectionResult:
    bundle: HMMBundle
    n_components: int
    covariance_type: str
    mean_cv_sharpe: float
    fold_sharpes: List[float]
    aic: float
    bic: float
    log_likelihood: float
    all_runs: List[dict] = field(default_factory=list)
    # Fold-val MC forecast predictions from the winning candidate, concatenated
    # across folds (sorted by session). Used as honest OOF predictions so we
    # don't run a second, redundant CV loop over the same training sessions.
    winner_oof: Optional[pd.DataFrame] = None

    def as_dict(self) -> dict:
        return {
            "n_components": int(self.n_components),
            "covariance_type": str(self.covariance_type),
            "mean_cv_sharpe": float(self.mean_cv_sharpe),
            "fold_sharpes": [float(x) for x in self.fold_sharpes],
            "aic": float(self.aic),
            "bic": float(self.bic),
            "log_likelihood": float(self.log_likelihood),
            "all_runs": self.all_runs,
        }


def _fold_sharpe_for_candidate(
    bundle: EmissionBundle,
    y: np.ndarray,
    hyper: HMMHyper,
    mc_cfg: MCConfig,
    sizing_cfg: SizingConfig,
    n_splits: int,
    random_state: int,
    emission_cfg: EmissionConfig,
    seen_bars: Optional[int] = None,
) -> Tuple[float, List[float], pd.DataFrame]:
    """Run session-level KFold: fit HMM on fold-train, forecast fold-val, Sharpe.

    Returns ``(mean_sharpe, per_fold_sharpes, concatenated_fold_val_predictions)``.
    The third element is the honest OOF frame for the winning candidate (each
    training session is predicted exactly once, when it sits in the
    validation fold).
    """
    sessions = bundle.sessions
    n = sessions.size
    if n < max(4, n_splits):
        # Fall back to a single train/val split when the data are very small.
        order = np.arange(n)
        split = max(1, int(n * 0.2))
        va_idx = order[-split:]
        tr_idx = order[:-split]
        indices = [(tr_idx, va_idx)]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices = list(kf.split(order := np.arange(n)))
    del order

    return_ix = emission_cfg.return_index()
    fold_scores: List[float] = []
    fold_frames: List[pd.DataFrame] = []
    n_folds = len(indices)
    for fold_ix, (tr_idx, va_idx) in enumerate(indices, start=1):
        # Build fold-train concatenation.
        tr_X = np.concatenate([bundle.per_session[i].features for i in tr_idx], axis=0)
        tr_lengths = np.array(
            [bundle.per_session[i].features.shape[0] for i in tr_idx],
            dtype=np.int64,
        )
        try:
            hmm_fold = fit_pooled_gaussian_hmm(tr_X, tr_lengths, hyper=hyper)
        except Exception as exc:
            fold_scores.append(float("nan"))
            _tick("selection", fold_ix, n_folds, f"CV fold failed: {type(exc).__name__}")
            continue

        va_sessions = [bundle.per_session[i] for i in va_idx]
        preds_va = forecast_sessions_mc(
            hmm_fold,
            va_sessions,
            return_index=return_ix,
            config=mc_cfg,
            seen_bars=seen_bars,
        )
        y_va = y[va_idx]
        # preds come back in va_idx order -> align with y_va directly before sorting.
        positions, _ = apply_sizing(preds_va, sizing_cfg)
        s = float(sharpe(positions * y_va))
        fold_scores.append(s)
        fold_frames.append(preds_va)
        _tick(
            "selection",
            fold_ix,
            n_folds,
            f"CV fold ({len(tr_idx)} train / {len(va_idx)} val) Sharpe={s:+.3f} "
            f"[HMM iters={int(getattr(hmm_fold.model.monitor_, 'iter', 0))} "
            f"converged={hmm_fold.converged}]",
        )

    finite = [s for s in fold_scores if np.isfinite(s)]
    mean_sharpe = float(np.mean(finite)) if finite else float("-inf")
    oof_frame = (
        pd.concat(fold_frames, axis=0, ignore_index=True)
        .sort_values("session")
        .reset_index(drop=True)
        if fold_frames
        else pd.DataFrame()
    )
    return mean_sharpe, fold_scores, oof_frame


def select_best_hmm(
    bundle: EmissionBundle,
    y: np.ndarray,
    base_hyper: HMMHyper,
    emission_cfg: EmissionConfig,
    mc_cfg: MCConfig,
    sizing_cfg: SizingConfig = SizingConfig(),
    select_cfg: SelectionConfig = SelectionConfig(),
    seen_bars: Optional[int] = None,
) -> SelectionResult:
    """Grid-search ``(n_components, covariance_type)`` by BIC + CV Sharpe.

    Parameters
    ----------
    bundle:
        Training :class:`EmissionBundle`.
    y:
        Per-session realized returns aligned with ``bundle.sessions``.
    base_hyper:
        Shared non-searched HMM hyperparameters (n_iter, tol, n_starts, ...).
    emission_cfg:
        Configuration of the emission featurizer (used to locate the
        log-return channel for the Monte-Carlo forecaster).
    mc_cfg:
        Monte-Carlo continuation configuration for the forecaster used during
        CV scoring.
    sizing_cfg:
        Sizing configuration applied during CV Sharpe computation. We do not
        re-tune the sizer inside the grid: a fixed "z_tanh with reasonable
        defaults" configuration keeps the selection signal focused on the
        HMM's forecasting quality rather than sizer tuning noise. The
        pipeline tunes the sizer on OOF predictions of the winning model.
    """
    if bundle.sessions.size != y.size:
        raise ValueError("y must be aligned with bundle.sessions (same length & order)")

    grid = select_cfg.grid
    runs: List[dict] = []
    candidates: List[Tuple[HMMHyper, HMMBundle, float, List[float], pd.DataFrame]] = []

    total_candidates = len(grid.covariance_types) * len(grid.n_components)
    _log(
        "selection",
        f"grid = {total_candidates} candidates "
        f"(K in {list(grid.n_components)}, cov in {list(grid.covariance_types)}), "
        f"CV = {select_cfg.cv_splits}-fold, starts = {grid.n_starts}",
    )
    cand_ix = 0
    for cov_type in grid.covariance_types:
        for K in grid.n_components:
            cand_ix += 1
            hyper = replace(
                base_hyper,
                n_components=int(K),
                covariance_type=str(cov_type),
                n_starts=int(grid.n_starts),
            )
            _log(
                "selection",
                f"candidate {cand_ix}/{total_candidates}: K={K} cov={cov_type} "
                f"-> fitting pooled HMM on {bundle.sessions.size} sessions",
            )
            try:
                bundle_full = fit_pooled_gaussian_hmm(
                    bundle.X,
                    bundle.lengths,
                    hyper=hyper,
                    progress_tag="selection",
                )
            except Exception as exc:
                runs.append(
                    {
                        "n_components": int(K),
                        "covariance_type": str(cov_type),
                        "error": repr(exc),
                    }
                )
                _log(
                    "selection",
                    f"candidate {cand_ix}/{total_candidates}: FAILED ({type(exc).__name__})",
                )
                continue
            _log(
                "selection",
                f"candidate {cand_ix}/{total_candidates}: pooled-fit LL={bundle_full.log_likelihood:,.0f} "
                f"BIC={bundle_full.bic:,.0f} AIC={bundle_full.aic:,.0f} "
                f"converged={bundle_full.converged}",
            )
            mean_sharpe, fold_sharpes, fold_preds = _fold_sharpe_for_candidate(
                bundle=bundle,
                y=y,
                hyper=hyper,
                mc_cfg=mc_cfg,
                sizing_cfg=sizing_cfg,
                n_splits=select_cfg.cv_splits,
                random_state=select_cfg.cv_random_state,
                emission_cfg=emission_cfg,
                seen_bars=seen_bars,
            )
            _log(
                "selection",
                f"candidate {cand_ix}/{total_candidates}: K={K} cov={cov_type} "
                f"mean CV Sharpe = {mean_sharpe:+.4f} (folds {fold_sharpes})",
            )
            runs.append(
                {
                    "n_components": int(K),
                    "covariance_type": str(cov_type),
                    "log_likelihood": float(bundle_full.log_likelihood),
                    "aic": float(bundle_full.aic),
                    "bic": float(bundle_full.bic),
                    "mean_cv_sharpe": float(mean_sharpe),
                    "fold_sharpes": [float(x) for x in fold_sharpes],
                    "converged": bool(bundle_full.converged),
                }
            )
            candidates.append((hyper, bundle_full, mean_sharpe, fold_sharpes, fold_preds))

    if not candidates:
        raise RuntimeError("No HMM candidate fit successfully during selection")

    best_bic = min(b.bic for _, b, _, _, _ in candidates)
    eligible = [c for c in candidates if c[1].bic <= best_bic + select_cfg.bic_tol]
    if not eligible:
        eligible = list(candidates)

    # Pick the Sharpe-maximising candidate; break ties by fewer states (BIC-lower on tie).
    def _key(entry):
        hyper, b, sharpe_val, _, _ = entry
        return (
            -(sharpe_val if np.isfinite(sharpe_val) else -1e18),
            int(hyper.n_components),
            float(b.bic),
        )

    eligible.sort(key=_key)
    winner = eligible[0]

    # Parsimony override: if a smaller-K candidate is within sharpe_tiebreak_tol
    # of the winner's Sharpe, prefer the smaller model.
    for cand in eligible:
        if cand[0].n_components < winner[0].n_components and np.isfinite(cand[2]):
            if winner[2] - cand[2] <= select_cfg.sharpe_tiebreak_tol:
                winner = cand
                break

    hyper, b, sharpe_val, fold_sharpes, fold_preds = winner
    _log(
        "selection",
        f"winner: K={hyper.n_components} cov={hyper.covariance_type} "
        f"CV Sharpe={sharpe_val:+.4f} BIC={b.bic:,.0f}",
    )
    return SelectionResult(
        bundle=b,
        n_components=int(hyper.n_components),
        covariance_type=str(hyper.covariance_type),
        mean_cv_sharpe=float(sharpe_val),
        fold_sharpes=[float(x) for x in fold_sharpes],
        aic=float(b.aic),
        bic=float(b.bic),
        log_likelihood=float(b.log_likelihood),
        all_runs=runs,
        winner_oof=fold_preds,
    )

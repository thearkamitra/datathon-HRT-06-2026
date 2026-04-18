"""Gaussian HMM wrapper with multi-start fitting and information criteria.

The pooled Method-1 model is a :class:`hmmlearn.hmm.GaussianHMM` trained on
the row-wise concatenation of all training sessions. EM (Baum-Welch) is
local-optimum sensitive, so :func:`fit_pooled_gaussian_hmm` runs several
random restarts and keeps the highest-likelihood converged model. The fit
returns an :class:`HMMBundle` that tracks:

* the fitted sklearn-compatible HMM object,
* the full multi-sequence training log-likelihood, AIC, BIC,
* the state-wise emission means / covariances / mixing,
* the starting-state distribution and transition matrix.

The wrapper deliberately stays very thin: model selection
(:mod:`selection`), inference (:mod:`forecast`) and clustering
(:mod:`clustering`) all consume this bundle rather than the raw hmmlearn
object so the pipeline stays swappable with other HMM backends later.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - surfaced as an import-time error
    raise ImportError(
        "hmmlearn is required for the regime pipeline. Install with "
        "`pip install --user hmmlearn`."
    ) from exc


# hmmlearn emits a ``logging.warning`` on every EM iteration whose
# log-likelihood is not *strictly* greater than the previous iteration's. For
# well-fit models near the EM fixed point, those "deltas" are dominated by
# floating-point noise (e.g. ``Delta is -0.0006`` on an LL of ~1.4M). The
# warnings are harmless and make terminal output unreadable during the
# selection / CV phases, so we silence the ``hmmlearn`` logger by default.
# The fit's actual convergence status stays reachable via
# ``HMMBundle.converged`` / ``.monitor_.converged``.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)


@dataclass(frozen=True)
class HMMHyper:
    """Hyperparameters for the pooled Gaussian HMM."""

    n_components: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-4
    min_covar: float = 1e-4
    n_starts: int = 5
    init_params: str = "stmc"   # start prob, transmat, means, covars all init-ed
    params: str = "stmc"        # fit all of them
    random_state: int = 0
    # If True, floor the estimated covariance after fitting to ``min_covar``
    # times the empirical feature variance. Helps on tiny states.
    floor_covariance: bool = True


@dataclass
class HMMBundle:
    """Fitted pooled Gaussian HMM plus book-keeping for downstream consumers."""

    model: GaussianHMM
    hyper: HMMHyper
    log_likelihood: float
    n_obs: int
    n_features: int
    aic: float
    bic: float
    starts: int
    converged: bool
    all_log_likelihoods: List[float] = field(default_factory=list)
    all_converged: List[bool] = field(default_factory=list)

    # Shortcut accessors ---------------------------------------------------
    @property
    def n_states(self) -> int:
        return int(self.model.n_components)

    @property
    def means(self) -> np.ndarray:
        return np.asarray(self.model.means_, dtype=np.float64)

    @property
    def covars(self) -> np.ndarray:
        # hmmlearn exposes the appropriate shape for the covariance_type
        return np.asarray(self.model.covars_, dtype=np.float64)

    @property
    def transmat(self) -> np.ndarray:
        return np.asarray(self.model.transmat_, dtype=np.float64)

    @property
    def startprob(self) -> np.ndarray:
        return np.asarray(self.model.startprob_, dtype=np.float64)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _free_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """Approximate free-parameter count for AIC / BIC of a Gaussian HMM.

    We count:
    * start probabilities: ``K - 1``
    * transition matrix:   ``K * (K - 1)``
    * emission means:      ``K * F``
    * emission covariances:
        - ``diag``:     ``K * F``
        - ``full``:     ``K * F * (F + 1) / 2``
        - ``spherical``: ``K``
        - ``tied``:     ``F * (F + 1) / 2``
    This closely follows the formulation in the hmmlearn AIC/BIC example.
    """
    params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features
    cov_type = covariance_type.lower()
    if cov_type == "diag":
        params += n_states * n_features
    elif cov_type == "full":
        params += n_states * n_features * (n_features + 1) // 2
    elif cov_type == "spherical":
        params += n_states
    elif cov_type == "tied":
        params += n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type!r}")
    return params


def _build_model(hyper: HMMHyper, seed: int) -> GaussianHMM:
    return GaussianHMM(
        n_components=int(hyper.n_components),
        covariance_type=str(hyper.covariance_type),
        n_iter=int(hyper.n_iter),
        tol=float(hyper.tol),
        min_covar=float(hyper.min_covar),
        init_params=str(hyper.init_params),
        params=str(hyper.params),
        random_state=int(seed),
        verbose=False,
    )


def _apply_covariance_floor(model: GaussianHMM, X: np.ndarray, eps: float) -> None:
    """Floor the estimated covariance to a fraction of the empirical variance.

    Prevents numerical catastrophes when a state collapses onto a small number
    of near-identical points and the estimated variance heads toward zero.

    hmmlearn's public ``covars_`` property expands ``'diag'`` covariances to
    full matrices on read but requires the compact ``(K, F)`` shape on write,
    so we go through the internal ``_covars_`` attribute which always matches
    the storage shape for the active covariance type.
    """
    if X.size == 0:
        return
    var = np.var(X, axis=0)
    floor_vec = np.maximum(var * eps, 1e-8)
    cov_type = model.covariance_type
    internal = np.asarray(model._covars_, dtype=np.float64).copy()
    if cov_type == "diag":
        # shape (K, F)
        internal = np.maximum(internal, floor_vec)
    elif cov_type == "full":
        # shape (K, F, F): floor only the diagonal
        for k in range(internal.shape[0]):
            internal[k].flat[:: internal.shape[1] + 1] = np.maximum(
                np.diag(internal[k]), floor_vec
            )
    elif cov_type == "spherical":
        # shape (K,)
        internal = np.maximum(internal, float(floor_vec.mean()))
    elif cov_type == "tied":
        # shape (F, F)
        internal.flat[:: internal.shape[0] + 1] = np.maximum(np.diag(internal), floor_vec)
    else:
        return
    model._covars_ = internal


def fit_pooled_gaussian_hmm(
    X: np.ndarray,
    lengths: Iterable[int],
    hyper: HMMHyper = HMMHyper(),
    *,
    progress_tag: Optional[str] = None,
) -> HMMBundle:
    """Fit a pooled Gaussian HMM to the multi-sequence training data.

    Runs ``hyper.n_starts`` EM restarts with distinct random seeds derived from
    ``hyper.random_state`` and keeps the highest-likelihood converged fit.
    Falls back to the highest-likelihood *any* fit when no restart converges
    (which should be rare for 1000 sequences of length 50).

    Passing ``progress_tag`` logs a one-line summary per EM restart to
    :mod:`progress` so the user can see the fit is advancing.
    """
    from progress import log as _log_progress

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    lengths = np.asarray(list(lengths), dtype=np.int64)
    if X.size == 0 or lengths.size == 0:
        raise ValueError("fit_pooled_gaussian_hmm requires non-empty training data")
    n_obs, n_features = X.shape

    best: Optional[Tuple[float, GaussianHMM, bool]] = None
    all_lls: List[float] = []
    all_conv: List[bool] = []

    for start_ix in range(max(1, hyper.n_starts)):
        seed = int(hyper.random_state + 10007 * start_ix)
        model = _build_model(hyper, seed=seed)
        with warnings.catch_warnings():
            # hmmlearn prints ConvergenceWarnings for marginal fits; we record
            # convergence ourselves via ``monitor_``.
            warnings.simplefilter("ignore")
            try:
                model.fit(X, lengths)
            except Exception as exc:  # pragma: no cover - surfaced in diagnostics
                all_lls.append(float("-inf"))
                all_conv.append(False)
                if best is None:
                    best = (float("-inf"), model, False)
                if progress_tag is not None:
                    _log_progress(
                        progress_tag,
                        f"restart {start_ix + 1}/{hyper.n_starts} failed: {type(exc).__name__}",
                    )
                continue
        if hyper.floor_covariance:
            _apply_covariance_floor(model, X, hyper.min_covar)
        try:
            ll = float(model.score(X, lengths))
        except Exception:  # pragma: no cover - fallback for degenerate fits
            ll = float("-inf")
        converged = bool(getattr(model.monitor_, "converged", False))
        iters = int(getattr(model.monitor_, "iter", 0))
        all_lls.append(ll)
        all_conv.append(converged)

        better = (
            best is None
            or (converged and not best[2] and np.isfinite(ll))
            or (converged == best[2] and np.isfinite(ll) and ll > best[0])
        )
        if better:
            best = (ll, model, converged)

        if progress_tag is not None:
            _log_progress(
                progress_tag,
                f"restart {start_ix + 1}/{hyper.n_starts}: "
                f"LL={ll:,.0f} iters={iters} converged={converged}",
            )

    assert best is not None
    best_ll, best_model, best_conv = best

    n_params = _free_parameters(hyper.n_components, n_features, hyper.covariance_type)
    aic = 2.0 * n_params - 2.0 * best_ll
    bic = n_params * float(np.log(max(n_obs, 2))) - 2.0 * best_ll

    return HMMBundle(
        model=best_model,
        hyper=hyper,
        log_likelihood=float(best_ll),
        n_obs=int(n_obs),
        n_features=int(n_features),
        aic=float(aic),
        bic=float(bic),
        starts=int(max(1, hyper.n_starts)),
        converged=bool(best_conv),
        all_log_likelihoods=[float(x) for x in all_lls],
        all_converged=[bool(x) for x in all_conv],
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def posterior_at_end(bundle: HMMBundle, sequence: np.ndarray) -> np.ndarray:
    """Return the filtered posterior ``P(s_{T-1} | x_{0:T-1})`` for a single sequence."""
    if sequence.ndim != 2:
        raise ValueError("sequence must be a (T, F) matrix")
    if sequence.shape[0] == 0:
        return bundle.startprob.copy()
    gamma = bundle.model.predict_proba(sequence)
    return np.asarray(gamma[-1], dtype=np.float64)


def occupancy_features(bundle: HMMBundle, sequence: np.ndarray) -> np.ndarray:
    """Concatenate useful posterior summaries for a single sequence.

    Returns a 1-D vector with:
    * final posterior ``P(s_{T-1} | x_{0:T-1})``        (K)
    * average occupancy ``(1/T) sum_t P(s_t | x_{0:T-1})`` (K)
    * expected next-step state distribution              (K)
    * sequence log-likelihood                            (1)
    * posterior entropy of the final state               (1)
    Total length: ``3K + 2``.
    """
    if sequence.ndim != 2:
        raise ValueError("sequence must be a (T, F) matrix")
    K = bundle.n_states
    if sequence.shape[0] == 0:
        final = bundle.startprob.copy()
        occ = bundle.startprob.copy()
        next_dist = bundle.startprob @ bundle.transmat
        ent = float(-np.sum(final * np.log(np.clip(final, 1e-12, 1.0))))
        return np.concatenate([final, occ, next_dist, np.array([0.0, ent])])

    gamma = np.asarray(bundle.model.predict_proba(sequence), dtype=np.float64)
    final = gamma[-1]
    occ = gamma.mean(axis=0)
    next_dist = final @ bundle.transmat
    ll = float(bundle.model.score(sequence))
    ent = float(-np.sum(final * np.log(np.clip(final, 1e-12, 1.0))))
    return np.concatenate([final, occ, next_dist, np.array([ll, ent])])


def score_sequence(bundle: HMMBundle, sequence: np.ndarray) -> float:
    if sequence.shape[0] == 0:
        return 0.0
    return float(bundle.model.score(sequence))


def _parse_extras(kwargs: dict) -> Any:  # pragma: no cover - reserved
    """Reserved hook for future per-bundle extras (GMM-HMM, MarkovAR)."""
    return kwargs

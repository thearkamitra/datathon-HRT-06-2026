"""Method-2 model-based clustering of sessions with cluster-specific HMMs.

The Method-2 pipeline:

1. Build coarse per-session summary features (:func:`emissions.session_summary_features`).
2. Initialise ``K`` clusters via K-means over those summaries.
3. For each cluster, fit a cluster-specific pooled Gaussian HMM on the
   concatenated emissions of its member sessions.
4. Re-assign each session to the cluster whose HMM gives the highest
   log-likelihood to its seen-half sequence.
5. Iterate steps 3-4 until assignments stabilise or ``max_iter`` is reached.

At inference, we score a new session's seen-half under *every* cluster HMM and
compute soft cluster responsibilities via softmax over log-likelihoods. The
final predictive distribution is the cluster-weighted mixture of the
cluster-specific MC forecasts (see :func:`forecast.mixture_forecast`).

This is strictly more expressive than Method 1 but also less robust; the
pipeline only runs it when the ``--method m2`` flag is passed on the CLI, and
it is always compared against Method 1's OOF Sharpe before being promoted as
the submission choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from emissions import EmissionBundle, SessionEmissions, session_summary_features
from hmm_model import HMMBundle, HMMHyper, fit_pooled_gaussian_hmm, score_sequence
from progress import log as _log


@dataclass(frozen=True)
class ClusteringConfig:
    """Hyperparameters for the Method-2 clustering loop."""

    n_clusters: int = 2
    max_iter: int = 5
    min_cluster_size: int = 50
    # Temperature for the softmax over per-cluster log-likelihoods when we
    # convert them to responsibilities. Lower -> harder assignments.
    responsibility_temperature: float = 1.0
    random_state: int = 0


@dataclass
class ClusterBundle:
    """Container for a single cluster's HMM and its training members."""

    hmm: HMMBundle
    member_sessions: np.ndarray
    size: int


@dataclass
class ClusteringResult:
    clusters: List[ClusterBundle]
    assignments: np.ndarray
    responsibilities: np.ndarray
    summary_features: pd.DataFrame
    history: List[dict] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


def _kmeans_init(
    summary: pd.DataFrame,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    sessions = summary["session"].to_numpy()
    feats = summary.drop(columns=["session"]).to_numpy(dtype=np.float64)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    Z = scaler.fit_transform(feats)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)
    # Re-index: align label order with session order of ``summary``.
    return np.asarray(labels, dtype=np.int64)


def _fit_cluster_hmm(
    sessions: Sequence[SessionEmissions],
    hyper: HMMHyper,
    min_size: int,
) -> HMMBundle:
    if len(sessions) == 0:
        raise ValueError("Cannot fit a cluster HMM with zero member sessions.")
    X = np.concatenate([s.features for s in sessions], axis=0)
    lengths = np.array([s.features.shape[0] for s in sessions], dtype=np.int64)
    # Tiny clusters need fewer restarts so we don't waste EM cycles.
    effective_hyper = hyper
    if len(sessions) < min_size:
        effective_hyper = HMMHyper(
            n_components=hyper.n_components,
            covariance_type=hyper.covariance_type,
            n_iter=hyper.n_iter,
            tol=hyper.tol,
            min_covar=hyper.min_covar,
            n_starts=max(1, min(2, hyper.n_starts)),
            init_params=hyper.init_params,
            params=hyper.params,
            random_state=hyper.random_state,
            floor_covariance=hyper.floor_covariance,
        )
    return fit_pooled_gaussian_hmm(X, lengths, hyper=effective_hyper)


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _responsibilities_from_ll(
    ll_matrix: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Convert a ``(n_sessions, K)`` log-likelihood matrix to responsibilities."""
    scaled = ll_matrix / max(temperature, 1e-6)
    lse = _logsumexp(scaled, axis=1)
    return np.exp(scaled - lse[:, None])


def fit_clustered_hmms(
    bundle: EmissionBundle,
    hyper: HMMHyper,
    config: ClusteringConfig = ClusteringConfig(),
) -> ClusteringResult:
    """Iteratively cluster sessions and fit cluster-specific HMMs.

    The initial cluster assignment is drawn from K-means over coarse session
    summaries; we then run up to ``config.max_iter`` reassign/refit rounds.
    """
    per_session = bundle.per_session
    n = len(per_session)
    if n == 0:
        raise ValueError("Empty emission bundle")

    summary = session_summary_features(per_session)
    assignments = _kmeans_init(
        summary, n_clusters=config.n_clusters, random_state=config.random_state
    )
    init_sizes = [int(np.sum(assignments == k)) for k in range(config.n_clusters)]
    _log(
        "clustering",
        f"initial K-means assignment: cluster_sizes={init_sizes}",
    )

    history: List[dict] = []
    responsibilities = np.zeros((n, config.n_clusters), dtype=np.float64)
    clusters: List[ClusterBundle] = []

    for iteration in range(max(1, config.max_iter)):
        clusters = []
        cluster_lls = np.full((n, config.n_clusters), -np.inf, dtype=np.float64)
        for k in range(config.n_clusters):
            members = [per_session[i] for i in range(n) if assignments[i] == k]
            if len(members) == 0:
                # Re-seed the empty cluster with the single session whose LL is
                # worst under the current winners so clusters don't collapse.
                worst_ix = int(np.argmin(cluster_lls.max(axis=1))) if iteration > 0 else 0
                members = [per_session[worst_ix]]
                assignments[worst_ix] = k
            hmm_k = _fit_cluster_hmm(members, hyper=hyper, min_size=config.min_cluster_size)
            clusters.append(
                ClusterBundle(
                    hmm=hmm_k,
                    member_sessions=np.array([m.session for m in members], dtype=np.int64),
                    size=len(members),
                )
            )
            # Score every session under this cluster's HMM.
            for i in range(n):
                cluster_lls[i, k] = score_sequence(hmm_k, per_session[i].features)

        new_assignments = np.argmax(cluster_lls, axis=1).astype(np.int64)
        responsibilities = _responsibilities_from_ll(
            cluster_lls, temperature=config.responsibility_temperature
        )
        changed = int((new_assignments != assignments).sum())
        mean_max_resp = float(responsibilities.max(axis=1).mean())
        history.append(
            {
                "iteration": int(iteration),
                "assignments_changed": changed,
                "cluster_sizes": [int(c.size) for c in clusters],
                "mean_max_responsibility": mean_max_resp,
            }
        )
        _log(
            "clustering",
            f"iter {iteration + 1}/{config.max_iter}: sizes={[c.size for c in clusters]} "
            f"reassigned={changed} mean_max_resp={mean_max_resp:.3f} "
            f"cluster LLs={[f'{c.hmm.log_likelihood:,.0f}' for c in clusters]}",
        )
        assignments = new_assignments
        if changed == 0:
            _log("clustering", f"converged after {iteration + 1} iteration(s)")
            break

    return ClusteringResult(
        clusters=clusters,
        assignments=assignments,
        responsibilities=responsibilities,
        summary_features=summary,
        history=history,
    )


def score_sessions_against_clusters(
    clusters: Sequence[ClusterBundle],
    sessions: Sequence[SessionEmissions],
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log-likelihood matrix, responsibilities) for new sessions."""
    K = len(clusters)
    n = len(sessions)
    ll = np.full((n, K), -np.inf, dtype=np.float64)
    for k, cluster in enumerate(clusters):
        for i, sess in enumerate(sessions):
            ll[i, k] = score_sequence(cluster.hmm, sess.features)
    resp = _responsibilities_from_ll(ll, temperature=temperature)
    return ll, resp

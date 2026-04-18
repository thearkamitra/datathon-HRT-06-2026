"""Continuation forecasting and posterior-feature extraction.

Given a fitted pooled Gaussian HMM (:class:`hmm_model.HMMBundle`) and the
seen-half emissions of a session, we need to produce a predictive distribution
over the second-half return :math:`R = C_{99} / C_{49} - 1`.

This module implements the two forecasting paths called out in the plan:

* :func:`forecast_sessions_mc` - **Monte-Carlo continuation** (Option 1).
  Seeds a state path from the filtered posterior at bar 49, simulates
  ``n_sim`` continuations under the fitted HMM dynamics, sums the simulated
  log-returns across the forecast horizon, and converts them to :math:`R`.
  Returns a dataframe with ``mu / p_up / q_lower / q_median / q_upper / u``
  columns matching the sizing layer contract.

* :func:`session_posterior_features` - **State-posterior feature block**
  (Option 2 / hybrid). Summarises the seen-half posterior into a compact
  per-session feature vector (final posterior, average occupancy, expected
  next-state distribution, log-likelihood, posterior entropy). The caller
  can concatenate this with downstream tabular heads if they want a hybrid
  MC + regression head later.

The two forecasters are intentionally decoupled: Method 1 baseline runs MC
only; the hybrid upgrade plugs posterior features into a downstream LightGBM
but can still reuse everything here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from emissions import SessionEmissions
from hmm_model import HMMBundle, occupancy_features  # noqa: F401  (re-exported)
from progress import log as _log


@dataclass(frozen=True)
class MCConfig:
    """Monte-Carlo continuation knobs.

    Attributes
    ----------
    horizon:
        Number of future bars to simulate after the seen window. For this
        competition the seen / unseen halves are both 50 bars long, so the
        default matches.
    n_sim:
        Number of Monte-Carlo paths to draw per session.
    emission_noise:
        Whether to sample the log-return under each simulated state from the
        fitted emission density. If False, we use the state-conditional mean
        (deterministic continuation) which makes ``u`` small but useful only
        as a diagnostic, not for Sharpe-aware sizing.
    quantiles:
        Triple of quantiles exposed as ``q_lower / q_median / q_upper`` in
        the returned frame.
    seed:
        RNG seed. The same seed is used across sessions so predictions are
        deterministic at a given bundle-level configuration.
    return_index:
        Override for the log-return column inside the emission vector. When
        None, the caller must pass it explicitly to :func:`forecast_sessions_mc`.
    """

    horizon: int = 50
    n_sim: int = 512
    emission_noise: bool = True
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    seed: int = 0
    return_index: Optional[int] = None


def _state_return_stats(bundle: HMMBundle, return_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-state mean and stddev of the log-return emission component.

    hmmlearn's public ``covars_`` property expands diagonal covariances to
    full matrices on read, so we branch on the covariance type and route all
    shape handling through the internal ``_covars_`` attribute whose layout
    matches the covariance type exactly.
    """
    means = bundle.means[:, return_index]
    internal = np.asarray(bundle.model._covars_, dtype=np.float64)
    cov_type = bundle.model.covariance_type
    if cov_type == "diag":
        # shape (K, F)
        sigma = np.sqrt(np.maximum(internal[:, return_index], 1e-16))
    elif cov_type == "full":
        # shape (K, F, F)
        sigma = np.sqrt(np.maximum(internal[:, return_index, return_index], 1e-16))
    elif cov_type == "spherical":
        # shape (K,)
        sigma = np.sqrt(np.maximum(internal, 1e-16))
    elif cov_type == "tied":
        # shape (F, F)
        sigma_scalar = float(np.sqrt(max(internal[return_index, return_index], 1e-16)))
        sigma = np.full_like(means, sigma_scalar)
    else:
        raise ValueError(f"Unsupported covariance_type: {cov_type}")
    return means.astype(np.float64), sigma.astype(np.float64)


def _batch_terminal_posteriors(
    bundle: HMMBundle,
    sessions: Sequence[SessionEmissions],
    seen_bars: Optional[int] = None,
) -> np.ndarray:
    """Return ``(N, K)`` array of terminal-bar posteriors for all sessions.

    Calls hmmlearn's ``predict_proba`` *once* on the concatenated multi-
    sequence input and slices the per-session terminal rows from the result,
    which avoids per-session Python overhead at scale (20k test sessions x
    ~10ms per individual call was the primary runtime hog).

    ``seen_bars`` caps the number of bars used for filtering. When training
    sessions carry 100 full bars but we want to mimic inference-time filtering
    over only the first 50, pass ``seen_bars=50``. Sessions shorter than
    ``seen_bars`` are used in full. ``None`` (default) uses every bar, which
    is the right setting for test sessions (already 50 bars) or for downstream
    feature extraction that wants the posterior over the full sequence.
    """
    K = bundle.n_states
    if not sessions:
        return np.zeros((0, K), dtype=np.float64)

    if seen_bars is None:
        feature_blocks = [s.features for s in sessions]
    else:
        cap = int(seen_bars)
        feature_blocks = [s.features[:cap] for s in sessions]
    lengths = np.array([fb.shape[0] for fb in feature_blocks], dtype=np.int64)
    non_empty_mask = lengths > 0
    if not non_empty_mask.any():
        return np.tile(bundle.startprob, (len(sessions), 1))

    X = np.concatenate(
        [fb for fb, has in zip(feature_blocks, non_empty_mask) if has], axis=0
    )
    gamma = np.asarray(
        bundle.model.predict_proba(X, lengths[non_empty_mask]), dtype=np.float64
    )
    end_idx = np.cumsum(lengths[non_empty_mask]) - 1
    terminal_non_empty = gamma[end_idx]  # (n_non_empty, K)

    out = np.empty((len(sessions), K), dtype=np.float64)
    ne_iter = iter(terminal_non_empty)
    for i, has in enumerate(non_empty_mask):
        if has:
            out[i] = next(ne_iter)
        else:
            out[i] = bundle.startprob
    return out


def _simulate_batch(
    start_post: np.ndarray,
    transmat: np.ndarray,
    state_mean: np.ndarray,
    state_sigma: np.ndarray,
    horizon: int,
    n_sim: int,
    emission_noise: bool,
    seed: int,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Vectorised Monte-Carlo simulation of summed log-returns per session.

    Returns a ``(N, n_sim)`` array. Sessions are simulated in chunks to cap
    peak memory (``chunk_size * n_sim`` states per step).
    """
    N, K = start_post.shape
    if N == 0:
        return np.zeros((0, n_sim), dtype=np.float64)
    cum_trans = np.cumsum(transmat, axis=1)
    cum_start = np.cumsum(start_post, axis=1)

    out = np.empty((N, n_sim), dtype=np.float64)
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(N, chunk_start + chunk_size)
        nc = chunk_end - chunk_start
        rng = np.random.default_rng(int(seed) + int(chunk_start))

        u0 = rng.random((nc, n_sim))
        states = (u0[:, :, None] < cum_start[chunk_start:chunk_end, None, :]).argmax(axis=2)
        states = np.clip(states, 0, K - 1).reshape(-1)  # (nc * n_sim,)

        total = np.zeros(nc * n_sim, dtype=np.float64)
        for _ in range(horizon):
            u = rng.random(states.size)
            cum_row = cum_trans[states]  # (nc*n_sim, K)
            states = np.clip((u[:, None] < cum_row).argmax(axis=1), 0, K - 1)
            mu = state_mean[states]
            if emission_noise:
                noise = rng.standard_normal(states.size) * state_sigma[states]
                total += mu + noise
            else:
                total += mu
        out[chunk_start:chunk_end] = total.reshape(nc, n_sim)
    return out


def forecast_sessions_mc(
    bundle: HMMBundle,
    sessions: Sequence[SessionEmissions],
    *,
    return_index: int,
    config: MCConfig = MCConfig(),
    seen_bars: Optional[int] = None,
    progress_tag: Optional[str] = None,
    return_samples: bool = False,
) -> pd.DataFrame:
    """Monte-Carlo forecast of ``R = C_end / C_half - 1`` for each session.

    The returned dataframe has the exact column contract the sizing layer
    expects (``session, mu, p_up, q_lower, q_median, q_upper, u``). Sessions
    are returned in the same order as the input sequence. The implementation
    is fully vectorised:

    1. A single batched ``predict_proba`` call over all sessions.
    2. Inverse-CDF sampling of initial states per session from their
       terminal-bar posteriors.
    3. A vectorised state walk of length ``config.horizon`` across all
       sessions x simulations simultaneously (chunked to cap memory).

    ``seen_bars`` caps the filtering window when training sessions ship with
    full 100-bar trajectories (the HMM is fit on bars 0..99 but inference must
    only condition on bars 0..49). Test sessions already only have 50 bars so
    this is a no-op for the submission path.

    ``return_samples=True`` attaches the underlying ``(N, n_sim)`` R-sample
    matrix to the dataframe ``attrs["R_samples"]`` so callers like
    :func:`mixture_forecast_from_samples` can resample from the true per-
    cluster predictive distribution instead of averaging summary quantiles.

    Passing ``progress_tag`` logs a single summary line with total elapsed
    time and key output statistics so large inference phases are visible.
    """
    import time as _time

    if not sessions:
        return pd.DataFrame(
            columns=["session", "mu", "p_up", "q_lower", "q_median", "q_upper", "u"]
        )

    t0 = _time.time()
    start_post = _batch_terminal_posteriors(bundle, sessions, seen_bars=seen_bars)
    state_mean, state_sigma = _state_return_stats(bundle, return_index=return_index)

    log_ret_future = _simulate_batch(
        start_post=start_post,
        transmat=bundle.transmat,
        state_mean=state_mean,
        state_sigma=state_sigma,
        horizon=int(config.horizon),
        n_sim=int(config.n_sim),
        emission_noise=bool(config.emission_noise),
        seed=int(config.seed) if config.seed is not None else 0,
    )
    R = np.exp(log_ret_future) - 1.0  # (N, n_sim)

    qs = np.asarray(config.quantiles, dtype=np.float64)
    mu = R.mean(axis=1)
    q = np.quantile(R, qs, axis=1)  # (len(qs), N)
    u = np.maximum(q[-1] - q[0], 1e-6)
    p_up = (R > 0.0).mean(axis=1)

    sess_ids = np.array([int(s.session) for s in sessions], dtype=np.int64)
    out = pd.DataFrame(
        {
            "session": sess_ids,
            "mu": mu,
            "p_up": p_up,
            "q_lower": q[0],
            "q_median": q[len(qs) // 2],
            "q_upper": q[-1],
            "u": u,
        }
    )
    if return_samples:
        out.attrs["R_samples"] = R
    if progress_tag is not None:
        _log(
            progress_tag,
            f"MC forecast {len(sessions)} sessions x {config.n_sim} sims "
            f"in {_time.time() - t0:.1f}s "
            f"(mu mean={float(out['mu'].mean()):+.5f}, p_up mean={float(out['p_up'].mean()):.3f})",
        )
    return out


def session_posterior_features(
    bundle: HMMBundle,
    sessions: Sequence[SessionEmissions],
    column_prefix: str = "hmm",
) -> pd.DataFrame:
    """Return a per-session posterior-summary feature frame.

    Columns are:

    * ``{prefix}_pfinal_k``      for k = 0..K-1 (final posterior)
    * ``{prefix}_pocc_k``        for k = 0..K-1 (average occupancy)
    * ``{prefix}_pnext_k``       for k = 0..K-1 (expected next-state dist)
    * ``{prefix}_loglik``         (sequence log-likelihood)
    * ``{prefix}_entropy``        (final-posterior entropy)

    This is ready to merge onto the OHLC tabular feature frame if we later
    want a hybrid HMM + LightGBM head.
    """
    if not sessions:
        return pd.DataFrame()
    K = bundle.n_states
    rows: List[dict] = []
    for sess in sessions:
        feats = occupancy_features(bundle, sess.features)
        row = {"session": int(sess.session)}
        for k in range(K):
            row[f"{column_prefix}_pfinal_{k}"] = float(feats[k])
            row[f"{column_prefix}_pocc_{k}"] = float(feats[K + k])
            row[f"{column_prefix}_pnext_{k}"] = float(feats[2 * K + k])
        row[f"{column_prefix}_loglik"] = float(feats[3 * K])
        row[f"{column_prefix}_entropy"] = float(feats[3 * K + 1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def mixture_forecast_from_samples(
    frames: Iterable[pd.DataFrame],
    weights: Iterable[Sequence[float]],
    *,
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Mathematically correct cluster-weighted mixture forecast.

    Each input frame must carry a per-session ``R_samples`` matrix of shape
    ``(N, n_sim)`` in ``frame.attrs["R_samples"]`` (produced by
    ``forecast_sessions_mc(..., return_samples=True)``). ``weights`` is a
    ``(K_clusters, N)`` array of per-session cluster responsibilities.

    For each session ``i`` we draw ``n_sim`` samples from the mixture

        p(R_i) = sum_k w_{ik} * p_k(R_i)

    by (1) sampling a cluster index ``k ~ Cat(w_{i,:})`` per simulation and
    (2) drawing an R sample from that cluster's predictive distribution.
    ``mu``, ``p_up``, and the quantile triple are then computed from the
    resulting mixture sample, which is the right thing to feed to the Sharpe-
    aware sizer.

    This replaces the previous ``mixture_forecast`` implementation which
    averaged per-cluster quantiles directly (mathematically wrong for
    multi-modal mixtures).
    """
    frames = list(frames)
    if not frames:
        raise ValueError("mixture_forecast_from_samples requires at least one frame")
    weights = np.asarray(list(weights), dtype=np.float64)
    if weights.shape[0] != len(frames):
        raise ValueError("weights must have one row per input frame")

    sessions = frames[0]["session"].to_numpy()
    n_sim = None
    sample_banks: List[np.ndarray] = []
    for f in frames:
        if not np.array_equal(f["session"].to_numpy(), sessions):
            raise ValueError("All mixture frames must share the same session ordering")
        bank = f.attrs.get("R_samples")
        if bank is None:
            raise ValueError(
                "Each cluster frame must carry its R_samples in .attrs. "
                "Pass return_samples=True to forecast_sessions_mc."
            )
        bank = np.asarray(bank, dtype=np.float64)
        if n_sim is None:
            n_sim = bank.shape[1]
        elif bank.shape[1] != n_sim:
            raise ValueError("Cluster R_samples banks must have matching n_sim")
        sample_banks.append(bank)

    K = len(frames)
    N = sessions.size
    # Normalise weights per session.
    wsum = weights.sum(axis=0, keepdims=True) + 1e-12
    w = (weights / wsum).T  # shape (N, K)
    cum_w = np.cumsum(w, axis=1)  # (N, K)
    stacked = np.stack(sample_banks, axis=0)  # (K, N, n_sim)

    rng = np.random.default_rng(int(rng_seed))
    # For each (i, j), pick k s.t. cum_w[i, k-1] <= u < cum_w[i, k].
    u = rng.random((N, n_sim))
    cluster_ix = np.sum(u[:, :, None] >= cum_w[:, None, :], axis=2)
    cluster_ix = np.clip(cluster_ix, 0, K - 1)
    # Pick a random within-cluster simulation index per draw.
    sim_ix = rng.integers(0, n_sim, size=(N, n_sim))
    sess_ix = np.broadcast_to(np.arange(N)[:, None], (N, n_sim))
    R_mix = stacked[cluster_ix, sess_ix, sim_ix]  # (N, n_sim)

    qs = np.asarray(quantiles, dtype=np.float64)
    mu = R_mix.mean(axis=1)
    q = np.quantile(R_mix, qs, axis=1)  # (3, N)
    spread = np.maximum(q[-1] - q[0], 1e-6)
    p_up = (R_mix > 0.0).mean(axis=1)

    out = pd.DataFrame(
        {
            "session": sessions,
            "mu": mu,
            "p_up": p_up,
            "q_lower": q[0],
            "q_median": q[len(qs) // 2],
            "q_upper": q[-1],
            "u": spread,
        }
    )
    out.attrs["R_samples"] = R_mix
    return out

"""End-to-end regime pipeline (Method 1 primary, Method 2 optional).

Phases mirror the plan:

1. Build per-session bar-level emission matrices from the seen-half bars.
2. Compute the per-session realized return ``R`` for the training set.
3. Fit the pooled Gaussian HMM (Method 1) with multi-start EM; select state
   count via BIC + session-level CV Sharpe.
4. Monte-Carlo forecast the second-half return distribution for every
   training session via session-level KFold (OOF predictions).
5. Tune the Sharpe-aware sizer on those OOF predictions.
6. Retrain the HMM on the full training set, forecast the 20000 test
   sessions, apply the tuned sizer, and emit a competition-format submission.

Method 2 (``method='m2'``) replaces step 3 with a clustering loop
(:mod:`clustering`) and step 4 with a cluster-weighted mixture forecast
(:func:`forecast.mixture_forecast`). All other stages stay identical.

News integration is stubbed at :mod:`news`: flipping ``use_news`` on today
raises a loud ``NotImplementedError`` so we cannot silently ship a submission
that forgot to read the headlines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from clustering import (
    ClusteringConfig,
    ClusteringResult,
    fit_clustered_hmms,
    score_sessions_against_clusters,
)
from emissions import (
    EmissionBundle,
    EmissionConfig,
    SessionEmissions,
    build_emission_bundle,
    build_session_emissions,
    session_summary_features,
)
from forecast import (
    MCConfig,
    forecast_sessions_mc,
    mixture_forecast_from_samples,
    session_posterior_features,
)
from linear_head import LinearHeadConfig, fit_and_predict, oof_linear_heads
from hmm_model import HMMHyper, fit_pooled_gaussian_hmm
from labels import load_full_train_bars, train_realized_returns
from news import (
    NewsConfig,
    NewsFeaturizer,
    build_news_regime_prior,
    load_test_news,
    load_train_news,
)
from paths import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
    BARS_UNSEEN_TRAIN,
)
from progress import log as _log, reset_clock
from selection import SelectionConfig, SelectionResult, select_best_hmm
from sizing import (
    SizingConfig,
    apply_sizing,
    build_ranking,
    sharpe,
    size_with_fallback,
    tune_sizing,
)


@dataclass
class RegimeConfig:
    """Top-level configuration for :func:`run_pipeline`.

    ``seen_bars`` is the number of bars available at inference time (50 for
    this competition). Training sessions carry the full 100-bar trajectories
    so the HMM learns the continuation dynamics, but all filtering / forecast
    calls clip the sequence to ``seen_bars`` to match test-time observability.
    """

    method: str = "m1"  # "m1" | "m1-linear" | "m2"
    random_state: int = 0
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    hmm: HMMHyper = field(default_factory=HMMHyper)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    linear: LinearHeadConfig = field(default_factory=LinearHeadConfig)
    oof_splits: int = 5
    target_scale: float = 1.0
    clip_quantile: float = 0.999
    seen_bars: int = 50


@dataclass
class RegimeResult:
    submission: pd.DataFrame
    rankings: pd.DataFrame
    diagnostics: dict
    selection: Optional[SelectionResult]
    clustering: Optional[ClusteringResult]
    tuned_sizing: SizingConfig
    config: RegimeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_y_to_bundle(
    labels: pd.DataFrame,
    bundle: EmissionBundle,
) -> np.ndarray:
    by_session = labels.set_index("session")["R"].astype(np.float64)
    try:
        return by_session.loc[bundle.sessions].to_numpy()
    except KeyError as exc:
        missing = [s for s in bundle.sessions if s not in by_session.index]
        raise KeyError(f"Training labels missing for sessions: {missing[:5]}") from exc


def _session_clipped(sess: SessionEmissions, seen_bars: int) -> SessionEmissions:
    """Return a shallow copy of ``sess`` whose feature matrix is clipped to ``seen_bars``."""
    if sess.features.shape[0] <= int(seen_bars):
        return sess
    clipped = SessionEmissions(
        session=sess.session,
        log_returns=sess.log_returns[: int(seen_bars)],
        features=sess.features[: int(seen_bars)],
        close_last=sess.close_last,
    )
    return clipped


def _subset_bundle(bundle: EmissionBundle, indices: np.ndarray) -> EmissionBundle:
    sessions = [bundle.per_session[i] for i in indices]
    X = np.concatenate([s.features for s in sessions], axis=0) if sessions else np.zeros((0, bundle.X.shape[1]))
    lengths = np.array([s.features.shape[0] for s in sessions], dtype=np.int64)
    sess_ids = np.array([s.session for s in sessions], dtype=np.int64)
    return EmissionBundle(
        X=X,
        lengths=lengths,
        sessions=sess_ids,
        columns=bundle.columns,
        per_session=sessions,
    )


# ---------------------------------------------------------------------------
# Method 1 OOF forecasting
# ---------------------------------------------------------------------------


def _oof_forecasts_m1(
    bundle: EmissionBundle,
    hyper: HMMHyper,
    mc_cfg: MCConfig,
    return_index: int,
    n_splits: int,
    random_state: int,
    seen_bars: int,
) -> pd.DataFrame:
    n = bundle.sessions.size
    order = np.arange(n)
    if n < max(4, n_splits):
        split = max(1, int(n * 0.2))
        splits = [(order[:-split], order[-split:])]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(order))

    oof_rows: List[pd.DataFrame] = []
    total = len(splits)
    for fold_ix, (tr_idx, va_idx) in enumerate(splits, start=1):
        _log(
            "oof",
            f"fold {fold_ix}/{total}: fitting HMM on {len(tr_idx)} full-trajectory "
            f"train sessions, forecasting {len(va_idx)} val sessions "
            f"(filtering on first {seen_bars} bars)",
        )
        tr_bundle = _subset_bundle(bundle, tr_idx)
        hmm_fold = fit_pooled_gaussian_hmm(
            tr_bundle.X, tr_bundle.lengths, hyper=hyper, progress_tag="oof"
        )
        va_sessions = [bundle.per_session[i] for i in va_idx]
        preds = forecast_sessions_mc(
            hmm_fold,
            va_sessions,
            return_index=return_index,
            config=mc_cfg,
            seen_bars=seen_bars,
            progress_tag="oof",
        )
        oof_rows.append(preds)
    oof = pd.concat(oof_rows, axis=0, ignore_index=True)
    return oof.sort_values("session").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Method 2 OOF forecasting
# ---------------------------------------------------------------------------


def _oof_forecasts_m2(
    bundle: EmissionBundle,
    hyper: HMMHyper,
    mc_cfg: MCConfig,
    return_index: int,
    cluster_cfg: ClusteringConfig,
    n_splits: int,
    random_state: int,
    seen_bars: int,
) -> pd.DataFrame:
    n = bundle.sessions.size
    order = np.arange(n)
    if n < max(4, n_splits):
        split = max(1, int(n * 0.2))
        splits = [(order[:-split], order[-split:])]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(order))

    oof_rows: List[pd.DataFrame] = []
    total = len(splits)
    for fold_ix, (tr_idx, va_idx) in enumerate(splits, start=1):
        _log(
            "oof",
            f"fold {fold_ix}/{total}: clustering {len(tr_idx)} full-trajectory train "
            f"sessions, forecasting {len(va_idx)} val sessions "
            f"(filtering on first {seen_bars} bars)",
        )
        tr_bundle = _subset_bundle(bundle, tr_idx)
        clustering = fit_clustered_hmms(tr_bundle, hyper=hyper, config=cluster_cfg)
        va_sessions = [bundle.per_session[i] for i in va_idx]
        _, resp = score_sessions_against_clusters(
            clustering.clusters,
            va_sessions,
            temperature=cluster_cfg.responsibility_temperature,
            seen_bars=seen_bars,
        )
        per_cluster_frames = [
            forecast_sessions_mc(
                c.hmm,
                va_sessions,
                return_index=return_index,
                config=mc_cfg,
                seen_bars=seen_bars,
                progress_tag="oof",
                return_samples=True,
            )
            for c in clustering.clusters
        ]
        # Shape (K, n_val)
        weights = resp.T
        mix = mixture_forecast_from_samples(
            per_cluster_frames,
            weights=weights,
            quantiles=mc_cfg.quantiles,
            rng_seed=int(mc_cfg.seed or 0) + fold_ix * 1009,
        )
        # Drop .attrs["R_samples"] before accumulating: pandas.concat() compares
        # frame .attrs element-wise, and numpy-array attrs make that comparison
        # raise "truth value is ambiguous".
        mix.attrs = {}
        oof_rows.append(mix)
    oof = pd.concat(oof_rows, axis=0, ignore_index=True)
    return oof.sort_values("session").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: Path,
    config: RegimeConfig = RegimeConfig(),
) -> RegimeResult:
    data_dir = Path(data_dir)
    reset_clock()
    _log(
        "pipeline",
        f"method={config.method} seed={config.random_state} "
        f"data_dir={data_dir} oof_splits={config.oof_splits} "
        f"mc(horizon={config.mc.horizon}, n_sim={config.mc.n_sim}, "
        f"emission_noise={config.mc.emission_noise})",
    )

    # ---- Train emissions + labels -----------------------------------------
    # Load FULL 100-bar training trajectories (seen bars 0..49 + unseen bars
    # 50..99). Fitting the HMM on only the seen half would prevent the model
    # from learning how latent states transition in the second half -- which
    # is exactly what we ask it to forecast at inference time.
    _log("emissions", f"loading full train trajectories ({BARS_SEEN_TRAIN} + {BARS_UNSEEN_TRAIN})")
    bars_tr = load_full_train_bars(data_dir)
    emissions_tr = build_emission_bundle(bars_tr, config.emission)
    if emissions_tr.sessions.size == 0:
        raise RuntimeError("No training sessions found while building emissions.")
    _log(
        "emissions",
        f"train: {emissions_tr.sessions.size} sessions x "
        f"{int(emissions_tr.lengths[0])} bars x {emissions_tr.X.shape[1]} features "
        f"(HMM fits on all bars; forecast filters to seen_bars={config.seen_bars})",
    )

    labels = train_realized_returns(data_dir)
    y_tr = _align_y_to_bundle(labels, emissions_tr)
    _log(
        "labels",
        f"train R: mean={float(np.mean(y_tr)):+.5f} std={float(np.std(y_tr, ddof=0)):.5f} "
        f"flat-long Sharpe={16.0 * float(np.mean(y_tr)) / max(float(np.std(y_tr, ddof=0)), 1e-12):+.3f}",
    )

    return_index = config.emission.return_index()

    # ---- News featurizer (first-half-only, fitted on train) --------------
    news_featurizer: Optional[NewsFeaturizer] = None
    news_tr: Optional[pd.DataFrame] = None
    news_te: Optional[pd.DataFrame] = None
    if config.news.enabled:
        if config.method != "m1-linear":
            raise NotImplementedError(
                f"News integration is wired only into m1-linear; got method={config.method!r}."
            )
        _log("news", "loading first-half headlines + sentiments (train + test)")
        headlines_tr, sentiments_tr = load_train_news(data_dir)
        headlines_te, sentiments_te = load_test_news(data_dir)
        _log(
            "news",
            f"train: {len(headlines_tr):,} headlines / {sentiments_tr['session'].nunique()} sessions; "
            f"test: {len(headlines_te):,} headlines / {sentiments_te['session'].nunique()} sessions",
        )
        news_featurizer = NewsFeaturizer(config.news).fit(headlines_tr, sentiments_tr)
        news_tr = news_featurizer.transform(
            headlines_tr, sentiments_tr, emissions_tr.sessions
        )
        _log(
            "news",
            f"train features: {news_tr.shape[1] - 1} columns "
            f"(top_entities={news_featurizer.top_entities_[:5]}... "
            f"svd_rank={getattr(news_featurizer.svd, 'n_components', 0)})",
        )

    selection_result: Optional[SelectionResult] = None
    clustering_result: Optional[ClusteringResult] = None
    final_hmm = None  # set for m1 / m1-linear below; m2 uses clustering_result
    post_tr: Optional[pd.DataFrame] = None
    feat_tr: Optional[pd.DataFrame] = None  # m1-linear: full stacked feature table

    if config.method in ("m1", "m1-linear"):
        # ---- Level-1 + Level-2 model selection -----------------------------
        # The selection CV is run at ``config.seen_bars`` filtering so it
        # matches the inference-time observability (and also trains on the
        # full 100-bar fold-train sessions). The winning candidate's
        # concatenated fold-val predictions are re-used as the OOF set for
        # sizer tuning -- this removes the previous double-CV leakage issue.
        _log(
            "pipeline",
            "phase 1/3: HMM state-count selection (BIC + CV Sharpe; "
            "winner's fold-val preds become OOF)",
        )
        selection_result = select_best_hmm(
            bundle=emissions_tr,
            y=y_tr,
            base_hyper=config.hmm,
            emission_cfg=config.emission,
            mc_cfg=config.mc,
            sizing_cfg=SizingConfig(
                target_scale=config.target_scale,
                alpha=1.0,
                mode="z_tanh",
                lam=3.0,
                tau_quantile=0.2,
            ),
            select_cfg=config.selection,
            seen_bars=config.seen_bars,
        )
        winning_hyper = HMMHyper(
            n_components=selection_result.n_components,
            covariance_type=selection_result.covariance_type,
            n_iter=config.hmm.n_iter,
            tol=config.hmm.tol,
            min_covar=config.hmm.min_covar,
            n_starts=config.hmm.n_starts,
            init_params=config.hmm.init_params,
            params=config.hmm.params,
            random_state=config.hmm.random_state,
            floor_covariance=config.hmm.floor_covariance,
        )
        if selection_result.winner_oof is None or selection_result.winner_oof.empty:
            raise RuntimeError("Selection winner produced no fold-val predictions.")

        _log(
            "final",
            f"fitting HMM on full {emissions_tr.sessions.size} train sessions "
            f"(K={winning_hyper.n_components}, cov={winning_hyper.covariance_type})",
        )
        final_hmm = fit_pooled_gaussian_hmm(
            emissions_tr.X,
            emissions_tr.lengths,
            hyper=winning_hyper,
            progress_tag="final",
        )
        _log(
            "final",
            f"HMM state means (return channel) = "
            f"{np.round(final_hmm.means[:, return_index], 5).tolist()}",
        )

        if config.method == "m1":
            oof_preds = selection_result.winner_oof.copy()
        else:
            # m1-linear: assemble the full "session identity" feature stack
            # that the linear head sees:
            #   A) HMM posterior features (pfinal, pocc, pnext, LL, entropy)
            #   B) coarse OHLC session-summary features (opt-in)
            #   C) first-half news features (opt-in via config.news.enabled):
            #      - directional sentiment aggregates
            #      - temporal profile
            #      - entity concentration
            #      - top-K entity footprint
            #      - TF-IDF + TruncatedSVD topic fingerprint
            # All blocks are joined on ``session``; sessions with no news
            # contribute zeros (honest absence signal).
            _log(
                "final",
                f"extracting posterior features on clipped train "
                f"({config.seen_bars}-bar filter)",
            )
            clipped_tr_sessions = [
                _session_clipped(s, config.seen_bars) for s in emissions_tr.per_session
            ]
            post_tr = session_posterior_features(final_hmm, clipped_tr_sessions)
            feature_blocks_tr: List[pd.DataFrame] = [post_tr]
            block_names = ["posterior"]
            if config.news.enabled and config.news.include_summary_features:
                summary_tr = session_summary_features(clipped_tr_sessions)
                feature_blocks_tr.append(summary_tr)
                block_names.append(f"summary[{summary_tr.shape[1] - 1}]")
            if config.news.enabled and news_tr is not None:
                feature_blocks_tr.append(news_tr)
                block_names.append(f"news[{news_tr.shape[1] - 1}]")
            feat_tr = feature_blocks_tr[0]
            for blk in feature_blocks_tr[1:]:
                feat_tr = feat_tr.merge(blk, on="session", how="left")
            feat_tr = feat_tr.fillna(0.0).sort_values("session").reset_index(drop=True)
            _log(
                "final",
                f"linear-head input: {feat_tr.shape[1] - 1} features "
                f"(blocks: {', '.join(block_names)})",
            )
            _log("final", "running linear-head OOF")
            align = (
                feat_tr.merge(
                    pd.DataFrame({"session": emissions_tr.sessions, "R": y_tr}),
                    on="session",
                    how="inner",
                )
                .sort_values("session")
                .reset_index(drop=True)
            )
            lh_oof = oof_linear_heads(
                align[[c for c in feat_tr.columns]],
                align["R"].to_numpy(dtype=np.float64),
                cfg=config.linear,
            )
            oof_preds = lh_oof.oof
            _log(
                "final",
                f"linear-head OOF: mu mean={float(oof_preds['mu'].mean()):+.5f} "
                f"p_up mean={float(oof_preds['p_up'].mean()):.3f} "
                f"mean u={float(oof_preds['u'].mean()):.4f}",
            )
    elif config.method == "m2":
        _log(
            "pipeline",
            f"phase 1/3: OOF clustered-HMM forecasts ({config.oof_splits}-fold, "
            f"K_clusters={config.clustering.n_clusters})",
        )
        oof_preds = _oof_forecasts_m2(
            emissions_tr,
            hyper=config.hmm,
            mc_cfg=config.mc,
            return_index=return_index,
            cluster_cfg=config.clustering,
            n_splits=config.oof_splits,
            random_state=config.random_state,
            seen_bars=config.seen_bars,
        )
        winning_hyper = config.hmm
    else:
        raise ValueError(
            f"Unknown method: {config.method!r}; expected 'm1', 'm1-linear', or 'm2'"
        )

    # Align oof_preds with y_tr order (session sorted order).
    oof_preds = oof_preds.sort_values("session").reset_index(drop=True)
    labels_sorted = labels.sort_values("session").reset_index(drop=True)
    y_aligned = labels_sorted["R"].to_numpy(dtype=np.float64)

    # ---- Tune the Sharpe-aware sizer --------------------------------------
    _log("pipeline", "phase 2/3: tune Sharpe-aware sizer on OOF predictions")
    tuned_cfg, tune_info = tune_sizing(
        oof_preds,
        y_aligned,
        target_scale=config.target_scale,
        clip_quantile=config.clip_quantile,
    )
    _log(
        "sizing",
        f"tuned: mode={tuned_cfg.mode} alpha={tuned_cfg.alpha} "
        f"lambda={tuned_cfg.lam} theta={tuned_cfg.theta} "
        f"tau_q={tuned_cfg.tau_quantile} allow_short={tuned_cfg.allow_short} "
        f"grid_size={tune_info['grid_size']} best_OOF_Sharpe={tune_info['grid_best_sharpe']:+.4f} "
        f"(flat-long {tune_info['flat_long_sharpe']:+.4f})",
    )

    # Supplementary OOF diagnostics.
    oof_positions, _ = apply_sizing(oof_preds, tuned_cfg)
    oof_sharpe = sharpe(oof_positions * y_aligned)
    always_long_sharpe = sharpe(y_aligned)
    sign_only_sharpe = sharpe(np.sign(oof_preds["mu"].to_numpy()) * y_aligned)
    p_sign_sharpe = sharpe(
        np.where(oof_preds["p_up"].to_numpy() > 0.5, 1.0, -1.0) * y_aligned
    )

    # ---- Test-time predictions --------------------------------------------
    _log(
        "pipeline",
        "phase 3/3: forecast test sessions "
        + (
            "(MC continuation from HMM posterior)"
            if config.method == "m1"
            else (
                "(ridge/logistic/quantile head on HMM posterior features)"
                if config.method == "m1-linear"
                else "(clustered HMMs + mixture)"
            )
        ),
    )
    _log("emissions", f"loading {BARS_SEEN_PUBLIC_TEST} + {BARS_SEEN_PRIVATE_TEST}")
    bars_pub = pd.read_parquet(data_dir / BARS_SEEN_PUBLIC_TEST)
    bars_priv = pd.read_parquet(data_dir / BARS_SEEN_PRIVATE_TEST)
    bars_te = pd.concat([bars_pub, bars_priv], ignore_index=True)
    emissions_te = build_session_emissions(bars_te, config.emission)
    _log("emissions", f"test: {len(emissions_te)} sessions built")

    if config.method == "m1":
        assert final_hmm is not None
        preds_te = forecast_sessions_mc(
            final_hmm,
            emissions_te,
            return_index=return_index,
            config=config.mc,
            seen_bars=config.seen_bars,
            progress_tag="final",
        )
        hmm_bundle_for_diag = final_hmm
    elif config.method == "m1-linear":
        assert final_hmm is not None and feat_tr is not None
        _log("final", "extracting posterior features on test sessions")
        clipped_te_sessions = [
            _session_clipped(s, config.seen_bars) for s in emissions_te
        ]
        post_te = session_posterior_features(final_hmm, clipped_te_sessions)
        feature_blocks_te: List[pd.DataFrame] = [post_te]
        if config.news.enabled and config.news.include_summary_features:
            summary_te = session_summary_features(clipped_te_sessions)
            feature_blocks_te.append(summary_te)
        if config.news.enabled and news_featurizer is not None:
            headlines_te, sentiments_te = load_test_news(data_dir)
            news_te = news_featurizer.transform(
                headlines_te,
                sentiments_te,
                pd.Series([int(s.session) for s in emissions_te]),
            )
            feature_blocks_te.append(news_te)
            _log(
                "news",
                f"test features: {news_te.shape[1] - 1} columns",
            )
        feat_te = feature_blocks_te[0]
        for blk in feature_blocks_te[1:]:
            feat_te = feat_te.merge(blk, on="session", how="left")
        feat_te = feat_te.fillna(0.0).sort_values("session").reset_index(drop=True)
        _log(
            "final",
            f"test linear-head input: {feat_te.shape[1] - 1} features for {len(feat_te)} sessions",
        )
        align = (
            feat_tr.merge(
                pd.DataFrame({"session": emissions_tr.sessions, "R": y_tr}),
                on="session",
                how="inner",
            )
            .sort_values("session")
            .reset_index(drop=True)
        )
        preds_te = fit_and_predict(
            align[[c for c in feat_tr.columns]],
            align["R"].to_numpy(dtype=np.float64),
            feat_te,
            cfg=config.linear,
        )
        _log(
            "final",
            f"linear-head test predictions: mu mean={float(preds_te['mu'].mean()):+.5f} "
            f"p_up mean={float(preds_te['p_up'].mean()):.3f}",
        )
        hmm_bundle_for_diag = final_hmm
    else:
        _log(
            "final",
            f"fitting clustered HMMs on full {emissions_tr.sessions.size} train sessions "
            f"(K_clusters={config.clustering.n_clusters}, K_states={winning_hyper.n_components})",
        )
        clustering_result = fit_clustered_hmms(
            emissions_tr, hyper=winning_hyper, config=config.clustering
        )
        _, test_resp = score_sessions_against_clusters(
            clustering_result.clusters,
            emissions_te,
            temperature=config.clustering.responsibility_temperature,
            seen_bars=config.seen_bars,
        )
        per_cluster_frames = [
            forecast_sessions_mc(
                c.hmm,
                emissions_te,
                return_index=return_index,
                config=config.mc,
                seen_bars=config.seen_bars,
                progress_tag=f"final/cluster{k}",
                return_samples=True,
            )
            for k, c in enumerate(clustering_result.clusters)
        ]
        preds_te = mixture_forecast_from_samples(
            per_cluster_frames,
            weights=test_resp.T,
            quantiles=config.mc.quantiles,
            rng_seed=int(config.mc.seed or 0) + 9991,
        )
        hmm_bundle_for_diag = clustering_result.clusters[0].hmm
        _log(
            "final",
            f"mixture forecast blended over {len(per_cluster_frames)} clusters "
            f"(true sample-mixture, not averaged quantiles)",
        )

    preds_te = preds_te.sort_values("session").reset_index(drop=True)
    positions_te = size_with_fallback(preds_te, tuned_cfg)
    rankings = build_ranking(
        preds_te["session"].to_numpy(), preds_te, positions_te
    )
    submission = rankings[["session", "target_position"]].copy()
    pos = rankings["target_position"].to_numpy()
    _log(
        "submission",
        f"sized positions: mean={pos.mean():+.4f} std={pos.std():.4f} "
        f"neg_frac={float((pos < 0).mean()):.3f} "
        f"|p|_range=[{abs(pos).min():.4f}, {abs(pos).max():.4f}]",
    )
    _log(
        "pipeline",
        f"OOF Sharpe (tuned) = {float(oof_sharpe):+.4f} vs flat-long {float(always_long_sharpe):+.4f} "
        f"(sign(mu)={float(sign_only_sharpe):+.3f}, sign(p_up)={float(p_sign_sharpe):+.3f})",
    )

    # ---- Diagnostics -------------------------------------------------------
    diagnostics: dict = {
        "method": config.method,
        "n_train_sessions": int(emissions_tr.sessions.size),
        "n_test_sessions": int(len(emissions_te)),
        "n_emission_features": int(emissions_tr.X.shape[1]),
        "train_R_mean": float(np.mean(y_aligned)),
        "train_R_std": float(np.std(y_aligned, ddof=0)),
        "oof_sharpe_tuned": float(oof_sharpe),
        "oof_sharpe_always_long": float(always_long_sharpe),
        "oof_sharpe_sign_mu": float(sign_only_sharpe),
        "oof_sharpe_sign_p": float(p_sign_sharpe),
        "tuned_mode": str(tuned_cfg.mode),
        "tuned_alpha": float(tuned_cfg.alpha),
        "tuned_lambda": float(tuned_cfg.lam),
        "tuned_theta": float(tuned_cfg.theta),
        "tuned_tau_quantile": float(tuned_cfg.tau_quantile),
        "tune_info": tune_info,
        "hmm_n_states": int(hmm_bundle_for_diag.n_states),
        "hmm_covariance_type": str(hmm_bundle_for_diag.model.covariance_type),
        "hmm_log_likelihood": float(hmm_bundle_for_diag.log_likelihood),
        "hmm_aic": float(hmm_bundle_for_diag.aic),
        "hmm_bic": float(hmm_bundle_for_diag.bic),
        "hmm_converged": bool(hmm_bundle_for_diag.converged),
        "use_news": bool(config.news.enabled),
    }
    if selection_result is not None:
        diagnostics["selection"] = selection_result.as_dict()
    if clustering_result is not None:
        diagnostics["clustering"] = {
            "n_clusters": int(clustering_result.n_clusters),
            "cluster_sizes": [int(c.size) for c in clustering_result.clusters],
            "history": clustering_result.history,
        }

    return RegimeResult(
        submission=submission,
        rankings=rankings,
        diagnostics=diagnostics,
        selection=selection_result,
        clustering=clustering_result,
        tuned_sizing=tuned_cfg,
        config=config,
    )

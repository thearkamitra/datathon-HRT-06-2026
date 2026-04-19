"""Regime-switching / HMM process library for the Zurich Datathon 2026.

Implements the two HMM-based directions recommended in
``plans/Brainstorming-regime.pdf``:

* **Method 1** (primary): a single pooled Gaussian HMM fit across all training
  sessions. Each session is treated as a realization of the same small library
  of hidden market states. At inference, we observe the first 50 bars of each
  test session, infer the posterior over latent regimes at bar 49, simulate a
  Monte-Carlo continuation of bars 50..99 under the fitted HMM dynamics, and
  convert the resulting return distribution into a Sharpe-aware position.

* **Method 2** (optional follow-up): model-based clustering. Sessions are
  softly partitioned into ``K`` archetypes, each getting its own cluster-local
  HMM. The continuation forecast for a new session becomes a mixture over
  cluster-conditional forecasts weighted by the cluster posterior.

News integration stays deliberately gated behind :mod:`news` so the regime
backbone can be built and validated before touching the headline signal.
"""

__version__ = "0.1.0"

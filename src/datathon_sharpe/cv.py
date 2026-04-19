"""Full-train fit with optional 25+25 session monitoring (Sharpe only where R exists)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_baseline.metrics import sharpe
from datathon_baseline.predict import Method, TrainResult
from datathon_sharpe.split import split_25_25
from datathon_sharpe.sharpe_label_transforms import SharpeOptimizerLabel
from datathon_sharpe.train_model import fit_full_train_and_submission


@dataclass
class CVReport:
    train_result: TrainResult
    # Sharpe on all rows in train_pred (1000 sessions): w * R with label column there.
    sharpe_train_all_sessions: float
    sharpe_block_train: float
    sharpe_block_label: float
    sessions_train_block: np.ndarray
    sessions_label_block: np.ndarray
    note: str | None = None


def _sharpe_on_sessions(pred: pd.DataFrame, sessions: np.ndarray) -> float:
    m = pred["session"].isin(sessions)
    sub = pred.loc[m]
    if sub.empty:
        return 0.0
    return float(sharpe(sub["w"].to_numpy() * sub["R"].to_numpy()))


def run_cv_report(
    data_dir: Path,
    method: Method,
    *,
    ridge_reg: float = 1.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    split_seed: int | None = None,
    within_session_split: bool = False,
    augment_test_with_proxy: bool = True,
    sharpe_optimizer_label: SharpeOptimizerLabel = "identity",
    use_cnn: bool = False,
    cnn_epochs: int = 40,
    mse_anchor_lambda: float = 0.0,
    distributional_policy: str = "prob_sign",
) -> tuple[pd.DataFrame, CVReport]:
    """
    1) Fit on **all** training sessions (with labels).
    2) Build submission for full public+private test.
    3) Draw 50 sessions from the **train** pool, split 25+25; report Sharpe of
       `w * R` on each block (disjoint holdout-style monitoring on train only).

    If ``within_session_split`` is True, training uses first-half features and
    proxy label close_49/close_24-1; reported Sharpes use that same ``R``.

    If ``augment_test_with_proxy`` is True, the fit includes test sessions (25-bar +
    proxy ``R``); block Sharpes still use train sessions and competition ``R`` only.

    Realized returns `R` are only available for train sessions (not for public/private
    test ids), so the 25+25 Sharpe uses train session ids only.

    ``mse_anchor_lambda``: Sharpe-linear only; passed to ``fit_full_train_and_submission``
    (0 = unit-sphere Sharpe; >0 adds MSE anchor to Ridge positions).

    ``distributional_policy``: ``distributional_mono`` only (``prob_sign``, ``prob_sign_sharpe``, …).
    """
    from datathon_sharpe.split import train_session_pool

    split_seed = split_seed if split_seed is not None else random_state
    pool = train_session_pool(data_dir)
    s_tr, s_lb = split_25_25(pool, split_seed)

    pred, sub, train_res = fit_full_train_and_submission(
        data_dir,
        method,
        ridge_reg=ridge_reg,
        l1_ratio=l1_ratio,
        random_state=random_state,
        within_session_split=within_session_split,
        augment_test_with_proxy=augment_test_with_proxy,
        sharpe_optimizer_label=sharpe_optimizer_label,
        use_cnn=use_cnn,
        cnn_epochs=cnn_epochs,
        mse_anchor_lambda=mse_anchor_lambda,
        distributional_policy=distributional_policy,
    )
    sh_a = _sharpe_on_sessions(pred, s_tr)
    sh_b = _sharpe_on_sessions(pred, s_lb)
    sh_full_train = float(sharpe(pred["w"].to_numpy() * pred["R"].to_numpy()))

    report = CVReport(
        train_result=train_res,
        sharpe_train_all_sessions=sh_full_train,
        sharpe_block_train=float(sh_a),
        sharpe_block_label=float(sh_b),
        sessions_train_block=s_tr,
        sessions_label_block=s_lb,
        note=None,
    )
    return sub, report


def split_test_csv_sessions_25_25(
    random_state: int = 0,
    *,
    public_csv: Path | None = None,
    private_csv: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    25+25 split over merged public/private **test** session ids from CSV exports.

    There is no official `R` for these sessions in the released data; use this for
    submission subsets or debugging, not for Sharpe unless you add your own labels.
    """
    from datathon_sharpe.split import merge_public_private_test_sessions

    pool = merge_public_private_test_sessions(public_csv, private_csv)
    return split_25_25(pool, random_state)

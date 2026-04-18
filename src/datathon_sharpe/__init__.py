"""Train on full session data; optional 25+25 session monitoring and test CSV splits."""

from datathon_sharpe.cv import CVReport, run_cv_report, split_test_csv_sessions_25_25
from datathon_sharpe.features_seen_split import (
    FIRST_HALF_LAST_BAR_IX,
    build_session_features_first_half,
)
from datathon_sharpe.path_features import (
    BASELINE_COLUMNS_SHARPE,
    FEATURE_COLUMNS_SHARPE,
    PATH_EXTRA_COLUMNS,
    build_session_features_with_path,
)
from datathon_sharpe.labels_seen_split import (
    proxy_returns_second_seen_half_from_bars,
    train_proxy_returns_second_seen_half,
)
from datathon_sharpe.training_table import load_training_feature_matrices
from datathon_sharpe.split import (
    merge_public_private_test_sessions,
    sessions_from_bars_csv,
    split_25_25,
    train_session_pool,
)
from datathon_sharpe.train_model import fit_full_train_and_submission, fit_full_train_predictions

__all__ = [
    "BASELINE_COLUMNS_SHARPE",
    "CVReport",
    "FEATURE_COLUMNS_SHARPE",
    "FIRST_HALF_LAST_BAR_IX",
    "PATH_EXTRA_COLUMNS",
    "build_session_features_first_half",
    "build_session_features_with_path",
    "fit_full_train_and_submission",
    "fit_full_train_predictions",
    "load_training_feature_matrices",
    "merge_public_private_test_sessions",
    "run_cv_report",
    "sessions_from_bars_csv",
    "split_25_25",
    "split_test_csv_sessions_25_25",
    "proxy_returns_second_seen_half_from_bars",
    "train_proxy_returns_second_seen_half",
    "train_session_pool",
]

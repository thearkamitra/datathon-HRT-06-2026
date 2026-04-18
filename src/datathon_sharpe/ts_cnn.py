"""1D CNN on per-session OHLC bar sequences; produces ``cnn_r_pred`` for Sharpe-linear stacking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datathon_baseline.io import (
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
    read_bars,
)

N_BARS = 50
N_CH = 4


def _session_ohlc_zscore(g: pd.DataFrame, n_bars: int = N_BARS) -> np.ndarray:
    g = g.sort_values("bar_ix")
    cols = ["open", "high", "low", "close"]
    x = g[cols].to_numpy(dtype=np.float64)
    if x.shape[0] < n_bars:
        pad = np.tile(x[-1:], (n_bars - x.shape[0], 1))
        x = np.vstack([x, pad])
    elif x.shape[0] > n_bars:
        x = x[:n_bars]
    for j in range(N_CH):
        m = float(np.mean(x[:, j]))
        s = float(np.std(x[:, j])) + 1e-8
        x[:, j] = (x[:, j] - m) / s
    return x


def bars_to_tensors(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(tensor, session_ids)`` with ``tensor`` shape (n_sess, n_bars, n_ch)."""
    rows: list[np.ndarray] = []
    sess_ids: list[int] = []
    for session, g in bars.groupby("session", sort=False):
        rows.append(_session_ohlc_zscore(g))
        sess_ids.append(int(session))
    if not rows:
        return np.zeros((0, N_BARS, N_CH), dtype=np.float64), np.array([], dtype=np.int64)
    return np.stack(rows, axis=0), np.array(sess_ids, dtype=np.int64)


def _load_all_seen_bars(data_dir: Path) -> pd.DataFrame:
    tr = read_bars(data_dir, BARS_SEEN_TRAIN)
    pub = read_bars(data_dir, BARS_SEEN_PUBLIC_TEST)
    priv = read_bars(data_dir, BARS_SEEN_PRIVATE_TEST)
    return pd.concat([tr, pub, priv], ignore_index=True)


def _train_mlp_flatten_fallback(
    X_all: np.ndarray,
    sid_all: np.ndarray,
    train_mask: np.ndarray,
    r_map: dict[int, float],
    *,
    epochs: int,
    seed: int,
) -> dict[int, float]:
    """If PyTorch is unavailable: same tensors, flattened → ``MLPRegressor`` (not a conv net)."""
    from sklearn.neural_network import MLPRegressor

    X_flat = X_all.reshape(len(X_all), -1).astype(np.float64)
    y_tr = np.array([r_map[int(s)] for s in sid_all[train_mask]], dtype=np.float64)
    X_tr = X_flat[train_mask]
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=max(200, int(epochs) * 25),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        alpha=1e-4,
    )
    mlp.fit(X_tr, y_tr)
    pred = mlp.predict(X_flat)
    return {int(s): float(p) for s, p in zip(sid_all, pred)}


def train_cnn_predict_r(
    data_dir: Path,
    feat_main: pd.DataFrame,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[int, float]:
    """
    Train a small 1D CNN to predict session ``R`` from OHLC (supervised MSE on ``feat_main`` rows).
    Returns mapping ``session -> cnn_r_pred`` for **all** sessions in the seen bar parquet union.

    If PyTorch is not installed (e.g. Python 3.13 without wheels), falls back to a **sklearn MLP**
    on flattened OHLC tensors (same inputs, not convolutional).
    """
    train_sessions = feat_main["session"].to_numpy()
    train_R = feat_main["R"].to_numpy(dtype=np.float64)
    r_map = {int(s): float(r) for s, r in zip(train_sessions, train_R)}

    bars = _load_all_seen_bars(data_dir)
    X_all, sid_all = bars_to_tensors(bars)
    n = len(sid_all)
    if n == 0:
        return {}

    train_sess_set = set(int(x) for x in train_sessions)
    train_mask = np.array([int(s) in train_sess_set for s in sid_all], dtype=bool)
    if not np.any(train_mask):
        return {int(s): 0.0 for s in sid_all}

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(
            "ts_cnn: PyTorch not found — using sklearn MLPRegressor on flattened OHLC "
            "(install torch for Conv1d CNN).",
            flush=True,
        )
        return _train_mlp_flatten_fallback(
            X_all, sid_all, train_mask, r_map, epochs=epochs, seed=seed
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = torch.from_numpy(X_all[train_mask]).float()
    y_tr = torch.tensor([r_map[int(s)] for s in sid_all[train_mask]], dtype=torch.float32)

    X_all_t = torch.from_numpy(X_all).float()

    class Session1DCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = 32
            self.conv = nn.Sequential(
                nn.Conv1d(N_CH, h, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(h, h, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(h, h // 2, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(h // 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            h = self.conv(x)
            h = h.mean(dim=-1)
            return self.head(h).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Session1DCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    n_tr = X_tr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds: list[float] = []
        for start in range(0, n, batch_size):
            batch = X_all_t[start : start + batch_size].to(device)
            p = model(batch).cpu().numpy()
            preds.extend(p.tolist())
    out = {int(s): float(p) for s, p in zip(sid_all, preds)}
    return out


def apply_cnn_r_pred_to_frame(df: pd.DataFrame, scores: dict[int, float]) -> None:
    """In-place: set ``cnn_r_pred`` from ``scores`` (missing sessions -> 0.0)."""
    s = df["session"].to_numpy()
    df["cnn_r_pred"] = np.array([float(scores.get(int(x), 0.0)) for x in s], dtype=np.float64)


CNN_EXTRA_COLUMNS: list[str] = ["cnn_r_pred"]

__all__ = [
    "CNN_EXTRA_COLUMNS",
    "N_BARS",
    "N_CH",
    "apply_cnn_r_pred_to_frame",
    "bars_to_tensors",
    "train_cnn_predict_r",
]

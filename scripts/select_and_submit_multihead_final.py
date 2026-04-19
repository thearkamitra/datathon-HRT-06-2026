#!/usr/bin/env python3
"""Multihead hybrid: weighted ensemble of tailored + prob_reg + regime base models.

==============================================================================
Final Locked Weights (seed 0, validation_fraction 0.2)
==============================================================================

Reference run: ``Submissions/multihead_final/20260419_101156/``

Candidate heads considered and their held-out validation Sharpe on 1000
augmented validation sessions (trained on the remaining 4000 augmented
sessions of ``data/augmented/``):

  - tailored/ohlc_news_hash64          validation Sharpe   3.3483
  - prob_reg/ridge_news_gaussian       validation Sharpe   3.7518
  - regime/m1_linear_news_fast         validation Sharpe   2.9960

Locked convex-combination weights (sum to 1, non-negative), chosen by
maximising the validation Sharpe of the weighted per-session positions on a
simplex grid (step 0.02, refined step 0.005 around the best point):

  - tailored/ohlc_news_hash64          weight   0.56
  - prob_reg/ridge_news_gaussian       weight   0.44
  - regime/m1_linear_news_fast         weight   0.00

Resulting held-out Sharpes:

  - Best single-base validation Sharpe            3.7518
  - Multihead validation Sharpe (locked blend)    3.8297

==============================================================================
Methodology
==============================================================================

1. Start from the augmented training set at ``data/augmented/``.
2. Make a random **session-wholesale** train/validation split: every augmented
   session's full seen and unseen bars stay together in exactly one side of
   the split (this avoids any bar-level leakage between train and validation).
3. Train each base model on the augmented-train split and predict the
   augmented-validation split. The per-base selection submission naturally has
   one row per validation session, which is what we score for weight learning.
4. Build a ``(n_val_sessions, n_heads)`` matrix of per-session ``target_position``
   values from each base, align it with the realised validation returns
   ``R = close_end / close_half - 1``, and pick the convex-combination weight
   vector on the simplex that maximises the competition Sharpe of the weighted
   per-session PnL. We sweep a coarse simplex grid (step 0.02) and then refine
   around the winner at a finer step (0.005) to avoid brittle single-point
   optima.
5. Lock those weights.
6. Refit every base model on the **full** augmented training set (5000
   sessions) and produce real public + private test predictions (20000 rows
   per base). No information from the test sets ever enters selection: the
   weights were already locked using only the augmented-validation split.
7. Combine the per-base 20000-row predictions using the locked weights and
   write a single 20000-row multihead submission to
   ``Submissions/multihead_final/<timestamp>/multihead_submission.csv``.

==============================================================================
Regime speed profile
==============================================================================

The regime family is the slowest piece. For the multihead suite we reuse the
trimmed regime settings from ``scripts/select_and_submit_separate_new.py`` so
the full run completes in a practical time without sacrificing the winning
regime structure we already validated:

  --select-states 3 --select-cov-types diag
  --select-cv-splits 2 --select-starts 1
  --n-starts 1 --n-sim 96 --oof-splits 3

==============================================================================
Outputs per run
==============================================================================

Under ``Submissions/multihead_final/<timestamp>/``:

  - ``multihead_submission.csv``   final 20000-row ensemble submission
  - ``final/<family>/<name>_submission.csv``   per-base 20000-row submissions
  - ``selection/<family>/<name>_submission.csv`` per-base validation predictions
  - ``weights.json``               locked weights + validation diagnostics
  - ``summary.json``               full structured summary (weights, row counts,
                                   per-base val Sharpes, final paths)
  - ``split_info.json``            session-wholesale split metadata
  - ``README.md``                  human-readable report
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUGMENTED_DATA = REPO_ROOT / "data" / "augmented"
DEFAULT_OUT_DIR = REPO_ROOT / "Submissions" / "multihead_final"

BARS_SEEN_TRAIN = "bars_seen_train.parquet"
BARS_UNSEEN_TRAIN = "bars_unseen_train.parquet"
BARS_SEEN_PUBLIC_TEST = "bars_seen_public_test.parquet"
BARS_SEEN_PRIVATE_TEST = "bars_seen_private_test.parquet"

HEADLINES_SEEN_TRAIN = "headlines_seen_train.parquet"
HEADLINES_UNSEEN_TRAIN = "headlines_unseen_train.parquet"
HEADLINES_SEEN_PUBLIC_TEST = "headlines_seen_public_test.parquet"
HEADLINES_SEEN_PRIVATE_TEST = "headlines_seen_private_test.parquet"

SENTIMENTS_SEEN_TRAIN = "sentiments_seen_train.csv"
SENTIMENTS_UNSEEN_TRAIN = "sentiments_unseen_train.csv"
SENTIMENTS_SEEN_PUBLIC_TEST = "sentiments_seen_public_test.csv"
SENTIMENTS_SEEN_PRIVATE_TEST = "sentiments_seen_private_test.csv"


# Same minimal regime settings used in separate_new.py
_REGIME_FAST = (
    "--select-states", "3",
    "--select-cov-types", "diag",
    "--select-cv-splits", "2",
    "--select-starts", "1",
    "--n-starts", "1",
    "--n-sim", "96",
    "--oof-splits", "3",
)


@dataclass(frozen=True)
class Base:
    family: str
    name: str
    script: Path
    args: tuple[str, ...]


def _bases() -> list[Base]:
    s = {
        "tailored": REPO_ROOT / "scripts" / "tailored-predictor",
        "prob_reg": REPO_ROOT / "scripts" / "prob-reg-predictor",
        "regime": REPO_ROOT / "scripts" / "regime-predictor",
    }
    # Each base is the per-family winner from the previous augmented-split run.
    return [
        Base(
            "tailored",
            "ohlc_news_hash64",
            s["tailored"],
            ("--use-news", "--news-hash-features", "64"),
        ),
        Base(
            "prob_reg",
            "ridge_news_gaussian",
            s["prob_reg"],
            (),
        ),
        Base(
            "regime",
            "m1_linear_news_fast",
            s["regime"],
            ("--method", "m1-linear", "--use-news", *_REGIME_FAST),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train tailored + prob_reg + regime, learn ensemble weights on validation, "
            "retrain on full augmented data, and combine real public+private predictions."
        )
    )
    parser.add_argument(
        "--augmented-data-dir",
        type=Path,
        default=DEFAULT_AUGMENTED_DATA,
        help="Source augmented data directory (defaults to data/augmented).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root for multihead_final artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the session split and base model runners.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of augmented sessions held out for validation.",
    )
    parser.add_argument(
        "--weight-grid-step",
        type=float,
        default=0.02,
        help="Resolution of the convex-combination weight grid (default 0.02).",
    )
    parser.add_argument(
        "--allow-zero-bias",
        action="store_true",
        help=(
            "If set, also include a flat-long baseline as a fourth ensemble head "
            "to provide a safe fallback signal."
        ),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )


def _run_command(cmd: Sequence[str], log_path: Path | None = None) -> None:
    print(f"\n[run] {' '.join(cmd)}\n", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            subprocess.run(
                cmd, check=True, cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT
            )
    else:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _load_train_labels(augmented_data_dir: Path) -> pd.DataFrame:
    seen = pd.read_parquet(
        augmented_data_dir / BARS_SEEN_TRAIN, columns=["session", "bar_ix", "close"]
    )
    unseen = pd.read_parquet(
        augmented_data_dir / BARS_UNSEEN_TRAIN, columns=["session", "bar_ix", "close"]
    )
    c_half = (
        seen.loc[seen["bar_ix"] == int(seen["bar_ix"].max())]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_half")
    )
    c_end = (
        unseen.loc[unseen["bar_ix"] == int(unseen["bar_ix"].max())]
        .groupby("session", sort=False)["close"]
        .first()
        .rename("close_end")
    )
    out = pd.concat([c_half, c_end], axis=1).dropna().reset_index()
    out["R"] = out["close_end"] / out["close_half"] - 1.0
    return out.sort_values("session").reset_index(drop=True)


def _select_sessions(
    augmented_data_dir: Path, seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    seen = pd.read_parquet(augmented_data_dir / BARS_SEEN_TRAIN, columns=["session"])
    sessions = np.sort(seen["session"].unique().astype(np.int64))
    if sessions.size < 2:
        raise RuntimeError("Need at least 2 augmented sessions to build a split.")

    rng = np.random.default_rng(int(seed))
    shuffled = sessions.copy()
    rng.shuffle(shuffled)

    n_val = int(round(float(validation_fraction) * float(shuffled.size)))
    n_val = max(1, min(shuffled.size - 1, n_val))
    val_sessions = np.sort(shuffled[:n_val])
    train_sessions = np.sort(shuffled[n_val:])
    return train_sessions, val_sessions


def _subset_rows(
    df: pd.DataFrame | None, sessions: Iterable[int]
) -> pd.DataFrame | None:
    if df is None:
        return None
    sess_set = {int(x) for x in sessions}
    if "session" not in df.columns:
        return df.copy()
    return df.loc[df["session"].astype(np.int64).isin(sess_set)].copy()


def _write_optional_parquet(
    df: pd.DataFrame | None, path: Path, template: pd.DataFrame | None
) -> None:
    if template is None and df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if df is not None else template.iloc[:0].copy()
    out.to_parquet(path, index=False)


def _write_optional_csv(
    df: pd.DataFrame | None, path: Path, template: pd.DataFrame | None
) -> None:
    if template is None and df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if df is not None else template.iloc[:0].copy()
    out.to_csv(path, index=False)


def _build_split_dataset(
    augmented_data_dir: Path,
    split_dir: Path,
    *,
    train_sessions: np.ndarray,
    val_sessions: np.ndarray,
) -> None:
    """Build a self-contained dataset where the validation sessions play the role
    of ``bars_seen_public_test.parquet`` so each base predictor emits exactly one
    row per validation session into its selection submission CSV.
    """
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    bars_seen = pd.read_parquet(augmented_data_dir / BARS_SEEN_TRAIN)
    bars_unseen = pd.read_parquet(augmented_data_dir / BARS_UNSEEN_TRAIN)

    headlines_seen = (
        pd.read_parquet(augmented_data_dir / HEADLINES_SEEN_TRAIN)
        if (augmented_data_dir / HEADLINES_SEEN_TRAIN).is_file()
        else None
    )
    headlines_unseen = (
        pd.read_parquet(augmented_data_dir / HEADLINES_UNSEEN_TRAIN)
        if (augmented_data_dir / HEADLINES_UNSEEN_TRAIN).is_file()
        else None
    )
    sentiments_seen = (
        pd.read_csv(augmented_data_dir / SENTIMENTS_SEEN_TRAIN)
        if (augmented_data_dir / SENTIMENTS_SEEN_TRAIN).is_file()
        else None
    )
    sentiments_unseen = (
        pd.read_csv(augmented_data_dir / SENTIMENTS_UNSEEN_TRAIN)
        if (augmented_data_dir / SENTIMENTS_UNSEEN_TRAIN).is_file()
        else None
    )

    train_seen = _subset_rows(bars_seen, train_sessions)
    train_unseen = _subset_rows(bars_unseen, train_sessions)
    val_seen = _subset_rows(bars_seen, val_sessions)
    assert train_seen is not None and train_unseen is not None and val_seen is not None

    train_seen.to_parquet(split_dir / BARS_SEEN_TRAIN, index=False)
    train_unseen.to_parquet(split_dir / BARS_UNSEEN_TRAIN, index=False)
    val_seen.to_parquet(split_dir / BARS_SEEN_PUBLIC_TEST, index=False)
    val_seen.iloc[:0].copy().to_parquet(split_dir / BARS_SEEN_PRIVATE_TEST, index=False)

    _write_optional_parquet(
        _subset_rows(headlines_seen, train_sessions),
        split_dir / HEADLINES_SEEN_TRAIN,
        headlines_seen,
    )
    _write_optional_parquet(
        _subset_rows(headlines_unseen, train_sessions),
        split_dir / HEADLINES_UNSEEN_TRAIN,
        headlines_unseen,
    )
    _write_optional_parquet(
        _subset_rows(headlines_seen, val_sessions),
        split_dir / HEADLINES_SEEN_PUBLIC_TEST,
        headlines_seen,
    )
    _write_optional_parquet(
        headlines_seen.iloc[:0].copy() if headlines_seen is not None else None,
        split_dir / HEADLINES_SEEN_PRIVATE_TEST,
        headlines_seen,
    )

    _write_optional_csv(
        _subset_rows(sentiments_seen, train_sessions),
        split_dir / SENTIMENTS_SEEN_TRAIN,
        sentiments_seen,
    )
    _write_optional_csv(
        _subset_rows(sentiments_unseen, train_sessions),
        split_dir / SENTIMENTS_UNSEEN_TRAIN,
        sentiments_unseen,
    )
    _write_optional_csv(
        _subset_rows(sentiments_seen, val_sessions),
        split_dir / SENTIMENTS_SEEN_PUBLIC_TEST,
        sentiments_seen,
    )
    _write_optional_csv(
        sentiments_seen.iloc[:0].copy() if sentiments_seen is not None else None,
        split_dir / SENTIMENTS_SEEN_PRIVATE_TEST,
        sentiments_seen,
    )

    source_readme = augmented_data_dir / "README.md"
    if source_readme.is_file():
        shutil.copy2(source_readme, split_dir / "README.md")


def _expected_test_session_count(data_dir: Path) -> int:
    total = 0
    for name in (BARS_SEEN_PUBLIC_TEST, BARS_SEEN_PRIVATE_TEST):
        path = data_dir / name
        if path.is_file():
            df = pd.read_parquet(path, columns=["session"])
            total += int(df["session"].nunique())
    return total


def _selection_artifacts(base_dir: Path, base: Base) -> tuple[Path, Path, Path]:
    family_dir = base_dir / "selection" / base.family
    return (
        family_dir / f"{base.name}_diagnostics.json",
        family_dir / f"{base.name}_submission.csv",
        family_dir / f"{base.name}_log.txt",
    )


def _final_artifacts(base_dir: Path, base: Base) -> tuple[Path, Path, Path]:
    family_dir = base_dir / "final" / base.family
    return (
        family_dir / f"{base.name}_diagnostics.json",
        family_dir / f"{base.name}_submission.csv",
        family_dir / f"{base.name}_log.txt",
    )


def _run_base_selection(
    base: Base,
    *,
    split_data_dir: Path,
    out_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    diag_path, submission_path, log_path = _selection_artifacts(out_dir, base)
    cmd = [
        "python",
        str(base.script),
        "--data-dir",
        str(split_data_dir),
        "--seed",
        str(seed),
        "--no-adversarial" if base.family == "tailored" else "",
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        f"{base.family}_{base.name}_multihead_augsplit",
        *base.args,
    ]
    cmd = [c for c in cmd if c]
    _run_command(cmd, log_path)
    return diag_path, submission_path


def _run_base_final(
    base: Base,
    *,
    augmented_data_dir: Path,
    out_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    diag_path, submission_path, log_path = _final_artifacts(out_dir, base)
    cmd = [
        "python",
        str(base.script),
        "--data-dir",
        str(augmented_data_dir),
        "--seed",
        str(seed),
        "--no-adversarial" if base.family == "tailored" else "",
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        f"{base.family}_{base.name}_multihead_fullaug",
        *base.args,
    ]
    cmd = [c for c in cmd if c]
    _run_command(cmd, log_path)
    return diag_path, submission_path


def _sharpe(pnl: np.ndarray) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    if x.size == 0:
        return 0.0
    sd = float(np.std(x, ddof=0))
    if sd == 0.0:
        return 0.0
    return float(np.mean(x)) / sd * 16.0


def _simplex_grid(n_heads: int, step: float) -> np.ndarray:
    step = max(min(float(step), 1.0), 0.001)
    K = int(round(1.0 / step))

    def gen(left: int, depth: int):
        if depth == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in gen(left - i, depth - 1):
                yield (i, *rest)

    pts = np.array(list(gen(K, n_heads)), dtype=np.float64) / float(K)
    return pts


def _optimise_weights(
    positions: np.ndarray, returns: np.ndarray, step: float
) -> dict:
    """positions: (n_sessions, n_heads). returns: (n_sessions,). Maximise Sharpe of the
    convex combination over a simplex grid, then refine around the winner.
    """
    if positions.shape[0] == 0:
        raise RuntimeError("Empty positions matrix; cannot optimise weights.")

    n_heads = positions.shape[1]
    grid = _simplex_grid(n_heads, step)
    pnl_matrix = positions * returns[:, None]
    combined = grid @ pnl_matrix.T
    sharpes = []
    for i in range(combined.shape[0]):
        sharpes.append(_sharpe(combined[i]))
    sharpes_arr = np.asarray(sharpes, dtype=np.float64)
    best_idx = int(np.argmax(sharpes_arr))
    best_w = grid[best_idx]
    best_sh = float(sharpes_arr[best_idx])

    # Small refinement around best_w on a finer simplex (step / 4), constrained to
    # a neighbourhood of width ~3 * step around best_w.
    refine_step = max(step / 4.0, 0.0025)
    fine_grid = _simplex_grid(n_heads, refine_step)
    radius = 3.0 * step
    mask = np.all(np.abs(fine_grid - best_w[None, :]) <= radius + 1e-9, axis=1)
    fine_grid = fine_grid[mask]
    if fine_grid.size > 0:
        combined_fine = fine_grid @ pnl_matrix.T
        sharpes_fine = np.asarray(
            [_sharpe(combined_fine[i]) for i in range(combined_fine.shape[0])],
            dtype=np.float64,
        )
        best_fine = int(np.argmax(sharpes_fine))
        if float(sharpes_fine[best_fine]) > best_sh:
            best_w = fine_grid[best_fine]
            best_sh = float(sharpes_fine[best_fine])

    return {
        "best_weights": best_w.tolist(),
        "best_validation_sharpe": best_sh,
        "grid_step": float(step),
        "grid_size": int(grid.shape[0]),
        "refined": True,
    }


def _load_positions(submission_path: Path) -> pd.DataFrame:
    df = pd.read_csv(submission_path)
    return df[["session", "target_position"]].copy()


def _build_position_matrix(
    submissions: list[Path], head_labels: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    merged: pd.DataFrame | None = None
    for path, label in zip(submissions, head_labels):
        df = _load_positions(path).rename(columns={"target_position": label})
        merged = df if merged is None else merged.merge(df, on="session", how="inner")
    assert merged is not None
    merged = merged.sort_values("session").reset_index(drop=True)
    matrix = merged[head_labels].to_numpy(dtype=np.float64)
    return merged, matrix


def _write_report(
    out_dir: Path,
    *,
    split_dir: Path,
    split_info: dict,
    expected_rows: int,
    base_val_records: list[dict],
    weights_record: dict,
    final_records: list[dict],
    multihead_path: Path,
    multihead_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# Multihead Final")
    lines.append("")
    lines.append("## Protocol")
    lines.append("- Start from `data/augmented/`.")
    lines.append(
        "- Session-wholesale random split: every augmented session's seen + unseen bars "
        "go entirely into either the augmented-train split or the augmented-validation split."
    )
    lines.append("- Train each base model on the augmented-train split.")
    lines.append(
        "- Optimise convex-combination ensemble weights (sum to 1, non-negative) on the "
        "held-out augmented-validation Sharpe."
    )
    lines.append(
        "- Refit each base model on the **full** augmented training set and predict the "
        "real public + private test sets."
    )
    lines.append(
        "- Combine those final per-base 20000-row predictions using the locked weights into "
        "a single multihead submission."
    )
    lines.append("")
    lines.append("## Split")
    lines.append(f"- Split data dir: `{split_dir}`")
    lines.append(f"- Seed: `{split_info['seed']}`")
    lines.append(f"- Train sessions: `{split_info['n_train_sessions']}`")
    lines.append(f"- Validation sessions: `{split_info['n_validation_sessions']}`")
    lines.append(f"- Validation fraction: `{split_info['validation_fraction']}`")
    lines.append(f"- Expected final submission rows: `{expected_rows}`")
    lines.append("")
    lines.append("## Base Validation Sharpes")
    for rec in base_val_records:
        lines.append(
            f"- `{rec['family']}/{rec['name']}`: validation Sharpe `{rec['validation_sharpe']:.4f}`"
        )
    lines.append("")
    lines.append("## Locked Ensemble Weights")
    for label, weight in weights_record["weights"].items():
        lines.append(f"- `{label}`: `{weight:.4f}`")
    lines.append("")
    lines.append(
        f"- Multihead validation Sharpe: `{weights_record['multihead_validation_sharpe']:.4f}`"
    )
    lines.append(
        f"- Best single-base validation Sharpe: `{weights_record['best_single_base_validation_sharpe']:.4f}`"
    )
    lines.append("")
    lines.append("## Per-Base Final Submissions (full augmented train → real test)")
    for rec in final_records:
        rows = int(rec.get("rows", 0))
        ok = "OK" if rows == expected_rows else f"MISMATCH (expected {expected_rows})"
        lines.append(
            f"- `{rec['family']}/{rec['name']}`: `{rec['submission_path']}` ({rows} rows, {ok})"
        )
    lines.append("")
    lines.append("## Final Multihead Submission")
    lines.append(f"- Path: `{multihead_path}`")
    lines.append(f"- Rows: `{multihead_rows}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    augmented_data_dir = args.augmented_data_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir.resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_rows = _expected_test_session_count(augmented_data_dir)
    if expected_rows <= 0:
        raise SystemExit(
            "Augmented data dir has no public/private test sessions; cannot build final submissions."
        )

    train_sessions, val_sessions = _select_sessions(
        augmented_data_dir, int(args.seed), float(args.validation_fraction)
    )
    split_dir = run_dir / "split_data"
    _build_split_dataset(
        augmented_data_dir,
        split_dir,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
    )
    split_info = {
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "n_train_sessions": int(len(train_sessions)),
        "n_validation_sessions": int(len(val_sessions)),
        "expected_final_submission_rows": int(expected_rows),
    }
    _write_json(run_dir / "split_info.json", split_info)

    bases = _bases()

    # ------------------------------------------------------------------
    # Phase 1: train each base on augmented-train split, predict val split.
    # ------------------------------------------------------------------
    selection_paths: list[Path] = []
    head_labels: list[str] = []
    base_val_records: list[dict] = []
    for base in bases:
        print(
            f"\n==== SELECTION [{base.family}/{base.name}] ====",
            flush=True,
        )
        _, sub_path = _run_base_selection(
            base,
            split_data_dir=split_dir,
            out_dir=run_dir,
            seed=int(args.seed),
        )
        selection_paths.append(sub_path)
        head_labels.append(f"{base.family}/{base.name}")

    labels = _load_train_labels(augmented_data_dir)
    val_labels = labels.loc[labels["session"].isin(val_sessions)].copy()

    merged_val, val_matrix = _build_position_matrix(selection_paths, head_labels)
    aligned_val = merged_val.merge(val_labels[["session", "R"]], on="session", how="inner")
    if aligned_val.empty:
        raise RuntimeError("No validation sessions could be aligned across base submissions.")
    val_returns = aligned_val["R"].to_numpy(dtype=np.float64)
    val_positions = aligned_val[head_labels].to_numpy(dtype=np.float64)

    for j, label in enumerate(head_labels):
        sh = _sharpe(val_positions[:, j] * val_returns)
        base_val_records.append(
            {"family": bases[j].family, "name": bases[j].name, "validation_sharpe": float(sh)}
        )

    # Optionally append a flat-long head as a safety baseline.
    if args.allow_zero_bias:
        head_labels.append("flat_long")
        val_positions = np.column_stack([val_positions, np.ones(val_positions.shape[0])])

    weights_info = _optimise_weights(
        val_positions, val_returns, float(args.weight_grid_step)
    )
    weights_dict = {
        label: float(w) for label, w in zip(head_labels, weights_info["best_weights"])
    }
    best_single_base = float(max(rec["validation_sharpe"] for rec in base_val_records))
    weights_record = {
        "weights": weights_dict,
        "multihead_validation_sharpe": float(weights_info["best_validation_sharpe"]),
        "best_single_base_validation_sharpe": best_single_base,
        "grid_step": float(weights_info["grid_step"]),
        "grid_size": int(weights_info["grid_size"]),
    }
    _write_json(run_dir / "weights.json", weights_record)

    # ------------------------------------------------------------------
    # Phase 2: retrain each base on the full augmented dataset.
    # ------------------------------------------------------------------
    final_paths: list[Path] = []
    final_records: list[dict] = []
    for base in bases:
        print(
            f"\n==== FINAL FIT [{base.family}/{base.name}] ====",
            flush=True,
        )
        _, sub_path = _run_base_final(
            base,
            augmented_data_dir=augmented_data_dir,
            out_dir=run_dir,
            seed=int(args.seed),
        )
        rows = int(len(pd.read_csv(sub_path)))
        final_paths.append(sub_path)
        final_records.append(
            {
                "family": base.family,
                "name": base.name,
                "submission_path": str(sub_path),
                "rows": rows,
            }
        )
    _write_json(run_dir / "final_progress.json", {"completed": final_records})

    # ------------------------------------------------------------------
    # Phase 3: combine using locked weights into a single 20000-row submission.
    # ------------------------------------------------------------------
    final_head_labels = [f"{b.family}/{b.name}" for b in bases]
    merged_test, test_matrix = _build_position_matrix(final_paths, final_head_labels)
    if args.allow_zero_bias:
        final_head_labels.append("flat_long")
        test_matrix = np.column_stack([test_matrix, np.ones(test_matrix.shape[0])])

    weight_vec = np.array(
        [weights_dict[label] for label in final_head_labels], dtype=np.float64
    )
    combined = test_matrix @ weight_vec
    out_df = pd.DataFrame(
        {
            "session": merged_test["session"].astype(np.int64),
            "target_position": combined,
        }
    )
    multihead_path = run_dir / "multihead_submission.csv"
    out_df.to_csv(multihead_path, index=False)
    multihead_rows = int(len(out_df))

    summary = {
        "weights": weights_dict,
        "multihead_validation_sharpe": weights_record["multihead_validation_sharpe"],
        "best_single_base_validation_sharpe": best_single_base,
        "expected_rows": int(expected_rows),
        "multihead_submission_path": str(multihead_path),
        "multihead_submission_rows": int(multihead_rows),
        "row_count_ok": bool(multihead_rows == expected_rows),
        "base_val_records": base_val_records,
        "final_records": final_records,
    }
    _write_json(run_dir / "summary.json", summary)

    _write_report(
        run_dir,
        split_dir=split_dir,
        split_info=split_info,
        expected_rows=int(expected_rows),
        base_val_records=base_val_records,
        weights_record=weights_record,
        final_records=final_records,
        multihead_path=multihead_path,
        multihead_rows=multihead_rows,
    )

    print(f"\nWrote multihead_final artifacts to {run_dir}")
    print(f"Multihead submission: {multihead_path} ({multihead_rows} rows)")
    print(f"Multihead validation Sharpe: {weights_record['multihead_validation_sharpe']:.4f}")
    print(f"Best single-base validation Sharpe: {best_single_base:.4f}")
    print("Locked weights:")
    for label, w in weights_dict.items():
        print(f"  {label}: {w:.4f}")


if __name__ == "__main__":
    main()

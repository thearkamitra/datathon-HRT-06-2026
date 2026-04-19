#!/usr/bin/env python3
"""Select on an augmented random split, then train all variants on full augmented data.

Protocol implemented exactly as requested by the user:

1. Start from the augmented training set.
2. Make a random train/validation split over augmented sessions.
3. Train each variant on the train split only.
4. Choose model variants using Sharpe on the held-out validation split.
5. Refit each variant on the full augmented training set.
6. Write fresh public/private submission CSVs into ``Submissions/separate_final``.

The selection score is an external held-out Sharpe computed on validation
sessions whose second-half labels are known from the augmented train set.
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
DEFAULT_OUT_DIR = REPO_ROOT / "Submissions" / "separate_final"

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


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str
    script: Path
    args: tuple[str, ...]


def _candidate_catalog() -> list[Candidate]:
    scripts = {
        "tailored": REPO_ROOT / "scripts" / "tailored-predictor",
        "prob_reg": REPO_ROOT / "scripts" / "prob-reg-predictor",
        "regime": REPO_ROOT / "scripts" / "regime-predictor",
    }
    return [
        Candidate("tailored", "full_ohlc_default", scripts["tailored"], ()),
        Candidate(
            "tailored",
            "full_ohlc_no_weights",
            scripts["tailored"],
            ("--no-sample-weights",),
        ),
        Candidate(
            "tailored",
            "full_ohlc_cv3",
            scripts["tailored"],
            ("--cv-repeats", "3"),
        ),
        Candidate(
            "tailored",
            "full_ohlc_news_hash64_cv3",
            scripts["tailored"],
            ("--use-news", "--news-hash-features", "64", "--cv-repeats", "3"),
        ),
        Candidate("prob_reg", "ridge_news_gaussian", scripts["prob_reg"], ()),
        Candidate(
            "prob_reg",
            "ridge_no_news_gaussian",
            scripts["prob_reg"],
            ("--no-news",),
        ),
        Candidate(
            "prob_reg",
            "elastic_news_gaussian",
            scripts["prob_reg"],
            ("--mean-model", "elastic_net"),
        ),
        Candidate(
            "prob_reg",
            "ridge_news_empirical",
            scripts["prob_reg"],
            ("--empirical-quantiles",),
        ),
        Candidate(
            "regime",
            "m1",
            scripts["regime"],
            (
                "--method",
                "m1",
                "--n-sim",
                "256",
                "--select-cv-splits",
                "3",
                "--select-starts",
                "3",
                "--oof-splits",
                "4",
            ),
        ),
        Candidate(
            "regime",
            "m1_linear",
            scripts["regime"],
            (
                "--method",
                "m1-linear",
                "--n-sim",
                "256",
                "--select-cv-splits",
                "3",
                "--select-starts",
                "3",
                "--oof-splits",
                "4",
            ),
        ),
        Candidate(
            "regime",
            "m1_linear_news",
            scripts["regime"],
            (
                "--method",
                "m1-linear",
                "--use-news",
                "--n-sim",
                "256",
                "--select-cv-splits",
                "3",
                "--select-starts",
                "3",
                "--oof-splits",
                "4",
            ),
        ),
        Candidate(
            "regime",
            "m2",
            scripts["regime"],
            (
                "--method",
                "m2",
                "--n-sim",
                "256",
                "--select-cv-splits",
                "3",
                "--select-starts",
                "3",
                "--oof-splits",
                "4",
            ),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run augmented split selection and refresh final submissions."
    )
    parser.add_argument(
        "--augmented-data-dir",
        type=Path,
        default=DEFAULT_AUGMENTED_DATA,
        help="Source augmented data directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root for separate_final artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the augmented session split and model runners.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of augmented sessions used as the held-out validation split.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="tailored,prob_reg,regime",
        help="Comma-separated subset of families to run.",
    )
    parser.add_argument(
        "--skip-final-fit",
        action="store_true",
        help="Run selection only; do not retrain variants on full augmented data.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def _run_command(cmd: Sequence[str]) -> None:
    print(f"\n[run] {' '.join(cmd)}\n", flush=True)
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


def _select_sessions(augmented_data_dir: Path, seed: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    seen = pd.read_parquet(augmented_data_dir / BARS_SEEN_TRAIN, columns=["session"])
    sessions = np.sort(seen["session"].unique().astype(np.int64))
    if sessions.size < 2:
        raise RuntimeError("Need at least 2 augmented sessions to build a train/validation split.")

    rng = np.random.default_rng(int(seed))
    shuffled = sessions.copy()
    rng.shuffle(shuffled)

    n_val = int(round(float(validation_fraction) * float(shuffled.size)))
    n_val = max(1, min(shuffled.size - 1, n_val))
    val_sessions = np.sort(shuffled[:n_val])
    train_sessions = np.sort(shuffled[n_val:])
    return train_sessions, val_sessions


def _subset_rows(df: pd.DataFrame | None, sessions: Iterable[int]) -> pd.DataFrame | None:
    if df is None:
        return None
    sess_set = set(int(x) for x in sessions)
    if "session" not in df.columns:
        return df.copy()
    return df.loc[df["session"].astype(np.int64).isin(sess_set)].copy()


def _write_optional_parquet(df: pd.DataFrame | None, path: Path, template: pd.DataFrame | None) -> None:
    if template is None and df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if df is not None else template.iloc[:0].copy()
    out.to_parquet(path, index=False)


def _write_optional_csv(df: pd.DataFrame | None, path: Path, template: pd.DataFrame | None) -> None:
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
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    bars_seen = pd.read_parquet(augmented_data_dir / BARS_SEEN_TRAIN)
    bars_unseen = pd.read_parquet(augmented_data_dir / BARS_UNSEEN_TRAIN)

    headlines_seen = None
    headlines_unseen = None
    sentiments_seen = None
    sentiments_unseen = None

    if (augmented_data_dir / HEADLINES_SEEN_TRAIN).is_file():
        headlines_seen = pd.read_parquet(augmented_data_dir / HEADLINES_SEEN_TRAIN)
    if (augmented_data_dir / HEADLINES_UNSEEN_TRAIN).is_file():
        headlines_unseen = pd.read_parquet(augmented_data_dir / HEADLINES_UNSEEN_TRAIN)
    if (augmented_data_dir / SENTIMENTS_SEEN_TRAIN).is_file():
        sentiments_seen = pd.read_csv(augmented_data_dir / SENTIMENTS_SEEN_TRAIN)
    if (augmented_data_dir / SENTIMENTS_UNSEEN_TRAIN).is_file():
        sentiments_unseen = pd.read_csv(augmented_data_dir / SENTIMENTS_UNSEEN_TRAIN)

    train_seen = _subset_rows(bars_seen, train_sessions)
    train_unseen = _subset_rows(bars_unseen, train_sessions)
    val_seen = _subset_rows(bars_seen, val_sessions)

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


def _validation_sharpe(submission_path: Path, val_labels: pd.DataFrame) -> float:
    sub = pd.read_csv(submission_path)
    merged = sub.merge(val_labels[["session", "R"]], on="session", how="inner")
    if merged.empty:
        raise RuntimeError(f"No validation sessions aligned in {submission_path}")
    pnl = merged["target_position"].to_numpy(dtype=np.float64) * merged["R"].to_numpy(dtype=np.float64)
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=0))
    return 0.0 if std == 0.0 else mean / std * 16.0


def _selection_artifacts(base_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    family_dir = base_dir / "selection" / candidate.family
    return (
        family_dir / f"{candidate.name}_diagnostics.json",
        family_dir / f"{candidate.name}_submission.csv",
    )


def _final_artifacts(base_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    family_dir = base_dir / "final" / candidate.family
    return (
        family_dir / f"{candidate.name}_diagnostics.json",
        family_dir / f"{candidate.name}_submission.csv",
    )


def _run_selection_candidate(
    candidate: Candidate,
    *,
    split_data_dir: Path,
    out_dir: Path,
    seed: int,
    val_labels: pd.DataFrame,
) -> dict:
    diag_path, submission_path = _selection_artifacts(out_dir, candidate)
    cmd = [
        "python",
        str(candidate.script),
        "--data-dir",
        str(split_data_dir),
        "--seed",
        str(seed),
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        f"{candidate.family}_{candidate.name}_augsplit",
        *candidate.args,
    ]
    _run_command(cmd)
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    return {
        "family": candidate.family,
        "candidate": candidate.name,
        "validation_sharpe": float(_validation_sharpe(submission_path, val_labels)),
        "train_internal_oof_sharpe": float(diag.get("oof_sharpe_tuned", 0.0)),
        "diagnostics_path": str(diag_path),
        "selection_submission_path": str(submission_path),
        "tuned_mode": diag.get("tuned_mode"),
        "use_news": bool(diag.get("use_news", diag.get("news_enabled", False))),
    }


def _run_final_candidate(
    candidate: Candidate,
    *,
    augmented_data_dir: Path,
    out_dir: Path,
    seed: int,
) -> dict:
    diag_path, submission_path = _final_artifacts(out_dir, candidate)
    cmd = [
        "python",
        str(candidate.script),
        "--data-dir",
        str(augmented_data_dir),
        "--seed",
        str(seed),
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        f"{candidate.family}_{candidate.name}_fullaug",
        *candidate.args,
    ]
    _run_command(cmd)
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    return {
        "family": candidate.family,
        "candidate": candidate.name,
        "final_submission_path": str(submission_path),
        "final_diagnostics_path": str(diag_path),
        "full_aug_internal_oof_sharpe": float(diag.get("oof_sharpe_tuned", 0.0)),
        "full_aug_train_sessions": int(diag.get("n_train_sessions", 0)),
    }


def _write_report(
    out_dir: Path,
    *,
    split_dir: Path,
    split_info: dict,
    selection_df: pd.DataFrame,
    family_best_df: pd.DataFrame,
    final_df: pd.DataFrame,
    overall_best: dict | None,
) -> None:
    lines: list[str] = []
    lines.append("# Separate Final")
    lines.append("")
    lines.append("## Protocol")
    lines.append("- Start from `data/augmented/`.")
    lines.append("- Make a random train/validation split over augmented sessions.")
    lines.append("- Train each candidate on the augmented-train split.")
    lines.append("- Choose variants by held-out validation Sharpe.")
    lines.append("- Retrain each variant on the full augmented set and generate fresh public/private submission files.")
    lines.append("")
    lines.append("## Split")
    lines.append(f"- Split data dir: `{split_dir}`")
    lines.append(f"- Seed: `{split_info['seed']}`")
    lines.append(f"- Train sessions: `{split_info['n_train_sessions']}`")
    lines.append(f"- Validation sessions: `{split_info['n_validation_sessions']}`")
    lines.append(f"- Validation fraction: `{split_info['validation_fraction']}`")
    lines.append("")
    lines.append("## Selection Scores")
    for _, row in selection_df.sort_values(["validation_sharpe", "family"], ascending=[False, True]).iterrows():
        lines.append(
            f"- `{row['family']}/{row['candidate']}`: validation Sharpe `{float(row['validation_sharpe']):.4f}` "
            f"(train-internal OOF `{float(row['train_internal_oof_sharpe']):.4f}`)"
        )
    lines.append("")
    lines.append("## Family Winners")
    for _, row in family_best_df.iterrows():
        lines.append(
            f"- `{row['family']}` winner: `{row['candidate']}` at validation Sharpe `{float(row['validation_sharpe']):.4f}`"
        )
    if overall_best is not None:
        lines.append("")
        lines.append("## Overall Winner")
        lines.append(
            f"- `{overall_best['family']}/{overall_best['candidate']}` at validation Sharpe `{float(overall_best['validation_sharpe']):.4f}`"
        )
        if overall_best.get("best_overall_submission_path"):
            lines.append(f"- Submission: `{overall_best['best_overall_submission_path']}`")
    lines.append("")
    lines.append("## Final Submission Files")
    for _, row in final_df.sort_values(["family", "candidate"]).iterrows():
        lines.append(f"- `{row['family']}/{row['candidate']}`: `{row['final_submission_path']}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    augmented_data_dir = args.augmented_data_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir.resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_families = {x.strip() for x in args.families.split(",") if x.strip()}
    candidates = [c for c in _candidate_catalog() if c.family in requested_families]
    if not candidates:
        raise SystemExit("No candidates selected.")

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
        "train_sessions": [int(x) for x in train_sessions.tolist()],
        "validation_sessions": [int(x) for x in val_sessions.tolist()],
    }
    _write_json(run_dir / "split_info.json", split_info)

    labels = _load_train_labels(augmented_data_dir)
    val_labels = labels.loc[labels["session"].isin(val_sessions)].copy()

    selection_records: list[dict] = []
    for candidate in candidates:
        selection_records.append(
            _run_selection_candidate(
                candidate,
                split_data_dir=split_dir,
                out_dir=run_dir,
                seed=int(args.seed),
                val_labels=val_labels,
            )
        )

    selection_df = pd.DataFrame(selection_records)
    selection_df.to_csv(run_dir / "selection_summary.csv", index=False)

    family_best_rows = []
    for family, family_rows in selection_df.groupby("family", sort=True):
        family_best_rows.append(
            family_rows.sort_values(
                ["validation_sharpe", "train_internal_oof_sharpe", "candidate"],
                ascending=[False, False, True],
            ).iloc[0].to_dict()
        )
    family_best_df = pd.DataFrame(family_best_rows)
    family_best_df.to_csv(run_dir / "family_winners.csv", index=False)

    final_records: list[dict] = []
    if not args.skip_final_fit:
        for candidate in candidates:
            final_records.append(
                _run_final_candidate(
                    candidate,
                    augmented_data_dir=augmented_data_dir,
                    out_dir=run_dir,
                    seed=int(args.seed),
                )
            )
    final_df = pd.DataFrame(final_records)
    if not final_df.empty:
        final_df.to_csv(run_dir / "final_submissions.csv", index=False)

    overall_best = None
    if not selection_df.empty:
        overall_best = (
            selection_df.sort_values(
                ["validation_sharpe", "train_internal_oof_sharpe", "family", "candidate"],
                ascending=[False, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        if not final_df.empty:
            match = final_df.loc[
                (final_df["family"] == overall_best["family"])
                & (final_df["candidate"] == overall_best["candidate"])
            ]
            if not match.empty:
                overall_best["best_overall_submission_path"] = match.iloc[0][
                    "final_submission_path"
                ]
                best_src = Path(str(match.iloc[0]["final_submission_path"]))
                shutil.copy2(best_src, run_dir / "best_overall_submission.csv")
        _write_json(run_dir / "overall_winner.json", overall_best)

    _write_report(
        run_dir,
        split_dir=split_dir,
        split_info=split_info,
        selection_df=selection_df,
        family_best_df=family_best_df,
        final_df=final_df,
        overall_best=overall_best,
    )
    print(f"\nWrote separate_final artifacts to {run_dir}")
    if overall_best is not None:
        print(
            f"Overall winner: {overall_best['family']}/{overall_best['candidate']} "
            f"(validation Sharpe={float(overall_best['validation_sharpe']):.4f})"
        )


if __name__ == "__main__":
    main()

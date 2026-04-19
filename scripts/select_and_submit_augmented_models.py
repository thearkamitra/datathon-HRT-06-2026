#!/usr/bin/env python3
"""Select honest configs on original train data, then fit final models on augmented data.

Workflow:

1. Model/config selection is performed on the original `data/` train set only.
2. The selection metric is each pipeline's honest OOF Sharpe on that original train set.
3. The tuned sizing config discovered on original data is frozen.
4. The chosen config for each family is refit on `data/augmented/`.
5. Final submission CSVs are emitted with descriptive strategy/mode tags.

This avoids the main pitfall discussed in the repo audit:
using public-test-like or synthetic rows to *choose* the model. Augmented rows
are used only for the final fit after the architecture/configuration is selected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTION_DATA = REPO_ROOT / "data"
DEFAULT_FINAL_DATA = REPO_ROOT / "data" / "augmented"


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str
    script: Path
    args: tuple[str, ...]
    score_key: str = "oof_sharpe_tuned"


def _augmentation_strategy_notes() -> list[tuple[str, str]]:
    return [
        (
            "path_noise",
            "Smooth zero-sum perturbations in log-return space inside each half-session, "
            "preserving the seen cutoff close and the final close.",
        ),
        (
            "path_noise_shape",
            "Same close-path perturbation plus mild open/high/low jitter so candlestick "
            "shape features vary without changing the label.",
        ),
        (
            "vol_scale",
            "Scales within-half volatility around the original drift, preserving the "
            "segment endpoint exactly.",
        ),
        (
            "time_warp",
            "Monotone time warping of the observed close path so the timing of moves "
            "changes while the path endpoints stay fixed.",
        ),
    ]


def _candidate_catalog() -> list[Candidate]:
    scripts = {
        "tailored": REPO_ROOT / "scripts" / "tailored-predictor",
        "prob_reg": REPO_ROOT / "scripts" / "prob-reg-predictor",
        "regime": REPO_ROOT / "scripts" / "regime-predictor",
    }
    return [
        Candidate("tailored", "full_ohlc_default", scripts["tailored"], ()),
        Candidate("tailored", "full_ohlc_no_weights", scripts["tailored"], ("--no-sample-weights",)),
        Candidate("tailored", "full_ohlc_cv3", scripts["tailored"], ("--cv-repeats", "3")),
        Candidate(
            "tailored",
            "full_ohlc_news_cv3",
            scripts["tailored"],
            ("--use-news", "--cv-repeats", "3"),
        ),
        Candidate("prob_reg", "ridge_news_gaussian", scripts["prob_reg"], ()),
        Candidate("prob_reg", "ridge_no_news_gaussian", scripts["prob_reg"], ("--no-news",)),
        Candidate("prob_reg", "elastic_news_gaussian", scripts["prob_reg"], ("--mean-model", "elastic_net")),
        Candidate("prob_reg", "ridge_news_empirical", scripts["prob_reg"], ("--empirical-quantiles",)),
        Candidate(
            "regime",
            "m1",
            scripts["regime"],
            ("--method", "m1", "--n-sim", "256", "--select-cv-splits", "3", "--select-starts", "3", "--oof-splits", "4"),
        ),
        Candidate(
            "regime",
            "m1_linear",
            scripts["regime"],
            ("--method", "m1-linear", "--n-sim", "256", "--select-cv-splits", "3", "--select-starts", "3", "--oof-splits", "4"),
        ),
        Candidate(
            "regime",
            "m1_linear_news",
            scripts["regime"],
            ("--method", "m1-linear", "--use-news", "--n-sim", "256", "--select-cv-splits", "3", "--select-starts", "3", "--oof-splits", "4"),
        ),
        Candidate(
            "regime",
            "m2",
            scripts["regime"],
            ("--method", "m2", "--n-sim", "256", "--select-cv-splits", "3", "--select-starts", "3", "--oof-splits", "4"),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run honest model selection on original data, then final augmented fits."
    )
    parser.add_argument(
        "--selection-data-dir",
        type=Path,
        default=DEFAULT_SELECTION_DATA,
        help="Original data directory used for honest model selection.",
    )
    parser.add_argument(
        "--final-data-dir",
        type=Path,
        default=DEFAULT_FINAL_DATA,
        help="Augmented data directory used for final refits.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "Submissions" / "augmented_model_suite",
        help="Output directory for reports, diagnostics, and submissions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Common random seed passed to all family runners.",
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
        help="Run selection only; do not fit the winning configs on augmented data.",
    )
    return parser.parse_args()


def _run_command(cmd: Sequence[str]) -> None:
    printable = " ".join(cmd)
    print(f"\n[run] {printable}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def _selection_artifacts(base_dir: Path, candidate: Candidate) -> tuple[Path, Path, Path]:
    family_dir = base_dir / "selection" / candidate.family
    return (
        family_dir / f"{candidate.name}_diagnostics.json",
        family_dir / f"{candidate.name}_sizing.json",
        family_dir / f"{candidate.name}_submission.csv",
    )


def _final_artifacts(base_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    family_dir = base_dir / "final" / candidate.family
    return (
        family_dir / f"{candidate.name}_diagnostics.json",
        family_dir / f"{candidate.name}_submission.csv",
    )


def _run_selection(
    candidate: Candidate,
    *,
    selection_data_dir: Path,
    out_dir: Path,
    seed: int,
) -> dict:
    diag_path, sizing_path, submission_path = _selection_artifacts(out_dir, candidate)
    tag = f"{candidate.family}_{candidate.name}_origcv"
    cmd = [
        "python",
        str(candidate.script),
        "--data-dir",
        str(selection_data_dir),
        "--seed",
        str(seed),
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        tag,
        *candidate.args,
    ]
    _run_command(cmd)
    diag = _load_json(diag_path)
    sizing = diag.get("tuned_sizing", {})
    _write_json(sizing_path, sizing)
    return {
        "family": candidate.family,
        "candidate": candidate.name,
        "selection_score": float(diag[candidate.score_key]),
        "diagnostics_path": str(diag_path),
        "sizing_path": str(sizing_path),
        "selection_submission_path": str(submission_path),
        "sizing_source": diag.get("sizing_source"),
        "tuned_mode": diag.get("tuned_mode"),
        "tuned_alpha": diag.get("tuned_alpha"),
        "tuned_lambda": diag.get("tuned_lambda"),
        "tuned_theta": diag.get("tuned_theta"),
        "tuned_tau_quantile": diag.get("tuned_tau_quantile"),
    }


def _run_final_fit(
    candidate: Candidate,
    *,
    final_data_dir: Path,
    out_dir: Path,
    seed: int,
    sizing_path: Path,
) -> dict:
    diag_path, submission_path = _final_artifacts(out_dir, candidate)
    tag = f"{candidate.family}_{candidate.name}_augfit"
    cmd = [
        "python",
        str(candidate.script),
        "--data-dir",
        str(final_data_dir),
        "--seed",
        str(seed),
        "--fixed-sizing-in",
        str(sizing_path),
        "--diagnostics-out",
        str(diag_path),
        "--submission-out",
        str(submission_path),
        "--tag",
        tag,
        *candidate.args,
    ]
    _run_command(cmd)
    diag = _load_json(diag_path)
    return {
        "family": candidate.family,
        "candidate": candidate.name,
        "final_submission_path": str(submission_path),
        "final_diagnostics_path": str(diag_path),
        "final_train_rows_metric": float(diag.get("n_train_sessions", 0)),
        "final_sizing_source": diag.get("sizing_source"),
    }


def _family_filter(candidates: list[Candidate], requested_families: set[str]) -> list[Candidate]:
    return [c for c in candidates if c.family in requested_families]


def _build_report(
    out_dir: Path,
    *,
    selection_data_dir: Path,
    final_data_dir: Path,
    selection_rows: pd.DataFrame,
    family_best: pd.DataFrame,
    final_rows: pd.DataFrame,
    overall_winner: dict | None,
) -> None:
    aug_manifest_path = final_data_dir / "augmentation_manifest.json"
    aug_manifest = _load_json(aug_manifest_path) if aug_manifest_path.is_file() else {}

    lines: list[str] = []
    lines.append("# Augmented Model Suite")
    lines.append("")
    lines.append("## Validation Logic")
    lines.append(f"- Honest model selection data: `{selection_data_dir}`.")
    lines.append(f"- Final refit data: `{final_data_dir}`.")
    lines.append("- Model architecture/configuration is selected on the original train set only.")
    lines.append("- Final submissions are produced after refitting the chosen config on the augmented train set.")
    lines.append("- The tuned sizing from honest selection is frozen and reused during final augmented fits.")
    lines.append("- Test files are never transformed and are never used to choose the model.")
    lines.append("")
    lines.append("## Legit Augmentations")
    for name, desc in _augmentation_strategy_notes():
        lines.append(f"- `{name}`: {desc}")
    lines.append("- News handling: seen-half headline/sentiment files are copied onto augmented session ids unchanged so `prob-reg` and `regime m1-linear` stay compatible, but no synthetic text is fabricated.")
    if aug_manifest:
        lines.append("")
        lines.append("## Augmented Data Summary")
        lines.append(f"- Original train sessions: `{aug_manifest.get('original_train_sessions')}`.")
        lines.append(f"- Augmented sessions: `{aug_manifest.get('augmented_sessions')}`.")
        lines.append(f"- Copies per session: `{aug_manifest.get('copies_per_session')}`.")
        lines.append(f"- Transform cycle: `{', '.join(aug_manifest.get('transforms', []))}`.")
    lines.append("")
    lines.append("## Selection Results")
    if not selection_rows.empty:
        for _, row in selection_rows.sort_values(["family", "selection_score"], ascending=[True, False]).iterrows():
            lines.append(
                f"- `{row['family']}/{row['candidate']}`: honest OOF Sharpe `{float(row['selection_score']):.4f}`, sizing mode `{row.get('tuned_mode')}`."
            )
    if not family_best.empty:
        lines.append("")
        lines.append("## Family Winners")
        for _, row in family_best.iterrows():
            lines.append(
                f"- `{row['family']}` winner: `{row['candidate']}` at `{float(row['selection_score']):.4f}`."
            )
    if overall_winner is not None:
        lines.append("")
        lines.append("## Overall Winner")
        lines.append(
            f"- `{overall_winner['family']}/{overall_winner['candidate']}` with honest selection Sharpe `{float(overall_winner['selection_score']):.4f}`."
        )
        if overall_winner.get("final_submission_path"):
            lines.append(f"- Final submission: `{overall_winner['final_submission_path']}`.")
    if not final_rows.empty:
        lines.append("")
        lines.append("## Final Submission Files")
        for _, row in final_rows.iterrows():
            lines.append(
                f"- `{row['family']}/{row['candidate']}`: `{row['final_submission_path']}`."
            )

    report_path = out_dir / "suite_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    selection_data_dir = args.selection_data_dir.resolve()
    final_data_dir = args.final_data_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_families = {x.strip() for x in args.families.split(",") if x.strip()}
    candidates = _family_filter(_candidate_catalog(), requested_families)
    if not candidates:
        raise SystemExit("No candidates selected.")

    selection_records: list[dict] = []
    for candidate in candidates:
        record = _run_selection(
            candidate,
            selection_data_dir=selection_data_dir,
            out_dir=out_dir,
            seed=int(args.seed),
        )
        selection_records.append(record)

    selection_df = pd.DataFrame(selection_records)
    selection_df.to_csv(out_dir / "selection_summary.csv", index=False)

    family_best_rows: list[dict] = []
    final_records: list[dict] = []
    for family, family_rows in selection_df.groupby("family", sort=True):
        winner = (
            family_rows.sort_values(["selection_score", "candidate"], ascending=[False, True])
            .iloc[0]
            .to_dict()
        )
        family_best_rows.append(winner)

        if args.skip_final_fit:
            continue

        candidate = next(
            c for c in candidates if c.family == family and c.name == winner["candidate"]
        )
        final_record = _run_final_fit(
            candidate,
            final_data_dir=final_data_dir,
            out_dir=out_dir,
            seed=int(args.seed),
            sizing_path=Path(str(winner["sizing_path"])),
        )
        merged = {**winner, **final_record}
        final_records.append(merged)

    family_best_df = pd.DataFrame(family_best_rows)
    family_best_df.to_csv(out_dir / "family_winners.csv", index=False)

    final_df = pd.DataFrame(final_records)
    if not final_df.empty:
        final_df.to_csv(out_dir / "final_submissions.csv", index=False)

    overall_winner = None
    if not family_best_df.empty:
        overall_winner = (
            family_best_df.sort_values(["selection_score", "family"], ascending=[False, True])
            .iloc[0]
            .to_dict()
        )
        if not final_df.empty:
            match = final_df.loc[
                (final_df["family"] == overall_winner["family"])
                & (final_df["candidate"] == overall_winner["candidate"])
            ]
            if not match.empty:
                overall_winner.update(match.iloc[0].to_dict())
        _write_json(out_dir / "overall_winner.json", overall_winner)

    _build_report(
        out_dir,
        selection_data_dir=selection_data_dir,
        final_data_dir=final_data_dir,
        selection_rows=selection_df,
        family_best=family_best_df,
        final_rows=final_df,
        overall_winner=overall_winner,
    )

    print(f"\nWrote suite artifacts to {out_dir}")
    if overall_winner is not None:
        print(
            f"Overall winner: {overall_winner['family']}/{overall_winner['candidate']} "
            f"(selection Sharpe={float(overall_winner['selection_score']):.4f})"
        )


if __name__ == "__main__":
    main()

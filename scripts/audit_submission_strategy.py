#!/usr/bin/env python3
"""Audit honest-vs-proxy evidence for final model selection.

This script is built for the exact team decision problem in this repository:

1. Load the saved honest OOF benchmark for the tailored family if available.
2. Optionally rerun the `datathon_sharpe` CV grid with and without
   `augment_test_with_proxy` on the same train-session folds.
3. Write a compact report that quantifies how much of the apparent edge comes
   from honest train-only validation versus transductive/proxy augmentation.

The goal is not to pick a winner from one public leaderboard score. The goal is
to estimate which family is most likely to survive the private leaderboard by
separating:

* unconditional market drift (`flat_long`),
* honest train-only excess Sharpe, and
* proxy/test-augmented CV gains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datathon_baseline.io import (  # noqa: E402
    BARS_SEEN_PRIVATE_TEST,
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_TRAIN,
    BARS_UNSEEN_TRAIN,
)
from datathon_sharpe.cv_grid import (  # noqa: E402
    cv_grid_results_to_dataframe,
    run_sharpe_linear_cv_grid,
)


DEFAULT_BENCHMARK_CSV = _REPO_ROOT / "Submissions" / "benchmarks" / "benchmark_table.csv"


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _required_data_files_present(data_dir: Path) -> bool:
    required = (
        BARS_SEEN_TRAIN,
        BARS_UNSEEN_TRAIN,
        BARS_SEEN_PUBLIC_TEST,
        BARS_SEEN_PRIVATE_TEST,
    )
    return all((data_dir / name).is_file() for name in required)


def _load_honest_reference(benchmark_csv: Path) -> tuple[dict, pd.DataFrame | None]:
    if not benchmark_csv.is_file():
        return {}, None

    table = pd.read_csv(benchmark_csv)
    if table.empty or "variant" not in table.columns:
        return {}, table

    ref: dict[str, dict] = {}

    flat = table.loc[table["variant"] == "flat_long"]
    if not flat.empty:
        ref["flat_long"] = flat.iloc[0].to_dict()

    candidates = table.loc[table["variant"] != "flat_long"].copy()
    if not candidates.empty:
        best_mean = candidates.sort_values(
            ["mean_sharpe", "t_vs_flat"],
            ascending=[False, False],
        ).iloc[0]
        best_stable = candidates.sort_values(
            ["t_vs_flat", "mean_sharpe"],
            ascending=[False, False],
        ).iloc[0]
        ref["best_honest_mean"] = best_mean.to_dict()
        ref["best_honest_stable"] = best_stable.to_dict()

    return ref, table


def _run_proxy_audit(
    data_dir: Path,
    *,
    seeds: list[int],
    n_splits: int,
    ridge_alphas: list[float],
    l1_ratios: list[float],
    mse_anchor_lambdas: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detailed_rows: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    for augment in (False, True):
        for seed in seeds:
            rows = run_sharpe_linear_cv_grid(
                data_dir,
                ridge_alphas=ridge_alphas,
                l1_ratios=l1_ratios,
                sharpe_optimizer_labels=("identity",),
                mse_anchor_lambdas=mse_anchor_lambdas,
                n_splits=n_splits,
                random_state=seed,
                augment_test_with_proxy=augment,
            )
            df = cv_grid_results_to_dataframe(rows)
            df.insert(0, "seed", int(seed))
            df.insert(1, "augment_test_with_proxy", bool(augment))
            detailed_rows.append(df)

            best = rows[0]
            best_rows.append(
                {
                    "seed": int(seed),
                    "augment_test_with_proxy": bool(augment),
                    "ridge_alpha": float(best.ridge_alpha),
                    "l1_ratio": float(best.l1_ratio),
                    "mse_anchor_lambda": float(best.mse_anchor_lambda),
                    "best_mean_val_sharpe": float(best.mean_val_sharpe),
                    "best_std_val_sharpe": float(best.std_val_sharpe),
                    "best_fold_val_sharpes": best.fold_val_sharpes,
                }
            )

    detailed = (
        pd.concat(detailed_rows, ignore_index=True)
        if detailed_rows
        else pd.DataFrame()
    )
    best_df = pd.DataFrame(best_rows)

    if best_df.empty:
        return detailed, best_df, pd.DataFrame()

    summary = (
        best_df.groupby("augment_test_with_proxy", dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_best_cv_sharpe=("best_mean_val_sharpe", "mean"),
            std_best_cv_sharpe=("best_mean_val_sharpe", "std"),
            min_best_cv_sharpe=("best_mean_val_sharpe", "min"),
            max_best_cv_sharpe=("best_mean_val_sharpe", "max"),
            mean_fold_std=("best_std_val_sharpe", "mean"),
        )
        .reset_index()
        .sort_values("augment_test_with_proxy")
    )
    summary["std_best_cv_sharpe"] = summary["std_best_cv_sharpe"].fillna(0.0)
    return detailed, best_df, summary


def _recommendation_lines(
    honest_ref: dict,
    proxy_summary: pd.DataFrame | None,
    proxy_best: pd.DataFrame | None,
    *,
    proxy_audit_ran: bool,
    missing_data_reason: str | None,
) -> list[str]:
    lines: list[str] = []
    lines.append("# Submission Strategy Audit")
    lines.append("")

    if honest_ref:
        lines.append("## Honest Reference")
        flat = honest_ref.get("flat_long")
        best_mean = honest_ref.get("best_honest_mean")
        best_stable = honest_ref.get("best_honest_stable")

        if flat is not None:
            lines.append(
                f"- `flat_long` honest OOF Sharpe: `{float(flat['mean_sharpe']):.4f}`."
            )
        if best_mean is not None:
            lines.append(
                "- Best mean honest tailored variant: "
                f"`{best_mean['variant']}` at `{float(best_mean['mean_sharpe']):.4f}`."
            )
        if best_stable is not None:
            lines.append(
                "- Most stable honest tailored variant: "
                f"`{best_stable['variant']}` with `t_vs_flat = {float(best_stable['t_vs_flat']):.2f}`."
            )
        if flat is not None and best_mean is not None:
            honest_edge = float(best_mean["mean_sharpe"]) - float(flat["mean_sharpe"])
            lines.append(
                f"- Honest excess over flat-long is only `{honest_edge:+.4f}`."
            )
            if honest_edge < 0.10:
                lines.append(
                    "- Interpretation: the train set appears strongly drift-dominated, "
                    "so a public score near `2.8` can be achieved without proving that "
                    "the model learned a robust volatility-aware rule."
                )
        lines.append("")
    else:
        lines.append("## Honest Reference")
        lines.append("- No saved honest benchmark table was found.")
        lines.append("")

    lines.append("## Proxy Audit")
    if not proxy_audit_ran:
        if missing_data_reason:
            lines.append(f"- Skipped: {missing_data_reason}")
        else:
            lines.append("- Skipped: source parquet files were not available.")
        lines.append("")
        lines.append("## Recommendation")
        lines.append(
            "- Treat a higher public score from a proxy/test-augmented strategy as "
            "weak evidence until you can rerun this audit with the raw parquet data."
        )
        return lines

    assert proxy_summary is not None
    if proxy_summary.empty:
        lines.append("- Proxy audit produced no rows.")
        return lines

    lines.extend(
        [
            f"- Train-only mean best CV Sharpe: `{float(proxy_summary.loc[proxy_summary['augment_test_with_proxy'] == False, 'mean_best_cv_sharpe'].iloc[0]):.4f}`."
            if (proxy_summary["augment_test_with_proxy"] == False).any()
            else "- Train-only run missing.",
            f"- Proxy-augmented mean best CV Sharpe: `{float(proxy_summary.loc[proxy_summary['augment_test_with_proxy'] == True, 'mean_best_cv_sharpe'].iloc[0]):.4f}`."
            if (proxy_summary["augment_test_with_proxy"] == True).any()
            else "- Proxy-augmented run missing.",
        ]
    )

    proxy_gain = None
    proxy_deltas = pd.Series(dtype=np.float64)
    if {
        False,
        True,
    }.issubset(set(proxy_summary["augment_test_with_proxy"].tolist())):
        train_only = float(
            proxy_summary.loc[
                proxy_summary["augment_test_with_proxy"] == False,
                "mean_best_cv_sharpe",
            ].iloc[0]
        )
        proxy = float(
            proxy_summary.loc[
                proxy_summary["augment_test_with_proxy"] == True,
                "mean_best_cv_sharpe",
            ].iloc[0]
        )
        proxy_gain = proxy - train_only
        lines.append(f"- Estimated proxy gain: `{proxy_gain:+.4f}`.")
    if proxy_best is not None and not proxy_best.empty:
        pivot = (
            proxy_best.pivot(
                index="seed",
                columns="augment_test_with_proxy",
                values="best_mean_val_sharpe",
            )
            .rename(columns={False: "train_only", True: "proxy"})
            .dropna()
        )
        if {"train_only", "proxy"}.issubset(set(pivot.columns)):
            proxy_deltas = pivot["proxy"] - pivot["train_only"]
            lines.append(
                f"- Per-seed proxy delta range: `{float(proxy_deltas.min()):+.4f}` to `{float(proxy_deltas.max()):+.4f}`."
            )
            if float(proxy_deltas.min()) < 0.0 < float(proxy_deltas.max()):
                lines.append(
                    "- The proxy effect flips sign across seeds, so the public-style edge is split-sensitive and unstable."
                )

    lines.append("")
    lines.append("## Recommendation")
    if proxy_gain is None:
        lines.append(
            "- Compare train-only and proxy-augmented runs once both are available; "
            "the final choice should favor the model with the smaller dependence on proxy rows."
        )
    elif not proxy_deltas.empty and float(proxy_deltas.min()) < 0.0 < float(proxy_deltas.max()):
        lines.append(
            "- Because the proxy effect changes sign across seeds, treat the proxy/test-augmented strategy as unstable rather than genuinely stronger."
        )
        lines.append(
            "- Prefer an honest train-only winner with consistent OOF evidence, and use public leaderboard gains only as supporting evidence."
        )
    elif proxy_gain > 0.05:
        lines.append(
            "- The proxy/test-augmented stack gains materially from transductive rows. "
            "That means its public performance is not strong evidence for private robustness."
        )
        lines.append(
            "- Prefer an honest train-only winner with stable excess over flat-long for the final submission."
        )
    elif proxy_gain < -0.05:
        lines.append(
            "- Proxy/test augmentation hurts honest validation here. That is a strong "
            "sign that the public-score-optimized transductive path is not the safer final submission."
        )
        lines.append(
            "- Prefer the train-only model family and use the public leaderboard only as a secondary check."
        )
    else:
        lines.append(
            "- Proxy augmentation does not materially change CV. In that case the "
            "public-vs-private concern is smaller, and you can choose the simpler "
            "train-only model with the stronger honest benchmark."
        )
    return lines


def _write_outputs(
    out_dir: Path,
    *,
    honest_ref: dict,
    honest_table: pd.DataFrame | None,
    detailed_proxy: pd.DataFrame,
    best_proxy: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    report_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "submission_strategy_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    (out_dir / "submission_strategy_summary.json").write_text(
        json.dumps(
            {
                "honest_reference": honest_ref,
                "proxy_summary": proxy_summary.to_dict(orient="records"),
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )

    if honest_table is not None and not honest_table.empty:
        honest_table.to_csv(out_dir / "honest_reference_table.csv", index=False)
    if not detailed_proxy.empty:
        detailed_proxy.to_csv(out_dir / "proxy_audit_detailed.csv", index=False)
    if not best_proxy.empty:
        best_proxy.to_csv(out_dir / "proxy_audit_best_per_seed.csv", index=False)
    if not proxy_summary.empty:
        proxy_summary.to_csv(out_dir / "proxy_audit_summary.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether public-score gains survive honest train-only validation."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Competition data directory (default: repo data/).",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_CSV,
        help="Saved honest benchmark CSV (default: Submissions/benchmarks/benchmark_table.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "Submissions" / "decision_audit",
        help="Output directory for markdown/CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds for the proxy audit CV runs.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="KFold splits for the proxy audit.",
    )
    parser.add_argument(
        "--ridge-alphas",
        default="0.5,1.0,2.0,5.0",
        help="Comma-separated ridge alpha grid for the proxy audit.",
    )
    parser.add_argument(
        "--l1-ratios",
        default="0.0",
        help="Comma-separated l1_ratio grid for the proxy audit.",
    )
    parser.add_argument(
        "--mse-anchor-lambdas",
        default="0.0",
        help="Comma-separated MSE-anchor lambdas for the proxy audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    benchmark_csv = args.benchmark_csv.resolve()

    honest_ref, honest_table = _load_honest_reference(benchmark_csv)

    detailed_proxy = pd.DataFrame()
    best_proxy = pd.DataFrame()
    proxy_summary = pd.DataFrame()
    proxy_audit_ran = False
    missing_data_reason = None

    if _required_data_files_present(data_dir):
        detailed_proxy, best_proxy, proxy_summary = _run_proxy_audit(
            data_dir,
            seeds=_parse_int_list(args.seeds),
            n_splits=int(args.n_splits),
            ridge_alphas=_parse_float_list(args.ridge_alphas),
            l1_ratios=_parse_float_list(args.l1_ratios),
            mse_anchor_lambdas=_parse_float_list(args.mse_anchor_lambdas),
        )
        proxy_audit_ran = True
    else:
        missing_data_reason = (
            f"required parquet files are missing under `{data_dir}` "
            f"({BARS_SEEN_TRAIN}, {BARS_UNSEEN_TRAIN}, "
            f"{BARS_SEEN_PUBLIC_TEST}, {BARS_SEEN_PRIVATE_TEST})."
        )

    report_lines = _recommendation_lines(
        honest_ref,
        proxy_summary if proxy_audit_ran else None,
        best_proxy if proxy_audit_ran else None,
        proxy_audit_ran=proxy_audit_ran,
        missing_data_reason=missing_data_reason,
    )
    _write_outputs(
        out_dir,
        honest_ref=honest_ref,
        honest_table=honest_table,
        detailed_proxy=detailed_proxy,
        best_proxy=best_proxy,
        proxy_summary=proxy_summary,
        report_lines=report_lines,
    )

    print("\n".join(report_lines))
    print(f"\nWrote audit artifacts to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a standalone augmented competition data directory.

The builder creates `data/augmented/`-style folders that remain directly usable
by the existing training scripts in this repository:

* test files are copied unchanged,
* train files keep the original rows,
* additional train sessions are appended under fresh session ids,
* the competition label `R = close_99 / close_49 - 1` is preserved exactly for
  each augmented session.

Safe train augmentations implemented here:

1. `path_noise`
   Smooth zero-sum noise in log-return space, applied separately to the seen and
   unseen halves while preserving the seen cutoff close and the final close.
2. `path_noise_shape`
   Same close-path augmentation, plus small intrabar open/high/low jitter to
   diversify candlestick/range features without changing the label.
3. `vol_scale`
   Scales the within-half return dispersion around the segment drift while
   preserving the segment endpoint exactly.
4. `time_warp`
   Re-allocates when moves happen inside each half by monotone time warping of
   the observed close path, again preserving segment endpoints.

News/sentiment files, when present, are duplicated onto the augmented session ids
unchanged. This keeps the augmented folder immediately usable by current models
without inventing synthetic text.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE = _REPO_ROOT / "data"
_DEFAULT_TARGET = _DEFAULT_SOURCE / "augmented"
_EPS = 1e-8


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

OPTIONAL_COPY_FILES = (
    BARS_SEEN_PUBLIC_TEST,
    BARS_SEEN_PRIVATE_TEST,
    HEADLINES_SEEN_PUBLIC_TEST,
    HEADLINES_SEEN_PRIVATE_TEST,
    SENTIMENTS_SEEN_PUBLIC_TEST,
    SENTIMENTS_SEEN_PRIVATE_TEST,
)


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _ensure_required_inputs(source_dir: Path) -> None:
    missing = [
        name
        for name in (BARS_SEEN_TRAIN, BARS_UNSEEN_TRAIN)
        if not (source_dir / name).is_file()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required source parquet files in {source_dir}: {joined}"
        )


def _prepare_output_dir(target_dir: Path, *, overwrite: bool) -> None:
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {target_dir}. "
                "Pass --overwrite to rebuild it."
            )
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _copy_if_present(source_dir: Path, target_dir: Path, name: str) -> bool:
    src = source_dir / name
    if not src.is_file():
        return False
    shutil.copy2(src, target_dir / name)
    return True


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _smoothed_zero_sum_noise(
    length: int,
    *,
    rng: np.random.Generator,
    sigma: float,
    smooth_window: int,
) -> np.ndarray:
    if length <= 0 or sigma <= 0.0:
        return np.zeros(length, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=length)
    if smooth_window > 1 and length > 1:
        width = max(1, min(int(smooth_window), length))
        kernel = np.ones(width, dtype=np.float64) / float(width)
        noise = np.convolve(noise, kernel, mode="same")
    noise = noise - float(noise.mean())
    return noise.astype(np.float64)


def _perturb_close_segment(
    close: np.ndarray,
    *,
    rng: np.random.Generator,
    close_noise: float,
    smooth_window: int,
    anchor_close: float | None = None,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if close.size <= 1:
        return close.copy()

    if anchor_close is None:
        base = close
        start_log = float(np.log(max(base[0], _EPS)))
        rets = np.diff(np.log(np.maximum(base, _EPS)))
        noise = _smoothed_zero_sum_noise(
            rets.size,
            rng=rng,
            sigma=close_noise,
            smooth_window=smooth_window,
        )
        log_prices = np.empty_like(base)
        log_prices[0] = start_log
        log_prices[1:] = start_log + np.cumsum(rets + noise)
        return np.exp(log_prices)

    base = np.concatenate([[float(anchor_close)], close])
    start_log = float(np.log(max(base[0], _EPS)))
    rets = np.diff(np.log(np.maximum(base, _EPS)))
    noise = _smoothed_zero_sum_noise(
        rets.size,
        rng=rng,
        sigma=close_noise,
        smooth_window=smooth_window,
    )
    log_prices = np.empty_like(base)
    log_prices[0] = start_log
    log_prices[1:] = start_log + np.cumsum(rets + noise)
    return np.exp(log_prices[1:])


def _vol_scale_close_segment(
    close: np.ndarray,
    *,
    rng: np.random.Generator,
    vol_scale_low: float,
    vol_scale_high: float,
    anchor_close: float | None = None,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if close.size <= 1:
        return close.copy()

    base = close if anchor_close is None else np.concatenate([[float(anchor_close)], close])
    log_base = np.log(np.maximum(base, _EPS))
    rets = np.diff(log_base)
    if rets.size == 0:
        return close.copy()

    scale = float(rng.uniform(vol_scale_low, vol_scale_high))
    drift = float(np.mean(rets))
    new_rets = drift + scale * (rets - drift)

    log_prices = np.empty_like(log_base)
    log_prices[0] = log_base[0]
    log_prices[1:] = log_base[0] + np.cumsum(new_rets)
    out = np.exp(log_prices)
    return out if anchor_close is None else out[1:]


def _time_warp_close_segment(
    close: np.ndarray,
    *,
    rng: np.random.Generator,
    warp_strength: float,
    anchor_close: float | None = None,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if close.size <= 2:
        return close.copy()

    base = close if anchor_close is None else np.concatenate([[float(anchor_close)], close])
    log_base = np.log(np.maximum(base, _EPS))
    n = log_base.size
    t = np.linspace(0.0, 1.0, n)

    increments = np.exp(rng.normal(0.0, warp_strength, size=n - 1))
    warped = np.concatenate([[0.0], np.cumsum(increments)])
    warped /= float(warped[-1])

    new_log = np.interp(warped, t, log_base)
    out = np.exp(new_log)
    return out if anchor_close is None else out[1:]


def _rebuild_bars_from_close(
    base: pd.DataFrame,
    *,
    new_session: int,
    new_close: np.ndarray,
    rng: np.random.Generator,
    shape_noise: float,
) -> pd.DataFrame:
    g = base.sort_values("bar_ix").reset_index(drop=True).copy()
    close_old = g["close"].to_numpy(dtype=np.float64)
    new_close = np.asarray(new_close, dtype=np.float64)

    ratio_open = g["open"].to_numpy(dtype=np.float64) / np.maximum(close_old, _EPS)
    ratio_high = g["high"].to_numpy(dtype=np.float64) / np.maximum(close_old, _EPS)
    ratio_low = g["low"].to_numpy(dtype=np.float64) / np.maximum(close_old, _EPS)

    open_new = new_close * ratio_open
    high_new = new_close * ratio_high
    low_new = new_close * ratio_low

    if shape_noise > 0.0 and len(g) > 0:
        body = open_new - new_close
        body *= np.exp(rng.normal(0.0, shape_noise, size=len(g)))

        upper = np.maximum(high_new - np.maximum(open_new, new_close), 0.0)
        lower = np.maximum(np.minimum(open_new, new_close) - low_new, 0.0)
        upper *= np.exp(rng.normal(0.0, shape_noise * 0.5, size=len(g)))
        lower *= np.exp(rng.normal(0.0, shape_noise * 0.5, size=len(g)))

        open_new = new_close + body
        high_new = np.maximum(open_new, new_close) + upper
        low_new = np.minimum(open_new, new_close) - lower

    low_new = np.maximum(low_new, _EPS)
    open_new = np.maximum(open_new, low_new)
    close_new = np.maximum(new_close, low_new)
    high_new = np.maximum(high_new, np.maximum(open_new, close_new))

    g["session"] = int(new_session)
    g["open"] = open_new.astype(np.float64)
    g["high"] = high_new.astype(np.float64)
    g["low"] = low_new.astype(np.float64)
    g["close"] = close_new.astype(np.float64)
    return g


def _duplicate_session_rows(
    df: pd.DataFrame | None,
    *,
    source_session: int,
    new_session: int,
) -> pd.DataFrame | None:
    if df is None or df.empty or "session" not in df.columns:
        return None
    out = df.loc[df["session"] == source_session].copy()
    if out.empty:
        return None
    out["session"] = int(new_session)
    return out


def _augment_session_pair(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    *,
    new_session: int,
    transform: str,
    rng: np.random.Generator,
    close_noise: float,
    shape_noise: float,
    smooth_window: int,
    vol_scale_low: float,
    vol_scale_high: float,
    warp_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seen = seen.sort_values("bar_ix").reset_index(drop=True)
    unseen = unseen.sort_values("bar_ix").reset_index(drop=True)

    if transform not in {"path_noise", "path_noise_shape", "vol_scale", "time_warp"}:
        raise ValueError(f"Unknown transform: {transform!r}")

    local_shape_noise = shape_noise if transform == "path_noise_shape" else 0.0
    if transform in {"path_noise", "path_noise_shape"}:
        seen_close_new = _perturb_close_segment(
            seen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            close_noise=close_noise,
            smooth_window=smooth_window,
        )
        unseen_close_new = _perturb_close_segment(
            unseen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            close_noise=close_noise,
            smooth_window=smooth_window,
            anchor_close=float(seen_close_new[-1]),
        )
    elif transform == "vol_scale":
        seen_close_new = _vol_scale_close_segment(
            seen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            vol_scale_low=vol_scale_low,
            vol_scale_high=vol_scale_high,
        )
        unseen_close_new = _vol_scale_close_segment(
            unseen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            vol_scale_low=vol_scale_low,
            vol_scale_high=vol_scale_high,
            anchor_close=float(seen_close_new[-1]),
        )
    else:
        seen_close_new = _time_warp_close_segment(
            seen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            warp_strength=warp_strength,
        )
        unseen_close_new = _time_warp_close_segment(
            unseen["close"].to_numpy(dtype=np.float64),
            rng=rng,
            warp_strength=warp_strength,
            anchor_close=float(seen_close_new[-1]),
        )

    seen_aug = _rebuild_bars_from_close(
        seen,
        new_session=new_session,
        new_close=seen_close_new,
        rng=rng,
        shape_noise=local_shape_noise,
    )
    unseen_aug = _rebuild_bars_from_close(
        unseen,
        new_session=new_session,
        new_close=unseen_close_new,
        rng=rng,
        shape_noise=local_shape_noise,
    )
    return seen_aug, unseen_aug


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone augmented competition data directory."
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=_DEFAULT_SOURCE,
        help="Original competition data directory (default: repo data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_TARGET,
        help="Destination augmented data directory (default: data/augmented/).",
    )
    parser.add_argument(
        "--copies-per-session",
        type=int,
        default=4,
        help="Number of augmented copies to create per original train session.",
    )
    parser.add_argument(
        "--transforms",
        default="path_noise,path_noise_shape,vol_scale,time_warp",
        help="Comma-separated transform cycle for augmented sessions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for augmentation.",
    )
    parser.add_argument(
        "--close-noise",
        type=float,
        default=0.015,
        help="Std-dev of smooth log-return perturbations.",
    )
    parser.add_argument(
        "--shape-noise",
        type=float,
        default=0.08,
        help="Intrabar shape jitter level for `path_noise_shape`.",
    )
    parser.add_argument(
        "--vol-scale-low",
        type=float,
        default=0.75,
        help="Lower bound for `vol_scale` dispersion factor.",
    )
    parser.add_argument(
        "--vol-scale-high",
        type=float,
        default=1.35,
        help="Upper bound for `vol_scale` dispersion factor.",
    )
    parser.add_argument(
        "--warp-strength",
        type=float,
        default=0.35,
        help="Strength of the monotone time-warp transform.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average width for return-noise smoothing.",
    )
    parser.add_argument(
        "--session-start",
        type=int,
        default=30000,
        help="First synthetic session id to assign.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_dir = args.source_data_dir.resolve()
    target_dir = args.output_dir.resolve()
    transforms = _parse_list(args.transforms)
    if not transforms:
        raise SystemExit("At least one transform must be provided.")
    if args.copies_per_session < 1:
        raise SystemExit("--copies-per-session must be >= 1.")

    _ensure_required_inputs(source_dir)
    _prepare_output_dir(target_dir, overwrite=bool(args.overwrite))

    bars_seen = pd.read_parquet(source_dir / BARS_SEEN_TRAIN)
    bars_unseen = pd.read_parquet(source_dir / BARS_UNSEEN_TRAIN)
    headlines_seen = _read_optional_parquet(source_dir / HEADLINES_SEEN_TRAIN)
    headlines_unseen = _read_optional_parquet(source_dir / HEADLINES_UNSEEN_TRAIN)
    sentiments_seen = _read_optional_csv(source_dir / SENTIMENTS_SEEN_TRAIN)
    sentiments_unseen = _read_optional_csv(source_dir / SENTIMENTS_UNSEEN_TRAIN)

    for name in OPTIONAL_COPY_FILES:
        _copy_if_present(source_dir, target_dir, name)

    source_readme = source_dir / "README.md"
    if source_readme.is_file():
        shutil.copy2(source_readme, target_dir / "original_competition_README.md")

    seen_sessions = set(int(x) for x in bars_seen["session"].unique())
    unseen_sessions = set(int(x) for x in bars_unseen["session"].unique())
    train_sessions = sorted(seen_sessions & unseen_sessions)
    if not train_sessions:
        raise RuntimeError("No shared train session ids were found across seen/unseen train bars.")

    by_seen = {
        int(sess): g.sort_values("bar_ix").reset_index(drop=True)
        for sess, g in bars_seen.groupby("session", sort=False)
    }
    by_unseen = {
        int(sess): g.sort_values("bar_ix").reset_index(drop=True)
        for sess, g in bars_unseen.groupby("session", sort=False)
    }

    rng = np.random.default_rng(int(args.seed))
    next_session = int(args.session_start)

    aug_seen_frames: list[pd.DataFrame] = []
    aug_unseen_frames: list[pd.DataFrame] = []
    aug_headlines_seen: list[pd.DataFrame] = []
    aug_headlines_unseen: list[pd.DataFrame] = []
    aug_sentiments_seen: list[pd.DataFrame] = []
    aug_sentiments_unseen: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []

    for session in train_sessions:
        base_seen = by_seen[session]
        base_unseen = by_unseen[session]

        for copy_idx in range(int(args.copies_per_session)):
            transform = transforms[copy_idx % len(transforms)]
            child_seed = int(rng.integers(0, 2**31 - 1))
            child_rng = np.random.default_rng(child_seed)
            new_session = next_session
            next_session += 1

            seen_aug, unseen_aug = _augment_session_pair(
                base_seen,
                base_unseen,
                new_session=new_session,
                transform=transform,
                rng=child_rng,
                close_noise=float(args.close_noise),
                shape_noise=float(args.shape_noise),
                smooth_window=int(args.smooth_window),
                vol_scale_low=float(args.vol_scale_low),
                vol_scale_high=float(args.vol_scale_high),
                warp_strength=float(args.warp_strength),
            )
            aug_seen_frames.append(seen_aug)
            aug_unseen_frames.append(unseen_aug)

            dup = _duplicate_session_rows(
                headlines_seen,
                source_session=session,
                new_session=new_session,
            )
            if dup is not None:
                aug_headlines_seen.append(dup)

            dup = _duplicate_session_rows(
                headlines_unseen,
                source_session=session,
                new_session=new_session,
            )
            if dup is not None:
                aug_headlines_unseen.append(dup)

            dup = _duplicate_session_rows(
                sentiments_seen,
                source_session=session,
                new_session=new_session,
            )
            if dup is not None:
                aug_sentiments_seen.append(dup)

            dup = _duplicate_session_rows(
                sentiments_unseen,
                source_session=session,
                new_session=new_session,
            )
            if dup is not None:
                aug_sentiments_unseen.append(dup)

            manifest_rows.append(
                {
                    "source_session": int(session),
                    "augmented_session": int(new_session),
                    "transform": transform,
                    "seed": child_seed,
                    "close_noise": float(args.close_noise),
                    "shape_noise": (
                        float(args.shape_noise)
                        if transform == "path_noise_shape"
                        else 0.0
                    ),
                    "vol_scale_low": float(args.vol_scale_low)
                    if transform == "vol_scale"
                    else np.nan,
                    "vol_scale_high": float(args.vol_scale_high)
                    if transform == "vol_scale"
                    else np.nan,
                    "warp_strength": float(args.warp_strength)
                    if transform == "time_warp"
                    else np.nan,
                }
            )

    bars_seen_aug = (
        pd.concat([bars_seen, *aug_seen_frames], ignore_index=True)
        .sort_values(["session", "bar_ix"])
        .reset_index(drop=True)
    )
    bars_unseen_aug = (
        pd.concat([bars_unseen, *aug_unseen_frames], ignore_index=True)
        .sort_values(["session", "bar_ix"])
        .reset_index(drop=True)
    )
    bars_seen_aug.to_parquet(target_dir / BARS_SEEN_TRAIN, index=False)
    bars_unseen_aug.to_parquet(target_dir / BARS_UNSEEN_TRAIN, index=False)

    if headlines_seen is not None:
        headlines_seen_aug = (
            pd.concat([headlines_seen, *aug_headlines_seen], ignore_index=True)
            if aug_headlines_seen
            else headlines_seen.copy()
        )
        headlines_seen_aug = headlines_seen_aug.sort_values(["session", "bar_ix"]).reset_index(drop=True)
        headlines_seen_aug.to_parquet(target_dir / HEADLINES_SEEN_TRAIN, index=False)

    if headlines_unseen is not None:
        headlines_unseen_aug = (
            pd.concat([headlines_unseen, *aug_headlines_unseen], ignore_index=True)
            if aug_headlines_unseen
            else headlines_unseen.copy()
        )
        headlines_unseen_aug = headlines_unseen_aug.sort_values(["session", "bar_ix"]).reset_index(drop=True)
        headlines_unseen_aug.to_parquet(target_dir / HEADLINES_UNSEEN_TRAIN, index=False)

    if sentiments_seen is not None:
        sentiments_seen_aug = (
            pd.concat([sentiments_seen, *aug_sentiments_seen], ignore_index=True)
            if aug_sentiments_seen
            else sentiments_seen.copy()
        )
        sentiments_seen_aug = sentiments_seen_aug.sort_values(["session", "bar_ix"]).reset_index(drop=True)
        sentiments_seen_aug.to_csv(target_dir / SENTIMENTS_SEEN_TRAIN, index=False)

    if sentiments_unseen is not None:
        sentiments_unseen_aug = (
            pd.concat([sentiments_unseen, *aug_sentiments_unseen], ignore_index=True)
            if aug_sentiments_unseen
            else sentiments_unseen.copy()
        )
        sentiments_unseen_aug = sentiments_unseen_aug.sort_values(["session", "bar_ix"]).reset_index(drop=True)
        sentiments_unseen_aug.to_csv(target_dir / SENTIMENTS_UNSEEN_TRAIN, index=False)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(target_dir / "augmentation_manifest.csv", index=False)
    (target_dir / "augmentation_manifest.json").write_text(
        json.dumps(
            {
                "source_data_dir": str(source_dir),
                "augmented_sessions": int(len(manifest_rows)),
                "original_train_sessions": int(len(train_sessions)),
                "copies_per_session": int(args.copies_per_session),
                "transforms": transforms,
                "seed": int(args.seed),
                "close_noise": float(args.close_noise),
                "shape_noise": float(args.shape_noise),
                "vol_scale_low": float(args.vol_scale_low),
                "vol_scale_high": float(args.vol_scale_high),
                "warp_strength": float(args.warp_strength),
                "smooth_window": int(args.smooth_window),
                "session_start": int(args.session_start),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (target_dir / "README.md").write_text(
        "\n".join(
            [
                "# Augmented Data Folder",
                "",
                "This folder was generated by `scripts/build_augmented_data.py`.",
                "",
                "It is intended to stay directly usable by the existing training scripts:",
                "- test files are copied unchanged",
                "- train files include the original rows plus synthetic train sessions",
                "- synthetic sessions preserve the train label `R = close_99 / close_49 - 1`",
                "",
                "Artifacts:",
                "- `augmentation_manifest.csv`",
                "- `augmentation_manifest.json`",
                "- `original_competition_README.md` when present in the source data directory",
                "",
                "Transforms used in this build:",
                f"- {', '.join(transforms)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Built augmented dataset:",
        f"{len(train_sessions)} original train sessions + {len(manifest_rows)} augmented sessions",
    )
    print(f"Wrote standalone data folder to {target_dir}")


if __name__ == "__main__":
    main()

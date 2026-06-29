from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


TOOL_REFERENCE = "reference"
TOOL_GLYCOSHIELD = "glycoshield"
TOOL_GLYCOSHAPE = "glycoshape"


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path} ({len(df)} rows)")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "None of these expected columns were found: "
        + ", ".join(candidates)
        + f"\nAvailable columns: {list(df.columns)}"
    )


def optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def circular_distance_deg(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean_deg(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan

    radians = np.deg2rad(values.to_numpy())

    if weights is None:
        w = np.ones(len(values), dtype=float)
    else:
        w = pd.to_numeric(weights.loc[values.index], errors="coerce").fillna(0).to_numpy()

    if np.sum(w) == 0:
        return np.nan

    x = np.sum(w * np.cos(radians))
    y = np.sum(w * np.sin(radians))

    return float((np.rad2deg(math.atan2(y, x)) + 360.0) % 360.0)


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce")
    mask = values.notna()

    if not mask.any():
        return np.nan

    if weights is None:
        return float(values[mask].mean())

    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    w = w[mask]
    values = values[mask]

    if w.sum() == 0:
        return np.nan

    return float(np.average(values, weights=w))


def weighted_median(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce")
    mask = values.notna()

    if not mask.any():
        return np.nan

    values = values[mask].to_numpy()

    if weights is None:
        return float(np.median(values))

    w = pd.to_numeric(weights, errors="coerce").fillna(0)[mask].to_numpy()
    if w.sum() == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    w = w[order]

    cutoff = 0.5 * w.sum()
    return float(values[np.searchsorted(np.cumsum(w), cutoff)])


def percentile_of_value(distribution: pd.Series, value: float) -> float:
    distribution = pd.to_numeric(distribution, errors="coerce").dropna()
    if distribution.empty or pd.isna(value):
        return np.nan
    return float((distribution <= value).mean() * 100.0)


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return np.nan
    if xy.iloc[:, 0].nunique() <= 1 or xy.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(xy.iloc[:, 0].corr(xy.iloc[:, 1], method="pearson"))


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return np.nan
    if xy.iloc[:, 0].nunique() <= 1 or xy.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(xy.iloc[:, 0].corr(xy.iloc[:, 1], method="spearman"))

"""Outlier and anomaly detection helpers."""

from __future__ import annotations

import pandas as pd


def _annotate_anomalies(df: pd.DataFrame, flags: pd.DataFrame, method: str) -> pd.DataFrame:
    row_mask = flags.any(axis=1).fillna(False)
    anomalies = df.loc[row_mask].copy()
    if anomalies.empty:
        return anomalies

    aligned_flags = flags.loc[row_mask].fillna(False)
    anomalies["_anomaly_method"] = method
    anomalies["_anomaly_columns"] = aligned_flags.apply(
        lambda row: ", ".join(row.index[row.astype(bool)]), axis=1
    )
    return anomalies


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Detect rows containing numeric values beyond a z-score threshold."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=list(df.columns) + ["_anomaly_method", "_anomaly_columns"])

    std = numeric.std(ddof=0).replace(0, pd.NA)
    z_scores = ((numeric - numeric.mean()) / std).abs()
    flags = z_scores.gt(threshold)
    return _annotate_anomalies(df, flags, f"z-score > {threshold:g}")


def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Detect rows outside the IQR fence for any numeric column."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=list(df.columns) + ["_anomaly_method", "_anomaly_columns"])

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags = (numeric.lt(lower) | numeric.gt(upper)) & iqr.ne(0)
    return _annotate_anomalies(df, flags, f"IQR x {multiplier:g}")


def outlier_summary(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Return IQR outlier counts by numeric column."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=["column", "outlier_count", "outlier_percent"])

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags = (numeric.lt(lower) | numeric.gt(upper)) & iqr.ne(0)
    counts = flags.sum().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "column": counts.index,
            "outlier_count": counts.values,
            "outlier_percent": (counts.values / max(len(df), 1) * 100).round(2),
        }
    )

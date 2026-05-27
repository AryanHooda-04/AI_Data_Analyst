"""Core descriptive analytics and insight generation."""

from __future__ import annotations

import pandas as pd


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns."""
    if df.empty:
        return pd.DataFrame()
    return df.describe(include="all").transpose()


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""
    total_rows = max(len(df), 1)
    missing = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_percent": (missing.values / total_rows * 100).round(2),
            }
        )
        .sort_values(["missing_count", "missing_percent"], ascending=False)
        .reset_index(drop=True)
    )


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return type, completeness, and cardinality information."""
    rows = []
    total_rows = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "null_percent": round(float(series.isna().sum() / total_rows * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def dataset_health(df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate simple quality indicators for the current dataset view."""
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    missing_penalty = missing_cells / total_cells * 70
    duplicate_penalty = duplicate_rows / max(len(df), 1) * 30
    score = max(0, round(100 - missing_penalty - duplicate_penalty))
    return {
        "score": score,
        "missing_cells": missing_cells,
        "missing_percent": round(missing_cells / total_cells * 100, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round(duplicate_rows / max(len(df), 1) * 100, 2),
    }


def column_profile(df: pd.DataFrame, column: str) -> dict[str, object]:
    """Return profile metadata and top values for one column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not present in the dataset.")

    series = df[column]
    profile: dict[str, object] = {
        "column": column,
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "missing": int(series.isna().sum()),
        "missing_percent": round(float(series.isna().mean() * 100), 2) if len(series) else 0.0,
        "unique": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        profile.update(
            {
                "mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "median": round(float(series.median()), 4) if series.notna().any() else None,
                "min": round(float(series.min()), 4) if series.notna().any() else None,
                "max": round(float(series.max()), 4) if series.notna().any() else None,
            }
        )

    return profile


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Pearson correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True).round(4)


def _outlier_counts_iqr(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.Series(dtype="int64")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flags = numeric.lt(lower) | numeric.gt(upper)
    flags = flags & iqr.ne(0)
    return flags.sum().sort_values(ascending=False)


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            candidates.append(column)
            continue
        name_hint = any(token in column.lower() for token in ("date", "time", "month", "year"))
        if not name_hint:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            candidates.append(column)
    return candidates


def generate_insights(df: pd.DataFrame, max_items: int = 12) -> list[str]:
    """Generate deterministic, offline insights about the uploaded data."""
    insights: list[str] = []
    if df.empty:
        return ["The dataset is empty, so there are no trends or anomalies to summarize."]

    insights.append(f"The dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        insights.append(f"There are {duplicate_count:,} duplicate rows that may need review.")

    missing = missing_values(df)
    missing = missing[missing["missing_count"] > 0]
    if not missing.empty:
        top_missing = missing.iloc[0]
        insights.append(
            f"{top_missing['column']} has the highest missingness "
            f"({int(top_missing['missing_count']):,} rows, {top_missing['missing_percent']}%)."
        )
    else:
        insights.append("No missing values were detected.")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        outlier_counts = _outlier_counts_iqr(df)
        outlier_counts = outlier_counts[outlier_counts > 0]
        if not outlier_counts.empty:
            column = outlier_counts.index[0]
            insights.append(
                f"{column} has the most IQR outliers ({int(outlier_counts.iloc[0]):,} rows)."
            )

        skew = numeric.skew(numeric_only=True).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        if not skew.empty and abs(float(skew.iloc[0])) >= 1:
            direction = "right-skewed" if skew.iloc[0] > 0 else "left-skewed"
            insights.append(f"{skew.index[0]} is strongly {direction}, suggesting an uneven distribution.")

        constant_columns = [col for col in numeric.columns if numeric[col].nunique(dropna=True) <= 1]
        if constant_columns:
            insights.append(
                "The following numeric columns have little or no variance: "
                + ", ".join(constant_columns[:5])
                + "."
            )

    corr = correlations(df)
    if not corr.empty:
        pairs = []
        columns = list(corr.columns)
        for idx, left in enumerate(columns):
            for right in columns[idx + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    pairs.append((abs(float(value)), float(value), left, right))
        pairs.sort(reverse=True)
        if pairs and pairs[0][0] >= 0.65:
            _, signed_value, left, right = pairs[0]
            insights.append(
                f"{left} and {right} show a strong correlation ({signed_value:.2f})."
            )

    categorical = df.select_dtypes(exclude="number")
    for column in categorical.columns[:5]:
        counts = df[column].value_counts(dropna=True)
        if counts.empty:
            continue
        top_value = counts.index[0]
        share = counts.iloc[0] / max(len(df), 1) * 100
        if share >= 35:
            insights.append(
                f"{column} is concentrated around '{top_value}' ({share:.1f}% of rows)."
            )

    datetime_columns = _detect_datetime_columns(df)
    if datetime_columns and not numeric.empty:
        date_column = datetime_columns[0]
        working = df.copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.dropna(subset=[date_column]).sort_values(date_column)
        for metric in numeric.columns[:5]:
            series = working[[date_column, metric]].dropna()
            if len(series) < 4:
                continue
            first = float(series[metric].head(max(1, len(series) // 4)).mean())
            last = float(series[metric].tail(max(1, len(series) // 4)).mean())
            if first == 0:
                continue
            change = (last - first) / abs(first) * 100
            if abs(change) >= 10:
                trend = "increased" if change > 0 else "decreased"
                insights.append(
                    f"{metric} {trend} by about {abs(change):.1f}% from early to late records."
                )
                break

    return insights[:max_items]

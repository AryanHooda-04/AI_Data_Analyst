"""Shared helpers for prompt construction and UI-safe formatting."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


MAX_PROMPT_CHARS = 16_000


def clean_text(text: str | None) -> str:
    """Normalize user text for prompts and UI display."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def truncate_text(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Trim long prompt sections while keeping the truncation explicit."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 120].rstrip() + "\n\n[Context truncated to stay within prompt limits.]"


def safe_example(value: object, max_chars: int = 60) -> str:
    """Return a compact, single-line example value for schema previews."""
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def format_schema(df: pd.DataFrame, max_columns: int = 80) -> str:
    """Create a concise schema summary suitable for model prompts."""
    rows: list[str] = []
    limited_columns = list(df.columns[:max_columns])
    for column in limited_columns:
        series = df[column]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        examples = series.dropna().head(3).tolist()
        example_text = ", ".join(safe_example(value) for value in examples)
        rows.append(
            f"- {column}: dtype={series.dtype}, non_null={non_null}, "
            f"missing={missing}, unique={unique}, examples=[{example_text}]"
        )

    if len(df.columns) > max_columns:
        rows.append(f"- ... {len(df.columns) - max_columns} additional columns omitted")

    return "\n".join(rows)


def dataframe_sample(df: pd.DataFrame, max_rows: int = 5, max_columns: int = 20) -> str:
    """Serialize a small sample without requiring optional tabulate dependencies."""
    sample = df.iloc[:max_rows, :max_columns].copy()
    if sample.empty:
        return "[Dataset has no rows.]"
    for column in sample.columns:
        sample[column] = sample[column].map(safe_example)
    text = sample.to_csv(index=False)
    if len(df.columns) > max_columns:
        text += f"\n[Only the first {max_columns} columns are shown.]"
    return text


def summary_for_prompt(df: pd.DataFrame, max_columns: int = 25) -> str:
    """Build compact descriptive statistics for prompt context."""
    if df.empty:
        return "[Dataset is empty.]"

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    sections: list[str] = []

    if not numeric.empty:
        stats = numeric.iloc[:, :max_columns].describe().round(4)
        sections.append("Numeric summary:\n" + stats.to_csv())

    if not categorical.empty:
        rows = []
        for column in categorical.columns[:max_columns]:
            mode = categorical[column].mode(dropna=True)
            top_value = safe_example(mode.iloc[0]) if not mode.empty else ""
            rows.append(
                {
                    "column": column,
                    "unique": int(categorical[column].nunique(dropna=True)),
                    "top_value": top_value,
                    "missing": int(categorical[column].isna().sum()),
                }
            )
        sections.append("Categorical summary:\n" + pd.DataFrame(rows).to_csv(index=False))

    return "\n".join(sections) if sections else "[No summary available.]"


def dataframe_context(df: pd.DataFrame, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Package the dataset metadata sent to the language model."""
    context = f"""
Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns

Schema:
{format_schema(df)}

First rows:
{dataframe_sample(df)}

Summary statistics:
{summary_for_prompt(df)}
""".strip()
    return truncate_text(context, max_chars=max_chars)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns as strings for Streamlit select boxes."""
    return list(df.select_dtypes(include="number").columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return likely categorical columns."""
    return list(df.select_dtypes(exclude="number").columns)


def first_available(items: Iterable[str], fallback: str | None = None) -> str | None:
    """Return the first item from an iterable, or a fallback."""
    for item in items:
        return item
    return fallback

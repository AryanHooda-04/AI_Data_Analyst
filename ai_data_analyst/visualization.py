"""Plotly visualizations for the Streamlit UI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


PLOT_TEMPLATE = "plotly_white"


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not present in the dataset.")


def plot_histogram(df: pd.DataFrame, column: str):
    """Create an interactive histogram."""
    _require_column(df, column)
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        template=PLOT_TEMPLATE,
        title=f"Distribution of {column}",
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_bar(df: pd.DataFrame, column: str):
    """Create a value-count bar chart."""
    _require_column(df, column)
    counts = df[column].value_counts(dropna=False).head(30).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts,
        x=column,
        y="count",
        template=PLOT_TEMPLATE,
        title=f"Top values for {column}",
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    return fig


def plot_aggregated_bar(df: pd.DataFrame, category: str, value: str, aggregation: str = "sum"):
    """Create a ranked bar chart from a category and numeric measure."""
    _require_column(df, category)
    _require_column(df, value)
    allowed = {"sum", "mean", "median", "min", "max", "count"}
    if aggregation not in allowed:
        raise ValueError(f"Unsupported aggregation '{aggregation}'.")

    if aggregation == "count":
        grouped = df.groupby(category, dropna=False).size().reset_index(name="count")
        y_column = "count"
    else:
        grouped = (
            df.groupby(category, dropna=False)[value]
            .agg(aggregation)
            .sort_values(ascending=False)
            .head(30)
            .reset_index()
        )
        y_column = value

    grouped = grouped.sort_values(y_column, ascending=False).head(30)
    fig = px.bar(
        grouped,
        x=category,
        y=y_column,
        template=PLOT_TEMPLATE,
        title=f"{aggregation.title()} of {value} by {category}",
    )
    fig.update_layout(xaxis_title=category, yaxis_title=y_column)
    return fig


def plot_line(df: pd.DataFrame, x: str, y: str):
    """Create an interactive line chart."""
    _require_column(df, x)
    _require_column(df, y)
    working = df[[x, y]].dropna().copy()
    parsed_x = pd.to_datetime(working[x], errors="coerce")
    if parsed_x.notna().mean() >= 0.8:
        working[x] = parsed_x
    working = working.sort_values(x)
    return px.line(
        working,
        x=x,
        y=y,
        markers=True,
        template=PLOT_TEMPLATE,
        title=f"{y} by {x}",
    )


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    """Create an interactive scatter plot."""
    _require_column(df, x)
    _require_column(df, y)
    if color:
        _require_column(df, color)
    return px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        template=PLOT_TEMPLATE,
        title=f"{y} vs {x}",
    )


def plot_box(df: pd.DataFrame, column: str, group_by: str | None = None):
    """Create a box plot for distribution and outlier review."""
    _require_column(df, column)
    if group_by:
        _require_column(df, group_by)
    return px.box(
        df,
        x=group_by,
        y=column,
        points="outliers",
        template=PLOT_TEMPLATE,
        title=f"Outlier spread for {column}",
    )


def plot_heatmap(df: pd.DataFrame):
    """Create a correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required for a correlation heatmap.")

    corr = numeric.corr(numeric_only=True).round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=PLOT_TEMPLATE,
        title="Correlation heatmap",
    )
    fig.update_layout(coloraxis_colorbar_title="r")
    return fig

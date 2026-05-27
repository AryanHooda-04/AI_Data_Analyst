"""Cleaned-data query execution for conversational analytics."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

import pandas as pd

from ai_engine import DEFAULT_MODEL, complete_with_messages
from utils import clean_text


TABLE_NAME = "cleaned_data"
MAX_RESULT_ROWS = 200
BLOCKED_SQL = re.compile(
    r"\b(attach|alter|create|delete|detach|drop|insert|pragma|replace|truncate|update|vacuum)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanedQueryResult:
    """Computed result from a natural-language analytical request."""

    result_df: pd.DataFrame
    sql: str
    cleaning_actions: list[str]
    mode: str
    plan_summary: str
    row_count: int


class QueryExecutionError(RuntimeError):
    """Raised when a natural-language query cannot be safely executed."""


def should_execute_query(message: str) -> bool:
    """Return whether a chat turn sounds like a computed data query."""
    text = clean_text(message).lower()
    if not text:
        return False

    clarification_markers = (
        "what do you mean",
        "explain",
        "help me understand",
        "why",
        "definition",
        "define",
    )
    if any(marker in text for marker in clarification_markers):
        return False

    execution_markers = (
        "average",
        "avg",
        "bottom",
        "calculate",
        "count",
        "group",
        "highest",
        "list",
        "max",
        "maximum",
        "mean",
        "median",
        "min",
        "minimum",
        "rank",
        "show",
        "sum",
        "total",
        "top",
        "by ",
        " per ",
    )
    return any(marker in text for marker in execution_markers)


def clean_column_name(value: object) -> str:
    """Convert a DataFrame column name into a stable SQL-safe identifier."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip()).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def _deduplicate_columns(columns: Iterable[object]) -> tuple[list[str], dict[str, str]]:
    seen: dict[str, int] = {}
    cleaned_columns: list[str] = []
    mapping: dict[str, str] = {}
    for column in columns:
        base = clean_column_name(column)
        seen[base] = seen.get(base, 0) + 1
        cleaned = base if seen[base] == 1 else f"{base}_{seen[base]}"
        cleaned_columns.append(cleaned)
        mapping[cleaned] = str(column)
    return cleaned_columns, mapping


def _missing_like(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin({"", "na", "n/a", "nan", "none", "null", "-", "--"})


def _coerce_numeric_text(series: pd.Series) -> tuple[pd.Series, float]:
    missing = _missing_like(series)
    text = series.astype("string").str.strip()
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    text = text.str.replace(r"[$£€₹,%]", "", regex=True)
    text = text.str.replace(",", "", regex=False)
    numeric = pd.to_numeric(text.mask(missing), errors="coerce")
    denominator = max(int((~missing).sum()), 1)
    return numeric, float(numeric.notna().sum() / denominator)


def _should_try_datetime(column: str) -> bool:
    return any(marker in column for marker in ("date", "time", "month", "year", "day"))


def clean_dataframe_for_query(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Return a cleaned copy of a DataFrame suitable for SQLite execution."""
    if df.empty:
        raise QueryExecutionError("The active dataset is empty, so there is nothing to query.")

    cleaned = df.copy()
    new_columns, mapping = _deduplicate_columns(cleaned.columns)
    if list(cleaned.columns) != new_columns:
        cleaned.columns = new_columns
        cleaned_changes = ["Standardized column names for safe SQL execution."]
    else:
        cleaned.columns = new_columns
        cleaned_changes = []

    before_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicate_count = before_rows - len(cleaned)
    if duplicate_count:
        cleaned_changes.append(f"Removed {duplicate_count:,} duplicate row(s).")

    for column in list(cleaned.columns):
        series = cleaned[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            missing_count = int(series.isna().sum())
            if missing_count:
                fill_value = series.median()
                if pd.isna(fill_value):
                    fill_value = 0
                cleaned[column] = series.fillna(fill_value)
                cleaned_changes.append(f"Filled {missing_count:,} missing value(s) in {column} with the median.")
            continue

        if _should_try_datetime(column):
            parsed_dates = pd.to_datetime(series, errors="coerce")
            non_missing = max(int((~_missing_like(series)).sum()), 1)
            if parsed_dates.notna().sum() / non_missing >= 0.75:
                missing_count = int(parsed_dates.isna().sum())
                if missing_count and parsed_dates.notna().any():
                    parsed_dates = parsed_dates.fillna(parsed_dates.dropna().median())
                cleaned[column] = parsed_dates
                cleaned_changes.append(f"Parsed {column} as a datetime field.")
                continue

        numeric, parse_rate = _coerce_numeric_text(series)
        if parse_rate >= 0.75:
            missing_count = int(numeric.isna().sum())
            fill_value = numeric.median()
            if pd.isna(fill_value):
                fill_value = 0
            cleaned[column] = numeric.fillna(fill_value)
            cleaned_changes.append(f"Converted {column} to numeric values before analysis.")
            continue

        missing_count = int(_missing_like(series).sum())
        cleaned[column] = series.astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
        if missing_count:
            cleaned_changes.append(f"Filled {missing_count:,} missing value(s) in {column} with Unknown.")

    if not cleaned_changes:
        cleaned_changes.append("No cleaning changes were required for this query.")
    return cleaned, cleaned_changes, mapping


def _sqlite_ready(df: pd.DataFrame) -> pd.DataFrame:
    ready = df.copy()
    for column in ready.columns:
        if pd.api.types.is_datetime64_any_dtype(ready[column]):
            ready[column] = ready[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return ready


def execute_sql(cleaned_df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Execute a validated read-only SQL query against the cleaned DataFrame."""
    validated = validate_select_sql(sql)
    with sqlite3.connect(":memory:") as conn:
        _sqlite_ready(cleaned_df).to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
        return pd.read_sql_query(validated, conn).head(MAX_RESULT_ROWS)


def validate_select_sql(sql: str) -> str:
    """Validate that generated SQL is a single read-only SELECT/WITH statement."""
    statement = clean_text(sql).strip().rstrip(";")
    if not statement:
        raise QueryExecutionError("The generated SQL was empty.")
    if ";" in statement:
        raise QueryExecutionError("Only one SQL statement can be executed at a time.")
    if BLOCKED_SQL.search(statement):
        raise QueryExecutionError("Only read-only SELECT queries are allowed.")
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise QueryExecutionError("Only SELECT queries can be executed.")
    if TABLE_NAME not in statement:
        raise QueryExecutionError(f"Query must use the {TABLE_NAME} table.")
    return statement


def _extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    select_match = re.search(r"((?:select|with)\b.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        return select_match.group(1).strip()
    raise QueryExecutionError("The model did not return an executable SQL query.")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())


def _column_from_text(text: str, columns: list[str], *, numeric_only: bool = False, df: pd.DataFrame | None = None) -> str | None:
    candidates = columns
    if numeric_only and df is not None:
        candidates = [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
    if not candidates:
        return None

    normalized_text = clean_column_name(text)
    for column in candidates:
        if column in normalized_text or column.replace("_", " ") in text.lower():
            return column

    aliases = {
        "expression": "impressions",
        "expressions": "impressions",
        "impression": "impressions",
        "sale": "sales",
        "sales": "sales",
        "amount": "revenue",
    }
    words = _tokens(text)
    for word in words:
        alias = aliases.get(word, word)
        exact = [column for column in candidates if column == alias or column.endswith(f"_{alias}")]
        if exact:
            return exact[0]
        close = get_close_matches(alias, candidates, n=1, cutoff=0.72)
        if close:
            return close[0]
    return None


def _simple_aggregation_sql(cleaned_df: pd.DataFrame, message: str) -> tuple[str, str] | None:
    text = clean_text(message).lower()
    aggregations = [
        ("max", ("maximum", "highest", "max")),
        ("min", ("minimum", "lowest", "min")),
        ("avg", ("average", "avg", "mean")),
        ("median", ("median",)),
        ("sum", ("sum", "total")),
        ("count", ("count", "number of")),
    ]
    agg = next((name for name, markers in aggregations if any(marker in text for marker in markers)), None)
    if not agg:
        return None

    columns = list(cleaned_df.columns)
    group_match = re.search(r"\b(?:per|by|for each)\s+([a-zA-Z][a-zA-Z0-9_ ]*)", text)
    group_col = _column_from_text(group_match.group(1), columns, df=cleaned_df) if group_match else None
    metric_col = _column_from_text(text, columns, numeric_only=True, df=cleaned_df)
    if metric_col == group_col:
        metric_col = None

    if agg == "count" and not metric_col:
        metric_expr = "*"
        result_name = "row_count"
    elif metric_col:
        sqlite_agg = "AVG" if agg in {"avg", "median"} else agg.upper()
        metric_expr = f"{sqlite_agg}({metric_col})"
        result_name = f"{agg}_{metric_col}"
    else:
        return None

    if agg == "median":
        return None

    if group_col:
        sort_direction = "ASC" if agg == "min" else "DESC"
        sql = (
            f"SELECT {group_col}, {metric_expr} AS {result_name} "
            f"FROM {TABLE_NAME} "
            f"GROUP BY {group_col} "
            f"ORDER BY {result_name} {sort_direction}"
        )
        return sql, f"{agg.upper()} analysis grouped by {group_col}."

    sql = f"SELECT {metric_expr} AS {result_name} FROM {TABLE_NAME}"
    return sql, f"{agg.upper()} analysis across the cleaned dataset."


def _schema_context(df: pd.DataFrame, mapping: dict[str, str], max_columns: int = 80) -> str:
    lines = [f"Table name: {TABLE_NAME}", "Columns:"]
    for column in list(df.columns)[:max_columns]:
        sample_values = [str(value)[:60] for value in df[column].dropna().head(4).tolist()]
        source = mapping.get(column, column)
        lines.append(
            f'- {column} | dtype={df[column].dtype} | original="{source}" | sample={sample_values}'
        )
    return "\n".join(lines)


def _generate_sql(
    cleaned_df: pd.DataFrame,
    mapping: dict[str, str],
    message: str,
    *,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
) -> str:
    system_prompt = (
        "You generate safe SQLite SELECT queries for an analytics app. "
        f"Use only the table named {TABLE_NAME}. "
        "Return only one SQL query, with no explanation. "
        "Never use mutating statements. "
        "If the user has a typo, choose the closest matching column from the schema."
    )
    user_prompt = (
        "Cleaned dataset schema:\n"
        f"{_schema_context(cleaned_df, mapping)}\n\n"
        f"User request: {clean_text(message)}\n\n"
        "Write the SQLite query that computes the numerical/table result for the request."
    )
    raw_sql = complete_with_messages(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return _extract_sql(raw_sql)


def answer_with_cleaned_sql(
    df: pd.DataFrame,
    message: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    max_tokens: int = 700,
) -> CleanedQueryResult:
    """Clean the dataset, execute a safe analytical query, and return the result."""
    cleaned_df, cleaning_actions, mapping = clean_dataframe_for_query(df)

    simple = _simple_aggregation_sql(cleaned_df, message)
    if simple:
        sql, plan_summary = simple
        mode = "deterministic"
    else:
        sql = _generate_sql(
            cleaned_df,
            mapping,
            message,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        plan_summary = "Generated and executed a read-only SQL query on the cleaned dataset."
        mode = "ai_sql"

    result_df = execute_sql(cleaned_df, sql)
    return CleanedQueryResult(
        result_df=result_df,
        sql=validate_select_sql(sql),
        cleaning_actions=cleaning_actions,
        mode=mode,
        plan_summary=plan_summary,
        row_count=len(result_df),
    )


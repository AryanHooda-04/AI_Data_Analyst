"""Specialized agents for the InsightFlow-style analysis pipeline."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analyzer import correlations, dataset_health, generate_insights, missing_values
from anomaly_detector import detect_outliers_iqr, outlier_summary
from pipeline_state import AgentResult, CleaningAction, PipelineRun
from utils import categorical_columns, numeric_columns


class BasePipelineAgent:
    """Base class with timing and result helpers."""

    name = "BaseAgent"

    def execute(self, run: PipelineRun) -> AgentResult:
        """Execute the agent and return an AgentResult."""
        started = time.time()
        result = AgentResult(
            name=self.name,
            status="running",
            started_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        try:
            payload = self._run(run)
            result.status = "completed"
            result.summary = payload.get("summary", "")
            result.findings = payload.get("findings", [])
            result.metrics = payload.get("metrics", {})
            result.artifacts = payload.get("artifacts", {})
        except Exception as exc:  # noqa: BLE001
            result.status = "failed"
            result.summary = f"{self.name} failed: {exc}"
            result.findings = [str(exc)]
        result.completed_at = datetime.utcnow().isoformat(timespec="seconds")
        result.duration_seconds = round(time.time() - started, 2)
        return result

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        raise NotImplementedError


def _date_like_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that can reasonably be treated as dates."""
    date_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue
        name_hint = any(token in column.lower() for token in ("date", "time", "month", "year"))
        if not name_hint:
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if len(series) and parsed.notna().mean() >= 0.75:
            date_columns.append(column)
    return date_columns


def _safe_number(value: Any) -> float | int | str:
    """Convert numpy scalar values into JSON/SQLite friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return value


class DataCleaningAgent(BasePipelineAgent):
    """Prepare a conservative cleaned dataset proposal."""

    name = "DataCleaningAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        raw = run.raw_df.copy()
        cleaned = raw.copy()
        actions: list[CleaningAction] = []

        duplicate_count = int(cleaned.duplicated().sum())
        if duplicate_count:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            actions.append(
                CleaningAction(
                    action="drop_duplicates",
                    target="all rows",
                    rows_affected=duplicate_count,
                    detail=f"Removed {duplicate_count:,} duplicate row(s).",
                    justification="Duplicate records can inflate counts, totals, and trend signals.",
                )
            )

        for column in list(cleaned.columns):
            series = cleaned[column]
            if not pd.api.types.is_object_dtype(series):
                continue
            name_hint = any(token in column.lower() for token in ("date", "time", "month", "year"))
            if not name_hint:
                continue
            parsed = pd.to_datetime(series, errors="coerce")
            if name_hint and len(series) and parsed.notna().mean() >= 0.75:
                cleaned[column] = parsed
                actions.append(
                    CleaningAction(
                        action="type_cast",
                        target=column,
                        rows_affected=int(parsed.notna().sum()),
                        detail=f"Converted {column} to datetime for trend analysis.",
                        justification="Date-aware columns improve time-series detection and chart recommendations.",
                    )
                )

        for column in cleaned.columns:
            missing_count = int(cleaned[column].isna().sum())
            if not missing_count:
                continue

            if pd.api.types.is_numeric_dtype(cleaned[column]):
                fill_value = cleaned[column].median()
                if pd.isna(fill_value):
                    fill_value = 0
                cleaned[column] = cleaned[column].fillna(fill_value)
                detail = f"Filled {missing_count:,} missing value(s) in {column} with median {fill_value:g}."
                action = "median_impute"
            elif pd.api.types.is_datetime64_any_dtype(cleaned[column]):
                fill_value = cleaned[column].dropna().median()
                if pd.isna(fill_value):
                    continue
                cleaned[column] = cleaned[column].fillna(fill_value)
                detail = f"Filled {missing_count:,} missing date value(s) in {column} with the median date."
                action = "date_impute"
            else:
                cleaned[column] = cleaned[column].fillna("Unknown")
                detail = f"Filled {missing_count:,} missing value(s) in {column} with 'Unknown'."
                action = "category_impute"

            actions.append(
                CleaningAction(
                    action=action,
                    target=column,
                    rows_affected=missing_count,
                    detail=detail,
                    justification="Missing values can break downstream summaries, charts, and agent analysis.",
                )
            )

        outliers = outlier_summary(cleaned)
        outlier_total = int(outliers["outlier_count"].sum()) if not outliers.empty else 0
        if outlier_total:
            top = outliers[outliers["outlier_count"] > 0].head(3)
            target = ", ".join(top["column"].astype(str).tolist())
            actions.append(
                CleaningAction(
                    action="flag_outliers",
                    target=target or "numeric columns",
                    rows_affected=outlier_total,
                    detail=f"Queued {outlier_total:,} IQR outlier flag(s) for anomaly review.",
                    justification="Outliers are preserved for auditability and reviewed by the anomaly agent.",
                )
            )

        run.cleaned_df = cleaned
        run.cleaning_actions = actions

        before = dataset_health(raw)
        after = dataset_health(cleaned)
        return {
            "summary": f"Prepared a cleaned dataset with {len(actions):,} auditable action(s).",
            "findings": [action.detail for action in actions] or ["No cleaning actions were required."],
            "metrics": {
                "raw_rows": len(raw),
                "cleaned_rows": len(cleaned),
                "raw_health_score": before["score"],
                "cleaned_health_score": after["score"],
                "missing_before": before["missing_cells"],
                "missing_after": after["missing_cells"],
                "duplicates_removed": duplicate_count,
                "outlier_flags": outlier_total,
            },
            "artifacts": {
                "actions": [action.__dict__ for action in actions],
                "outlier_summary": outliers.to_dict("records") if not outliers.empty else [],
            },
        }


class VerificationAgent(BasePipelineAgent):
    """Validate proposed cleaning before user approval."""

    name = "VerificationAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        if run.cleaned_df is None:
            raise ValueError("No cleaned dataset is available for verification.")

        raw_health = dataset_health(run.raw_df)
        clean_health = dataset_health(run.cleaned_df)
        rows_removed = len(run.raw_df) - len(run.cleaned_df)
        removal_percent = round(rows_removed / max(len(run.raw_df), 1) * 100, 2)

        issues: list[str] = []
        if removal_percent > 15:
            issues.append(f"Cleaning removed {removal_percent}% of rows, which needs review.")
        if clean_health["missing_cells"] > raw_health["missing_cells"]:
            issues.append("Missing-cell count increased after cleaning.")
        if list(run.raw_df.columns) != list(run.cleaned_df.columns):
            issues.append("Column names changed during cleaning.")
        if not issues:
            issues.append("Cleaning proposal passed integrity checks.")

        approved = removal_percent <= 15 and clean_health["missing_cells"] <= raw_health["missing_cells"]
        severity = "low" if approved else "medium"
        if removal_percent > 30:
            severity = "high"

        return {
            "summary": "Cleaning proposal approved by verification checks." if approved else "Cleaning proposal needs review.",
            "findings": issues,
            "metrics": {
                "approved": approved,
                "severity": severity,
                "rows_removed": rows_removed,
                "removal_percent": removal_percent,
                "raw_health_score": raw_health["score"],
                "cleaned_health_score": clean_health["score"],
            },
        }


class TrendAgent(BasePipelineAgent):
    """Detect simple trend direction across time or row order."""

    name = "TrendAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        df = run.active_df.copy()
        nums = numeric_columns(df)
        date_columns = _date_like_columns(df)
        findings: list[str] = []
        trend_rows: list[dict[str, Any]] = []

        if not nums:
            return {"summary": "No numeric columns were available for trend analysis.", "findings": findings}

        sort_column = date_columns[0] if date_columns else None
        if sort_column:
            df[sort_column] = pd.to_datetime(df[sort_column], errors="coerce")
            df = df.dropna(subset=[sort_column]).sort_values(sort_column)

        window = max(1, len(df) // 4)
        for column in nums[:8]:
            series = df[column].dropna()
            if len(series) < 4:
                continue
            early = float(series.head(window).mean())
            late = float(series.tail(window).mean())
            if early == 0:
                continue
            change = (late - early) / abs(early) * 100
            direction = "increased" if change > 5 else "decreased" if change < -5 else "stayed relatively stable"
            trend_rows.append(
                {
                    "column": column,
                    "early_average": round(early, 4),
                    "late_average": round(late, 4),
                    "change_percent": round(change, 2),
                    "direction": direction,
                }
            )
            if abs(change) >= 10:
                findings.append(f"{column} {direction} by about {abs(change):.1f}% across the dataset.")

        if not findings and trend_rows:
            findings.append("No major directional shifts exceeded the 10% trend threshold.")

        return {
            "summary": f"Analyzed {len(trend_rows):,} numeric trend candidate(s).",
            "findings": findings,
            "metrics": {"trend_count": len(trend_rows), "date_column": sort_column or "row order"},
            "artifacts": {"trends": trend_rows},
        }


class AnomalyAgent(BasePipelineAgent):
    """Detect numeric anomalies with IQR fences."""

    name = "AnomalyAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        df = run.active_df
        summary = outlier_summary(df)
        anomalies = detect_outliers_iqr(df)
        total_rows = len(anomalies)
        findings: list[str] = []
        if summary.empty or int(summary["outlier_count"].sum()) == 0:
            findings.append("No IQR outliers were detected in numeric columns.")
        else:
            for item in summary[summary["outlier_count"] > 0].head(5).to_dict("records"):
                findings.append(
                    f"{item['column']} has {int(item['outlier_count']):,} outlier flag(s) "
                    f"({item['outlier_percent']}% of rows)."
                )

        return {
            "summary": f"Flagged {total_rows:,} row(s) with at least one numeric anomaly.",
            "findings": findings,
            "metrics": {
                "anomaly_rows": total_rows,
                "anomaly_rate": round(total_rows / max(len(df), 1) * 100, 2),
            },
            "artifacts": {
                "summary": summary.to_dict("records") if not summary.empty else [],
                "sample_rows": anomalies.head(20).astype(str).to_dict("records") if not anomalies.empty else [],
            },
        }


class CorrelationAgent(BasePipelineAgent):
    """Find strongest numeric relationships."""

    name = "CorrelationAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        corr = correlations(run.active_df)
        if corr.empty:
            return {
                "summary": "At least two numeric columns are required for correlation analysis.",
                "findings": ["Correlation analysis skipped because fewer than two numeric columns were found."],
            }

        pairs: list[dict[str, Any]] = []
        columns = list(corr.columns)
        for idx, left in enumerate(columns):
            for right in columns[idx + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    pairs.append({"left": left, "right": right, "correlation": round(float(value), 4)})

        pairs = sorted(pairs, key=lambda row: abs(row["correlation"]), reverse=True)
        findings = [
            f"{row['left']} and {row['right']} have correlation {row['correlation']:.2f}."
            for row in pairs[:5]
            if abs(row["correlation"]) >= 0.5
        ]
        if not findings:
            findings.append("No numeric pair exceeded the 0.50 correlation threshold.")

        return {
            "summary": f"Evaluated {len(pairs):,} numeric relationship(s).",
            "findings": findings,
            "metrics": {"pair_count": len(pairs), "strong_pair_count": len(findings)},
            "artifacts": {"top_pairs": pairs[:10], "matrix": corr.to_dict()},
        }


class InsightsAgent(BasePipelineAgent):
    """Generate deterministic business-readable findings."""

    name = "InsightsAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        insights = generate_insights(run.active_df, max_items=10)
        return {
            "summary": f"Generated {len(insights):,} cross-sectional insight(s).",
            "findings": insights,
            "metrics": {"insight_count": len(insights)},
        }


class VisualizationAgent(BasePipelineAgent):
    """Recommend visualizations for the final report."""

    name = "VisualizationAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        df = run.active_df
        nums = numeric_columns(df)
        cats = categorical_columns(df)
        dates = _date_like_columns(df)
        specs: list[dict[str, Any]] = []

        metric = next((col for col in nums if any(hint in col.lower() for hint in ("revenue", "sales", "salary", "amount", "score"))), nums[0] if nums else None)
        category = next((col for col in cats if any(hint in col.lower() for hint in ("region", "department", "product", "channel", "segment"))), cats[0] if cats else None)

        if dates and metric:
            specs.append({"type": "line", "title": f"{metric} trend", "x": dates[0], "y": metric})
        if category and metric:
            specs.append({"type": "bar", "title": f"{metric} by {category}", "x": category, "y": metric, "aggregation": "sum"})
        if metric:
            specs.append({"type": "box", "title": f"{metric} outlier spread", "x": category, "y": metric})
        if len(nums) >= 2:
            specs.append({"type": "heatmap", "title": "Correlation heatmap"})
        if metric:
            specs.append({"type": "histogram", "title": f"{metric} distribution", "x": metric})

        run.chart_specs = specs[:5]
        return {
            "summary": f"Recommended {len(run.chart_specs):,} chart(s) for the report.",
            "findings": [spec["title"] for spec in run.chart_specs],
            "metrics": {"chart_count": len(run.chart_specs)},
            "artifacts": {"chart_specs": run.chart_specs},
        }


class ReportSynthesisAgent(BasePipelineAgent):
    """Merge agent outputs into an executive report."""

    name = "ReportSynthesisAgent"

    def _run(self, run: PipelineRun) -> dict[str, Any]:
        health = dataset_health(run.active_df)
        findings: list[str] = []
        for agent_name in ("TrendAgent", "AnomalyAgent", "CorrelationAgent", "InsightsAgent"):
            result = run.agent_results.get(agent_name)
            if result:
                findings.extend(result.findings[:3])

        recommendations = [
            "Review high-impact anomaly rows before publishing decisions.",
            "Use the strongest correlated metric pairs to guide deeper root-cause analysis.",
            "Refresh the pipeline after applying business filters or uploading a newer dataset.",
        ]
        if health["missing_cells"]:
            recommendations.insert(0, "Address remaining missing values before operational reporting.")
        if not findings:
            findings = ["The agents completed without identifying major issues or directional shifts."]

        executive_summary = (
            f"The pipeline analyzed {run.active_df.shape[0]:,} rows across {run.active_df.shape[1]:,} columns. "
            f"The current data health score is {int(health['score'])}/100, with "
            f"{int(health['missing_cells']):,} missing cells and {int(health['duplicate_rows']):,} duplicate rows."
        )

        report_lines = [
            f"# Agentic Analysis Report: {run.dataset_name}",
            "",
            f"Run ID: `{run.run_id}`",
            f"Created: `{run.created_at}`",
            "",
            "## Executive Summary",
            "",
            executive_summary,
            "",
            "## Key Findings",
            "",
        ]
        report_lines.extend(f"- {finding}" for finding in findings[:10])
        report_lines.extend(["", "## Agent Summaries", ""])
        for name, result in run.agent_results.items():
            report_lines.extend(
                [
                    f"### {name}",
                    "",
                    f"- Status: `{result.status}`",
                    f"- Duration: `{result.duration_seconds}s`",
                    f"- Summary: {result.summary}",
                    "",
                ]
            )
        report_lines.extend(["## Recommendations", ""])
        report_lines.extend(f"- {item}" for item in recommendations)
        report_lines.extend(["", "## Cleaning Audit", ""])
        if run.cleaning_actions:
            report_lines.extend(
                f"- {action.action} on `{action.target}`: {action.detail}"
                for action in run.cleaning_actions
            )
        else:
            report_lines.append("- No cleaning actions were required.")

        run.executive_summary = executive_summary
        run.recommendations = recommendations
        run.report_markdown = "\n".join(report_lines)

        return {
            "summary": "Merged agent outputs into an executive-ready report.",
            "findings": findings[:10],
            "metrics": {
                "health_score": health["score"],
                "recommendation_count": len(recommendations),
                "report_chars": len(run.report_markdown),
            },
            "artifacts": {"recommendations": recommendations},
        }

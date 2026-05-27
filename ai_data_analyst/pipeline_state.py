"""State objects for the agentic InsightFlow-style pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import pandas as pd


class PipelineStage(str, Enum):
    """High-level stages in the agent workflow."""

    UPLOAD = "upload"
    CLEANING = "cleaning"
    CLEANING_APPROVAL = "cleaning_approval"
    VERIFICATION = "verification"
    ANALYSIS = "analysis"
    REPORT = "report"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    """Human approval status for proposed cleaning."""

    PENDING = "pending"
    APPROVED_CLEANED = "approved_cleaned"
    APPROVED_RAW = "approved_raw"
    REJECTED = "rejected"


@dataclass
class CleaningAction:
    """Auditable record of a cleaning operation."""

    action: str
    target: str
    rows_affected: int
    detail: str
    justification: str


@dataclass
class AgentResult:
    """Result returned by one pipeline agent."""

    name: str
    status: str = "pending"
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


@dataclass
class PipelineRun:
    """Complete pipeline state stored in Streamlit session state."""

    dataset_name: str
    raw_df: pd.DataFrame
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    completed_at: str = ""
    current_stage: PipelineStage = PipelineStage.UPLOAD
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    cleaned_df: pd.DataFrame | None = None
    analysis_df: pd.DataFrame | None = None
    cleaning_actions: list[CleaningAction] = field(default_factory=list)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    chart_specs: list[dict[str, Any]] = field(default_factory=list)
    report_markdown: str = ""
    executive_summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def raw_shape(self) -> tuple[int, int]:
        """Return raw dataset shape."""
        return self.raw_df.shape

    @property
    def cleaned_shape(self) -> tuple[int, int] | None:
        """Return cleaned dataset shape if available."""
        if self.cleaned_df is None:
            return None
        return self.cleaned_df.shape

    @property
    def active_df(self) -> pd.DataFrame:
        """Return the DataFrame currently used by downstream analysis."""
        if self.analysis_df is not None:
            return self.analysis_df
        if self.cleaned_df is not None:
            return self.cleaned_df
        return self.raw_df

    def add_result(self, result: AgentResult) -> None:
        """Store an agent result by agent name."""
        self.agent_results[result.name] = result

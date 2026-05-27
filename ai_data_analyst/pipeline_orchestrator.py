"""Orchestration for the agentic InsightFlow-style pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from pipeline_agents import (
    AnomalyAgent,
    CorrelationAgent,
    DataCleaningAgent,
    InsightsAgent,
    ReportSynthesisAgent,
    TrendAgent,
    VerificationAgent,
    VisualizationAgent,
)
from pipeline_history import save_pipeline_run
from pipeline_state import ApprovalStatus, PipelineRun, PipelineStage


def start_pipeline(df: pd.DataFrame, dataset_name: str) -> PipelineRun:
    """Run cleaning and verification, then pause for human approval."""
    run = PipelineRun(dataset_name=dataset_name, raw_df=df.copy())
    run.current_stage = PipelineStage.CLEANING

    cleaning = DataCleaningAgent().execute(run)
    run.add_result(cleaning)

    run.current_stage = PipelineStage.VERIFICATION
    verification = VerificationAgent().execute(run)
    run.add_result(verification)

    run.current_stage = PipelineStage.CLEANING_APPROVAL
    return run


def run_analysis(run: PipelineRun, *, use_cleaned_data: bool = True) -> PipelineRun:
    """Run parallel analysis agents and synthesize a final report."""
    run.approval_status = ApprovalStatus.APPROVED_CLEANED if use_cleaned_data else ApprovalStatus.APPROVED_RAW
    run.analysis_df = run.cleaned_df.copy() if use_cleaned_data and run.cleaned_df is not None else run.raw_df.copy()
    run.current_stage = PipelineStage.ANALYSIS

    agents = [TrendAgent(), AnomalyAgent(), CorrelationAgent(), InsightsAgent(), VisualizationAgent()]
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(agent.execute, run): agent.name for agent in agents}
        for future in as_completed(futures):
            result = future.result()
            run.add_result(result)

    run.current_stage = PipelineStage.REPORT
    synthesis = ReportSynthesisAgent().execute(run)
    run.add_result(synthesis)

    run.current_stage = PipelineStage.COMPLETED
    run.completed_at = datetime.utcnow().isoformat(timespec="seconds")
    save_pipeline_run(run)
    return run


def reject_cleaning(run: PipelineRun) -> PipelineRun:
    """Mark the cleaning proposal as rejected."""
    run.approval_status = ApprovalStatus.REJECTED
    run.current_stage = PipelineStage.CLEANING_APPROVAL
    return run

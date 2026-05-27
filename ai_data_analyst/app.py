"""Streamlit entry point for the AI Data Analyst application."""

from __future__ import annotations

import base64
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analyzer import (
    column_info,
    column_profile,
    correlations,
    dataset_health,
    generate_insights,
    missing_values,
    summary_stats,
)
from anomaly_detector import detect_outliers_iqr, detect_outliers_zscore, outlier_summary
from ai_engine import conversation_ai, openai_ssl_mode, text_to_speech, transcribe_audio
from code_generator import generate_code
from data_loader import load_file
from pipeline_history import load_recent_runs
from pipeline_orchestrator import reject_cleaning, run_analysis, start_pipeline
from pipeline_state import ApprovalStatus, PipelineRun, PipelineStage
from utils import categorical_columns, numeric_columns
from visualization import (
    plot_aggregated_bar,
    plot_bar,
    plot_box,
    plot_heatmap,
    plot_histogram,
    plot_line,
    plot_scatter,
)


APP_DIR = Path(__file__).parent
SAMPLE_DATA = APP_DIR / "sample_data.csv"
HEADER_ART = APP_DIR / "assets" / "analytics_header.svg"
VOICE_RECORDER = components.declare_component("voice_recorder", path=str(APP_DIR / "voice_recorder"))

NAV_ITEMS = [
    {"label": "Overview", "icon": ":material/dashboard:"},
    {"label": "Conversation AI", "icon": ":material/forum:"},
    {"label": "Visualizations", "icon": ":material/monitoring:"},
    {"label": "Insights & Anomalies", "icon": ":material/troubleshoot:"},
    {"label": "Agent Pipeline", "icon": ":material/account_tree:"},
    {"label": "Code Generator", "icon": ":material/code:"},
    {"label": "Presentation Mode", "icon": ":material/present_to_all:"},
]

PAGE_COPY = {
    "Overview": ("Overview", "Dataset health, preview, and core quality checks."),
    "Conversation AI": ("Conversation AI", "Chat with a dataset-aware analyst and ask follow-up questions."),
    "Visualizations": ("Visualizations", "Build charts from the active dataset view."),
    "Insights & Anomalies": ("Insights", "Review patterns, quality signals, and anomalies."),
    "Agent Pipeline": ("Agent Pipeline", "Clean, verify, analyze, and synthesize a report."),
    "Code Generator": ("Code Generator", "Generate SQL and Pandas from a plain-language request."),
    "Presentation Mode": ("Presentation", "A clean executive view for demos and walkthroughs."),
}

SUGGESTED_QUESTIONS = [
    "What are the three most important patterns in this dataset?",
    "Which segments are underperforming and what might explain that?",
    "What data quality issues should I fix before reporting?",
]

OVERVIEW_ACTIONS = [
    "Find missing data",
    "Profile the best metric",
    "Review correlations",
]

VISUAL_ACTIONS = [
    "Chart revenue by region",
    "Show trend over time",
    "Inspect outliers",
]

INSIGHT_ACTIONS = [
    "Review data quality",
    "Inspect anomalies",
    "Prepare executive summary",
]

CODE_PROMPTS = [
    "Group revenue by month and region, then rank the top performers.",
    "Find rows with unusually high revenue and return their key columns.",
    "Create a customer satisfaction summary by product and channel.",
]

AI_RESPONSE_SECTIONS = [
    "Summary",
    "Evidence",
    "Caveats",
    "Recommended Next Steps",
]

MODEL_OPTIONS = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4.1",
    "Custom",
]

REASONING_OPTIONS = ["none", "low", "medium", "high", "xhigh"]

VOICE_OPTIONS = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]

TRANSCRIPTION_OPTIONS = [
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "whisper-1",
]

DEMO_REASONING_OPTIONS = ["none", "low"]


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag with friendly true/false values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer environment setting while keeping a safe lower bound."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def asset_data_uri(path: Path, mime_type: str) -> str:
    """Return a local visual asset as a browser-safe data URI."""
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


STREAMLIT_DEMO_MODE = env_flag("STREAMLIT_DEMO_MODE", True)
DEMO_AI_CALL_LIMIT = env_int("DEMO_AI_CALL_LIMIT", 18, minimum=1)
DEMO_SESSION_TOKEN_BUDGET = env_int("DEMO_SESSION_TOKEN_BUDGET", 80_000, minimum=1_000)
DEMO_CONTEXT_CHAR_LIMIT = env_int("DEMO_CONTEXT_CHAR_LIMIT", 12_000, minimum=1_000)
DEMO_MAX_REQUEST_CHARS = env_int("DEMO_MAX_REQUEST_CHARS", 3_000, minimum=200)
DEMO_TEXT_OUTPUT_TOKENS = env_int("DEMO_TEXT_OUTPUT_TOKENS", 1_200, minimum=128)
DEMO_CODE_OUTPUT_TOKENS = env_int("DEMO_CODE_OUTPUT_TOKENS", 1_800, minimum=128)
DEMO_TRANSCRIPTION_TOKEN_COST = env_int("DEMO_TRANSCRIPTION_TOKEN_COST", 500, minimum=50)
DEMO_TTS_CHAR_LIMIT = env_int("DEMO_TTS_CHAR_LIMIT", 3_000, minimum=200)
DEMO_MAX_AUDIO_MB = env_int("DEMO_MAX_AUDIO_MB", 8, minimum=1)
DEMO_MAX_AUDIO_BYTES = DEMO_MAX_AUDIO_MB * 1024 * 1024
TOKEN_CHARS_ESTIMATE = 4


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Apply a stable, professional visual shell regardless of browser theme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --app-bg: #f7f9fc;
            --panel: #ffffff;
            --panel-soft: #f9fbff;
            --panel-border: #d9e4f2;
            --ink: #102033;
            --muted: #63748a;
            --sidebar-bg: #f8fbff;
            --sidebar-panel: #ffffff;
            --sidebar-line: #d7e4f1;
            --sidebar-ink: #102033;
            --sidebar-muted: #64748b;
            --accent: #2563eb;
            --accent-2: #0f766e;
            --accent-3: #e11d48;
            --accent-warm: #f59e0b;
            --accent-soft: #e8f1ff;
            --success: #0f8a55;
            --warning: #b45309;
            --danger: #b91c1c;
            --shadow-sm: 0 1px 2px rgba(16, 32, 51, 0.06);
            --shadow-md: 0 12px 30px rgba(16, 32, 51, 0.08);
        }

        @keyframes page-rise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes soft-float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }

        @keyframes accent-sweep {
            from { transform: translateX(-120%); }
            to { transform: translateX(120%); }
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background:
                linear-gradient(180deg, #fbfdff 0%, #f2faf6 42%, #f8f7ff 100%) !important;
            color: var(--ink) !important;
        }

        .block-container {
            padding-top: 0.85rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
            animation: page-rise 340ms ease-out;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: var(--ink) !important;
            font-weight: 750 !important;
        }

        h2 {
            font-size: 1.45rem !important;
            margin-top: 1.25rem !important;
            margin-bottom: 0.35rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
            margin-top: 1rem !important;
        }

        p, label, span, div {
            color: inherit;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #ffffff 0%, #f2faf6 48%, #f7f8ff 100%) !important;
            border-right: 1px solid var(--sidebar-line);
            box-shadow: 4px 0 28px rgba(16, 32, 51, 0.06);
        }

        [data-testid="stSidebar"] > div {
            padding-top: 1.1rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--sidebar-ink) !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"] small {
            color: var(--sidebar-muted) !important;
        }

        .sidebar-brand {
            padding: 0 0 0.85rem 0;
            border-bottom: 1px solid var(--sidebar-line);
            margin-bottom: 0.85rem;
        }

        .sidebar-brand-title {
            color: #0f2f2b !important;
            font-size: 1.35rem;
            font-weight: 760;
            line-height: 1.2;
        }

        .sidebar-brand-subtitle {
            color: var(--sidebar-muted) !important;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }

        .sidebar-section-title {
            color: #52677d !important;
            font-size: 0.74rem;
            font-weight: 760;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 0.9rem 0 0.4rem 0;
        }

        .sidebar-card {
            background: var(--sidebar-panel);
            border: 1px solid var(--sidebar-line);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin: 0.3rem 0 0.65rem 0;
        }

        .sidebar-card-title {
            color: var(--sidebar-muted) !important;
            font-size: 0.74rem;
            margin-bottom: 0.25rem;
        }

        .sidebar-card-value {
            color: var(--sidebar-ink) !important;
            font-size: 0.98rem;
            font-weight: 720;
            overflow-wrap: anywhere;
        }

        .sidebar-card-meta {
            color: var(--sidebar-muted) !important;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border: 1px dashed #99c7bd !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] button *,
        [data-testid="stSidebar"] .stDownloadButton button,
        [data-testid="stSidebar"] .stDownloadButton button * {
            color: #111827 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 7px;
            padding: 0.38rem 0.45rem;
            margin: 0.1rem 0;
            border: 1px solid transparent;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #eef8f6;
            border-color: #b9dad3;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #dff7f1;
            border-color: #3b82f6;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: var(--sidebar-panel) !important;
            border: 1px solid var(--sidebar-line) !important;
            border-radius: 8px !important;
            overflow: hidden;
            margin-bottom: 0.7rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: var(--sidebar-panel) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpander"] svg {
            color: var(--sidebar-ink) !important;
            fill: var(--sidebar-ink) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background: #edf7f5 !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: var(--sidebar-panel) !important;
            border: 1px solid var(--sidebar-line) !important;
            color: var(--sidebar-ink) !important;
            justify-content: flex-start;
            align-items: center;
            gap: 0.38rem;
            min-height: 2.35rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
            white-space: nowrap;
            box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
            transition: border-color 150ms ease, background 150ms ease, transform 150ms ease, box-shadow 150ms ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #eef8f6 !important;
            border-color: #a8d5cc !important;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.12);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            border-color: transparent !important;
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button svg {
            color: inherit !important;
            fill: currentColor !important;
        }

        [data-testid="stSidebar"] .stButton > button p {
            white-space: nowrap;
        }

        .app-topbar {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
        }

        .app-eyebrow {
            color: var(--muted) !important;
            font-size: 0.72rem;
            font-weight: 720;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .app-title {
            color: var(--ink) !important;
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 780;
            margin-top: 0.15rem;
        }

        .app-subtitle {
            color: var(--muted) !important;
            font-size: 0.86rem;
            margin-top: 0.18rem;
        }

        .app-visual-card {
            position: relative;
            min-height: 112px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #cfe4f4;
            background: linear-gradient(135deg, #eff6ff 0%, #ecfdf5 55%, #fff7ed 100%);
            box-shadow: var(--shadow-sm);
        }

        .app-visual-card::after {
            content: "";
            position: absolute;
            inset: auto -35% 0 -35%;
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), var(--accent-warm), transparent);
            animation: accent-sweep 4.5s ease-in-out infinite;
            opacity: 0.75;
        }

        .app-visual-card img {
            display: block;
            width: 100%;
            height: 112px;
            object-fit: cover;
            animation: soft-float 7s ease-in-out infinite;
        }

        .app-meta-row,
        .filter-chip-row,
        .readiness-strip,
        .suggestion-row,
        .workflow-rail,
        .data-story-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
        }

        .app-meta-row {
            margin-top: 0.45rem;
        }

        .meta-pill,
        .filter-chip,
        .readiness-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-size: 0.73rem;
            font-weight: 680;
            border: 1px solid #c9ddf4;
            background: #f2f8ff;
            color: #1e3a5f !important;
        }

        .filter-chip-row {
            margin: 0 0 0.75rem 0;
        }

        .filter-chip {
            background: #eef6ff;
            border-color: #bfdbfe;
            color: #1e40af !important;
        }

        .filter-chip-empty {
            background: #f8fafc;
            color: var(--muted) !important;
        }

        .readiness-band {
            background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 55%, #eff6ff 100%);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.1rem 0 0.75rem 0;
            box-shadow: var(--shadow-sm);
        }

        .workflow-rail {
            margin-top: 0.35rem;
            padding-top: 0.45rem;
            border-top: 1px solid var(--panel-border);
        }

        .workflow-step {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid #dbe3ef;
            background: #f8fafc;
            color: #475569 !important;
            border-radius: 999px;
            padding: 0.24rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .workflow-step-active {
            background: #dbeafe;
            border-color: #93c5fd;
            color: #1e40af !important;
        }

        .readiness-heading {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .readiness-title {
            color: var(--ink) !important;
            font-weight: 740;
            font-size: 1rem;
        }

        .readiness-subtitle {
            color: var(--muted) !important;
            font-size: 0.82rem;
        }

        .readiness-pill-ready {
            background: #ecfdf5;
            border-color: #bbf7d0;
            color: #166534 !important;
        }

        .readiness-pill-warn {
            background: #fff7ed;
            border-color: #fed7aa;
            color: #9a3412 !important;
        }

        .section-kicker {
            color: var(--muted) !important;
            font-size: 0.88rem;
            margin-top: -0.35rem;
            margin-bottom: 0.9rem;
        }

        .suggestion-panel {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.15rem 0 0.85rem 0;
        }

        .suggestion-title {
            color: var(--muted) !important;
            font-size: 0.72rem;
            font-weight: 720;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }

        .empty-workspace {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1.3rem;
            max-width: 860px;
        }

        .empty-title {
            color: var(--ink) !important;
            font-size: 1.8rem;
            font-weight: 780;
            line-height: 1.2;
        }

        .empty-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .empty-item {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem;
            background: #f8fafc;
        }

        .empty-item-title,
        .presentation-heading {
            color: var(--ink) !important;
            font-weight: 740;
        }

        .ai-response-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
            box-shadow: var(--shadow-sm);
        }

        .ai-response-card-title {
            color: var(--ink) !important;
            font-weight: 740;
            margin-bottom: 0.35rem;
        }

        .ai-response-card-body {
            color: #334155 !important;
            line-height: 1.55;
        }

        .conversation-empty {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0 0.95rem 0;
            box-shadow: var(--shadow-sm);
        }

        .conversation-empty-title {
            color: var(--ink) !important;
            font-size: 1rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .conversation-empty-body {
            color: var(--muted) !important;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .conversation-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin: 0.2rem 0 0.7rem 0;
        }

        .presentation-band {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .data-story-grid {
            align-items: stretch;
            margin: 0.9rem 0 1rem 0;
        }

        .data-story-card {
            flex: 1 1 180px;
            min-width: 180px;
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }

        .data-story-label {
            color: var(--muted) !important;
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .data-story-value {
            color: var(--ink) !important;
            font-size: 1rem;
            font-weight: 760;
            margin-top: 0.2rem;
        }

        [data-testid="stMetric"] {
            position: relative;
            overflow: hidden;
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.72rem 0.85rem;
            box-shadow: var(--shadow-sm);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-warm));
        }

        [data-testid="stMetric"]:hover {
            border-color: #b6d7ef;
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] label * {
            color: var(--muted) !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: var(--ink) !important;
            font-size: 1.65rem !important;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 680;
            background: #e8f7f4;
            color: #0f5f59 !important;
            border: 1px solid #b7e1d8;
        }

        .status-pill-ok {
            background: #dcfce7;
            border-color: #bbf7d0;
            color: #166534 !important;
        }

        .status-pill-pending {
            background: #eff6ff;
            border-color: #bfdbfe;
            color: #1d4ed8 !important;
        }

        .status-pill-failed {
            background: #fee2e2;
            border-color: #fecaca;
            color: #991b1b !important;
        }

        .insight-card {
            background: linear-gradient(135deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid var(--panel-border);
            border-left: 4px solid var(--accent-2);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.55rem;
            color: var(--ink) !important;
            box-shadow: var(--shadow-sm);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .insight-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .insight-badge {
            display: inline-flex;
            margin-bottom: 0.4rem;
            border-radius: 999px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            color: #334155 !important;
            padding: 0.2rem 0.48rem;
            font-size: 0.72rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .glossary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1rem 0;
        }

        .glossary-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }

        .glossary-title {
            color: var(--ink) !important;
            font-weight: 760;
            margin-bottom: 0.3rem;
        }

        .glossary-body {
            color: var(--muted) !important;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .agent-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }

        .agent-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            min-height: 120px;
        }

        .agent-card-completed {
            border-left: 4px solid #15803d;
        }

        .agent-card-pending {
            border-left: 4px solid #94a3b8;
        }

        .agent-card-failed {
            border-left: 4px solid #dc2626;
        }

        .agent-card-title {
            color: var(--ink) !important;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .agent-card-meta {
            color: var(--muted) !important;
            font-size: 0.78rem;
            margin-bottom: 0.45rem;
        }

        .agent-card-body {
            color: #334155 !important;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .approval-panel {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin: 0.65rem 0 0.85rem 0;
        }

        .history-row {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
        }

        .muted {
            color: var(--muted) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 6px !important;
            border: 1px solid #c9d2e3 !important;
            min-height: 2.35rem;
            font-weight: 650 !important;
            transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease, background 150ms ease;
        }

        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            border-color: transparent !important;
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"] *,
        [data-testid="stFormSubmitButton"] button[kind="primary"] * {
            color: #ffffff !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.12);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.42rem 0.65rem;
        }

        @media (max-width: 900px) {
            .empty-grid,
            .agent-grid,
            .glossary-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_sample_dataset() -> pd.DataFrame:
    """Load bundled demo data."""
    return load_file(SAMPLE_DATA)


def initialize_state() -> None:
    """Initialize durable widget state."""
    env_model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    default_model_choice = env_model if env_model in MODEL_OPTIONS else "Custom"
    defaults = {
        "ai_messages": [],
        "generated_code": "",
        "conversation_draft": "",
        "code_request": "",
        "navigation": "Overview",
        "use_sample_dataset": True,
        "chart_intent": "Auto",
        "overview_focus": "",
        "insight_focus": "",
        "model_choice": default_model_choice,
        "custom_model": env_model,
        "reasoning_effort": "none",
        "voice_output_enabled": False,
        "voice_autoplay": False,
        "tts_voice": "coral",
        "transcription_model": "gpt-4o-mini-transcribe",
        "last_voice_audio": None,
        "last_transcript": "",
        "table_density": "Comfortable",
        "show_term_glossary": True,
        "demo_ai_calls": 0,
        "demo_estimated_tokens": 0,
        "demo_last_usage": "",
        "pipeline_run": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_chat_if_dataset_changed(df: pd.DataFrame) -> None:
    """Clear AI state when the schema changes."""
    fingerprint = f"{df.shape}-{tuple(df.columns)}"
    if st.session_state.get("dataset_fingerprint") != fingerprint:
        st.session_state["dataset_fingerprint"] = fingerprint
        st.session_state["ai_messages"] = []
        st.session_state["generated_code"] = ""
        st.session_state["conversation_draft"] = ""
        st.session_state["code_request"] = ""
        st.session_state["pipeline_run"] = None


def health_label(score: int | float) -> str:
    """Map health score to a short label."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Needs review"
    return "High risk"


def ai_status_label() -> str:
    """Return a concise AI connection label."""
    return "AI connected" if os.getenv("OPENAI_API_KEY") else "AI key missing"


def selected_ai_model() -> str:
    """Return the active text model selected by the user."""
    choice = st.session_state.get("model_choice", "gpt-5.2")
    if choice == "Custom":
        return st.session_state.get("custom_model", "gpt-5.2").strip() or "gpt-5.2"
    return choice


def selected_reasoning_effort() -> str:
    """Return the active reasoning effort."""
    effort = st.session_state.get("reasoning_effort", "none")
    if STREAMLIT_DEMO_MODE and effort not in DEMO_REASONING_OPTIONS:
        return "low"
    return effort


def demo_mode_enabled() -> bool:
    """Return whether public-demo cost controls are active."""
    return STREAMLIT_DEMO_MODE


def estimate_tokens(text: str | None) -> int:
    """Estimate token count from character length for lightweight budget checks."""
    compact = str(text or "")
    return max(1, (len(compact) + TOKEN_CHARS_ESTIMATE - 1) // TOKEN_CHARS_ESTIMATE)


def demo_context_chars() -> int:
    """Return the context window sent to OpenAI for the current mode."""
    return DEMO_CONTEXT_CHAR_LIMIT if demo_mode_enabled() else 16_000


def ask_output_tokens() -> int:
    """Return the text answer token cap for the current mode."""
    return DEMO_TEXT_OUTPUT_TOKENS if demo_mode_enabled() else 1_200


def code_output_tokens() -> int:
    """Return the code-generation answer token cap for the current mode."""
    return DEMO_CODE_OUTPUT_TOKENS if demo_mode_enabled() else 2_000


def demo_tokens_used() -> int:
    """Return estimated demo tokens consumed in this browser session."""
    return int(st.session_state.get("demo_estimated_tokens", 0))


def demo_tokens_remaining() -> int:
    """Return estimated demo tokens remaining in this browser session."""
    return max(0, DEMO_SESSION_TOKEN_BUDGET - demo_tokens_used())


def projected_demo_tokens(user_text: str, max_output_tokens: int, *, include_context: bool = True) -> int:
    """Estimate the upper-bound token use for a pending AI action."""
    context_tokens = estimate_tokens("x" * DEMO_CONTEXT_CHAR_LIMIT) if include_context else 0
    return estimate_tokens(user_text) + context_tokens + max_output_tokens + 150


def demo_guard_allows(
    action: str,
    user_text: str,
    max_output_tokens: int,
    *,
    include_context: bool = True,
) -> bool:
    """Block AI calls that would exceed the public demo session guardrail."""
    if not demo_mode_enabled():
        return True

    if len(user_text) > DEMO_MAX_REQUEST_CHARS:
        st.warning(f"Demo mode limits prompts to {DEMO_MAX_REQUEST_CHARS:,} characters. Shorten the request.")
        return False

    calls = int(st.session_state.get("demo_ai_calls", 0))
    if calls >= DEMO_AI_CALL_LIMIT:
        st.warning(
            f"Demo AI limit reached: {DEMO_AI_CALL_LIMIT:,} AI actions per browser session. "
            "This protects the hosted demo from runaway API spend."
        )
        return False

    projected = projected_demo_tokens(user_text, max_output_tokens, include_context=include_context)
    if demo_tokens_used() + projected > DEMO_SESSION_TOKEN_BUDGET:
        st.warning(
            f"Demo token budget reached for this browser session. "
            f"Remaining estimate: {demo_tokens_remaining():,} tokens."
        )
        return False

    st.session_state["demo_pending_action"] = action
    return True


def record_demo_usage(
    action: str,
    user_text: str,
    response_text: str,
    max_output_tokens: int,
    *,
    include_context: bool = True,
) -> None:
    """Record estimated usage after a successful AI call."""
    if not demo_mode_enabled():
        return

    context_tokens = estimate_tokens("x" * DEMO_CONTEXT_CHAR_LIMIT) if include_context else 0
    output_tokens = min(estimate_tokens(response_text), max_output_tokens)
    used = estimate_tokens(user_text) + context_tokens + output_tokens + 150
    st.session_state["demo_ai_calls"] = int(st.session_state.get("demo_ai_calls", 0)) + 1
    st.session_state["demo_estimated_tokens"] = min(
        DEMO_SESSION_TOKEN_BUDGET,
        demo_tokens_used() + used,
    )
    st.session_state["demo_last_usage"] = f"{action}: about {used:,} estimated tokens"


def truncate_demo_text(text: str) -> str:
    """Trim user/voice text to the demo prompt length."""
    if not demo_mode_enabled() or len(text) <= DEMO_MAX_REQUEST_CHARS:
        return text
    return text[:DEMO_MAX_REQUEST_CHARS].rstrip()


def render_demo_guard_status() -> None:
    """Show public-demo cost controls in the sidebar."""
    if not demo_mode_enabled():
        st.sidebar.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">Demo guard</div>
                <div class="sidebar-card-value">Off</div>
                <div class="sidebar-card-meta">AI calls use the configured model without demo caps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    calls = int(st.session_state.get("demo_ai_calls", 0))
    used = demo_tokens_used()
    remaining = demo_tokens_remaining()
    last_usage = st.session_state.get("demo_last_usage") or "No AI actions yet"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Demo guard</div>
            <div class="sidebar-card-value">{calls:,}/{DEMO_AI_CALL_LIMIT:,} AI actions</div>
            <div class="sidebar-card-meta">{used:,}/{DEMO_SESSION_TOKEN_BUDGET:,} estimated tokens used</div>
            <div class="sidebar-card-meta">{remaining:,} estimated tokens remaining</div>
            <div class="sidebar-card-meta">Output caps: {DEMO_TEXT_OUTPUT_TOKENS:,} answer / {DEMO_CODE_OUTPUT_TOKENS:,} code tokens</div>
            <div class="sidebar-card-meta">{escape(last_usage)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_stage_label(stage: PipelineStage | str) -> str:
    """Return a readable pipeline stage label."""
    value = stage.value if isinstance(stage, PipelineStage) else str(stage)
    return value.replace("_", " ").title()


def pipeline_result_status(run: PipelineRun | None, agent_name: str) -> str:
    """Return one agent's status for compact display."""
    if run is None:
        return "pending"
    result = run.agent_results.get(agent_name)
    return result.status if result else "pending"


def render_pipeline_sidebar_status() -> None:
    """Render current agent workflow progress in the sidebar."""
    run = st.session_state.get("pipeline_run")
    if not isinstance(run, PipelineRun):
        return

    agents = [
        "DataCleaningAgent",
        "VerificationAgent",
        "TrendAgent",
        "AnomalyAgent",
        "CorrelationAgent",
        "InsightsAgent",
        "VisualizationAgent",
        "ReportSynthesisAgent",
    ]
    completed = sum(1 for agent in agents if pipeline_result_status(run, agent) == "completed")
    st.sidebar.markdown('<div class="sidebar-section-title">Agent pipeline</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Current run</div>
            <div class="sidebar-card-value">{escape(run.dataset_name)}</div>
            <div class="sidebar-card-meta">Run ID: {escape(run.run_id)}</div>
            <div class="sidebar-card-meta">Stage: {escape(pipeline_stage_label(run.current_stage))}</div>
            <div class="sidebar-card-meta">{completed}/{len(agents)} agents completed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_navigation(label: str) -> None:
    """Set the active app destination before rerendering navigation buttons."""
    st.session_state["navigation"] = label


def clear_ai_chat() -> None:
    """Clear chat and generated voice output."""
    st.session_state["ai_messages"] = []
    st.session_state["last_voice_audio"] = None
    st.session_state["last_transcript"] = ""
    st.session_state["conversation_draft"] = ""


def reset_workspace_state() -> None:
    """Reset UI choices without unloading the dataset."""
    for key, value in {
        "ai_messages": [],
        "generated_code": "",
        "conversation_draft": "",
        "code_request": "",
        "navigation": "Overview",
        "chart_intent": "Auto",
        "overview_focus": "",
        "insight_focus": "",
        "last_voice_audio": None,
        "last_transcript": "",
        "table_density": "Comfortable",
        "pipeline_run": None,
    }.items():
        st.session_state[key] = value


def insight_category(insight: str) -> str:
    """Classify deterministic insights for clearer scanning."""
    text = insight.lower()
    if "missing" in text or "duplicate" in text:
        return "Quality"
    if "outlier" in text or "anomal" in text:
        return "Outlier"
    if "correlation" in text:
        return "Relationship"
    if "skew" in text or "distribution" in text or "concentrated" in text:
        return "Distribution"
    if "increased" in text or "decreased" in text or "trend" in text:
        return "Trend"
    return "Summary"


def table_row_height() -> int:
    """Return preferred dataframe row height from density state."""
    return 28 if st.session_state.get("table_density") == "Compact" else 36


def audio_extension_from_name(name: str | None, fallback: str = "webm") -> str:
    """Return a safe file extension for uploaded or recorded audio."""
    if not name or "." not in name:
        return fallback
    return name.rsplit(".", 1)[-1].lower()


def decode_audio_data_url(data_url: str) -> bytes:
    """Decode a browser-recorded audio data URL."""
    if "," not in data_url:
        raise ValueError("The recorded audio payload was not a valid data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


def speech_safe_text(text: str, max_chars: int | None = None) -> str:
    """Remove some Markdown noise before TTS."""
    cleaned = text.replace("###", "").replace("##", "").replace("**", "")
    cleaned = cleaned.strip()
    if max_chars is not None:
        return cleaned[:max_chars].rstrip()
    return cleaned


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Find columns that are likely usable as dates."""
    candidates: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(column)
            continue
        name_hint = any(token in column.lower() for token in ("date", "time", "month", "year"))
        if not name_hint:
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if len(series) and parsed.notna().mean() >= 0.75:
            candidates.append(column)
    return candidates


def choose_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    """Select the best column by preferred name hints."""
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        for lower_name, original in lowered.items():
            if hint in lower_name:
                return original
    return columns[0] if columns else None


def readiness_checks(df: pd.DataFrame) -> list[dict[str, str | bool]]:
    """Return compact dataset readiness checks for AI/data analysis."""
    health = dataset_health(df)
    nums = numeric_columns(df)
    dates = detect_datetime_columns(df)
    return [
        {
            "label": "Schema loaded",
            "ready": df.shape[1] > 0 and df.shape[0] > 0,
            "detail": f"{df.shape[0]:,} rows x {df.shape[1]:,} columns",
        },
        {
            "label": "Missing values",
            "ready": int(health["missing_cells"]) == 0,
            "detail": f'{int(health["missing_cells"]):,} cells',
        },
        {
            "label": "Duplicates",
            "ready": int(health["duplicate_rows"]) == 0,
            "detail": f'{int(health["duplicate_rows"]):,} rows',
        },
        {
            "label": "Numeric metrics",
            "ready": len(nums) > 0,
            "detail": f"{len(nums):,} columns",
        },
        {
            "label": "Time fields",
            "ready": len(dates) > 0,
            "detail": f"{len(dates):,} detected",
        },
        {
            "label": "OpenAI key",
            "ready": bool(os.getenv("OPENAI_API_KEY")),
            "detail": "available" if os.getenv("OPENAI_API_KEY") else "missing",
        },
    ]


def recommended_chart_config(df: pd.DataFrame, intent: str = "Auto") -> dict[str, str | None]:
    """Recommend chart type and columns from schema and optional user intent."""
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    dates = detect_datetime_columns(df)
    metric = choose_column(nums, ("revenue", "sales", "amount", "profit", "units", "score"))
    category = choose_column(cats, ("region", "category", "product", "channel", "segment"))
    date_col = choose_column(dates, ("date", "month", "time", "year"))

    if intent == "Chart revenue by region" and category and metric:
        return {"chart_type": "Aggregated bar", "x": category, "y": metric, "aggregation": "sum", "color": None}
    if intent == "Show trend over time" and date_col and metric:
        return {"chart_type": "Line chart", "x": date_col, "y": metric, "aggregation": None, "color": None}
    if intent == "Inspect outliers" and metric:
        return {"chart_type": "Box plot", "x": category, "y": metric, "aggregation": None, "color": None}
    if date_col and metric:
        return {"chart_type": "Line chart", "x": date_col, "y": metric, "aggregation": None, "color": None}
    if category and metric:
        return {"chart_type": "Aggregated bar", "x": category, "y": metric, "aggregation": "sum", "color": None}
    if len(nums) >= 2:
        return {"chart_type": "Scatter plot", "x": nums[0], "y": nums[1], "aggregation": None, "color": category}
    if metric:
        return {"chart_type": "Histogram", "x": metric, "y": None, "aggregation": None, "color": None}
    if cats:
        return {"chart_type": "Bar chart", "x": cats[0], "y": None, "aggregation": None, "color": None}
    return {"chart_type": "Correlation heatmap", "x": None, "y": None, "aggregation": None, "color": None}


def render_sidebar_source(df: pd.DataFrame | None, source_name: str | None, filtered_rows: int | None) -> None:
    """Render source and quality status in the sidebar."""
    if df is None:
        st.sidebar.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">Status</div>
                <div class="sidebar-card-value">No dataset loaded</div>
                <div class="sidebar-card-meta">Upload a CSV/XLSX file or use the sample dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    health = dataset_health(df)
    filtered_note = ""
    if filtered_rows is not None and filtered_rows != len(df):
        filtered_note = f"<div class='sidebar-card-meta'>Filtered view: {filtered_rows:,} rows</div>"

    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Active dataset</div>
            <div class="sidebar-card-value">{escape(source_name or "Uploaded data")}</div>
            <div class="sidebar-card-meta">{df.shape[0]:,} rows x {df.shape[1]:,} columns</div>
            <div class="sidebar-card-meta">Health score: {int(health["score"])}/100 ({health_label(health["score"])})</div>
            {filtered_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply global filters configured from the sidebar."""
    filtered = df.copy()
    active_filters: list[str] = []
    nums = numeric_columns(df)
    cats = [
        column
        for column in categorical_columns(df)
        if 1 < df[column].nunique(dropna=True) <= 60
    ][:8]

    st.sidebar.markdown('<div class="sidebar-section-title">Global filters</div>', unsafe_allow_html=True)
    with st.sidebar.expander("Configure filters", expanded=True):
        if not cats and not nums:
            st.caption("No filterable columns were detected.")
        if cats:
            selected_cats = st.multiselect("Categorical columns", cats, default=[])
            for column in selected_cats:
                options = sorted(df[column].dropna().unique().tolist(), key=lambda value: str(value))
                selected_values = st.multiselect(f"{column} values", options, default=[], key=f"cat_filter_{column}")
                if selected_values:
                    filtered = filtered[filtered[column].isin(selected_values)]
                    active_filters.append(f"{column}: {len(selected_values)} value(s)")

        if nums:
            selected_nums = st.multiselect("Numeric ranges", nums, default=[])
            for column in selected_nums:
                series = df[column].dropna()
                if series.empty:
                    continue
                min_value = float(series.min())
                max_value = float(series.max())
                if min_value == max_value:
                    continue
                selected_range = st.slider(
                    f"{column} range",
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    key=f"num_filter_{column}",
                )
                filtered = filtered[filtered[column].between(selected_range[0], selected_range[1])]
                if selected_range != (min_value, max_value):
                    active_filters.append(f"{column}: {selected_range[0]:g} to {selected_range[1]:g}")

    if active_filters:
        st.sidebar.caption(f"{len(active_filters)} filter(s) active")
    else:
        st.sidebar.caption("No filters active")

    return filtered, active_filters


def render_sidebar() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str, str | None, str | None, list[str]]:
    """Render sidebar controls and return dataset state."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">AI Data Analyst</div>
            <div class="sidebar-brand-subtitle">Analytics workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-title">Data source</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )
    use_sample = st.sidebar.toggle(
        "Use sample dataset",
        key="use_sample_dataset",
        disabled=uploaded_file is not None,
    )

    raw_df: pd.DataFrame | None = None
    source_name: str | None = None
    load_error: str | None = None
    try:
        if uploaded_file is not None:
            raw_df = load_file(uploaded_file)
            source_name = uploaded_file.name
        elif use_sample:
            raw_df = load_sample_dataset()
            source_name = "sample_data.csv"
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)

    filtered_df: pd.DataFrame | None = None
    active_filters: list[str] = []
    if raw_df is not None:
        filtered_df, active_filters = apply_sidebar_filters(raw_df)
        render_sidebar_source(raw_df, source_name, len(filtered_df))
        st.sidebar.download_button(
            "Download current view",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="ai_data_analyst_view.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        render_sidebar_source(None, None, None)
        if load_error:
            st.sidebar.error(load_error)

    st.sidebar.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)
    if st.session_state.get("navigation") == "Ask AI":
        st.session_state["navigation"] = "Conversation AI"
    if st.session_state.get("navigation") not in [item["label"] for item in NAV_ITEMS]:
        st.session_state["navigation"] = "Overview"
    for item in NAV_ITEMS:
        selected = st.session_state["navigation"] == item["label"]
        st.sidebar.button(
            item["label"],
            key=f"nav_{item['label']}",
            icon=item["icon"],
            type="primary" if selected else "secondary",
            width="stretch",
            on_click=set_navigation,
            args=(item["label"],),
        )
    navigation = st.session_state["navigation"]

    st.sidebar.markdown('<div class="sidebar-section-title">Session tools</div>', unsafe_allow_html=True)
    tool_cols = st.sidebar.columns([0.9, 1.15])
    tool_cols[0].button("Reset", icon=":material/restart_alt:", on_click=reset_workspace_state, width="stretch")
    tool_cols[1].button("Clear chat", icon=":material/delete_sweep:", on_click=clear_ai_chat, width="stretch")

    st.sidebar.markdown('<div class="sidebar-section-title">AI status</div>', unsafe_allow_html=True)
    st.sidebar.selectbox("GPT model", MODEL_OPTIONS, key="model_choice")
    if st.session_state.get("model_choice") == "Custom":
        st.sidebar.text_input("Custom model ID", key="custom_model")
    reasoning_options = DEMO_REASONING_OPTIONS if demo_mode_enabled() else REASONING_OPTIONS
    if st.session_state.get("reasoning_effort") not in reasoning_options:
        st.session_state["reasoning_effort"] = reasoning_options[0]
    if selected_ai_model().lower().startswith("gpt-5"):
        st.sidebar.selectbox("Reasoning effort", reasoning_options, key="reasoning_effort")
        if demo_mode_enabled():
            st.sidebar.caption("Demo mode caps reasoning to avoid hidden token spend.")
    else:
        st.sidebar.caption("Reasoning effort applies to GPT-5 family models.")

    api_status = "Connected" if os.getenv("OPENAI_API_KEY") else "API key not set"
    ssl_status = openai_ssl_mode()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">OpenAI</div>
            <div class="sidebar-card-value">{api_status}</div>
            <div class="sidebar-card-meta">Model: {escape(selected_ai_model())}</div>
            <div class="sidebar-card-meta">SSL mode: {escape(ssl_status)}</div>
            <div class="sidebar-card-meta">AI tabs use OPENAI_API_KEY from environment or .env.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_demo_guard_status()
    render_pipeline_sidebar_status()

    return raw_df, filtered_df, navigation, source_name, load_error, active_filters


def search_dataframe(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search across all columns using plain-text matching."""
    query = query.strip()
    if not query:
        return df

    mask = pd.Series(False, index=df.index)
    for column in df.columns:
        mask = mask | df[column].astype(str).str.contains(query, case=False, na=False, regex=False)
    return df[mask]


def render_app_topbar(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    source_name: str | None,
    active_filters: list[str],
) -> None:
    """Render compact app header with source, filters, AI status, and export."""
    filter_label = f"{len(active_filters)} filter(s)" if active_filters else "Unfiltered"
    source = escape(source_name or "Dataset")
    navigation = st.session_state.get("navigation", "Overview")
    page_title, page_subtitle = PAGE_COPY.get(navigation, ("Analysis Workspace", "Explore the active dataset."))
    art_uri = asset_data_uri(HEADER_ART, "image/svg+xml")
    with st.container(border=True):
        left, right = st.columns([4.4, 1.6], vertical_alignment="center")
        with left:
            st.markdown(
                f"""
                <div class="app-eyebrow">AI Data Analyst</div>
                <div class="app-title">{escape(page_title)}</div>
                <div class="app-subtitle">{escape(page_subtitle)}</div>
                <div class="app-meta-row">
                    <span class="meta-pill">{source}</span>
                    <span class="meta-pill">{df.shape[0]:,}/{raw_df.shape[0]:,} rows</span>
                    <span class="meta-pill">{filter_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            if art_uri:
                st.markdown(
                    f'<div class="app-visual-card"><img src="{art_uri}" alt="Analytics dashboard visual" /></div>',
                    unsafe_allow_html=True,
                )
            st.download_button(
                "Export CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ai_data_analyst_export.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )


def render_filter_chips(active_filters: list[str]) -> None:
    """Display active filters as always-visible chips."""
    if not active_filters:
        return
    chip_html = "".join(f'<span class="filter-chip">{escape(chip)}</span>' for chip in active_filters)
    st.markdown(f'<div class="filter-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_workflow_rail(active_navigation: str) -> None:
    """Show a compact workflow map under the app header."""
    steps = []
    for item in NAV_ITEMS:
        label = item["label"]
        css_class = "workflow-step workflow-step-active" if label == active_navigation else "workflow-step"
        steps.append(f'<span class="{css_class}">{escape(label)}</span>')
    st.markdown(f'<div class="workflow-rail">{"".join(steps)}</div>', unsafe_allow_html=True)


def render_data_story(df: pd.DataFrame) -> None:
    """Render a short data story strip above detailed tools."""
    health = dataset_health(df)
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    dates = detect_datetime_columns(df)
    anomalies = outlier_summary(df)
    anomaly_total = int(anomalies["outlier_count"].sum()) if not anomalies.empty else 0
    cards = [
        ("Data Shape", f"{df.shape[0]:,} rows across {df.shape[1]:,} fields"),
        ("Analysis Fit", f"{len(nums)} metrics, {len(cats)} dimensions, {len(dates)} time fields"),
        ("Quality Signal", f'{int(health["score"])}/100 health score'),
        ("Review Queue", f"{anomaly_total:,} numeric outlier flags"),
    ]
    html = "".join(
        f'<div class="data-story-card"><div class="data-story-label">{escape(label)}</div>'
        f'<div class="data-story-value">{escape(value)}</div></div>'
        for label, value in cards
    )
    st.markdown(f'<div class="data-story-grid">{html}</div>', unsafe_allow_html=True)


def render_term_glossary() -> None:
    """Render compact explanations of recurring analytics terms."""
    glossary = [
        (
            "IQR Outlier",
            "A value outside Q1 - 1.5 x IQR or Q3 + 1.5 x IQR. It deserves review, but is not automatically wrong.",
        ),
        (
            "Right-Skewed",
            "Most values sit lower, with a smaller number of unusually high values stretching the distribution rightward.",
        ),
        (
            "Correlation",
            "A relationship score from -1 to 1. Values near 1 move together; values near -1 move in opposite directions.",
        ),
    ]
    cards = "".join(
        f'<div class="glossary-card"><div class="glossary-title">{escape(title)}</div>'
        f'<div class="glossary-body">{escape(body)}</div></div>'
        for title, body in glossary
    )
    st.markdown(f'<div class="glossary-grid">{cards}</div>', unsafe_allow_html=True)


def render_readiness_panel(df: pd.DataFrame) -> None:
    """Show dataset readiness checks for AI analysis."""
    checks = readiness_checks(df)
    ready_count = sum(1 for item in checks if item["ready"])
    status = "Ready" if ready_count >= 5 else "Needs review"
    pill_html = "".join(
        (
            f'<span class="readiness-pill readiness-pill-ready">{escape(str(item["label"]))}: '
            f'{escape(str(item["detail"]))}</span>'
            if item["ready"]
            else f'<span class="readiness-pill readiness-pill-warn">{escape(str(item["label"]))}: '
            f'{escape(str(item["detail"]))}</span>'
        )
        for item in checks
    )
    st.markdown(
        f"""
        <div class="readiness-band">
            <div class="readiness-heading">
                <div>
                    <div class="readiness-title">Readiness: {status}</div>
                    <div class="readiness-subtitle">{ready_count}/{len(checks)} checks passed</div>
                </div>
            </div>
            <div class="readiness-strip">{pill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    source_name: str | None,
    active_filters: list[str],
    *,
    show_readiness: bool = True,
) -> None:
    """Render the compact app header and KPI row."""
    health = dataset_health(df)
    active_label = "Active rows" if len(df) != len(raw_df) else "Rows"

    render_app_topbar(df, raw_df, source_name, active_filters)
    render_filter_chips(active_filters)
    if show_readiness and st.session_state.get("navigation") == "Overview":
        with st.expander("Dataset readiness", expanded=False):
            render_readiness_panel(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(active_label, f"{df.shape[0]:,}", help=f"{len(raw_df):,} rows in the source dataset")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric(
        "Quality issues",
        f'{int(health["missing_cells"]) + int(health["duplicate_rows"]):,}',
        help=f'{int(health["missing_cells"]):,} missing cells; {int(health["duplicate_rows"]):,} duplicate rows',
    )
    col4.metric("Health score", f'{int(health["score"])}/100', help=health_label(health["score"]))


def render_insight_cards(df: pd.DataFrame, limit: int = 4) -> None:
    """Show compact deterministic insight cards."""
    for insight in generate_insights(df, max_items=limit):
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-badge">{escape(insight_category(insight))}</div>
                <div>{escape(insight)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_action_bar(title: str, actions: list[str], state_key: str, key_prefix: str) -> None:
    """Render page-level suggested actions."""
    st.markdown(f'<div class="suggestion-title">{escape(title)}</div>', unsafe_allow_html=True)
    cols = st.columns(len(actions))
    for idx, action in enumerate(actions):
        if cols[idx].button(action, key=f"{key_prefix}_{idx}", width="stretch"):
            st.session_state[state_key] = action


def parse_ai_sections(content: str) -> list[tuple[str, str]]:
    """Parse model responses into productized UI sections when possible."""
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []
    section_names = {name.lower(): name for name in AI_RESPONSE_SECTIONS}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        normalized = line.strip("#*: ").lower()
        normalized = normalized.removesuffix(":")
        if normalized in section_names:
            if current_title:
                sections.append((current_title, current_body))
            current_title = section_names[normalized]
            current_body = []
            continue
        if current_title:
            current_body.append(raw_line)

    if current_title:
        sections.append((current_title, current_body))

    parsed = [(title, "\n".join(body).strip()) for title, body in sections]
    return [(title, body) for title, body in parsed if body]


def render_ai_response(content: str) -> None:
    """Render AI output as structured product cards."""
    sections = parse_ai_sections(content)
    if len(sections) < 2:
        st.markdown(content)
        return

    for title, body in sections:
        safe_body = escape(body).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="ai-response-card">
                <div class="ai-response-card-title">{escape(title)}</div>
                <div class="ai-response-card-body">{safe_body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_column_profiler(df: pd.DataFrame) -> None:
    """Render a focused one-column profile panel."""
    st.subheader("Column Profiler")
    selected_column = st.selectbox("Profile column", list(df.columns))
    profile = column_profile(df, selected_column)

    profile_cols = st.columns(4)
    profile_cols[0].metric("Type", str(profile["dtype"]))
    profile_cols[1].metric("Unique", f'{int(profile["unique"]):,}')
    profile_cols[2].metric("Missing", f'{int(profile["missing"]):,}', f'{profile["missing_percent"]}%')
    profile_cols[3].metric("Rows", f'{int(profile["rows"]):,}')

    left, right = st.columns([1, 1])
    with left:
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            stats = pd.DataFrame([profile]).transpose().rename(columns={0: "value"}).astype(str)
            st.dataframe(stats, width="stretch", height=260, row_height=table_row_height())
        else:
            top_values = df[selected_column].value_counts(dropna=False).head(15).reset_index()
            top_values.columns = [selected_column, "count"]
            st.dataframe(top_values, width="stretch", height=260, row_height=table_row_height())

    with right:
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.plotly_chart(
                plot_histogram(df, selected_column),
                width="stretch",
                key=f"column_profile_histogram_{selected_column}",
            )
        else:
            st.plotly_chart(
                plot_bar(df, selected_column),
                width="stretch",
                key=f"column_profile_bar_{selected_column}",
            )


def render_overview(df: pd.DataFrame) -> None:
    """Render the overview workspace."""
    st.subheader("Snapshot")
    st.markdown('<div class="section-kicker">Start with the highest-signal checks before drilling into raw rows.</div>', unsafe_allow_html=True)
    render_action_bar("Suggested actions", OVERVIEW_ACTIONS, "overview_focus", "overview_action")
    if st.session_state.get("overview_focus") == "Find missing data":
        st.info("Missing-value review is active.")
        st.dataframe(missing_values(df), width="stretch", height=260, row_height=table_row_height())
    elif st.session_state.get("overview_focus") == "Profile the best metric":
        best_metric = choose_column(numeric_columns(df), ("revenue", "sales", "amount", "profit", "units", "score"))
        if best_metric:
            st.info(f"Profiling the most relevant metric detected: {best_metric}.")
            st.plotly_chart(
                plot_histogram(df, best_metric),
                width="stretch",
                key=f"overview_focus_histogram_{best_metric}",
            )
    elif st.session_state.get("overview_focus") == "Review correlations":
        corr = correlations(df)
        if corr.empty:
            st.info("Correlation review needs at least two numeric columns.")
        else:
            st.info("Correlation review is active.")
            st.plotly_chart(plot_heatmap(df), width="stretch", key="overview_focus_correlation_heatmap")
    render_insight_cards(df, limit=3)

    st.subheader("Dataset Preview")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("Search preview", placeholder="Search across all columns")
    with col2:
        max_preview = min(max(len(df), 10), 500)
        preview_rows = st.slider("Rows to show", 10, max_preview, min(50, max_preview), step=10)
    with col3:
        st.radio("Table density", ["Comfortable", "Compact"], horizontal=True, key="table_density")

    preview_df = search_dataframe(df, query)
    st.caption(f"Showing {min(preview_rows, len(preview_df)):,} of {len(preview_df):,} matching rows")
    st.dataframe(preview_df.head(preview_rows), width="stretch", height=350, row_height=table_row_height())

    render_column_profiler(df)

    tab1, tab2, tab3 = st.tabs(["Column Types", "Missing Values", "Summary Stats"])
    with tab1:
        st.dataframe(column_info(df), width="stretch", height=360, row_height=table_row_height())
    with tab2:
        st.dataframe(missing_values(df), width="stretch", height=360, row_height=table_row_height())
    with tab3:
        st.dataframe(summary_stats(df).astype(str), width="stretch", row_height=table_row_height())

    st.subheader("Correlation Matrix")
    corr = correlations(df)
    if corr.empty:
        st.info("At least two numeric columns are required for correlations.")
    else:
        st.plotly_chart(plot_heatmap(df), width="stretch", key="overview_correlation_heatmap")
        with st.expander("Show correlation values"):
            st.dataframe(corr, width="stretch", row_height=table_row_height())


def render_prompt_buttons(prompts: list[str], state_key: str, prefix: str) -> str | None:
    """Render reusable prompt shortcut buttons and return the selected prompt."""
    selected_prompt: str | None = None
    cols = st.columns(len(prompts))
    for idx, prompt in enumerate(prompts):
        if cols[idx].button(prompt, key=f"{prefix}_{idx}", width="stretch"):
            st.session_state[state_key] = prompt
            selected_prompt = prompt
    return selected_prompt


def render_voice_controls() -> str | None:
    """Render voice input and voice output controls for AI chat."""
    transcript_to_send: str | None = None
    with st.expander("Voice input and output", expanded=False):
        st.caption("Voice responses are AI-generated.")
        control_cols = st.columns(3)
        control_cols[0].toggle("Speak AI responses", key="voice_output_enabled")
        control_cols[1].toggle("Autoplay voice", key="voice_autoplay")
        control_cols[2].selectbox("Voice", VOICE_OPTIONS, key="tts_voice")
        st.selectbox("Transcription model", TRANSCRIPTION_OPTIONS, key="transcription_model")

        recorded_audio = VOICE_RECORDER(key="conversation_voice_recorder")

        upload = st.file_uploader(
            "Or upload an audio question",
            type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
            key="conversation_audio_upload",
        )

        audio_bytes: bytes | None = None
        filename = "voice_input.webm"
        if recorded_audio and isinstance(recorded_audio, dict) and recorded_audio.get("dataUrl"):
            audio_bytes = decode_audio_data_url(recorded_audio["dataUrl"])
            mime_type = recorded_audio.get("mimeType") or "audio/webm"
            filename = recorded_audio.get("filename") or "voice_input.webm"
            st.audio(audio_bytes, format=mime_type)
        elif upload is not None:
            audio_bytes = upload.getvalue()
            extension = audio_extension_from_name(upload.name)
            filename = upload.name or f"voice_input.{extension}"
            st.audio(audio_bytes, format=f"audio/{extension}")

        if st.button("Transcribe to message", icon=":material/keyboard_voice:", disabled=audio_bytes is None):
            if audio_bytes and demo_mode_enabled() and len(audio_bytes) > DEMO_MAX_AUDIO_BYTES:
                st.warning(f"Demo mode limits audio input to {DEMO_MAX_AUDIO_MB:,} MB.")
                return None
            if not demo_guard_allows(
                "Voice transcription",
                filename,
                DEMO_TRANSCRIPTION_TOKEN_COST,
                include_context=False,
            ):
                return None
            with st.spinner("Transcribing voice input..."):
                try:
                    transcript = transcribe_audio(
                        audio_bytes or b"",
                        filename=filename,
                        model=st.session_state.get("transcription_model", "gpt-4o-mini-transcribe"),
                    )
                    trimmed_transcript = truncate_demo_text(transcript)
                    if trimmed_transcript != transcript:
                        st.warning("Transcript was shortened to fit the hosted demo prompt limit.")
                    transcript = trimmed_transcript
                    record_demo_usage(
                        "Voice transcription",
                        filename,
                        transcript,
                        DEMO_TRANSCRIPTION_TOKEN_COST,
                        include_context=False,
                    )
                    st.session_state["last_transcript"] = transcript
                    st.session_state["conversation_draft"] = transcript
                    st.success("Transcript is ready to send.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        if st.session_state.get("last_transcript"):
            st.caption("Latest transcript")
            st.code(st.session_state["last_transcript"], language="text")
            transcript_cols = st.columns(2)
            if transcript_cols[0].button("Send transcript", icon=":material/send:", type="primary", width="stretch"):
                transcript_to_send = st.session_state["last_transcript"]
            if transcript_cols[1].button("Clear transcript", icon=":material/close:", width="stretch"):
                st.session_state["last_transcript"] = ""
                st.session_state["conversation_draft"] = ""
                st.rerun()

        if st.session_state.get("last_voice_audio"):
            st.caption("Latest voice response")
            st.audio(
                st.session_state["last_voice_audio"],
                format="audio/mp3",
                autoplay=st.session_state.get("voice_autoplay", False),
            )
    return transcript_to_send


def render_conversation_empty_state() -> None:
    """Render a light empty state for a new AI conversation."""
    st.markdown(
        """
        <div class="conversation-empty">
            <div class="conversation-empty-title">Start a conversation with your dataset</div>
            <div class="conversation-empty-body">
                Ask for trends, definitions, outlier explanations, reporting caveats, or follow-up analysis.
                The assistant keeps recent chat context and uses the active filtered dataset view.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def submit_conversation_message(df: pd.DataFrame, question: str) -> bool:
    """Submit one conversational turn and render the assistant response."""
    question = truncate_demo_text(question.strip())
    if not question:
        st.warning("Enter a message before sending.")
        return False
    if not demo_guard_allows("Conversation AI", question, ask_output_tokens()):
        return False

    messages = st.session_state.setdefault("ai_messages", [])
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the dataset..."):
            try:
                response = conversation_ai(
                    df,
                    question,
                    history=messages[:-1],
                    model=selected_ai_model(),
                    reasoning_effort=selected_reasoning_effort(),
                    max_tokens=ask_output_tokens(),
                    context_max_chars=demo_context_chars(),
                )
                render_ai_response(response)
                messages.append({"role": "assistant", "content": response})
                record_demo_usage("Conversation AI", question, response, ask_output_tokens())
                if st.session_state.get("voice_output_enabled"):
                    with st.spinner("Generating voice response..."):
                        audio = text_to_speech(
                            speech_safe_text(
                                response,
                                max_chars=DEMO_TTS_CHAR_LIMIT if demo_mode_enabled() else 12_000,
                            ),
                            voice=st.session_state.get("tts_voice", "coral"),
                        )
                    st.session_state["last_voice_audio"] = audio
                    st.audio(
                        audio,
                        format="audio/mp3",
                        autoplay=st.session_state.get("voice_autoplay", False),
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return False
    return True


def render_conversation_ai(df: pd.DataFrame) -> None:
    """Render conversational AI analyst chat."""
    st.markdown(
        f"""
        <div class="conversation-toolbar">
            <span class="status-pill">Using {escape(selected_ai_model())}</span>
            <span class="status-pill">{len(st.session_state.get("ai_messages", [])) // 2:,} conversation turns</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    voice_prompt = render_voice_controls()
    shortcut_prompt = render_prompt_buttons(SUGGESTED_QUESTIONS, "conversation_draft", "conversation_prompt")

    messages = st.session_state.setdefault("ai_messages", [])
    if not messages:
        render_conversation_empty_state()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_ai_response(message["content"])
            else:
                st.markdown(message["content"])

    chat_prompt = st.chat_input(
        "Message Conversation AI...",
        max_chars=DEMO_MAX_REQUEST_CHARS if demo_mode_enabled() else None,
        key="conversation_chat_input",
    )
    prompt = voice_prompt or shortcut_prompt or chat_prompt
    if prompt and submit_conversation_message(df, str(prompt)):
        st.session_state["conversation_draft"] = ""
        if voice_prompt:
            st.session_state["last_transcript"] = ""


def render_visualizations(df: pd.DataFrame) -> None:
    """Render interactive chart builder."""
    render_action_bar("Suggested chart actions", VISUAL_ACTIONS, "chart_intent", "visual_action")

    nums = numeric_columns(df)
    cats = categorical_columns(df)
    chart_options = [
        "Histogram",
        "Bar chart",
        "Aggregated bar",
        "Line chart",
        "Scatter plot",
        "Box plot",
        "Correlation heatmap",
    ]
    intent = st.session_state.get("chart_intent", "Auto")
    recommendation = recommended_chart_config(df, intent)
    recommended_type = str(recommendation["chart_type"])
    recommended_index = chart_options.index(recommended_type) if recommended_type in chart_options else 0

    left, right = st.columns([1, 2.2])
    with left:
        st.markdown(
            f'<span class="status-pill">Recommended: {escape(recommended_type)}</span>',
            unsafe_allow_html=True,
        )
        chart_type = st.selectbox(
            "Chart type",
            chart_options,
            index=recommended_index,
        )

        try:
            if chart_type == "Histogram":
                options = nums or list(df.columns)
                default = recommendation["x"] if recommendation["x"] in options else options[0]
                column = st.selectbox("Column", options, index=options.index(default))
                fig = plot_histogram(df, column)
            elif chart_type == "Bar chart":
                options = cats or list(df.columns)
                default = recommendation["x"] if recommendation["x"] in options else options[0]
                column = st.selectbox("Column", options, index=options.index(default))
                fig = plot_bar(df, column)
            elif chart_type == "Aggregated bar":
                if not nums:
                    st.info("Aggregated bars need at least one numeric column.")
                    return
                category_options = cats or list(df.columns)
                category_default = recommendation["x"] if recommendation["x"] in category_options else category_options[0]
                value_default = recommendation["y"] if recommendation["y"] in nums else nums[0]
                agg_options = ["sum", "mean", "median", "min", "max", "count"]
                agg_default = recommendation["aggregation"] if recommendation["aggregation"] in agg_options else "sum"
                category = st.selectbox("Category", category_options, index=category_options.index(category_default))
                value = st.selectbox("Value", nums, index=nums.index(value_default))
                aggregation = st.selectbox("Aggregation", agg_options, index=agg_options.index(str(agg_default)))
                fig = plot_aggregated_bar(df, category, value, aggregation)
            elif chart_type == "Line chart":
                if not nums:
                    st.info("Line charts need at least one numeric Y column.")
                    return
                x_options = list(df.columns)
                x_default = recommendation["x"] if recommendation["x"] in x_options else x_options[0]
                y_default = recommendation["y"] if recommendation["y"] in nums else nums[0]
                x = st.selectbox("X axis", x_options, index=x_options.index(x_default))
                y = st.selectbox("Y axis", nums, index=nums.index(y_default))
                fig = plot_line(df, x, y)
            elif chart_type == "Scatter plot":
                if len(nums) < 2:
                    st.info("Scatter plots need at least two numeric columns.")
                    return
                x_default = recommendation["x"] if recommendation["x"] in nums else nums[0]
                y_default = recommendation["y"] if recommendation["y"] in nums else nums[1]
                x = st.selectbox("X axis", nums, index=nums.index(x_default))
                y = st.selectbox("Y axis", nums, index=nums.index(y_default))
                color_options = ["None"] + cats
                color_default = recommendation["color"] if recommendation["color"] in color_options else "None"
                color = st.selectbox("Color by", color_options, index=color_options.index(str(color_default)))
                fig = plot_scatter(df, x, y, None if color == "None" else color)
            elif chart_type == "Box plot":
                if not nums:
                    st.info("Box plots need at least one numeric column.")
                    return
                value_default = recommendation["y"] if recommendation["y"] in nums else nums[0]
                column = st.selectbox("Numeric column", nums, index=nums.index(value_default))
                group_options = ["None"] + cats
                group_default = recommendation["x"] if recommendation["x"] in group_options else "None"
                group_by = st.selectbox("Group by", group_options, index=group_options.index(str(group_default)))
                fig = plot_box(df, column, None if group_by == "None" else group_by)
            else:
                fig = plot_heatmap(df)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

    with right:
        st.plotly_chart(fig, width="stretch", key=f"visualization_builder_{chart_type}")

    with st.expander("Chart source data"):
        st.dataframe(df.head(250), width="stretch", height=260, row_height=table_row_height())
        st.download_button(
            "Download chart source data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="chart_source_data.csv",
            mime="text/csv",
        )


def render_insights(df: pd.DataFrame) -> None:
    """Render deterministic insights and anomaly review."""
    render_action_bar("Suggested insight actions", INSIGHT_ACTIONS, "insight_focus", "insight_action")
    st.toggle("Show analytics glossary", key="show_term_glossary")
    if st.session_state.get("show_term_glossary"):
        render_term_glossary()
    if st.session_state.get("insight_focus") == "Prepare executive summary":
        st.info("Executive summary mode is active. Use Presentation Mode for a cleaner share-ready view.")
    render_insight_cards(df, limit=8)

    st.subheader("Data Quality")
    health = dataset_health(df)
    quality = pd.DataFrame(
        [
            {"metric": "Health score", "value": f'{int(health["score"])}/100'},
            {"metric": "Missing cells", "value": f'{int(health["missing_cells"]):,}'},
            {"metric": "Missing percent", "value": f'{health["missing_percent"]}%'},
            {"metric": "Duplicate rows", "value": f'{int(health["duplicate_rows"]):,}'},
            {"metric": "Duplicate percent", "value": f'{health["duplicate_percent"]}%'},
        ]
    )
    st.dataframe(quality, width="stretch", hide_index=True, row_height=table_row_height())

    st.subheader("Anomaly Detection")
    nums = numeric_columns(df)
    if not nums:
        st.info("Anomaly detection requires at least one numeric column.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        method = st.selectbox("Method", ["IQR", "Z-score"])
        if method == "Z-score":
            threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
            anomalies = detect_outliers_zscore(df, threshold=threshold)
        else:
            multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
            anomalies = detect_outliers_iqr(df, multiplier=multiplier)

    with col2:
        st.dataframe(outlier_summary(df), width="stretch", height=220, row_height=table_row_height())

    if anomalies.empty:
        st.success("No anomalous rows were detected with the current settings.")
    else:
        st.warning(f"Detected {len(anomalies):,} anomalous rows.")
        styled = anomalies.style.apply(lambda _: ["background-color: #fff7ed"] * len(anomalies.columns), axis=1)
        st.dataframe(styled, width="stretch", height=360, row_height=table_row_height())
        st.download_button(
            "Download anomalies as CSV",
            data=anomalies.to_csv(index=False).encode("utf-8"),
            file_name="anomalies.csv",
            mime="text/csv",
        )


def render_presentation_mode(df: pd.DataFrame) -> None:
    """Render a clean dashboard-style view for demos and stakeholder review."""
    st.markdown(
        """
        <div class="presentation-band">
            <div class="presentation-heading">Executive Snapshot</div>
            <div class="muted">Use this page for a clean demo narrative without raw tables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_insight_cards(df, limit=4)

    nums = numeric_columns(df)
    cats = categorical_columns(df)
    dates = detect_datetime_columns(df)
    metric = choose_column(nums, ("revenue", "sales", "amount", "profit", "units", "score"))
    category = choose_column(cats, ("region", "product", "channel", "segment", "category"))
    date_col = choose_column(dates, ("date", "month", "time", "year"))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        if date_col and metric:
            st.plotly_chart(
                plot_line(df, date_col, metric),
                width="stretch",
                key="presentation_primary_line",
            )
        elif metric:
            st.plotly_chart(
                plot_histogram(df, metric),
                width="stretch",
                key="presentation_primary_histogram",
            )
        else:
            st.info("Add a numeric column to show a KPI trend or distribution.")

    with chart_right:
        if category and metric:
            st.plotly_chart(
                plot_aggregated_bar(df, category, metric, "sum"),
                width="stretch",
                key="presentation_segment_bar",
            )
        elif len(nums) >= 2:
            st.plotly_chart(plot_heatmap(df), width="stretch", key="presentation_heatmap")
        else:
            st.info("Add category and numeric columns to show segment performance.")

    if metric:
        st.plotly_chart(plot_box(df, metric, category), width="stretch", key="presentation_box")


def agent_display_name(agent_name: str) -> str:
    """Return a friendlier agent name."""
    return agent_name.replace("Agent", " Agent").replace("DataCleaning", "Data Cleaning")


def render_agent_cards(run: PipelineRun) -> None:
    """Render compact stage-level pipeline status cards."""
    analysis_agents = [
        "TrendAgent",
        "AnomalyAgent",
        "CorrelationAgent",
        "InsightsAgent",
        "VisualizationAgent",
    ]
    analysis_done = sum(
        1 for agent in analysis_agents if run.agent_results.get(agent) and run.agent_results[agent].status == "completed"
    )
    stages = [
        ("Cleaning", run.agent_results.get("DataCleaningAgent"), None),
        ("Verification", run.agent_results.get("VerificationAgent"), None),
        (
            "Analysis",
            None,
            {
                "status": "completed" if analysis_done == len(analysis_agents) else "pending",
                "summary": f"{analysis_done}/{len(analysis_agents)} analysis agents completed",
            },
        ),
        ("Report", run.agent_results.get("ReportSynthesisAgent"), None),
    ]
    columns = st.columns(4)
    for column, (label, result, synthetic) in zip(columns, stages):
        status = synthetic["status"] if synthetic else result.status if result else "pending"
        summary = synthetic["summary"] if synthetic else result.summary if result else "Waiting for this stage."
        duration = "" if synthetic or not result else f" · {result.duration_seconds}s"
        status_class = {
            "completed": "status-pill-ok",
            "failed": "status-pill-failed",
            "pending": "status-pill-pending",
        }.get(str(status), "status-pill-pending")
        status_text = "Complete" if status == "completed" else "Failed" if status == "failed" else "Pending"
        with column.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(
                f'<span class="status-pill {status_class}">{escape(status_text + duration)}</span>',
                unsafe_allow_html=True,
            )
            st.caption(summary)


def render_agent_details(run: PipelineRun) -> None:
    """Render detailed agent findings in one compact expander."""
    if not run.agent_results:
        return
    with st.expander("Agent details", expanded=False):
        agent_names = list(run.agent_results)
        selected = st.selectbox(
            "Agent",
            agent_names,
            format_func=agent_display_name,
            key="pipeline_agent_detail_select",
        )
        result = run.agent_results[selected]
        st.write(result.summary)
        if result.findings:
            st.markdown("**Findings**")
            for finding in result.findings:
                st.write(f"- {finding}")
        if result.metrics:
            st.markdown("**Metrics**")
            st.dataframe(
                pd.DataFrame([{"metric": key, "value": value} for key, value in result.metrics.items()]),
                width="stretch",
                row_height=table_row_height(),
            )


def render_cleaning_actions(run: PipelineRun) -> None:
    """Render the cleaning audit table."""
    if not run.cleaning_actions:
        st.info("The cleaning agent did not need to modify the dataset.")
        return
    st.dataframe(
        pd.DataFrame([action.__dict__ for action in run.cleaning_actions]),
        width="stretch",
        row_height=table_row_height(),
    )


def render_pipeline_chart(df: pd.DataFrame, spec: dict[str, object], idx: int) -> None:
    """Render a chart recommendation from the visualization agent."""
    chart_type = str(spec.get("type", ""))
    title = str(spec.get("title", f"Pipeline chart {idx + 1}"))
    try:
        if chart_type == "line":
            fig = plot_line(df, str(spec["x"]), str(spec["y"]))
        elif chart_type == "bar":
            fig = plot_aggregated_bar(
                df,
                str(spec["x"]),
                str(spec["y"]),
                str(spec.get("aggregation", "sum")),
            )
        elif chart_type == "box":
            group_by = spec.get("x")
            fig = plot_box(df, str(spec["y"]), str(group_by) if group_by else None)
        elif chart_type == "heatmap":
            fig = plot_heatmap(df)
        elif chart_type == "histogram":
            fig = plot_histogram(df, str(spec["x"]))
        else:
            st.info(f"Unsupported chart recommendation: {chart_type}")
            return
        fig.update_layout(title=title)
        st.plotly_chart(fig, width="stretch", key=f"pipeline_chart_{idx}_{chart_type}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render {title}: {exc}")


def render_pipeline_history() -> None:
    """Render persisted pipeline history."""
    runs = load_recent_runs(limit=8)
    if not runs:
        st.caption("No completed pipeline runs have been saved yet.")
        return
    for item in runs:
        with st.container(border=True):
            st.markdown(f"**{item['dataset_name']}**")
            st.caption(
                f"Run {item['run_id']} | {item['created_at']} | "
                f"{item['rows']:,} rows x {item['columns']:,} columns"
            )
            st.write(item.get("summary") or "No summary saved.")


def render_agent_pipeline(df: pd.DataFrame, source_name: str | None) -> None:
    """Render the InsightFlow-style agentic pipeline."""
    run = st.session_state.get("pipeline_run")
    dataset_name = source_name or "Filtered dataset"

    control_cols = st.columns([1.1, 1, 3.5], vertical_alignment="center")
    if control_cols[0].button("Start pipeline", icon=":material/play_arrow:", type="primary", width="stretch"):
        with st.spinner("Running cleaning and verification agents..."):
            st.session_state["pipeline_run"] = start_pipeline(df, dataset_name)
        st.rerun()
    if control_cols[1].button("Reset pipeline", icon=":material/restart_alt:", width="stretch"):
        st.session_state["pipeline_run"] = None
        st.rerun()
    control_cols[2].caption(f"Source: {dataset_name} · {df.shape[0]:,} rows · {df.shape[1]:,} columns")

    if not isinstance(run, PipelineRun):
        with st.container(border=True):
            st.markdown("**Ready to run**")
            st.write("The pipeline pauses after cleaning so you can approve the proposal before analysis.")
        with st.expander("Previous pipeline runs", expanded=False):
            render_pipeline_history()
        return

    render_agent_cards(run)
    render_agent_details(run)

    stage_cols = st.columns(3)
    stage_cols[0].metric("Stage", pipeline_stage_label(run.current_stage))
    stage_cols[1].metric("Rows", f"{run.raw_shape[0]:,}")
    stage_cols[2].metric("Approval", run.approval_status.value.replace("_", " ").title())

    st.subheader("Cleaning Proposal")
    render_cleaning_actions(run)

    if run.cleaned_df is not None:
        preview_tabs = st.tabs(["Cleaned Preview", "Before/After Quality"])
        with preview_tabs[0]:
            st.dataframe(run.cleaned_df.head(50), width="stretch", row_height=table_row_height())
            st.download_button(
                "Download cleaned proposal",
                data=run.cleaned_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_pipeline_dataset.csv",
                mime="text/csv",
            )
        with preview_tabs[1]:
            raw_health = dataset_health(run.raw_df)
            clean_health = dataset_health(run.cleaned_df)
            comparison = pd.DataFrame(
                [
                    {"metric": "Health score", "raw": raw_health["score"], "cleaned": clean_health["score"]},
                    {"metric": "Missing cells", "raw": raw_health["missing_cells"], "cleaned": clean_health["missing_cells"]},
                    {"metric": "Duplicate rows", "raw": raw_health["duplicate_rows"], "cleaned": clean_health["duplicate_rows"]},
                    {"metric": "Rows", "raw": len(run.raw_df), "cleaned": len(run.cleaned_df)},
                ]
            )
            st.dataframe(comparison, width="stretch", row_height=table_row_height())

    verification = run.agent_results.get("VerificationAgent")
    if verification:
        st.subheader("Verification")
        severity = verification.metrics.get("severity", "low")
        st.markdown(f'<span class="status-pill">Severity: {escape(str(severity).title())}</span>', unsafe_allow_html=True)
        for finding in verification.findings:
            st.write(f"- {finding}")

    if run.current_stage == PipelineStage.CLEANING_APPROVAL:
        st.markdown(
            """
            <div class="approval-panel">
                <div class="presentation-heading">Human approval required</div>
                <div class="muted">Review the cleaning proposal before analysis agents use it. This keeps the workflow auditable and explainable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        approve_cols = st.columns(3)
        if approve_cols[0].button("Approve cleaned data", icon=":material/check_circle:", type="primary", width="stretch"):
            with st.spinner("Running analysis agents in parallel..."):
                st.session_state["pipeline_run"] = run_analysis(run, use_cleaned_data=True)
            st.rerun()
        if approve_cols[1].button("Analyze raw data", icon=":material/database:", width="stretch"):
            with st.spinner("Running analysis agents on raw data..."):
                st.session_state["pipeline_run"] = run_analysis(run, use_cleaned_data=False)
            st.rerun()
        if approve_cols[2].button("Reject proposal", icon=":material/block:", width="stretch"):
            st.session_state["pipeline_run"] = reject_cleaning(run)
            st.warning("Cleaning proposal marked as rejected. Start a new pipeline run to retry.")
            st.rerun()

    if run.current_stage != PipelineStage.COMPLETED:
        with st.expander("Previous pipeline runs", expanded=False):
            render_pipeline_history()
        return

    st.subheader("Final Agent Report")
    report_tabs = st.tabs(["Executive Report", "Trends", "Anomalies", "Correlations", "Charts", "History"])

    with report_tabs[0]:
        st.markdown(run.report_markdown)
        st.download_button(
            "Download executive report",
            data=run.report_markdown.encode("utf-8"),
            file_name=f"agentic_report_{run.run_id}.md",
            mime="text/markdown",
            type="primary",
        )

    with report_tabs[1]:
        trend = run.agent_results.get("TrendAgent")
        trends = trend.artifacts.get("trends", []) if trend else []
        if trends:
            st.dataframe(pd.DataFrame(trends), width="stretch", row_height=table_row_height())
        elif trend:
            st.info(trend.summary)

    with report_tabs[2]:
        anomaly = run.agent_results.get("AnomalyAgent")
        if anomaly:
            summary = anomaly.artifacts.get("summary", [])
            if summary:
                st.dataframe(pd.DataFrame(summary), width="stretch", row_height=table_row_height())
            sample_rows = anomaly.artifacts.get("sample_rows", [])
            if sample_rows:
                with st.expander("Sample anomaly rows", expanded=False):
                    st.dataframe(pd.DataFrame(sample_rows), width="stretch", row_height=table_row_height())

    with report_tabs[3]:
        corr = run.agent_results.get("CorrelationAgent")
        pairs = corr.artifacts.get("top_pairs", []) if corr else []
        if pairs:
            st.dataframe(pd.DataFrame(pairs), width="stretch", row_height=table_row_height())
        elif corr:
            st.info(corr.summary)

    with report_tabs[4]:
        if run.chart_specs:
            for idx, spec in enumerate(run.chart_specs):
                render_pipeline_chart(run.active_df, spec, idx)
        else:
            st.info("The visualization agent did not recommend charts for this dataset.")

    with report_tabs[5]:
        render_pipeline_history()


def render_code_generator(df: pd.DataFrame) -> None:
    """Render SQL and Pandas generator."""
    render_prompt_buttons(CODE_PROMPTS, "code_request", "code_prompt")

    with st.form("code_generator_form"):
        request = st.text_area(
            "Generate SQL or Pandas code",
            placeholder="Example: Find monthly revenue by region and show the top 5 months.",
            height=130,
            max_chars=DEMO_MAX_REQUEST_CHARS if demo_mode_enabled() else None,
            key="code_request",
        )
        submitted = st.form_submit_button("Generate code", type="primary")

    if submitted:
        request = truncate_demo_text(request.strip())
        if not request:
            st.warning("Describe the code you want generated.")
            return
        if not demo_guard_allows("Code Generator", request, code_output_tokens()):
            return
        with st.spinner("Generating code..."):
            try:
                generated = generate_code(
                    df,
                    request,
                    model=selected_ai_model(),
                    reasoning_effort=selected_reasoning_effort(),
                    max_tokens=code_output_tokens(),
                    context_max_chars=demo_context_chars(),
                )
                st.session_state["generated_code"] = generated
                record_demo_usage("Code Generator", request, generated, code_output_tokens())
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    if st.session_state.get("generated_code"):
        st.markdown(st.session_state["generated_code"])
        st.download_button(
            "Download generated answer",
            data=st.session_state["generated_code"].encode("utf-8"),
            file_name="generated_sql_pandas.md",
            mime="text/markdown",
        )


def render_empty_state(load_error: str | None = None) -> None:
    """Render state before a dataset is available."""
    st.markdown(
        """
        <div class="empty-workspace">
            <div class="app-eyebrow">AI Data Analyst</div>
            <div class="empty-title">Start with a dataset</div>
            <div class="muted">Upload a CSV or Excel workbook from the sidebar, or load the bundled sample dataset for a guided demo.</div>
            <div class="empty-grid">
                <div class="empty-item">
                    <div class="empty-item-title">Supported files</div>
                    <div class="muted">CSV and XLSX files up to Streamlit's configured upload limit.</div>
                </div>
                <div class="empty-item">
                    <div class="empty-item-title">Private by default</div>
                    <div class="muted">Local exploration works offline. AI features only send compact schema and sample context.</div>
                </div>
                <div class="empty-item">
                    <div class="empty-item-title">Workflow</div>
                    <div class="muted">Load data, inspect readiness, chat with Conversation AI, build charts, review anomalies, then export code.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if load_error:
        st.error(load_error)
    if st.button("Load sample dataset", icon=":material/table_chart:", type="primary"):
        st.session_state["use_sample_dataset"] = True
        st.rerun()


def main() -> None:
    """Run the Streamlit app."""
    inject_css()
    initialize_state()

    raw_df, filtered_df, navigation, source_name, load_error, active_filters = render_sidebar()

    if raw_df is None or filtered_df is None:
        render_empty_state(load_error)
        return

    reset_chat_if_dataset_changed(raw_df)

    if filtered_df.empty:
        render_header(filtered_df, raw_df, source_name, active_filters, show_readiness=False)
        st.warning("The current filters returned no rows. Adjust filters in the sidebar.")
        return

    render_header(
        filtered_df,
        raw_df,
        source_name,
        active_filters,
        show_readiness=navigation != "Presentation Mode",
    )

    if navigation == "Overview":
        render_overview(filtered_df)
    elif navigation == "Conversation AI":
        render_conversation_ai(filtered_df)
    elif navigation == "Visualizations":
        render_visualizations(filtered_df)
    elif navigation == "Insights & Anomalies":
        render_insights(filtered_df)
    elif navigation == "Agent Pipeline":
        render_agent_pipeline(filtered_df, source_name)
    elif navigation == "Presentation Mode":
        render_presentation_mode(filtered_df)
    else:
        render_code_generator(filtered_df)


if __name__ == "__main__":
    main()

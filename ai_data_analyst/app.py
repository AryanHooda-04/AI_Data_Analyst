"""Streamlit entry point for the InsightAnalytica application."""

from __future__ import annotations

import base64
import os
import re
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
from query_engine import CleanedQueryResult, answer_with_cleaned_sql, should_execute_query
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
APP_NAME = "InsightAnalytica"
APP_TAGLINE = "Enterprise AI data intelligence"
HEADER_ART = APP_DIR / "assets" / "insightanalytica_logo.png"
BRAND_MARK = APP_DIR / "assets" / "insightanalytica_mark.png"
VOICE_RECORDER = components.declare_component("voice_recorder", path=str(APP_DIR / "voice_recorder"))

NAV_ITEMS = [
    {"label": "Overview", "short": "Overview", "icon": ":material/dashboard:"},
    {"label": "Conversation AI", "short": "AI Chat", "icon": ":material/forum:"},
    {"label": "Visualizations", "short": "Charts", "icon": ":material/monitoring:"},
    {"label": "Insights & Anomalies", "short": "Insights", "icon": ":material/troubleshoot:"},
    {"label": "Agent Pipeline", "short": "Pipeline", "icon": ":material/account_tree:"},
    {"label": "Code Generator", "short": "Code", "icon": ":material/code:"},
    {"label": "Presentation Mode", "short": "Present", "icon": ":material/present_to_all:"},
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
    "Query Result",
    "Evidence",
    "Caveats",
    "Guardrails",
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
THEME_OPTIONS = ["Light", "Dark"]


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


ASSISTANT_CHAT_AVATAR = asset_data_uri(BRAND_MARK, "image/png")
USER_CHAT_AVATAR = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
          <defs>
            <linearGradient id="g" x1="10" x2="54" y1="8" y2="58" gradientUnits="userSpaceOnUse">
              <stop stop-color="#7c3aed"/>
              <stop offset="1" stop-color="#2563eb"/>
            </linearGradient>
          </defs>
          <rect width="64" height="64" rx="16" fill="url(#g)"/>
          <circle cx="32" cy="25" r="10" fill="#fff" opacity=".95"/>
          <path d="M16 52c2.7-10 10-16 16-16s13.3 6 16 16" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>
        </svg>
        """
    ).decode("ascii")
)


def chat_avatar(role: str) -> str:
    """Return the branded avatar for a chat role."""
    normalized_role = str(role or "").lower()
    if normalized_role in {"assistant", "ai"}:
        return ASSISTANT_CHAT_AVATAR or ":material/auto_awesome:"
    return USER_CHAT_AVATAR


def active_theme_mode() -> str:
    """Return the selected display theme."""
    mode = st.session_state.get("theme_mode", "Dark")
    return mode if mode in THEME_OPTIONS else "Dark"


def toggle_theme_mode() -> None:
    """Switch between light and dark display modes."""
    st.session_state["theme_mode"] = "Light" if active_theme_mode() == "Dark" else "Dark"


def theme_css_variables() -> str:
    """Return CSS custom properties for the active app theme."""
    if active_theme_mode() == "Dark":
        return """
            color-scheme: dark;
            --app-bg: #0b1220;
            --panel: #111827;
            --panel-soft: #0f172a;
            --panel-border: #26374f;
            --ink: #e7eef8;
            --muted: #9fb0c5;
            --eyebrow: #d7e7fb;
            --sidebar-bg: #0b1220;
            --sidebar-panel: #111827;
            --sidebar-line: #26374f;
            --sidebar-ink: #e7eef8;
            --sidebar-muted: #9fb0c5;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --accent-3: #fb7185;
            --accent-warm: #fbbf24;
            --accent-soft: #0d2a48;
            --code-bg: #08111f;
            --code-inline-bg: #17233a;
            --code-border: #2b405d;
            --code-ink: #dbeafe;
            --quote-bg: rgba(56, 189, 248, 0.09);
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #fb7185;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.24);
            --shadow-md: 0 18px 42px rgba(0, 0, 0, 0.28);
        """
    return """
            color-scheme: light;
            --app-bg: #f4f7fb;
            --panel: #ffffff;
            --panel-soft: #f9fbff;
            --panel-border: #d9e4f2;
            --ink: #102033;
            --muted: #63748a;
            --eyebrow: #425875;
            --sidebar-bg: #f8fbff;
            --sidebar-panel: #ffffff;
            --sidebar-line: #d7e4f1;
            --sidebar-ink: #102033;
            --sidebar-muted: #64748b;
            --accent: #1d4ed8;
            --accent-2: #0891b2;
            --accent-3: #e11d48;
            --accent-warm: #f59e0b;
            --accent-soft: #e8f1ff;
            --code-bg: #f8fbff;
            --code-inline-bg: #edf4ff;
            --code-border: #cbdcf1;
            --code-ink: #17324d;
            --quote-bg: #f0f7ff;
            --success: #0f8a55;
            --warning: #b45309;
            --danger: #b91c1c;
            --shadow-sm: 0 1px 2px rgba(16, 32, 51, 0.06);
            --shadow-md: 0 12px 30px rgba(16, 32, 51, 0.08);
        """


def theme_override_css() -> str:
    """Return targeted overrides so the app theme wins over Streamlit's native theme."""
    if active_theme_mode() == "Light":
        return """
        html,
        body,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        [data-testid="stMain"] > div,
        section.main,
        .main,
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            color-scheme: light !important;
            background:
                linear-gradient(180deg, #fbfdff 0%, #f2faf6 42%, #f8f7ff 100%) !important;
            color: var(--ink) !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            border-bottom: 0 !important;
            backdrop-filter: none !important;
        }

        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatFloatingInputContainer"] {
            background: #f6f9fd !important;
            background-color: #f6f9fd !important;
            background-image: none !important;
            border-top: 1px solid var(--panel-border) !important;
            box-shadow: 0 -18px 44px rgba(16, 32, 51, 0.10) !important;
        }

        [data-testid="stBottom"]::before,
        [data-testid="stBottom"]::after,
        [data-testid="stBottomBlockContainer"]::before,
        [data-testid="stBottomBlockContainer"]::after,
        [data-testid="stChatFloatingInputContainer"]::before,
        [data-testid="stChatFloatingInputContainer"]::after {
            background: transparent !important;
            background-image: none !important;
        }

        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] > div,
        [data-testid="stChatFloatingInputContainer"] > div,
        div:has(> [data-testid="stChatInput"]) {
            background: #f6f9fd !important;
            background-color: #f6f9fd !important;
            background-image: none !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 54%, #f8fbff 100%) !important;
            border-right-color: var(--sidebar-line) !important;
            box-shadow: 4px 0 28px rgba(16, 32, 51, 0.06);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] em {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        .app-subtitle,
        .sidebar-brand-subtitle,
        .sidebar-card-meta,
        .sidebar-card-title,
        .section-kicker,
        .conversation-empty-body,
        .ai-response-card-body,
        .agent-card-body,
        .glossary-body,
        .empty-lead,
        .muted,
        [data-testid="stCaptionContainer"] *,
        .stCaptionContainer,
        .stCaptionContainer *,
        small {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        .sidebar-card,
        .topbar-command-panel,
        .readiness-band,
        .presentation-band,
        .suggestion-panel,
        .insight-card,
        .data-story-card,
        .ai-response-card,
        .conversation-empty,
        .empty-workspace,
        .glossary-card,
        .agent-card,
        .history-row,
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        .st-key-top_workspace_nav,
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
            background: var(--panel) !important;
            border-color: var(--panel-border) !important;
            color: var(--ink) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        .st-key-theme_toggle_button button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
        [data-testid="stSidebar"] .stDownloadButton button,
        .st-key-top_workspace_nav .stButton > button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-color: #c9d2e3 !important;
            color: var(--ink) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        .st-key-theme_toggle_button button *,
        [data-testid="stFormSubmitButton"] button *,
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button svg,
        [data-testid="stSidebar"] .stDownloadButton button *,
        .st-key-top_workspace_nav .stButton > button * {
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .st-key-theme_toggle_button button:hover,
        [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stSidebar"] .stButton > button:hover,
        .st-key-top_workspace_nav .stButton > button:hover {
            background: #f3f8ff !important;
            border-color: #9ec8f7 !important;
            color: var(--accent) !important;
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.10) !important;
        }

        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        .st-key-top_workspace_nav .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            background-color: var(--accent) !important;
            border-color: transparent !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.16) !important;
        }

        .stButton > button[kind="primary"] *,
        [data-testid="stFormSubmitButton"] button[kind="primary"] *,
        [data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        .st-key-top_workspace_nav .stButton > button[kind="primary"] * {
            color: #ffffff !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }

        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] label *,
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label *,
        [data-testid="stMultiSelect"] label,
        [data-testid="stMultiSelect"] label *,
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] label *,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label *,
        [data-baseweb="radio"] *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"] p {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        input,
        textarea,
        [contenteditable="true"],
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"],
        [data-baseweb="select"] > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] [contenteditable="true"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--ink) !important;
            border-color: #c9d2e3 !important;
        }

        [data-baseweb="select"] *,
        [data-testid="stSelectbox"] *,
        [data-testid="stMultiSelect"] *,
        [data-testid="stTextInput"] *,
        [data-testid="stNumberInput"] * {
            color: var(--ink) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"],
        [role="option"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--panel-border) !important;
        }

        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background: #eef6ff !important;
            color: var(--accent) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        [data-testid="stPlotlyChart"] {
            background: #ffffff !important;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.35rem;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] svg,
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stMetric"],
        [data-testid="stMetric"] div {
            background-color: transparent !important;
            color: var(--ink) !important;
        }

        [data-testid="stMetric"] {
            background: #ffffff !important;
            border-color: var(--panel-border) !important;
        }

        [data-testid="stChatMessage"] {
            background: transparent !important;
            color: var(--ink) !important;
        }

        [data-testid="stChatMessage"] > div {
            background: transparent !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
            background: linear-gradient(135deg, #f2f8ff 0%, #eefbf7 100%) !important;
            border: 1px solid #c9ddf4 !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-sm) !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #ffffff !important;
            border-color: var(--panel-border) !important;
        }

        [data-testid="stChatInput"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            background-image: none !important;
            border: 1px solid #c9d2e3 !important;
            border-radius: 10px !important;
            box-shadow: 0 14px 34px rgba(16, 32, 51, 0.10) !important;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form,
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInput"] [data-baseweb="base-input"] > div {
            background: #ffffff !important;
            background-color: #ffffff !important;
            background-image: none !important;
            border-color: transparent !important;
        }

        [data-testid="stChatInput"] button {
            width: 2.25rem !important;
            height: 2.25rem !important;
            min-width: 2.25rem !important;
            min-height: 2.25rem !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.18) !important;
        }

        [data-testid="stChatInput"] button *,
        [data-testid="stChatInput"] button svg,
        [data-testid="stChatInput"] button svg * {
            background: transparent !important;
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button svg {
            width: 1.05rem !important;
            height: 1.05rem !important;
        }

        [data-testid="stChatInput"] button:disabled,
        [data-testid="stChatInput"] button[disabled],
        [data-testid="stChatInput"] button[aria-disabled="true"] {
            background: #e8eef7 !important;
            border-color: #c9d2e3 !important;
            color: #64748b !important;
            box-shadow: none !important;
        }

        .meta-pill,
        .filter-chip,
        .readiness-pill {
            background: #f2f8ff !important;
            border-color: #c9ddf4 !important;
            color: #1e3a5f !important;
        }

        .status-pill,
        .readiness-pill-ready {
            background: #ecfdf5 !important;
            border-color: #bbf7d0 !important;
            color: #166534 !important;
        }

        .readiness-pill-warn {
            background: #fff7ed !important;
            border-color: #fed7aa !important;
            color: #9a3412 !important;
        }
        """
    if active_theme_mode() != "Dark":
        return ""
    return """
        html,
        body,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        [data-testid="stMain"] > div,
        section.main,
        .main,
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            background:
                radial-gradient(circle at 12% 0%, rgba(56, 189, 248, 0.13), transparent 32%),
                linear-gradient(180deg, #0b1220 0%, #0f172a 100%) !important;
            color: var(--ink) !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            border-bottom: 0 !important;
            backdrop-filter: none !important;
        }

        [data-testid="stHeader"]::before,
        [data-testid="stHeader"]::after {
            background: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
        }

        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatFloatingInputContainer"] {
            background: #0b1220 !important;
            background-color: #0b1220 !important;
            background-image: none !important;
            border-top: 1px solid var(--panel-border) !important;
            box-shadow: 0 -18px 44px rgba(0, 0, 0, 0.34) !important;
        }

        [data-testid="stBottom"]::before,
        [data-testid="stBottom"]::after,
        [data-testid="stBottomBlockContainer"]::before,
        [data-testid="stBottomBlockContainer"]::after,
        [data-testid="stChatFloatingInputContainer"]::before,
        [data-testid="stChatFloatingInputContainer"]::after {
            background: transparent !important;
            background-image: none !important;
        }

        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] > div,
        [data-testid="stChatFloatingInputContainer"] > div,
        div:has(> [data-testid="stChatInput"]) {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
        }

        [data-testid="stBottom"] *,
        [data-testid="stBottomBlockContainer"] *,
        [data-testid="stChatFloatingInputContainer"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 54%, #0f172a 100%) !important;
            border-right-color: var(--sidebar-line) !important;
            box-shadow: 4px 0 28px rgba(0, 0, 0, 0.28);
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        .sidebar-card,
        .topbar-command-panel,
        .readiness-band,
        .presentation-band,
        .suggestion-panel,
        .insight-card,
        .data-story-card,
        .ai-response-card,
        .conversation-empty,
        .empty-workspace,
        .glossary-card,
        .agent-card,
        .history-row,
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        .st-key-top_workspace_nav,
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
            background: var(--panel) !important;
            border-color: var(--panel-border) !important;
        }

        .brand-mark,
        .sidebar-brand-logo,
        .app-logo-card {
            background: linear-gradient(135deg, #082f49 0%, #0f172a 100%) !important;
            border-color: #24506d !important;
            box-shadow: 0 14px 28px rgba(56, 189, 248, 0.14);
        }

        .empty-workspace {
            background: linear-gradient(135deg, #111827 0%, #0f2238 100%) !important;
        }

        .glossary-grid .glossary-card {
            background: #111827 !important;
            border-color: var(--panel-border) !important;
            box-shadow: var(--shadow-sm);
        }

        .glossary-grid .glossary-card,
        .glossary-grid .glossary-card * {
            color: var(--ink) !important;
        }

        .glossary-grid .glossary-body {
            color: #b8c8dc !important;
        }

        .presentation-band {
            background: linear-gradient(135deg, #111827 0%, #0f2238 100%) !important;
            border-color: var(--panel-border) !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        }

        .approval-panel {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.13), rgba(17, 24, 39, 0.96)) !important;
            border-color: rgba(251, 191, 36, 0.48) !important;
            color: #fef3c7 !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        }

        .approval-panel .presentation-heading {
            color: #fde68a !important;
        }

        .approval-panel .muted {
            color: #f8dfaa !important;
        }

        [data-testid="stPlotlyChart"] {
            background: var(--panel) !important;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.35rem;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stPlotlyChart"] > div {
            background: transparent !important;
        }

        [data-testid="stDataFrame"] {
            background: var(--panel) !important;
            border: 1px solid var(--panel-border) !important;
            border-radius: 8px;
            padding: 0.12rem;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stDataFrame"] [data-testid="data-grid-canvas"] {
            border-radius: 6px;
            filter: invert(0.92) hue-rotate(180deg) saturate(0.78) brightness(0.92);
        }

        [data-testid="stElementToolbarButtonContainer"] {
            background: #111827 !important;
            border: 1px solid var(--panel-border) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        [data-testid="stElementToolbarButtonContainer"] *,
        [data-testid="stElementToolbarButtonIcon"] {
            color: var(--ink) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }

        [data-testid="stChatMessage"] {
            background: transparent !important;
            color: var(--ink) !important;
        }

        [data-testid="stChatMessage"] > div {
            background: transparent !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] em {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: #d6e3f4 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code,
        [data-testid="stMarkdownContainer"] code {
            background: var(--code-inline-bg) !important;
            border: 1px solid var(--code-border) !important;
            color: var(--code-ink) !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre,
        [data-testid="stMarkdownContainer"] pre {
            background: var(--code-bg) !important;
            border-color: var(--code-border) !important;
            border-left-color: var(--accent) !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code,
        [data-testid="stMarkdownContainer"] pre code {
            background: transparent !important;
            border: 0 !important;
            color: var(--code-ink) !important;
        }

        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #111827 !important;
            border-color: var(--panel-border) !important;
        }

        .app-title,
        .app-eyebrow,
        .sidebar-brand-title,
        .sidebar-card-value,
        .data-story-value,
        .readiness-title,
        .presentation-heading,
        .agent-card-title,
        .glossary-title,
        .conversation-empty-title,
        .empty-title,
        .ai-response-card-title,
        h1, h2, h3, h4, h5, h6 {
            color: var(--ink) !important;
        }

        .app-brandline .app-eyebrow {
            color: #e7eef8 !important;
            opacity: 1 !important;
            text-shadow: 0 1px 12px rgba(15, 23, 42, 0.42);
        }

        .app-subtitle,
        .sidebar-brand-subtitle,
        .sidebar-card-meta,
        .sidebar-card-title,
        .section-kicker,
        .conversation-empty-body,
        .ai-response-card-body,
        .agent-card-body,
        .glossary-body,
        .empty-lead,
        .muted,
        [data-testid="stCaptionContainer"] *,
        .stCaptionContainer,
        .stCaptionContainer *,
        small {
            color: #a9bdd5 !important;
        }

        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] label *,
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] label *,
        [data-testid="stSlider"] *,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label *,
        [data-baseweb="radio"] *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"] p {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        .meta-pill,
        .filter-chip,
        .readiness-pill,
        .status-pill,
        .workflow-step,
        .empty-steps span {
            background: #0d2a48 !important;
            border-color: #28577a !important;
            color: #d9ecff !important;
        }

        .readiness-pill-ready {
            background: rgba(34, 197, 94, 0.12) !important;
            border-color: rgba(34, 197, 94, 0.36) !important;
            color: #a7f3d0 !important;
        }

        .readiness-pill-warn {
            background: rgba(251, 191, 36, 0.12) !important;
            border-color: rgba(251, 191, 36, 0.36) !important;
            color: #fde68a !important;
        }

        [data-testid="stFileUploader"] section {
            background: #0f172a !important;
            border: 1px dashed #2b405d !important;
            border-radius: 8px !important;
            color: var(--ink) !important;
        }

        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button:disabled,
        [data-testid="stFileUploader"] button[disabled],
        [data-testid="stFileUploader"] button[aria-disabled="true"] {
            background: #111827 !important;
            border: 1px solid #2b405d !important;
            color: #e7eef8 !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] button svg,
        [data-testid="stFileUploader"] button:disabled *,
        [data-testid="stFileUploader"] button:disabled svg,
        [data-testid="stFileUploader"] button[disabled] *,
        [data-testid="stFileUploader"] button[disabled] svg,
        [data-testid="stFileUploader"] button[aria-disabled="true"] *,
        [data-testid="stFileUploader"] button[aria-disabled="true"] svg {
            color: #e7eef8 !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        .st-key-theme_toggle_button button,
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
        [data-testid="stSidebar"] .stDownloadButton button,
        .st-key-top_workspace_nav .stButton > button {
            background: #111827 !important;
            border-color: var(--panel-border) !important;
            color: var(--ink) !important;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        .st-key-theme_toggle_button button *,
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button svg,
        [data-testid="stSidebar"] .stDownloadButton button *,
        .st-key-top_workspace_nav .stButton > button * {
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .st-key-theme_toggle_button button:hover,
        [data-testid="stSidebar"] .stButton > button:hover,
        .st-key-top_workspace_nav .stButton > button:hover {
            background: #17233a !important;
            border-color: #3b82f6 !important;
            box-shadow: 0 10px 22px rgba(56, 189, 248, 0.12);
        }

        .stButton > button[kind="primary"],
        .st-key-theme_toggle_button button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        .st-key-top_workspace_nav .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb, #0891b2) !important;
            border-color: transparent !important;
            color: #ffffff !important;
        }

        input,
        textarea,
        [contenteditable="true"],
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] *,
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] [contenteditable="true"],
        [data-baseweb="select"] > div,
        [data-testid="stSelectbox"] div,
        [data-testid="stMultiSelect"] div {
            background-color: #0f172a !important;
            color: var(--ink) !important;
            border-color: var(--panel-border) !important;
        }

        [data-testid="stChatInput"] {
            background: #0f172a !important;
            background-color: #0f172a !important;
            background-image: none !important;
            border: 1px solid var(--panel-border) !important;
            border-radius: 10px !important;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.34) !important;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form {
            background: #0f172a !important;
            background-color: #0f172a !important;
            background-image: none !important;
            border-color: var(--panel-border) !important;
        }

        [data-testid="stChatInput"] button {
            width: 2.25rem !important;
            height: 2.25rem !important;
            min-width: 2.25rem !important;
            min-height: 2.25rem !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(56, 189, 248, 0.18) !important;
        }

        [data-testid="stChatInput"] button *,
        [data-testid="stChatInput"] button svg,
        [data-testid="stChatInput"] button svg * {
            background: transparent !important;
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button svg {
            width: 1.05rem !important;
            height: 1.05rem !important;
        }

        [data-testid="stChatInput"] button:disabled,
        [data-testid="stChatInput"] button[disabled],
        [data-testid="stChatInput"] button[aria-disabled="true"] {
            background: #1e293b !important;
            color: #cbd5e1 !important;
            border-color: var(--panel-border) !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button:disabled *,
        [data-testid="stChatInput"] button[disabled] *,
        [data-testid="stChatInput"] button[aria-disabled="true"] *,
        [data-testid="stChatInput"] button:disabled svg,
        [data-testid="stChatInput"] button[disabled] svg,
        [data-testid="stChatInput"] button[aria-disabled="true"] svg {
            color: #cbd5e1 !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stCode"],
        [data-testid="stCodeBlock"],
        [data-testid="stMarkdownContainer"] pre {
            background: #08111f !important;
            border-color: var(--code-border) !important;
        }

        [data-testid="stCode"] button,
        [data-testid="stCodeBlock"] button,
        [data-testid="stMarkdownContainer"] pre button,
        button[title*="Copy"],
        button[aria-label*="Copy"] {
            background: #1e293b !important;
            border: 1px solid var(--panel-border) !important;
            color: #e7eef8 !important;
            opacity: 1 !important;
        }

        [data-testid="stCode"] button *,
        [data-testid="stCodeBlock"] button *,
        [data-testid="stMarkdownContainer"] pre button *,
        [data-testid="stCode"] button svg,
        [data-testid="stCodeBlock"] button svg,
        [data-testid="stMarkdownContainer"] pre button svg,
        button[title*="Copy"] *,
        button[aria-label*="Copy"] * {
            color: #e7eef8 !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stMetric"],
        [data-testid="stMetric"] div {
            color: var(--ink) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            border-bottom-color: var(--panel-border) !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"],
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {
            color: var(--accent) !important;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
        }

        label,
        p,
        span,
        div {
            color: inherit;
        }
    """


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
    page_title=APP_NAME,
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    """Apply a stable, professional visual shell regardless of browser theme."""
    st.markdown(
        """
        <style>
        :root {
        """ + theme_css_variables() + """
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
            visibility: visible !important;
            pointer-events: auto !important;
        }

        [data-testid="stSidebarCollapseButton"] {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            position: fixed !important;
            top: 0.9rem !important;
            left: 0.9rem !important;
            z-index: 999999 !important;
            align-items: center;
            gap: 0.45rem;
            min-height: 2.45rem;
            padding: 0.2rem 0.75rem !important;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            box-shadow: 0 12px 26px rgba(16, 32, 51, 0.22);
        }

        [data-testid="collapsedControl"]::after,
        [data-testid="stSidebarCollapsedControl"]::after,
        [data-testid="stExpandSidebarButton"]::after {
            content: "Show sidebar";
            color: #ffffff;
            font-size: 0.82rem;
            font-weight: 720;
            white-space: nowrap;
        }

        [data-testid="collapsedControl"] *,
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stSidebarCollapseButton"] svg {
            color: #ffffff !important;
            fill: currentColor !important;
            stroke: currentColor !important;
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
                linear-gradient(180deg, #ffffff 0%, #f3f8ff 54%, #f8fbff 100%) !important;
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
            padding: 0 0 0.95rem 0;
            border-bottom: 1px solid var(--sidebar-line);
            margin-bottom: 0.85rem;
        }

        .sidebar-brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.72rem;
        }

        .brand-mark,
        .sidebar-brand-logo {
            flex: 0 0 auto;
            width: 48px;
            height: 48px;
            border: 1px solid #d5e8f6;
            border-radius: 12px;
            background: linear-gradient(135deg, #dff4ff 0%, #f8fbff 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            box-shadow: 0 9px 22px rgba(29, 78, 216, 0.16);
        }

        .brand-mark img,
        .sidebar-brand-logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
        }

        .sidebar-brand-title {
            color: #0b2545 !important;
            font-size: 1.12rem;
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
            color: var(--eyebrow) !important;
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

        .app-brandline {
            display: inline-flex;
            align-items: center;
            gap: 0.62rem;
        }

        .app-brandline .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 11px;
        }

        .topbar-command-panel {
            background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%);
            border: 1px solid #d7e7f7;
            border-radius: 8px;
            padding: 0.72rem;
            box-shadow: var(--shadow-sm);
        }

        .topbar-command-title {
            color: #334155 !important;
            font-size: 0.72rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }

        .topbar-command-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.38rem;
        }

        .topbar-action-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 0.45rem;
            margin-top: 0.5rem;
        }

        .st-key-theme_toggle_button button {
            min-height: 2.25rem;
            justify-content: center;
            border-radius: 7px !important;
            border: 1px solid var(--panel-border) !important;
            background: var(--panel) !important;
            color: var(--ink) !important;
            font-weight: 720;
            box-shadow: var(--shadow-sm);
        }

        .st-key-theme_toggle_button button:hover {
            border-color: var(--accent) !important;
            background: var(--accent-soft) !important;
            transform: translateY(-1px);
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

        .app-logo-card {
            background: linear-gradient(135deg, #dff4ff 0%, #f8fbff 54%, #ffffff 100%);
        }

        .app-logo-card img {
            object-fit: contain;
            object-position: center;
        }

        .st-key-top_workspace_nav {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.45rem;
            margin: 0.4rem 0 0.85rem 0;
            box-shadow: var(--shadow-sm);
        }

        .st-key-top_workspace_nav .stButton > button {
            min-height: 2.15rem;
            border-radius: 7px !important;
            border: 1px solid #d9e4f2 !important;
            background: #ffffff !important;
            color: #1e293b !important;
            justify-content: center;
            font-size: 0.86rem;
            font-weight: 680;
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        .st-key-top_workspace_nav .stButton > button p,
        .st-key-theme_toggle_button button p,
        [data-testid="stSidebar"] .stButton > button p,
        .stDownloadButton > button p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            line-height: 1.1 !important;
        }

        .st-key-top_workspace_nav .stButton > button:hover {
            border-color: #9ec8f7 !important;
            background: #f3f8ff !important;
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.10);
        }

        .st-key-top_workspace_nav .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            color: #ffffff !important;
            border-color: transparent !important;
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
            padding: 1.35rem 1.45rem;
            max-width: 760px;
            margin: 1.1rem auto 0;
            box-shadow: var(--shadow-sm);
        }

        .empty-title {
            color: var(--ink) !important;
            font-size: 1.65rem;
            font-weight: 780;
            line-height: 1.2;
            margin-top: 0.25rem;
        }

        .empty-lead {
            color: var(--muted) !important;
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 620px;
            margin-top: 0.45rem;
        }

        .empty-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.95rem;
        }

        .empty-steps span {
            display: inline-flex;
            align-items: center;
            border: 1px solid #dbe7f3;
            background: #f8fafc;
            border-radius: 999px;
            color: #475569 !important;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.28rem 0.6rem;
        }

        .st-key-empty_actions {
            max-width: 760px;
            margin: 0.75rem auto 0;
        }

        .st-key-empty_actions .stButton > button {
            min-height: 2.45rem;
            border-radius: 8px;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
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

        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel) !important;
            border: 1px solid var(--panel-border) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: transparent !important;
        }

        [data-testid="stChatMessageAvatarCustom"],
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        img[alt="assistant avatar"] {
            width: 2.35rem !important;
            height: 2.35rem !important;
            border-radius: 10px !important;
            border: 1px solid var(--panel-border) !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stChatMessageAvatarCustom"] {
            background: linear-gradient(135deg, #312e81 0%, #7c3aed 100%) !important;
            color: #f5f3ff !important;
        }

        [data-testid="stChatMessageAvatarCustom"] *,
        [data-testid="stChatMessageAvatarCustom"] svg {
            color: #f5f3ff !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }

        img[alt="assistant avatar"] {
            background: linear-gradient(135deg, #082f49 0%, #0f172a 100%) !important;
            object-fit: cover !important;
            object-position: center !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: var(--ink) !important;
            font-size: 0.95rem;
            line-height: 1.66;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
            color: var(--ink) !important;
            margin: 0.85rem 0 0.35rem 0 !important;
            line-height: 1.28 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1:first-child,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2:first-child,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3:first-child,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4:first-child,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:first-child {
            margin-top: 0 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            margin: 0.45rem 0 0.78rem 0;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
            margin: 0.35rem 0 0.9rem 0;
            padding-left: 1.35rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            margin: 0.28rem 0;
            padding-left: 0.16rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li::marker {
            color: var(--accent) !important;
            font-weight: 760;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
            color: var(--ink) !important;
            font-weight: 780;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a[href^="#"] {
            display: none !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
            margin: 0.75rem 0;
            padding: 0.65rem 0.8rem;
            border-left: 3px solid var(--accent);
            border-radius: 0 8px 8px 0;
            background: var(--quote-bg);
            color: var(--muted) !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
            background: var(--code-inline-bg) !important;
            border: 1px solid var(--code-border) !important;
            border-radius: 6px !important;
            color: var(--code-ink) !important;
            font-size: 0.86em !important;
            font-weight: 650;
            padding: 0.12rem 0.32rem !important;
            white-space: break-spaces;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
            position: relative;
            overflow-x: auto;
            margin: 0.85rem 0 1rem 0;
            padding: 0.9rem 1rem !important;
            border: 1px solid var(--code-border) !important;
            border-left: 4px solid var(--accent) !important;
            border-radius: 10px !important;
            background: var(--code-bg) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), var(--shadow-sm);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
            display: block;
            width: max-content;
            min-width: 100%;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            color: var(--code-ink) !important;
            font-size: 0.88rem !important;
            font-weight: 600;
            line-height: 1.55;
            padding: 0 !important;
            white-space: pre;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code::selection,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code::selection {
            background: rgba(56, 189, 248, 0.32);
            color: #ffffff;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel);
            margin: 0.75rem 0 1rem 0;
            font-size: 0.9rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
            border-bottom: 1px solid var(--panel-border);
            padding: 0.5rem 0.6rem;
            text-align: left;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
            background: var(--panel-soft);
            color: var(--ink) !important;
            font-weight: 760;
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

        .st-key-analysis_context_bar [data-testid="stVerticalBlockBorderWrapper"] {
            background:
                linear-gradient(135deg, rgba(56, 189, 248, 0.08), transparent 46%),
                var(--panel) !important;
            border-color: var(--panel-border) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-sm);
        }

        .st-key-analysis_context_bar [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        .st-key-analysis_context_bar [data-testid="column"] {
            min-width: 8.5rem;
        }

        .st-key-analysis_context_bar [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-soft) !important;
            border-color: var(--panel-border) !important;
            border-radius: 8px !important;
        }

        .st-key-analysis_context_bar [data-testid="stCaptionContainer"],
        .st-key-analysis_context_bar [data-testid="stCaptionContainer"] * {
            color: var(--muted) !important;
            font-size: 0.72rem !important;
            font-weight: 720 !important;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .st-key-analysis_context_bar [data-testid="stMarkdownContainer"] p {
            color: var(--ink) !important;
            margin-bottom: 0 !important;
        }

        .analysis-context-grid,
        .guardrail-grid,
        .quality-score-grid,
        .executive-summary-grid,
        .ai-insight-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.55rem 0 0.85rem 0;
        }

        .context-card,
        .guardrail-card,
        .quality-score-card,
        .executive-summary-card,
        .ai-insight-card,
        .table-shell,
        .chart-control-panel,
        .chart-preview-panel,
        .loading-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
        }

        .context-card,
        .guardrail-card,
        .quality-score-card,
        .executive-summary-card,
        .ai-insight-card {
            padding: 0.72rem 0.78rem;
        }

        .context-label,
        .guardrail-label,
        .quality-score-label,
        .executive-summary-label,
        .ai-insight-label,
        .table-label,
        .chart-builder-label {
            color: var(--muted) !important;
            font-size: 0.68rem;
            font-weight: 760;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .context-value,
        .quality-score-value,
        .executive-summary-value,
        .ai-insight-title {
            color: var(--ink) !important;
            font-size: 0.94rem;
            font-weight: 760;
            line-height: 1.28;
        }

        .guardrail-value,
        .ai-insight-body,
        .executive-summary-body {
            color: var(--muted) !important;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .guardrail-card {
            border-left: 3px solid var(--accent-2);
        }

        .ai-insight-card {
            border-top: 3px solid var(--accent);
        }

        .ai-insight-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            margin-top: 0.58rem;
        }

        .executive-dashboard {
            margin: 0.85rem 0 1.2rem 0;
        }

        .executive-kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.7rem;
            margin-bottom: 0.9rem;
        }

        .executive-kpi-card,
        .executive-panel,
        .ai-finding-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
        }

        .executive-kpi-card {
            min-height: 5rem;
            padding: 0.86rem 0.9rem;
            border-top: 3px solid var(--accent);
        }

        .executive-kpi-label,
        .executive-section-eyebrow,
        .ai-finding-label,
        .ai-finding-meta {
            color: var(--muted) !important;
            font-size: 0.68rem;
            font-weight: 780;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }

        .executive-kpi-value {
            color: var(--ink) !important;
            font-size: 1.42rem;
            font-weight: 780;
            line-height: 1.12;
            margin-top: 0.5rem;
        }

        .executive-story-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(17rem, 0.9fr);
            gap: 0.9rem;
        }

        .executive-panel {
            padding: 1rem;
        }

        .executive-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.78rem;
        }

        .executive-panel-title,
        .ai-findings-title {
            color: var(--ink) !important;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .executive-panel-count {
            color: var(--accent) !important;
            background: rgba(0, 148, 255, 0.13);
            border: 1px solid rgba(0, 148, 255, 0.28);
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 760;
            white-space: nowrap;
        }

        .executive-list {
            display: grid;
            gap: 0.62rem;
        }

        .executive-list-row {
            display: grid;
            grid-template-columns: 2rem minmax(0, 1fr);
            gap: 0.68rem;
            align-items: flex-start;
            color: var(--ink) !important;
            background: var(--panel-soft);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.74rem 0.78rem;
            line-height: 1.45;
        }

        .executive-list-index {
            width: 1.55rem;
            height: 1.55rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            color: #ffffff !important;
            background: var(--accent);
            font-size: 0.74rem;
            font-weight: 820;
        }

        .executive-risk .executive-list-index {
            background: var(--warning);
        }

        .executive-action-card {
            margin-top: 0.75rem;
            background: rgba(0, 148, 255, 0.12);
            border: 1px solid rgba(0, 148, 255, 0.28);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            padding: 0.82rem 0.9rem;
        }

        .executive-action-card .executive-panel-title {
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }

        .executive-action-card p,
        .executive-list-row p {
            margin: 0 !important;
        }

        .ai-findings-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.4rem 0 0.7rem 0;
        }

        .ai-findings-subtitle {
            color: var(--muted) !important;
            font-size: 0.84rem;
            margin-top: 0.18rem;
        }

        .ai-findings-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 1.1rem;
        }

        .ai-finding-card {
            padding: 0.95rem;
            border-top: 3px solid var(--accent-2);
        }

        .ai-finding-title {
            color: var(--ink) !important;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.35;
            margin: 0.4rem 0 0.6rem 0;
        }

        .ai-finding-evidence {
            color: var(--muted) !important;
            font-size: 0.86rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .ai-finding-footer {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 0.65rem;
            align-items: flex-start;
            border-top: 1px solid var(--panel-border);
            padding-top: 0.72rem;
        }

        .ai-finding-action {
            color: var(--ink) !important;
            font-size: 0.84rem;
            line-height: 1.42;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--panel-border);
            color: var(--ink) !important;
            background: var(--panel-soft);
            padding: 0.16rem 0.42rem;
            font-size: 0.7rem;
            font-weight: 720;
        }

        .table-shell {
            padding: 0.82rem;
            margin: 0.5rem 0 1rem 0;
        }

        .table-header-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-end;
            margin-bottom: 0.65rem;
        }

        .table-title {
            color: var(--ink) !important;
            font-size: 1.04rem;
            font-weight: 780;
        }

        .table-count {
            color: var(--muted) !important;
            font-size: 0.82rem;
        }

        .chart-builder-grid {
            display: grid;
            grid-template-columns: minmax(18rem, 0.85fr) minmax(0, 1.85fr);
            gap: 1rem;
            align-items: start;
            margin-top: 0.65rem;
        }

        .chart-control-panel,
        .chart-preview-panel {
            padding: 0.85rem;
        }

        .chart-preview-panel {
            min-width: 0;
        }

        .loading-card {
            position: relative;
            overflow: hidden;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 0.75rem 0;
        }

        .loading-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, var(--accent), var(--accent-2), var(--accent-warm));
        }

        .loading-title {
            color: var(--ink) !important;
            font-weight: 780;
            margin-bottom: 0.45rem;
        }

        .loading-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .loading-step {
            color: var(--ink) !important;
            background: var(--panel-soft);
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .query-result-caption {
            color: var(--muted) !important;
            font-size: 0.8rem;
            margin-bottom: 0.4rem;
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

        @media (max-width: 1180px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                max-width: 100% !important;
            }

            .st-key-top_workspace_nav [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.55rem !important;
            }

            .st-key-top_workspace_nav [data-testid="column"] {
                flex: 1 1 8.8rem !important;
                min-width: 8.8rem !important;
                width: auto !important;
            }

            .st-key-top_workspace_nav .stButton > button {
                min-height: 2.35rem;
            }
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }

            .empty-grid,
            .agent-grid,
            .glossary-grid,
            .analysis-context-grid,
            .guardrail-grid,
            .quality-score-grid,
            .executive-summary-grid,
            .ai-insight-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .st-key-app_topbar [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.7rem !important;
            }

            .st-key-app_topbar [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }

            .st-key-app_topbar .stButton > button,
            .st-key-app_topbar .stDownloadButton > button {
                min-height: 2.55rem;
            }

            .app-title {
                font-size: clamp(1.28rem, 4vw, 1.45rem);
            }

            .app-subtitle {
                max-width: 100%;
            }

            .data-story-card {
                min-width: min(100%, 13rem);
            }
        }

        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] {
                padding: 0.85rem 0.75rem 6.5rem !important;
            }

            [data-testid="stSidebar"] {
                width: min(88vw, 20rem) !important;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.65rem !important;
            }

            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }

            .st-key-top_workspace_nav [data-testid="column"] {
                flex: 1 1 calc(50% - 0.35rem) !important;
                min-width: calc(50% - 0.35rem) !important;
                width: calc(50% - 0.35rem) !important;
            }

            .st-key-app_topbar {
                padding: 0.8rem !important;
            }

            .app-brandline {
                align-items: flex-start;
                gap: 0.55rem;
            }

            .app-brandline .brand-mark {
                width: 36px;
                height: 36px;
                border-radius: 9px;
            }

            .app-meta-row,
            .filter-chip-row,
            .workflow-rail,
            .empty-steps {
                gap: 0.38rem;
            }

            .meta-pill,
            .filter-chip,
            .readiness-pill,
            .status-pill {
                max-width: 100%;
                white-space: normal;
                line-height: 1.25;
            }

            [data-testid="stMetric"] {
                padding: 0.72rem 0.78rem;
            }

            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
                font-size: 1.42rem !important;
            }

            .stButton > button,
            .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] button {
                width: 100% !important;
                min-height: 2.7rem;
                white-space: normal !important;
            }

            .stButton > button p,
            .stDownloadButton > button p,
            [data-testid="stFormSubmitButton"] button p {
                white-space: normal !important;
                overflow-wrap: anywhere;
                line-height: 1.18;
            }

            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto;
                flex-wrap: nowrap;
                scrollbar-width: thin;
            }

            .stTabs [data-baseweb="tab"] {
                min-width: max-content;
                padding: 0.45rem 0.7rem;
            }

            div[data-testid="stDataFrame"],
            [data-testid="stPlotlyChart"] {
                max-width: 100%;
                overflow-x: auto;
            }

            [data-testid="stPlotlyChart"] {
                padding: 0.25rem;
            }

            .glossary-grid,
            .agent-grid,
            .empty-grid,
            .executive-kpi-grid,
            .executive-story-grid,
            .ai-findings-grid,
            .analysis-context-grid,
            .guardrail-grid,
            .quality-score-grid,
            .executive-summary-grid,
            .ai-insight-grid,
            .chart-builder-grid {
                grid-template-columns: 1fr;
                gap: 0.65rem;
            }

            .insight-card,
            .conversation-empty,
            .presentation-band,
            .approval-panel,
            .ai-response-card,
            .context-card,
            .guardrail-card,
            .quality-score-card,
            .executive-summary-card,
            .ai-insight-card,
            .table-shell,
            .chart-control-panel,
            .chart-preview-panel {
                padding: 0.78rem 0.82rem;
            }

            [data-testid="stChatMessage"] {
                gap: 0.45rem;
            }

            [data-testid="stBottom"],
            [data-testid="stBottomBlockContainer"],
            [data-testid="stChatFloatingInputContainer"] {
                padding-left: 0.65rem !important;
                padding-right: 0.65rem !important;
            }

            [data-testid="stChatInput"] {
                width: 100% !important;
                min-height: 3rem;
            }
        }

        @media (max-width: 480px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 0.6rem !important;
                padding-right: 0.6rem !important;
            }

            .st-key-top_workspace_nav [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }

            .app-brandline {
                display: grid;
                grid-template-columns: 34px minmax(0, 1fr);
            }

            .app-brandline .brand-mark {
                width: 34px;
                height: 34px;
            }

            .app-title {
                font-size: 1.22rem;
                overflow-wrap: anywhere;
            }

            .app-subtitle,
            .conversation-empty-body,
            .glossary-body,
            .agent-card-body {
                font-size: 0.84rem;
                line-height: 1.4;
            }

            .meta-pill,
            .filter-chip,
            .readiness-pill,
            .status-pill {
                font-size: 0.72rem;
                padding: 0.22rem 0.48rem;
            }

            .sidebar-card,
            .glossary-card,
            .agent-card,
            .history-row,
            .data-story-card {
                padding: 0.72rem 0.78rem;
            }

            [data-testid="stMetric"] {
                min-height: auto;
            }

            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
                font-size: 1.3rem !important;
            }
        }
        """ + theme_override_css() + """

        /* Native Streamlit control repairs: keep copy/send icons crisp in both themes. */
        [data-testid="stCode"] button,
        [data-testid="stCodeBlock"] button,
        [data-testid="stMarkdownContainer"] pre button,
        button[title*="Copy"],
        button[aria-label*="Copy"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border-radius: 8px !important;
            background: var(--panel-soft) !important;
            background-color: var(--panel-soft) !important;
            border: 1px solid var(--code-border) !important;
            color: var(--code-ink) !important;
            box-shadow: var(--shadow-sm) !important;
            opacity: 1 !important;
        }

        [data-testid="stCode"] button *,
        [data-testid="stCodeBlock"] button *,
        [data-testid="stMarkdownContainer"] pre button *,
        button[title*="Copy"] *,
        button[aria-label*="Copy"] *,
        [data-testid="stCode"] button svg,
        [data-testid="stCodeBlock"] button svg,
        [data-testid="stMarkdownContainer"] pre button svg,
        button[title*="Copy"] svg,
        button[aria-label*="Copy"] svg {
            background: transparent !important;
            background-color: transparent !important;
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stCode"] button svg,
        [data-testid="stCodeBlock"] button svg,
        [data-testid="stMarkdownContainer"] pre button svg,
        button[title*="Copy"] svg,
        button[aria-label*="Copy"] svg {
            width: 1.05rem !important;
            height: 1.05rem !important;
        }

        [data-testid="stChatInput"] button,
        [data-testid="stChatInputSubmitButton"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2.25rem !important;
            height: 2.25rem !important;
            min-width: 2.25rem !important;
            min-height: 2.25rem !important;
            padding: 0 !important;
            border-radius: 9px !important;
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            background-color: var(--accent) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(0, 148, 255, 0.18) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button *,
        [data-testid="stChatInput"] button svg,
        [data-testid="stChatInput"] button svg *,
        [data-testid="stChatInputSubmitButton"] *,
        [data-testid="stChatInputSubmitButton"] svg,
        [data-testid="stChatInputSubmitButton"] svg * {
            background: transparent !important;
            background-color: transparent !important;
            color: inherit !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button svg,
        [data-testid="stChatInputSubmitButton"] svg {
            width: 1.08rem !important;
            height: 1.08rem !important;
        }

        [data-testid="stChatInput"] button:disabled,
        [data-testid="stChatInput"] button[disabled],
        [data-testid="stChatInput"] button[aria-disabled="true"],
        [data-testid="stChatInputSubmitButton"]:disabled,
        [data-testid="stChatInputSubmitButton"][disabled],
        [data-testid="stChatInputSubmitButton"][aria-disabled="true"] {
            background: var(--panel-soft) !important;
            background-color: var(--panel-soft) !important;
            border-color: var(--panel-border) !important;
            color: var(--muted) !important;
            box-shadow: none !important;
            opacity: 1 !important;
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
        "pending_ai_prompt": "",
        "pending_ai_context": "",
        "analysis_history": [],
        "saved_ai_insights": [],
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
        "theme_mode": "Dark",
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
        st.session_state["pending_ai_prompt"] = ""
        st.session_state["pending_ai_context"] = ""
        st.session_state["analysis_history"] = []
        st.session_state["saved_ai_insights"] = []
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


def enable_sample_dataset() -> None:
    """Enable the sample dataset before sidebar widgets are instantiated."""
    st.session_state["use_sample_dataset"] = True


def clear_ai_chat() -> None:
    """Clear chat and generated voice output."""
    st.session_state["ai_messages"] = []
    st.session_state["last_voice_audio"] = None
    st.session_state["last_transcript"] = ""
    st.session_state["conversation_draft"] = ""
    st.session_state["pending_ai_prompt"] = ""
    st.session_state["pending_ai_context"] = ""


def reset_workspace_state() -> None:
    """Reset UI choices without unloading the dataset."""
    for key, value in {
        "ai_messages": [],
        "generated_code": "",
        "conversation_draft": "",
        "code_request": "",
        "pending_ai_prompt": "",
        "pending_ai_context": "",
        "analysis_history": [],
        "saved_ai_insights": [],
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


def apply_plotly_theme(fig):
    """Align Plotly figures with the active Streamlit theme."""
    if active_theme_mode() != "Dark":
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#102033"},
            title_font={"color": "#102033"},
        )
        fig.update_xaxes(gridcolor="#e5edf6", linecolor="#cbd5e1", tickfont={"color": "#425875"})
        fig.update_yaxes(gridcolor="#e5edf6", linecolor="#cbd5e1", tickfont={"color": "#425875"})
        return fig

    ink = "#e7eef8"
    muted = "#a9bdd5"
    grid = "#26374f"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font={"color": ink},
        title_font={"color": ink, "size": 16},
        legend={"bgcolor": "rgba(17, 24, 39, 0)", "font": {"color": ink}},
        margin={"l": 42, "r": 24, "t": 58, "b": 42},
    )
    fig.update_xaxes(
        gridcolor=grid,
        linecolor="#334155",
        zerolinecolor="#334155",
        tickfont={"color": muted},
        title_font={"color": ink},
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=grid,
        linecolor="#334155",
        zerolinecolor="#334155",
        tickfont={"color": muted},
        title_font={"color": ink},
        automargin=True,
    )
    fig.update_coloraxes(colorbar={"tickfont": {"color": muted}, "title": {"font": {"color": ink}}})
    fig.update_traces(textfont={"color": ink}, selector={"type": "heatmap"})
    return fig


def render_plotly_chart(fig, *, key: str, width: str = "stretch") -> None:
    """Render a Plotly chart with app theme-aware styling."""
    st.plotly_chart(apply_plotly_theme(fig), width=width, key=key)


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
    logo_uri = asset_data_uri(BRAND_MARK, "image/png")
    logo_html = f'<img src="{logo_uri}" alt="{APP_NAME} logo" />' if logo_uri else ""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-lockup">
                <div class="sidebar-brand-logo">{logo_html}</div>
                <div>
                    <div class="sidebar-brand-title">{APP_NAME}</div>
                    <div class="sidebar-brand-subtitle">{APP_TAGLINE}</div>
                </div>
            </div>
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
            file_name="insightanalytica_view.csv",
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


def compact_value(value: object, *, max_chars: int = 48) -> str:
    """Return a compact display value for UI cards."""
    text = str(value or "")
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}..."


def render_current_analysis_context(
    df: pd.DataFrame,
    source_name: str | None,
    active_filters: list[str],
    *,
    compact: bool = False,
) -> None:
    """Show the exact context used by AI analysis."""
    filter_value = f"{len(active_filters)} active" if active_filters else "None"
    cards = [
        ("Dataset", compact_value(source_name or "Active dataset")),
        ("Rows", f"{len(df):,} active"),
        ("Filters", filter_value),
        ("Model", selected_ai_model()),
        ("SQL execution", "Enabled" if should_execute_query("maximum value") else "Enabled"),
    ]
    if compact:
        cards = cards[:4]
    with st.container(border=True, key="analysis_context_bar"):
        st.markdown("**Current Analysis Context**")
        st.caption("This is the exact dataset view used for AI answers and SQL execution.")
        cols = st.columns(len(cards))
        for idx, (label, value) in enumerate(cards):
            with cols[idx]:
                with st.container(border=True):
                    st.caption(label)
                    st.markdown(f"**{value}**")


def render_table_workspace(
    title: str,
    df: pd.DataFrame,
    key_prefix: str,
    *,
    default_rows: int = 50,
    download_name: str = "dataset_view.csv",
) -> pd.DataFrame:
    """Render a polished dataframe workspace with search, density, rows, and download."""
    st.markdown(
        f"""
        <div class="table-shell">
            <div class="table-header-row">
                <div>
                    <div class="table-label">Table Workspace</div>
                    <div class="table-title">{escape(title)}</div>
                </div>
                <div class="table-count">{len(df):,} available rows</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    controls = st.columns([2.0, 1.0, 1.0, 1.0], vertical_alignment="bottom")
    query = controls[0].text_input(
        "Search",
        placeholder="Search across all columns",
        key=f"{key_prefix}_search",
    )
    filtered = search_dataframe(df, query)
    max_rows = max(1, min(max(len(filtered), 1), 500))
    default_value = min(max(default_rows, 1), max_rows)
    if max_rows <= 1:
        rows_to_show = 1
        controls[1].caption("Rows to show")
        controls[1].markdown("1")
    else:
        rows_to_show = controls[1].slider(
            "Rows to show",
            min_value=1,
            max_value=max_rows,
            value=default_value,
            key=f"{key_prefix}_rows",
        )
    controls[2].radio(
        "Density",
        ["Comfortable", "Compact"],
        horizontal=True,
        key=f"{key_prefix}_density",
    )
    current_density = st.session_state.get(f"{key_prefix}_density", st.session_state.get("table_density", "Comfortable"))
    row_height = 28 if current_density == "Compact" else 36
    controls[3].download_button(
        "Download",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
        key=f"{key_prefix}_download",
    )
    shown = filtered.head(rows_to_show)
    st.caption(f"Showing {len(shown):,} of {len(filtered):,} matching rows")
    st.dataframe(shown, width="stretch", height=350, row_height=row_height)
    return filtered


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
    mark_uri = asset_data_uri(BRAND_MARK, "image/png")
    mark_html = f'<span class="brand-mark"><img src="{mark_uri}" alt="{APP_NAME} mark" /></span>' if mark_uri else ""
    is_dark = active_theme_mode() == "Dark"
    theme_label = "Light" if is_dark else "Dark"
    theme_icon = ":material/light_mode:" if is_dark else ":material/dark_mode:"
    with st.container(border=True, key="app_topbar"):
        left, right = st.columns([4.2, 1.8], vertical_alignment="center")
        with left:
            st.markdown(
                f"""
                <div class="app-brandline">
                    {mark_html}
                    <div>
                        <div class="app-eyebrow">{APP_NAME}</div>
                        <div class="app-title">{escape(page_title)}</div>
                    </div>
                </div>
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
            action_cols = st.columns(2)
            action_cols[0].button(
                theme_label,
                key="theme_toggle_button",
                icon=theme_icon,
                width="stretch",
                on_click=toggle_theme_mode,
            )
            action_cols[1].download_button(
                "Export",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="insightanalytica_export.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )


def render_top_workspace_nav(active_navigation: str) -> None:
    """Render a clickable workspace switcher near the app header."""
    with st.container(key="top_workspace_nav"):
        st.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)
        cols = st.columns(len(NAV_ITEMS))
        for idx, item in enumerate(NAV_ITEMS):
            label = item["label"]
            cols[idx].button(
                item.get("short", label),
                key=f"top_nav_{label}",
                icon=item["icon"],
                type="primary" if active_navigation == label else "secondary",
                width="stretch",
                on_click=set_navigation,
                args=(label,),
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
    render_top_workspace_nav(st.session_state.get("navigation", "Overview"))
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


def canonical_ai_section_title(line: str) -> str | None:
    """Return a known AI response section title from a markdown-ish heading."""
    raw = line.strip()
    if not raw:
        return None

    heading_match = re.match(r"^#{1,6}\s+(?P<title>.+)$", raw)
    title = heading_match.group("title") if heading_match else raw
    title = re.sub(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)", "", title).strip()
    title = title.strip("*_` ").rstrip(":").strip()
    normalized = re.sub(r"\s+", " ", re.sub(r"^[^a-z0-9]+", "", title.lower()))

    exact_titles = {section.lower(): section for section in AI_RESPONSE_SECTIONS}
    aliases = {
        "next steps": "Recommended Next Steps",
        "recommendations": "Recommended Next Steps",
        "recommended actions": "Recommended Next Steps",
        "risks": "Caveats",
        "limitations": "Caveats",
        "query results": "Query Result",
        "result": "Query Result",
        "results": "Query Result",
        "data": "Query Result",
        "trust guardrails": "Guardrails",
        "guardrail": "Guardrails",
        "evidence": "Evidence",
        "summary": "Summary",
    }
    if normalized in exact_titles:
        return exact_titles[normalized]
    if normalized in aliases:
        return aliases[normalized]

    # Flexible matches are only treated as section breaks when the model used
    # an explicit heading marker, preventing ordinary body lines such as
    # "Summary statistics:" from being split into a new card.
    if heading_match or raw.startswith("**"):
        flexible_prefixes = [
            ("recommended next steps", "Recommended Next Steps"),
            ("next steps", "Recommended Next Steps"),
            ("recommendations", "Recommended Next Steps"),
            ("query result", "Query Result"),
            ("query results", "Query Result"),
            ("guardrails", "Guardrails"),
            ("evidence", "Evidence"),
            ("caveats", "Caveats"),
            ("limitations", "Caveats"),
            ("summary", "Summary"),
        ]
        for prefix, canonical in flexible_prefixes:
            if normalized == prefix or normalized.startswith(f"{prefix} "):
                return canonical
    return None


def parse_ai_sections(content: str) -> list[tuple[str, str]]:
    """Parse model responses into productized UI sections when possible."""
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for raw_line in content.splitlines():
        title = canonical_ai_section_title(raw_line)
        if title:
            if current_title:
                sections.append((current_title, current_body))
            current_title = title
            current_body = []
            continue
        if current_title:
            current_body.append(raw_line)

    if current_title:
        sections.append((current_title, current_body))

    parsed = [(title, "\n".join(body).strip()) for title, body in sections]
    return [(title, body) for title, body in parsed if body]


def render_guardrail_cards(body: str) -> None:
    """Render guardrail bullets as compact trust cards."""
    lines = [line.strip(" -*") for line in str(body or "").splitlines() if line.strip(" -*")]
    if not lines:
        st.markdown(body)
        return
    cards = []
    for line in lines[:6]:
        if ":" in line:
            label, value = line.split(":", 1)
        else:
            label, value = "Guardrail", line
        cards.append(
            f"""
            <div class="guardrail-card">
                <div class="guardrail-label">{escape(short_text(label, max_chars=44))}</div>
                <div class="guardrail-value">{escape(short_text(value, max_chars=120))}</div>
            </div>
            """
        )
    st.markdown(f'<div class="guardrail-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_next_steps_card(body: str) -> None:
    """Render recommendations as a full, non-collapsible Markdown card."""
    cleaned_body = str(body or "").strip()
    if not cleaned_body:
        return

    with st.container(border=True):
        st.markdown('<div class="ai-response-card-title">Recommended Next Steps</div>', unsafe_allow_html=True)
        st.caption("Prioritized follow-up actions from this analysis.")
        st.markdown(cleaned_body)


def render_query_result_table(result: CleanedQueryResult, key_prefix: str) -> None:
    """Render a computed query result as a real table with export."""
    st.markdown(
        f'<div class="query-result-caption">{result.row_count:,} row(s) returned from the cleaned query workspace.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        result.result_df,
        width="stretch",
        hide_index=True,
        row_height=table_row_height(),
    )
    st.download_button(
        "Download query result",
        data=result.result_df.to_csv(index=False).encode("utf-8"),
        file_name="query_result.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"{key_prefix}_query_result_download",
    )


def render_ai_response(
    content: str,
    *,
    result: CleanedQueryResult | None = None,
    key_prefix: str = "ai_response",
) -> None:
    """Render AI output as structured product cards."""
    sections = parse_ai_sections(content)
    if len(sections) < 2:
        with st.container(border=True):
            st.markdown(content)
        return

    for title, body in sections:
        if title == "Query Result" and result is not None:
            with st.container(border=True):
                st.markdown(f'<div class="ai-response-card-title">{escape(title)}</div>', unsafe_allow_html=True)
                render_query_result_table(result, key_prefix)
            continue
        if title == "Guardrails":
            with st.container(border=True):
                st.markdown(f'<div class="ai-response-card-title">{escape(title)}</div>', unsafe_allow_html=True)
                render_guardrail_cards(body)
            continue
        if title == "Recommended Next Steps":
            render_next_steps_card(body)
            continue
        if title == "Evidence":
            with st.container(border=True):
                st.markdown(f'<div class="ai-response-card-title">{escape(title)}</div>', unsafe_allow_html=True)
                st.markdown(body)
            continue

        is_long = len(body) > 1_200 or body.count("\n") > 18
        if is_long and title not in {"Summary", "Query Result"}:
            with st.expander(title, expanded=False):
                st.markdown(body)
        else:
            with st.container(border=True):
                st.markdown(f'<div class="ai-response-card-title">{escape(title)}</div>', unsafe_allow_html=True)
                st.markdown(body)


def section_body(content: str, wanted_title: str) -> str:
    """Return a parsed AI section body by title."""
    wanted = wanted_title.lower()
    for title, body in parse_ai_sections(content):
        if title.lower() == wanted:
            return body
    return ""


def first_nonempty_line(text: str, fallback: str) -> str:
    """Return the first useful non-empty line from a block of text."""
    for line in str(text or "").splitlines():
        stripped = line.strip(" -*#`")
        if stripped:
            return stripped
    return fallback


def short_text(text: str, *, max_chars: int = 150) -> str:
    """Shorten text for dashboard cards."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned if len(cleaned) <= max_chars else f"{cleaned[: max_chars - 1]}..."


def plain_text(text: object, *, max_chars: int | None = None) -> str:
    """Convert markdown-ish model text into clean card copy."""
    cleaned = str(text or "")
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"^[\s>*#-]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return short_text(cleaned, max_chars=max_chars) if max_chars else cleaned


def save_analysis_artifacts(
    question: str,
    response: str,
    result: CleanedQueryResult | None = None,
) -> None:
    """Save reusable AI insights and analysis history for workspace recall."""
    history = st.session_state.setdefault("analysis_history", [])
    history.insert(
        0,
        {
            "question": question,
            "summary": short_text(section_body(response, "Summary") or response, max_chars=220),
            "rows": result.row_count if result else None,
            "mode": result.mode if result else "AI narrative",
        },
    )
    del history[12:]

    summary = section_body(response, "Summary") or response
    evidence = section_body(response, "Evidence") or section_body(response, "Query Result")
    next_steps = section_body(response, "Recommended Next Steps")
    caveats = section_body(response, "Caveats")
    insights = st.session_state.setdefault("saved_ai_insights", [])
    insights.insert(
        0,
        {
            "title": plain_text(first_nonempty_line(summary, question), max_chars=110),
            "metric": f"{result.row_count:,} row(s)" if result else "Narrative insight",
            "evidence": plain_text(first_nonempty_line(evidence, "Based on the active dataset profile."), max_chars=220),
            "risk": "Medium" if caveats else "Low",
            "action": plain_text(first_nonempty_line(next_steps, "Review and validate before reporting."), max_chars=170),
        },
    )
    del insights[8:]


def render_saved_analysis_history() -> None:
    """Render saved AI analysis history."""
    history = st.session_state.get("analysis_history", [])
    if not history:
        return
    with st.expander("Saved analysis history", expanded=False):
        for idx, item in enumerate(history[:8], start=1):
            rows = item.get("rows")
            rows_label = f"{rows:,} rows returned" if isinstance(rows, int) else str(item.get("mode", "AI narrative"))
            st.markdown(
                f"""
                <div class="history-row">
                    <div class="data-story-label">Analysis {idx} - {escape(rows_label)}</div>
                    <div class="data-story-value">{escape(short_text(str(item.get("question", "Analysis")), max_chars=110))}</div>
                    <div class="muted">{escape(str(item.get("summary", "")))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        markdown = "\n\n".join(
            f"## {item.get('question', 'Analysis')}\n\n{item.get('summary', '')}" for item in history
        )
        st.download_button(
            "Download history",
            data=markdown.encode("utf-8"),
            file_name="analysis_history.md",
            mime="text/markdown",
            icon=":material/download:",
        )


def render_saved_ai_insight_cards() -> None:
    """Render reusable AI-generated insight cards."""
    insights = st.session_state.get("saved_ai_insights", [])
    if not insights:
        return
    visible_insights = insights[:4]
    cards = []
    for item in visible_insights:
        metric = plain_text(item.get("metric", "AI finding"), max_chars=48)
        title = plain_text(item.get("title", "Saved finding"), max_chars=120)
        evidence = plain_text(item.get("evidence", "Based on the active dataset."), max_chars=230)
        risk = plain_text(item.get("risk", "Low"), max_chars=24)
        action = plain_text(item.get("action", "Review before reporting."), max_chars=180)
        cards.append(
            f"""
            <article class="ai-finding-card">
                <div class="ai-finding-label">{escape(metric)}</div>
                <div class="ai-finding-title">{escape(title)}</div>
                <div class="ai-finding-evidence">{escape(evidence)}</div>
                <div class="ai-finding-footer">
                    <span class="risk-pill">Risk: {escape(risk)}</span>
                    <div class="ai-finding-action">{escape(action)}</div>
                </div>
            </article>
            """
        )
    st.markdown(
        f"""
        <div class="ai-findings-header">
            <div>
                <div class="ai-findings-title">Saved AI Findings</div>
                <div class="ai-findings-subtitle">Reusable conclusions from the latest Conversation AI analyses.</div>
            </div>
            <span class="executive-panel-count">{len(visible_insights)} shown</span>
        </div>
        <div class="ai-findings-grid">
            {''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_score_explanation(df: pd.DataFrame) -> None:
    """Explain the dataset quality score using visible ingredients."""
    health = dataset_health(df)
    checks = readiness_checks(df)
    anomalies = outlier_summary(df)
    anomaly_total = int(anomalies["outlier_count"].sum()) if not anomalies.empty else 0
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    type_issue_count = len(object_columns)
    date_count = len(detect_datetime_columns(df))
    ai_ready = sum(1 for item in checks if item["ready"])
    cards = [
        ("Health score", f'{int(health["score"])}/100', health_label(health["score"])),
        ("Missing values", f'{int(health["missing_cells"]):,}', f'{health["missing_percent"]}% of cells'),
        ("Duplicate rows", f'{int(health["duplicate_rows"]):,}', f'{health["duplicate_percent"]}% of rows'),
        ("Type review", f"{type_issue_count:,}", "object/text columns to validate"),
        ("Outliers", f"{anomaly_total:,}", "numeric flags using IQR"),
        ("Date readiness", f"{date_count:,}", "date-like fields detected"),
        ("AI readiness", f"{ai_ready}/{len(checks)}", "checks passed"),
    ]
    with st.expander("Health score explanation", expanded=False):
        for start in range(0, len(cards), 3):
            row_cards = cards[start : start + 3]
            cols = st.columns(len(row_cards))
            for idx, (label, value, detail) in enumerate(row_cards):
                with cols[idx]:
                    with st.container(border=True):
                        st.caption(label)
                        st.markdown(f"**{value}**")
                        st.caption(detail)


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
            render_plotly_chart(
                plot_histogram(df, selected_column),
                key=f"column_profile_histogram_{selected_column}",
            )
        else:
            render_plotly_chart(
                plot_bar(df, selected_column),
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
            render_plotly_chart(
                plot_histogram(df, best_metric),
                key=f"overview_focus_histogram_{best_metric}",
            )
    elif st.session_state.get("overview_focus") == "Review correlations":
        corr = correlations(df)
        if corr.empty:
            st.info("Correlation review needs at least two numeric columns.")
        else:
            st.info("Correlation review is active.")
            render_plotly_chart(plot_heatmap(df), key="overview_focus_correlation_heatmap")
    render_insight_cards(df, limit=3)
    render_saved_ai_insight_cards()

    render_table_workspace(
        "Dataset Preview",
        df,
        "overview_preview",
        default_rows=50,
        download_name="dataset_preview.csv",
    )

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
        render_plotly_chart(plot_heatmap(df), key="overview_correlation_heatmap")
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
                For numerical questions, it cleans a query workspace and returns computed result tables.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_result_value(value: object) -> str:
    """Format query output values for compact markdown tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    """Render a small DataFrame as a markdown table without optional dependencies."""
    if df.empty:
        return "_The query returned no rows._"

    display_df = df.head(max_rows)
    columns = [str(column) for column in display_df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display_df.iterrows():
        values = [_format_result_value(row[column]).replace("|", "\\|") for column in display_df.columns]
        rows.append("| " + " | ".join(values) + " |")
    table = "\n".join([header, separator, *rows])
    if len(df) > max_rows:
        table += f"\n\n_Showing first {max_rows:,} rows of {len(df):,} returned rows._"
    return table


def _string_group_values(series: pd.Series) -> pd.Series:
    """Return grouping labels with missing values made explicit."""
    return series.astype("object").where(series.notna(), "Missing")


def chart_ai_context(
    df: pd.DataFrame,
    chart_type: str,
    chart_description: str,
    params: dict[str, object],
    *,
    max_rows: int = 45,
) -> str:
    """Build compact chart data context for Conversation AI chart explanations."""
    lines = [
        "Chart context:",
        f"- Chart type: {chart_type}",
        f"- Chart description: {chart_description}",
        f"- Active dataset rows behind chart: {len(df):,}",
    ]
    chart_df = pd.DataFrame()

    try:
        if chart_type == "Histogram":
            column = str(params["column"])
            series = df[column].dropna()
            lines.append(f"- Column plotted: {column}")
            if pd.api.types.is_numeric_dtype(series):
                stats = series.describe(percentiles=[0.25, 0.5, 0.75]).reset_index()
                stats.columns = ["statistic", column]
                chart_df = stats
            else:
                chart_df = _string_group_values(df[column]).value_counts(dropna=False).head(25).rename_axis(column).reset_index(name="count")

        elif chart_type == "Bar chart":
            column = str(params["column"])
            lines.append(f"- Bar categories: {column}")
            chart_df = _string_group_values(df[column]).value_counts(dropna=False).head(30).rename_axis(column).reset_index(name="count")

        elif chart_type == "Aggregated bar":
            category = str(params["category"])
            value = str(params["value"])
            aggregation = str(params["aggregation"])
            lines.append(f"- Aggregation: {aggregation}({value}) by {category}")
            working = df[[category, value]].copy()
            working[category] = _string_group_values(working[category])
            if aggregation == "count":
                chart_df = working.groupby(category, dropna=False).size().reset_index(name="count")
                sort_col = "count"
            else:
                chart_df = working.groupby(category, dropna=False)[value].agg(aggregation).reset_index()
                sort_col = value
            chart_df = chart_df.sort_values(sort_col, ascending=False).head(30)

        elif chart_type == "Line chart":
            x = str(params["x"])
            y = str(params["y"])
            lines.append(f"- X axis: {x}")
            lines.append(f"- Y axis: {y}")
            working = df[[x, y]].dropna().copy()
            parsed_x = pd.to_datetime(working[x], errors="coerce")
            if not working.empty and parsed_x.notna().mean() >= 0.8:
                working[x] = parsed_x
            working = working.sort_values(x)
            numeric_y = pd.to_numeric(working[y], errors="coerce")
            if numeric_y.notna().any():
                high_idx = numeric_y.idxmax()
                low_idx = numeric_y.idxmin()
                lines.append(f"- Highest point: {working.loc[high_idx, x]} = {_format_result_value(working.loc[high_idx, y])}")
                lines.append(f"- Lowest point: {working.loc[low_idx, x]} = {_format_result_value(working.loc[low_idx, y])}")
                if len(working) >= 2:
                    first_y = numeric_y.dropna().iloc[0]
                    last_y = numeric_y.dropna().iloc[-1]
                    direction = "increased" if last_y > first_y else "decreased" if last_y < first_y else "stayed flat"
                    lines.append(f"- First-to-last direction: {direction} from {_format_result_value(first_y)} to {_format_result_value(last_y)}")
            chart_df = working.head(max_rows)

        elif chart_type == "Scatter plot":
            x = str(params["x"])
            y = str(params["y"])
            color = params.get("color")
            columns = [x, y] + ([str(color)] if color else [])
            lines.append(f"- X axis: {x}")
            lines.append(f"- Y axis: {y}")
            if color:
                lines.append(f"- Color grouping: {color}")
            working = df[columns].dropna().copy()
            if pd.api.types.is_numeric_dtype(working[x]) and pd.api.types.is_numeric_dtype(working[y]) and len(working) >= 2:
                lines.append(f"- Correlation between axes: {working[x].corr(working[y]):.3f}")
            chart_df = working.head(max_rows)

        elif chart_type == "Box plot":
            column = str(params["column"])
            group_by = params.get("group_by")
            lines.append(f"- Numeric distribution: {column}")
            if group_by:
                group_by = str(group_by)
                lines.append(f"- Grouped by: {group_by}")
                working = df[[group_by, column]].dropna().copy()
                working[group_by] = _string_group_values(working[group_by])
                chart_df = (
                    working.groupby(group_by, dropna=False)[column]
                    .agg(
                        count="count",
                        min="min",
                        q1=lambda value: value.quantile(0.25),
                        median="median",
                        q3=lambda value: value.quantile(0.75),
                        max="max",
                    )
                    .reset_index()
                )
            else:
                stats = df[column].dropna().describe(percentiles=[0.25, 0.5, 0.75]).reset_index()
                stats.columns = ["statistic", column]
                chart_df = stats

        elif chart_type == "Correlation heatmap":
            corr = correlations(df)
            lines.append("- Values are Pearson correlations from -1 to 1.")
            chart_df = corr.round(4).reset_index().rename(columns={"index": "metric"})

    except Exception as exc:  # noqa: BLE001
        lines.append(f"- Chart context generation warning: {exc}")

    if not chart_df.empty:
        lines.append("\nChart values:")
        lines.append(dataframe_to_markdown_table(chart_df, max_rows=max_rows))
    else:
        lines.append("- No compact chart table could be generated.")

    return "\n".join(lines)[:7_500]


def format_cleaned_query_response(result: CleanedQueryResult) -> str:
    """Format an executed analytical query as a productized AI response."""
    cleaning = "\n".join(f"- {action}" for action in result.cleaning_actions[:6])
    if len(result.cleaning_actions) > 6:
        cleaning += f"\n- {len(result.cleaning_actions) - 6:,} additional cleaning action(s) were applied."
    if not cleaning:
        cleaning = "- No cleaning changes were required before execution."
    mode_label = "rule-based aggregation" if result.mode == "deterministic" else "validated AI-generated SQL"
    return (
        "Summary\n"
        f"I cleaned a working copy of the active dataset and computed the result using {mode_label}. "
        f"The query returned {result.row_count:,} row(s).\n\n"
        "Query Result\n"
        f"{dataframe_to_markdown_table(result.result_df)}\n\n"
        "Evidence\n"
        f"- Execution mode: {mode_label}.\n"
        f"- Validated SQL was executed against the cleaned in-memory dataset.\n"
        f"- Rows returned: {result.row_count:,}.\n\n"
        "Caveats\n"
        f"{cleaning}\n"
        "- Results reflect the active filters currently applied in the workspace.\n\n"
        "Guardrails\n"
        "- Data was cleaned before query execution.\n"
        "- Source dataset unchanged.\n"
        "- SQL was validated before execution.\n"
        f"- Rows returned: {result.row_count:,}.\n"
        "- Assumptions made: ambiguous business terms were mapped to the closest matching column names when confidence was sufficient.\n\n"
        "Recommended Next Steps\n"
        "- Use the result table for reporting, or adjust filters and ask a follow-up question for a narrower cut."
    )


def render_execution_details(result: CleanedQueryResult) -> None:
    """Show optional execution metadata without making code the primary answer."""
    with st.expander("Execution details", expanded=False):
        st.caption(result.plan_summary)
        st.code(result.sql, language="sql")


def render_ai_loading_state(is_query: bool) -> None:
    """Render a branded AI loading state with visible analysis steps."""
    steps = [
        "Reading schema",
        "Cleaning query workspace" if is_query else "Preparing analysis context",
        "Running analysis" if is_query else "Reasoning over profile",
        "Preparing answer",
    ]
    step_html = "".join(f'<span class="loading-step">{escape(step)}</span>' for step in steps)
    st.markdown(
        f"""
        <div class="loading-card">
            <div class="loading-title">Analyzing dataset...</div>
            <div class="loading-steps">{step_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def should_execute_contextual_query(message: str, history: list[dict[str, object]] | None) -> bool:
    """Route follow-up analytical requests to the query executor when context is needed."""
    if should_execute_query(message):
        return True
    if not history:
        return False

    text = re.sub(r"\s+", " ", str(message or "")).strip().lower()
    if not text:
        return False

    clarification_markers = (
        "what do you mean",
        "what does",
        "explain",
        "help me understand",
        "why",
        "definition",
        "define",
    )
    if any(marker in text for marker in clarification_markers):
        return False

    has_recent_assistant_context = any(item.get("role") == "assistant" for item in history[-8:])
    if not has_recent_assistant_context:
        return False

    context_references = ("same", "that", "those", "them", "it", "previous", "above", "again", "now")
    query_actions = (
        "average",
        "break down",
        "breakdown",
        "compare",
        "count",
        "group",
        "list",
        "maximum",
        "minimum",
        "rank",
        "show",
        "split",
        "sum",
        "top",
        "total",
        "trend",
    )
    has_context_reference = any(marker in text for marker in context_references)
    has_query_action = any(marker in text for marker in query_actions)
    return has_context_reference and has_query_action


def submit_conversation_message(df: pd.DataFrame, question: str, *, extra_context: str | None = None) -> bool:
    """Submit one conversational turn and render the assistant response."""
    question = truncate_demo_text(question.strip())
    if not question:
        st.warning("Enter a message before sending.")
        return False
    context = str(extra_context or "").strip()
    model_question = question
    if context:
        model_question = (
            f"{question}\n\n"
            "Hidden chart context for the assistant. Use this as the source of truth for the chart explanation; "
            "do not say the chart or values are unavailable unless this context is insufficient. "
            "Explain with one key pattern, one or two concrete values, an optional caveat only if meaningful, "
            "and one recommended action.\n\n"
            f"{context}"
        )
    if not demo_guard_allows("Conversation AI", model_question, ask_output_tokens()):
        return False

    messages = st.session_state.setdefault("ai_messages", [])
    user_message_id = f"user_{len(messages)}"
    messages.append({"role": "user", "content": question, "id": user_message_id})
    conversation_history = messages[:-1]
    with st.chat_message("user", avatar=chat_avatar("user")):
        st.markdown(question)

    with st.chat_message("assistant", avatar=chat_avatar("assistant")):
        query_mode = should_execute_contextual_query(question, conversation_history)
        loading_slot = st.empty()
        try:
            with loading_slot.container():
                render_ai_loading_state(query_mode)
            query_result: CleanedQueryResult | None = None
            if query_mode:
                query_result = answer_with_cleaned_sql(
                    df,
                    model_question,
                    history=conversation_history,
                    model=selected_ai_model(),
                    reasoning_effort=selected_reasoning_effort(),
                    max_tokens=min(ask_output_tokens(), 700),
                )
                response = format_cleaned_query_response(query_result)
            else:
                response = conversation_ai(
                    df,
                    model_question,
                    history=conversation_history,
                    model=selected_ai_model(),
                    reasoning_effort=selected_reasoning_effort(),
                    max_tokens=ask_output_tokens(),
                    context_max_chars=demo_context_chars(),
                )
            loading_slot.empty()
            assistant_message_id = f"assistant_{len(messages)}"
            render_ai_response(response, result=query_result, key_prefix=assistant_message_id)
            if query_result is not None:
                render_execution_details(query_result)
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "id": assistant_message_id,
                    "query_result": query_result,
                }
            )
            save_analysis_artifacts(question, response, query_result)
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
            loading_slot.empty()
            st.error(str(exc))
            return False
    return True


def render_conversation_ai(
    df: pd.DataFrame,
    source_name: str | None,
    active_filters: list[str],
) -> None:
    """Render conversational AI analyst chat."""
    render_current_analysis_context(df, source_name, active_filters)
    st.markdown(
        f"""
        <div class="conversation-toolbar">
            <span class="status-pill">Using {escape(selected_ai_model())}</span>
            <span class="status-pill">{len(st.session_state.get("ai_messages", [])) // 2:,} conversation turns</span>
            <span class="status-pill">SQL execution enabled</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    voice_prompt = render_voice_controls()
    shortcut_prompt = render_prompt_buttons(SUGGESTED_QUESTIONS, "conversation_draft", "conversation_prompt")
    pending_prompt = st.session_state.get("pending_ai_prompt", "")
    pending_prompt_to_run = str(pending_prompt).strip() if pending_prompt else ""
    pending_context_to_run = str(st.session_state.get("pending_ai_context", "")).strip() if pending_prompt else ""
    if pending_prompt:
        st.session_state["pending_ai_prompt"] = ""
        st.session_state["pending_ai_context"] = ""
        st.caption("Running suggested analysis from another workspace.")

    messages = st.session_state.setdefault("ai_messages", [])
    if not messages:
        render_conversation_empty_state()

    for message in messages:
        role = str(message.get("role", "assistant"))
        with st.chat_message(role, avatar=chat_avatar(role)):
            if message["role"] == "assistant":
                render_ai_response(
                    message["content"],
                    result=message.get("query_result"),
                    key_prefix=str(message.get("id", f"assistant_{id(message)}")),
                )
            else:
                st.markdown(message["content"])
    render_saved_analysis_history()

    chat_prompt = st.chat_input(
        "Message Conversation AI...",
        max_chars=DEMO_MAX_REQUEST_CHARS if demo_mode_enabled() else None,
        key="conversation_chat_input",
    )
    prompt = pending_prompt_to_run or voice_prompt or shortcut_prompt or chat_prompt
    extra_context = pending_context_to_run if pending_prompt_to_run else None
    if prompt and submit_conversation_message(df, str(prompt), extra_context=extra_context):
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
        with st.container(border=True):
            st.markdown('<div class="chart-builder-label">Builder controls</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="status-pill">Recommended: {escape(recommended_type)}</span>',
                unsafe_allow_html=True,
            )
            chart_type = st.selectbox(
                "Chart type",
                chart_options,
                index=recommended_index,
            )

            chart_description = chart_type
            chart_params: dict[str, object] = {}
            try:
                if chart_type == "Histogram":
                    options = nums or list(df.columns)
                    default = recommendation["x"] if recommendation["x"] in options else options[0]
                    column = st.selectbox("Column", options, index=options.index(default))
                    chart_description = f"{chart_type} of {column}"
                    chart_params = {"column": column}
                    fig = plot_histogram(df, column)
                elif chart_type == "Bar chart":
                    options = cats or list(df.columns)
                    default = recommendation["x"] if recommendation["x"] in options else options[0]
                    column = st.selectbox("Column", options, index=options.index(default))
                    chart_description = f"{chart_type} of {column}"
                    chart_params = {"column": column}
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
                    chart_description = f"{aggregation} of {value} by {category}"
                    chart_params = {"category": category, "value": value, "aggregation": aggregation}
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
                    chart_description = f"{y} over {x}"
                    chart_params = {"x": x, "y": y}
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
                    color_phrase = "" if color == "None" else f" colored by {color}"
                    chart_description = f"{y} vs {x}{color_phrase}"
                    chart_params = {"x": x, "y": y, "color": None if color == "None" else color}
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
                    group_phrase = "" if group_by == "None" else f" grouped by {group_by}"
                    chart_description = f"distribution of {column}{group_phrase}"
                    chart_params = {"column": column, "group_by": None if group_by == "None" else group_by}
                    fig = plot_box(df, column, None if group_by == "None" else group_by)
                else:
                    chart_description = "correlation heatmap of numeric columns"
                    chart_params = {}
                    fig = plot_heatmap(df)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

    with right:
        with st.container(border=True):
            st.markdown('<div class="chart-builder-label">Chart preview</div>', unsafe_allow_html=True)
            render_plotly_chart(fig, key=f"visualization_builder_{chart_type}")
            if st.button("Explain this chart in AI Chat", icon=":material/auto_awesome:", width="stretch"):
                st.session_state["pending_ai_context"] = chart_ai_context(df, chart_type, chart_description, chart_params)
                st.session_state["pending_ai_prompt"] = (
                    f"Explain this chart for a business audience: {chart_description}."
                )
                st.session_state["navigation"] = "Conversation AI"
                st.rerun()

    with st.expander("Chart source data"):
        render_table_workspace(
            "Chart Source Data",
            df,
            "chart_source",
            default_rows=100,
            download_name="chart_source_data.csv",
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
    render_quality_score_explanation(df)

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


def executive_summary_data(df: pd.DataFrame) -> dict[str, object]:
    """Build a deterministic executive summary payload."""
    health = dataset_health(df)
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    dates = detect_datetime_columns(df)
    insights = generate_insights(df, max_items=5)
    anomalies = outlier_summary(df)
    anomaly_total = int(anomalies["outlier_count"].sum()) if not anomalies.empty else 0
    risks = []
    if int(health["missing_cells"]) > 0:
        risks.append(f'{int(health["missing_cells"]):,} missing cells may affect reporting completeness.')
    if int(health["duplicate_rows"]) > 0:
        risks.append(f'{int(health["duplicate_rows"]):,} duplicate rows may inflate metrics.')
    if anomaly_total > 0:
        risks.append(f"{anomaly_total:,} numeric outlier flags should be reviewed before final reporting.")
    if not risks:
        risks.append("No major quality risk was detected in the active dataset view.")
    recommendation = "Use Conversation AI to validate the top insight and convert it into a stakeholder-ready narrative."
    if int(health["score"]) < 90:
        recommendation = "Resolve the highest-impact quality issue before using this view for formal reporting."
    return {
        "kpis": [
            ("Rows", f"{len(df):,}"),
            ("Fields", f"{df.shape[1]:,}"),
            ("Metrics", f"{len(nums):,}"),
            ("Dimensions", f"{len(cats):,}"),
            ("Time fields", f"{len(dates):,}"),
            ("Health", f'{int(health["score"])}/100'),
        ],
        "insights": insights[:3] or ["Dataset is loaded and ready for exploration."],
        "risks": risks[:2],
        "recommendation": recommendation,
    }


def executive_summary_markdown(df: pd.DataFrame) -> str:
    """Render executive summary data as Markdown."""
    data = executive_summary_data(df)
    kpis = "\n".join(f"- **{label}:** {value}" for label, value in data["kpis"])
    insights = "\n".join(f"{idx}. {item}" for idx, item in enumerate(data["insights"], start=1))
    risks = "\n".join(f"{idx}. {item}" for idx, item in enumerate(data["risks"], start=1))
    return (
        "# InsightAnalytica Executive Summary\n\n"
        "## KPIs\n"
        f"{kpis}\n\n"
        "## Key Insights\n"
        f"{insights}\n\n"
        "## Risks\n"
        f"{risks}\n\n"
        "## Recommended Action\n"
        f"{data['recommendation']}\n"
    )


def render_one_click_executive_summary(df: pd.DataFrame) -> None:
    """Render a polished one-click executive summary."""
    data = executive_summary_data(df)
    kpi_cards = "".join(
        f"""
        <div class="executive-kpi-card">
            <div class="executive-kpi-label">{escape(str(label))}</div>
            <div class="executive-kpi-value">{escape(str(value))}</div>
        </div>
        """
        for label, value in data["kpis"]
    )
    insight_rows = "".join(
        f"""
        <div class="executive-list-row">
            <span class="executive-list-index">{idx}</span>
            <p>{escape(plain_text(insight, max_chars=210))}</p>
        </div>
        """
        for idx, insight in enumerate(data["insights"], start=1)
    )
    risk_rows = "".join(
        f"""
        <div class="executive-list-row executive-risk">
            <span class="executive-list-index">{idx}</span>
            <p>{escape(plain_text(risk, max_chars=190))}</p>
        </div>
        """
        for idx, risk in enumerate(data["risks"], start=1)
    )
    st.markdown(
        f"""
        <div class="executive-dashboard">
            <div class="executive-kpi-grid">
                {kpi_cards}
            </div>
            <div class="executive-story-grid">
                <section class="executive-panel">
                    <div class="executive-panel-header">
                        <div>
                            <div class="executive-section-eyebrow">Decision summary</div>
                            <div class="executive-panel-title">Key insights</div>
                        </div>
                        <span class="executive-panel-count">{len(data["insights"])} insights</span>
                    </div>
                    <div class="executive-list">{insight_rows}</div>
                </section>
                <section class="executive-panel">
                    <div class="executive-panel-header">
                        <div>
                            <div class="executive-section-eyebrow">Review queue</div>
                            <div class="executive-panel-title">Risks and action</div>
                        </div>
                        <span class="executive-panel-count">{len(data["risks"])} risks</span>
                    </div>
                    <div class="executive-list">{risk_rows}</div>
                    <div class="executive-action-card">
                        <div class="executive-panel-title">Recommended action</div>
                        <p>{escape(plain_text(data["recommendation"], max_chars=220))}</p>
                    </div>
                </section>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Export summary Markdown",
        data=executive_summary_markdown(df).encode("utf-8"),
        file_name="executive_summary.md",
        mime="text/markdown",
        icon=":material/download:",
        width="stretch",
    )
    summary_rows = (
        [{"section": "kpi", "label": label, "value": value} for label, value in data["kpis"]]
        + [{"section": "insight", "label": f"Insight {idx}", "value": item} for idx, item in enumerate(data["insights"], start=1)]
        + [{"section": "risk", "label": f"Risk {idx}", "value": item} for idx, item in enumerate(data["risks"], start=1)]
        + [{"section": "action", "label": "Recommended action", "value": data["recommendation"]}]
    )
    export_cols[1].download_button(
        "Export summary CSV",
        data=pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8"),
        file_name="executive_summary.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
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
    render_one_click_executive_summary(df)
    render_saved_ai_insight_cards()
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
            render_plotly_chart(
                plot_line(df, date_col, metric),
                key="presentation_primary_line",
            )
        elif metric:
            render_plotly_chart(
                plot_histogram(df, metric),
                key="presentation_primary_histogram",
            )
        else:
            st.info("Add a numeric column to show a KPI trend or distribution.")

    with chart_right:
        if category and metric:
            render_plotly_chart(
                plot_aggregated_bar(df, category, metric, "sum"),
                key="presentation_segment_bar",
            )
        elif len(nums) >= 2:
            render_plotly_chart(plot_heatmap(df), key="presentation_heatmap")
        else:
            st.info("Add category and numeric columns to show segment performance.")

    if metric:
        render_plotly_chart(plot_box(df, metric, category), key="presentation_box")


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
        render_plotly_chart(fig, key=f"pipeline_chart_{idx}_{chart_type}")
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
                history = st.session_state.setdefault("analysis_history", [])
                history.insert(
                    0,
                    {
                        "question": request,
                        "summary": short_text(generated, max_chars=220),
                        "rows": None,
                        "mode": "Generated code",
                    },
                )
                del history[12:]
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
    render_saved_analysis_history()


def render_empty_state(load_error: str | None = None) -> None:
    """Render state before a dataset is available."""
    st.markdown(
        f"""
        <div class="empty-workspace">
            <div class="app-eyebrow">{APP_NAME}</div>
            <div class="empty-title">Start with a dataset</div>
            <div class="empty-lead">Upload a CSV or Excel workbook from the sidebar, or try the sample dataset to launch the enterprise analytics workspace.</div>
            <div class="empty-steps">
                <span>CSV and XLSX</span>
                <span>Governed preview</span>
                <span>AI-assisted intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="empty_actions"):
        if load_error:
            st.error(load_error)
        st.button(
            "Load sample dataset",
            icon=":material/table_chart:",
            type="primary",
            on_click=enable_sample_dataset,
        )


def main() -> None:
    """Run the Streamlit app."""
    initialize_state()
    inject_css()

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
        render_conversation_ai(filtered_df, source_name, active_filters)
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

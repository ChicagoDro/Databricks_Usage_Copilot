# src/app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime, timedelta
import io

import streamlit as st

from src.rag_context_aware_prompts.chat_orchestrator import DatabricksUsageAssistant
from src.reports.base import SelectionLike
from src.reports.registry import get_reports, get_report_map, get_default_report_key


# ============================
# Deterministic Chip System
# ============================

@dataclass(frozen=True)
class Chip:
    """
    A deterministic, UI-stable action chip.

    id: stable identifier used for Streamlit keying (prevents index-shift weirdness)
    label: button label
    prompt: prompt to run when clicked
    focus: whether the chip is selection-focused (parity with existing chip models)
    group: taxonomy lane (Understand / Diagnose / Optimize / Monitor)
    """
    id: str
    label: str
    prompt: str
    focus: bool = True
    group: str = "Diagnose"


GROUP_ORDER = ["Understand", "Diagnose", "Optimize", "Monitor"]


def _safe_slug(x: str) -> str:
    return (
        str(x)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("\n", "_")
    )


def _default_chips_for_selection(report_name: str, sel: SelectionLike) -> List[Chip]:
    """
    Minimal, CONDITIONAL default chips.
    Only show chips when they're actually relevant.
    """
    et = str(sel.entity_type)
    eid = str(sel.entity_id)
    
    base: List[Chip] = []
    
    # Check if there's actually a spike worth investigating
    payload = sel.payload or {}
    
    # Spike detection logic - adapt based on what's in the payload
    has_spike = False
    
    # Method 1: Check for explicit spike indicators
    if 'pct_deviation' in payload:
        # From anomaly report
        has_spike = abs(payload.get('pct_deviation', 0)) > 20  # >20% deviation
    
    elif 'pct_of_total_cost' in payload:
        # From cost reports - if this entity is >15% of total, investigate
        has_spike = payload.get('pct_of_total_cost', 0) > 15
    
    elif 'cost_volatility_ratio' in payload:
        # From compute type report - high volatility = investigate
        has_spike = payload.get('cost_volatility_ratio', 0) > 1.0
    
    elif 'failure_rate_pct' in payload:
        # Failures can cause cost spikes via retries
        has_spike = payload.get('failure_rate_pct', 0) > 15
    
    # CONDITIONAL: Only add "Why a spike?" if there's evidence of one
    if has_spike:
        base.append(
            Chip(
                id=f"core:spike:{_safe_slug(et)}:{_safe_slug(eid)}",
                label="📈 Why a spike?",
                group="Diagnose",
                prompt=(
                    f"Analyze the recent spike for {et} ({eid}). "
                    f"Walk through likely causes: "
                    f"(1) Data volume growth "
                    f"(2) Retries or failures "
                    f"(3) Spot evictions forcing on-demand "
                    f"(4) Configuration changes "
                    f"(5) Schedule changes or increased frequency. "
                    f"For each hypothesis, explain how to verify it with the telemetry data."
                ),
            )
        )
    
    # ALWAYS add: Next steps (always relevant)
    base.append(
        Chip(
            id=f"core:next:{_safe_slug(et)}:{_safe_slug(eid)}",
            label="✅ Next steps",
            group="Monitor",
            prompt=(
                f"Give me a short action plan for {et} ({eid}) based on the "
                f"'{report_name}' report: quick wins, deeper investigation steps, "
                "and what to monitor going forward."
            ),
        )
    )
    
    return base


# RESULT:
# 
# For normal jobs:
#   - 💰 Cost Breakdown (report)
#   - 💡 How to Optimize (report)
#   - ✅ Next steps (default)
#   Total: 3 chips
#
# For jobs with spikes (>15% of cost OR >20% deviation):
#   - 💰 Cost Breakdown (report)
#   - 💡 How to Optimize (report)
#   - 📈 Why a spike? (default - CONDITIONAL)
#   - ✅ Next steps (default)
#   Total: 4 chips
#
# For failing jobs with spikes:
#   - 💰 Cost Breakdown (report)
#   - 🔍 Why Is This Failing? (report - CONDITIONAL)
#   - 💡 How to Optimize (report)
#   - 📈 Why a spike? (default - CONDITIONAL)
#   - ✅ Next steps (default)
#   Total: 5 chips
#
# Every chip is EARNED by the data, not shown blindly!


# WHAT THIS ELIMINATES:
#
# ❌ REMOVED from defaults:
#   - 📌 Explain this (redundant with Cost Breakdown context)
#   - 🧾 Main drivers (redundant with Cost Breakdown analysis)
#   - 💸 Optimize cost (redundant with "How to Optimize")
#   - 🛡️ Reliability check (redundant with "Why Is This Failing?")
#   - 🧠 Utilization (no compute reports yet to conflict)
#   - ⚠️ Stability (no compute reports yet to conflict)
#
# ✅ KEPT (truly generic):
#   - 📈 Why a spike? (temporal analysis - different from cost breakdown)
#   - ✅ Next steps (action planning - different from optimization recs)
#
# RESULT:
# - Job reports control 100% of their domain-specific chips
# - Defaults add ONLY cross-cutting temporal/planning analysis
# - Zero redundancy possible


def _render_chip_row(chips: List[Chip], key_prefix: str, columns: int = 3) -> None:
    if not chips:
        return

    cols = st.columns(min(columns, len(chips)))
    for i, chip in enumerate(chips):
        with cols[i % len(cols)]:
            if st.button(chip.label, key=f"{key_prefix}:{chip.id}"):
                st.session_state.pending_prompt = chip.prompt
                st.rerun()


def _render_chip_groups(chips: List[Chip], key_prefix: str) -> None:
    """
    Render chips grouped by taxonomy lane in a deterministic order.
    Unknown groups fall into Diagnose.
    """
    if not chips:
        return

    grouped: Dict[str, List[Chip]] = {g: [] for g in GROUP_ORDER}
    for c in chips:
        g = c.group if c.group in grouped else "Diagnose"
        grouped[g].append(c)

    for g in GROUP_ORDER:
        if not grouped[g]:
            continue
        st.markdown(f"**{g}**")
        _render_chip_row(grouped[g], key_prefix=f"{key_prefix}:{g}", columns=3)


# ============================
# Report Catalog (Pillars)
# ============================

# WITH this simplified version:

PILLAR_CATALOG: List[Tuple[str, str, List[str], List[Tuple[str, str]]]] = [
    (
        "Cost Management",
        "Spend, spot exposure, and cost concentration.",
        [
            "Job Cost & Reliability",
            "Compute Type Analysis",
            "Cost Concentration (Pareto)",
            "Spot Risk & Evictions",
            "Cost Anomaly Detection",
        ],
        [],  # Empty TODO list - keeps sidebar clean!
    ),
    # Future pillars can be added here as reports are implemented
]

# ============================================================================
# RESULT: Clean sidebar showing only your 4 implemented Cost Management reports!
# ============================================================================

# Sidebar-only renames (do NOT change report implementation)
REPORT_NAME_ALIASES = {
    "Job Cost": "Job Cost Breakdown",
}


def _resolve_report_key(identifier: str, report_map: Dict[str, object]) -> Optional[str]:
    """
    Identifier can be either:
      - an actual report key, OR
      - a report.name
    Returns the report key if found.
    """
    if identifier in report_map:
        return identifier
    for k, r in report_map.items():
        if getattr(r, "name", None) == identifier:
            return k
    return None


def _display_report_name(report_obj: object) -> str:
    n = getattr(report_obj, "name", "Unknown Report")
    return REPORT_NAME_ALIASES.get(n, n)


def _build_uncategorized_report_names(report_map: Dict[str, object]) -> List[str]:
    categorized_names = set()
    for _, _, active_identifiers, _ in PILLAR_CATALOG:
        for ident in active_identifiers:
            categorized_names.add(ident)

    all_names = [getattr(r, "name", "") for r in report_map.values()]
    return sorted([n for n in all_names if n and n not in categorized_names])


def _select_report(key: str) -> None:
    st.session_state.selected_report_key = key
    st.session_state.pending_prompt = None
    st.session_state.selection = None
    st.session_state.commentary = []  # Clear commentary when switching reports
    st.rerun()


def _render_sidebar_report_nav(report_map: Dict[str, object]) -> None:
    """
    Pillar-based navigation:
    - Active reports are clickable
    - TODO reports are disabled + greyed out
    - Unmapped reports show up under "Uncategorized"
    """
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
        }

        .pillar-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin: 0.25rem 0 0.10rem 0;
        }
        .pillar-sub {
            font-size: 0.75rem;
            opacity: 0.70;
            margin: 0 0 0.35rem 0;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 10px;
            padding: 0.35rem 0.55rem;
            margin: 0.12rem 0;
            font-size: 0.80rem;
            text-align: left;
            white-space: normal !important;
            height: auto !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        .selected-report {
            font-size: 0.75rem;
            opacity: 0.75;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Reports (Pillars)")
    q = st.text_input("Search reports", key="report_search", placeholder="e.g. cost, reliability…")

    def matches(text: str) -> bool:
        if not q:
            return True
        return q.lower() in text.lower()

    for pillar, desc, active_idents, todo_items in PILLAR_CATALOG:
        # Filter pillars based on search
        if q and not matches(pillar) and not matches(desc):
            any_hit = False
            for ident in active_idents:
                k = _resolve_report_key(ident, report_map)
                if k:
                    shown = _display_report_name(report_map[k])
                    if matches(shown):
                        any_hit = True
                        break
            if not any_hit:
                for name, d in todo_items:
                    if matches(name) or matches(d):
                        any_hit = True
                        break
            if not any_hit:
                continue

        st.markdown(f"<div class='pillar-title'>{pillar}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pillar-sub'>{desc}</div>", unsafe_allow_html=True)

        # Active reports (clickable)
        for ident in active_idents:
            k = _resolve_report_key(ident, report_map)
            if not k:
                continue
            r = report_map[k]
            shown_name = _display_report_name(r)

            if q and not matches(shown_name):
                continue

            is_selected = (k == st.session_state.selected_report_key)
            label = f"✅ {shown_name}" if is_selected else shown_name

            if st.button(label, key=f"nav:{pillar}:{k}"):
                _select_report(k)

        st.divider()

    # Uncategorized (auto)
    uncat = _build_uncategorized_report_names(report_map)
    if uncat:
        st.markdown("<div class='pillar-title'>Uncategorized</div>", unsafe_allow_html=True)
        st.markdown("<div class='pillar-sub'>Reports not yet mapped to a pillar.</div>", unsafe_allow_html=True)

        for name in uncat:
            k = _resolve_report_key(name, report_map)
            if not k:
                continue
            r = report_map[k]
            shown_name = _display_report_name(r)

            if q and not matches(shown_name):
                continue

            is_selected = (k == st.session_state.selected_report_key)
            label = f"✅ {shown_name}" if is_selected else shown_name

            if st.button(label, key=f"nav:uncat:{k}"):
                _select_report(k)

        st.divider()

    cur_key = st.session_state.selected_report_key
    cur_name = _display_report_name(report_map[cur_key]) if cur_key in report_map else cur_key
    st.markdown(f"<div class='selected-report'>Selected: <b>{cur_name}</b></div>", unsafe_allow_html=True)


# ============================
# App State / Assistant
# ============================

def init_state() -> None:
    if "assistant" not in st.session_state:
        st.session_state.assistant = DatabricksUsageAssistant.from_local()

    if "selected_report_key" not in st.session_state:
        st.session_state.selected_report_key = get_default_report_key()

    if "filters" not in st.session_state:
        st.session_state.filters = {}

    if "selection" not in st.session_state:
        st.session_state.selection = None  # type: Optional[SelectionLike]

    if "commentary" not in st.session_state:
        st.session_state.commentary = []  # list of {"prompt": str, "response": str}

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "db_path" not in st.session_state:
        repo_root = Path(__file__).resolve().parents[1]
        default_db = repo_root / "data" / "usage_rag_data.db"
        st.session_state.db_path = os.getenv("DB_PATH", str(default_db))

    # Debug buffers used later
    if "_debug_graph" not in st.session_state:
        st.session_state._debug_graph = None
    if "_debug_prompt" not in st.session_state:
        st.session_state._debug_prompt = None
    if "_debug_context" not in st.session_state:
        st.session_state._debug_context = None
    
    # Filters
    if "date_filter_start" not in st.session_state:
        # Default to last 30 days
        st.session_state.date_filter_start = datetime.now() - timedelta(days=30)
    
    if "date_filter_end" not in st.session_state:
        st.session_state.date_filter_end = datetime.now()
    
    if "workspace_filter" not in st.session_state:
        st.session_state.workspace_filter = []  # Empty = all workspaces
    
    if "apply_filters" not in st.session_state:
        st.session_state.apply_filters = True


def assistant() -> DatabricksUsageAssistant:
    return st.session_state.assistant


def run_commentary(prompt: str) -> None:
    focus = None
    sel = st.session_state.selection
    if sel is not None:
        focus = {"entity_type": sel.entity_type, "entity_id": sel.entity_id}

    # ADD LOADING INDICATOR:
    with st.spinner("Generating AI commentary..."):
        result = assistant().answer(prompt, focus=focus)
    
    st.session_state.commentary.append({"prompt": prompt, "response": result.answer})

    if st.session_state.debug_mode:
        st.session_state._debug_graph = result.graph_explanation
        st.session_state._debug_prompt = result.llm_prompt
        st.session_state._debug_context = result.llm_context
    else:
        st.session_state._debug_graph = None
        st.session_state._debug_prompt = None
        st.session_state._debug_context = None

# ============================
# Chip rendering (taxonomy + deterministic)
# ============================

def render_action_chips(report, sel: SelectionLike) -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] {
            width: 100%;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 999px;
            padding: 0.30rem 0.65rem;
            margin: 0.15rem 0;
            border: 1px solid rgba(49, 51, 63, 0.25);
            background-color: rgba(240, 242, 246, 0.6);
            font-size: 0.78rem;
            line-height: 1.05rem;
            font-weight: 500;
            white-space: normal !important;
            height: auto !important;
            text-align: center;
        }
        div[data-testid="stButton"] > button:hover {
            background-color: rgba(240, 242, 246, 0.9);
            border-color: rgba(49, 51, 63, 0.45);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    report_chips_raw = report.build_action_chips(sel, st.session_state.filters) or []

    report_chips: List[Chip] = []
    for idx, rc in enumerate(report_chips_raw):
        rc_id = getattr(rc, "id", None)
        stable_id = rc_id or f"report:{_safe_slug(report.key)}:{_safe_slug(sel.entity_type)}:{_safe_slug(sel.entity_id)}:{idx}"
        grp = getattr(rc, "group", None) or "Diagnose"
        report_chips.append(
            Chip(
                id=stable_id,
                label=rc.label,
                prompt=rc.prompt,
                focus=getattr(rc, "focus", True),
                group=grp,
            )
        )

    core_chips = _default_chips_for_selection(report.name, sel)

    seen = set()
    combined: List[Chip] = []
    for c in (report_chips + core_chips):
        if c.id in seen:
            continue
        seen.add(c.id)
        combined.append(c)

    if not combined:
        return

    st.markdown("**Actions:**")
    _render_chip_groups(combined, key_prefix=f"chip:{report.key}")


def get_available_workspaces() -> list:
    """Query DB for available workspaces"""
    import sqlite3
    try:
        conn = sqlite3.connect(st.session_state.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT workspace_id, workspace_name FROM workspace ORDER BY workspace_name")
        workspaces = cursor.fetchall()
        conn.close()
        return [(ws_id, ws_name) for ws_id, ws_name in workspaces]
    except Exception as e:
        st.sidebar.error(f"Failed to load workspaces: {e}")
        return []
    
# ============================
# App UI
# ============================

st.set_page_config(page_title="Databricks Copilot", page_icon="📊", layout="wide")
init_state()

reports = get_reports()
report_map = get_report_map()

if st.session_state.selected_report_key not in report_map:
    st.session_state.selected_report_key = get_default_report_key()

current_report = report_map[st.session_state.selected_report_key]

st.title("📊 Databricks Copilot")
st.caption("Deterministic reporting + contextual AI commentary")

with st.sidebar:
    _render_sidebar_report_nav(report_map)

    st.markdown("---")
    
    # =========================
    # FILTERS SECTION
    # =========================
    st.header("📅 Filters")
    
    with st.expander("Date Range & Workspace", expanded=False):
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.date_filter_start,
                key="filter_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=st.session_state.date_filter_end,
                key="filter_end_date"
            )
        
        # Quick date range buttons
        st.caption("Quick ranges:")
        qr_col1, qr_col2, qr_col3 = st.columns(3)
        with qr_col1:
            if st.button("Last 7d", key="qr_7d"):
                st.session_state.date_filter_start = datetime.now() - timedelta(days=7)
                st.session_state.date_filter_end = datetime.now()
                st.rerun()
        with qr_col2:
            if st.button("Last 30d", key="qr_30d"):
                st.session_state.date_filter_start = datetime.now() - timedelta(days=30)
                st.session_state.date_filter_end = datetime.now()
                st.rerun()
        with qr_col3:
            if st.button("Last 90d", key="qr_90d"):
                st.session_state.date_filter_start = datetime.now() - timedelta(days=90)
                st.session_state.date_filter_end = datetime.now()
                st.rerun()
        
        # Workspace filter
        workspaces = get_available_workspaces()
        if workspaces:
            workspace_options = [f"{ws_name} ({ws_id})" for ws_id, ws_name in workspaces]
            selected_workspaces = st.multiselect(
                "Workspaces",
                options=workspace_options,
                default=st.session_state.workspace_filter,
                key="filter_workspaces",
                help="Leave empty to show all workspaces"
            )
            st.session_state.workspace_filter = selected_workspaces
        
        # Apply filters button
        if st.button("🔄 Apply Filters", type="primary", use_container_width=True):
            st.session_state.date_filter_start = start_date
            st.session_state.date_filter_end = end_date
            st.session_state.apply_filters = True
            st.cache_data.clear()  # Clear cached data
            st.rerun()
        
        # Reset filters
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.date_filter_start = datetime.now() - timedelta(days=30)
            st.session_state.date_filter_end = datetime.now()
            st.session_state.workspace_filter = []
            st.cache_data.clear()
            st.rerun()
    
    # Show active filters summary
    if st.session_state.workspace_filter or st.session_state.apply_filters:
        st.caption("**Active Filters:**")
        st.caption(f"📅 {st.session_state.date_filter_start.strftime('%Y-%m-%d')} to {st.session_state.date_filter_end.strftime('%Y-%m-%d')}")
        if st.session_state.workspace_filter:
            st.caption(f"🏢 {len(st.session_state.workspace_filter)} workspace(s)")
    
    st.markdown("---")   

    st.header("Controls")
    st.checkbox("Debug mode", key="debug_mode")
    st.caption(f"DB: `{st.session_state.db_path}`")

    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload"):
        st.cache_data.clear()
        st.rerun()   

    if st.button("Clear selection"):
        st.session_state.selection = None
        st.rerun()

    if st.button("Clear commentary"):
        st.session_state.commentary = []
        st.session_state.pending_prompt = None
        st.rerun()




viz_col, comm_col = st.columns([2.2, 1.0], gap="large")

with viz_col:
    st.subheader(current_report.name)
    st.caption(current_report.description)

    active_filters = {
        **st.session_state.filters,  # Existing filters
        "date_start": st.session_state.date_filter_start.strftime('%Y-%m-%d'),
        "date_end": st.session_state.date_filter_end.strftime('%Y-%m-%d'),
        "workspaces": [
            ws.split('(')[1].rstrip(')')  # Extract workspace_id from "Name (ID)"
            for ws in st.session_state.workspace_filter
        ] if st.session_state.workspace_filter else []
    }

    # Load data with loading indicator
    with st.spinner("Loading report data..."):
        df = current_report.load_df(st.session_state.db_path, active_filters)

    # Render with loading indicator
    with st.spinner("Rendering visualization..."):
        current_report.render_viz(df, active_filters)

    # Export functionality
    st.markdown("---")
    export_col1, export_col2 = st.columns([3, 1])

    with export_col1:
        st.caption(f"Showing data from {active_filters['date_start']} to {active_filters['date_end']}")

    with export_col2:
        if not df.empty:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"{current_report.key}_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                key=f"export_{current_report.key}",
                help="Download this report's data as CSV"
            )
        else:
            st.button("📥 Export CSV", disabled=True, help="No data to export")

    selections = current_report.build_selections(df, active_filters)
    if selections:
        st.markdown("**Select an item:**")
        cols = st.columns(3)
        for i, sel in enumerate(selections):
            sel_key = f"select:{current_report.key}:{_safe_slug(sel.entity_type)}:{_safe_slug(sel.entity_id)}"
            with cols[i % 3]:
                if st.button(sel.label, key=sel_key):
                    st.session_state.selection = sel
                    st.session_state.pending_prompt = f"Tell me more about {sel.entity_type} {sel.entity_id}."
                    st.rerun()

with comm_col:
    st.subheader("Commentary")

    sel = st.session_state.selection
    if sel is None:
        st.info("Select an item in the report to generate commentary.")
    else:
        st.success(f"Selection: {sel.entity_type} • {sel.label}")
        render_action_chips(current_report, sel)

    st.markdown("---")

    if st.session_state.commentary:
        last = st.session_state.commentary[-1]
        st.markdown(last["response"])
        with st.expander("Show prompt", expanded=False):
            st.code(last["prompt"])
    else:
        st.caption("No commentary yet.")

    if st.session_state.debug_mode:
        with st.expander("🔍 Debug", expanded=False):
            if getattr(current_report, "debug_sql", None):
                st.markdown("**Report SQL**")
                st.code(current_report.debug_sql, language="sql")

            if st.session_state.commentary:
                if st.session_state._debug_graph:
                    st.markdown("**Graph / retrieval explanation**")
                    st.markdown(st.session_state._debug_graph)
                if st.session_state._debug_prompt:
                    st.markdown("**LLM prompt**")
                    st.code(st.session_state._debug_prompt)
                if st.session_state._debug_context:
                    st.markdown("**LLM context**")
                    st.code(st.session_state._debug_context)

    st.markdown("---")

    with st.form("freeform", clear_on_submit=False):
        free = st.text_area(
            "Ask a follow-up",
            placeholder="Ask a follow-up about this report or selection…",
            height=110,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and free.strip():
        context = [f"Report: {current_report.name}"]
        if sel:
            context.append(f"Selected: {sel.entity_type} {sel.entity_id}")
        prompt = f"{free.strip()}\n\nContext:\n" + "\n".join(context)
        st.session_state.pending_prompt = prompt
        st.rerun()

if st.session_state.pending_prompt:
    p = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    run_commentary(p)
    st.rerun()

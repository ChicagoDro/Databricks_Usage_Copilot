# src/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from src.chat_orchestrator import DatabricksUsageAssistant
from src.reports.base import SelectionLike
from src.reports.registry import get_reports, get_report_map, get_default_report_key


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
        # Robust default path from repo root (works regardless of how streamlit is launched)
        repo_root = Path(__file__).resolve().parents[1]
        default_db = repo_root / "data" / "usage_rag_data.db"
        st.session_state.db_path = os.getenv("DB_PATH", str(default_db))


def assistant() -> DatabricksUsageAssistant:
    return st.session_state.assistant


def run_commentary(prompt: str) -> None:
    focus = None
    sel = st.session_state.selection
    if sel is not None:
        focus = {"entity_type": sel.entity_type, "entity_id": sel.entity_id}

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


def render_action_chips(report, sel: SelectionLike) -> None:
    st.markdown(
        """
        <style>
        /* Base button container */
        div[data-testid="stButton"] {
            width: 100%;
        }

        /* Actual button */
        div[data-testid="stButton"] > button {
            width: 100%;

            border-radius: 999px;
            padding: 0.30rem 0.65rem;
            margin: 0.15rem 0;

            border: 1px solid rgba(49, 51, 63, 0.25);
            background-color: rgba(240, 242, 246, 0.6);

            font-size: 0.78rem;     /* smaller text */
            line-height: 1.05rem;   /* tight vertical spacing */
            font-weight: 500;

            white-space: normal !important;  /* allow wrapping */
            height: auto !important;         /* grow vertically */
            text-align: center;
        }

        /* Hover / focus polish */
        div[data-testid="stButton"] > button:hover {
            background-color: rgba(240, 242, 246, 0.9);
            border-color: rgba(49, 51, 63, 0.45);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    chips = report.build_action_chips(sel, st.session_state.filters)
    if not chips:
        return

    st.markdown("**Actions:**")
    cols = st.columns(min(3, len(chips)))
    for i, chip in enumerate(chips):
        with cols[i % len(cols)]:
            if st.button(chip.label, key=f"chip-{report.key}-{i}"):
                if chip.focus:
                    st.session_state.selection = sel
                st.session_state.pending_prompt = chip.prompt
                st.rerun()


st.set_page_config(page_title="Databricks Usage Copilot", page_icon="📊", layout="wide")
init_state()

reports = get_reports()
report_map = get_report_map()
current_report = report_map[st.session_state.selected_report_key]

st.title("📊 Databricks Usage Copilot")
st.caption("Deterministic reporting + contextual AI commentary")

with st.sidebar:
    st.header("Reports")

    report_labels = {r.name: r.key for r in reports}
    selected_label = st.radio(
        "Choose report",
        options=list(report_labels.keys()),
        index=list(report_labels.values()).index(st.session_state.selected_report_key),
        label_visibility="collapsed",
    )
    st.session_state.selected_report_key = report_labels[selected_label]
    current_report = report_map[st.session_state.selected_report_key]

    st.divider()

    st.header("Controls")
    st.checkbox("Debug mode", key="debug_mode")

    # Read-only DB path (no input)
    st.caption(f"DB: `{st.session_state.db_path}`")

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

    df = current_report.load_df(st.session_state.db_path, st.session_state.filters)
    current_report.render_viz(df, st.session_state.filters)

    selections = current_report.build_selections(df, st.session_state.filters)
    if selections:
        st.markdown("**Select an item:**")
        cols = st.columns(3)
        for i, sel in enumerate(selections):
            with cols[i % 3]:
                if st.button(sel.label, key=f"select-{current_report.key}-{i}"):
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
            # Show report SQL FIRST (because it’s what you need when df is empty)
            if current_report.debug_sql:
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

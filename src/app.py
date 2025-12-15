# src/app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from src.chat_orchestrator import DatabricksUsageAssistant
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
    focus: whether the chip is selection-focused (kept for parity with your existing chip model)
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
        x.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("\n", "_")
    )


def _default_chips_for_selection(report_name: str, sel: SelectionLike) -> List[Chip]:
    """
    Deterministic baseline chips that ALWAYS appear for a selection,
    even if the report didn't define any action chips.
    """
    et = str(sel.entity_type)
    eid = str(sel.entity_id)

    base: List[Chip] = [
        Chip(
            id=f"core:about:{_safe_slug(et)}:{_safe_slug(eid)}",
            label="📌 Explain this",
            group="Understand",
            prompt=(
                f"Explain what this {et} ({eid}) represents in Databricks usage telemetry, "
                f"and summarize what matters most in the context of the '{report_name}' report."
            ),
        ),
        Chip(
            id=f"core:drivers:{_safe_slug(et)}:{_safe_slug(eid)}",
            label="🧾 Main drivers",
            group="Diagnose",
            prompt=(
                f"For {et} ({eid}), identify the biggest drivers behind what I'm seeing in the '{report_name}' report. "
                "Be specific, and reference the underlying telemetry patterns (runs, compute usage, events) where applicable."
            ),
        ),
        Chip(
            id=f"core:spike:{_safe_slug(et)}:{_safe_slug(eid)}",
            label="📈 Why a spike?",
            group="Diagnose",
            prompt=(
                f"Did {et} ({eid}) spike recently? If so, give the most likely causes. "
                "Walk through a few hypotheses (data growth, retries, evictions, sizing, schedule change) and how to verify each."
            ),
        ),
        Chip(
            id=f"core:next:{_safe_slug(et)}:{_safe_slug(eid)}",
            label="✅ Next steps",
            group="Monitor",
            prompt=(
                f"Give me a short action plan for {et} ({eid}) based on the '{report_name}' report: "
                "quick wins, deeper investigation steps, and what to monitor going forward."
            ),
        ),
    ]

    et_norm = et.lower()

    # Job-ish entity types
    if "job" in et_norm:
        base.extend(
            [
                Chip(
                    id=f"job:cost:{_safe_slug(eid)}",
                    label="💸 Optimize cost",
                    group="Optimize",
                    prompt=(
                        f"For job ({eid}), what are the top cost drivers and the highest-confidence way to reduce cost "
                        "without harming SLA? Include tradeoffs and verification steps."
                    ),
                ),
                Chip(
                    id=f"job:reliability:{_safe_slug(eid)}",
                    label="🛡️ Reliability check",
                    group="Optimize",
                    prompt=(
                        f"For job ({eid}), assess reliability risks (failures, retries, long tail runtimes, evictions). "
                        "Recommend fixes and how to validate improvement."
                    ),
                ),
            ]
        )

    # Compute-ish entity types
    if any(x in et_norm for x in ["cluster", "compute", "warehouse"]):
        base.extend(
            [
                Chip(
                    id=f"compute:util:{_safe_slug(et)}:{_safe_slug(eid)}",
                    label="🧠 Utilization",
                    group="Optimize",
                    prompt=(
                        f"For {et} ({eid}), assess utilization efficiency (CPU/memory patterns, over/under-provisioning). "
                        "Recommend sizing/autoscaling changes and how to validate improvements."
                    ),
                ),
                Chip(
                    id=f"compute:stability:{_safe_slug(et)}:{_safe_slug(eid)}",
                    label="⚠️ Stability",
                    group="Diagnose",
                    prompt=(
                        f"For {et} ({eid}), identify stability risks (spot/eviction behavior, node churn, driver OOM, GC pressure). "
                        "Give mitigation steps and what telemetry would confirm the root cause."
                    ),
                ),
            ]
        )

    return base


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

    grouped = {g: [] for g in GROUP_ORDER}
    for c in chips:
        g = c.group if c.group in grouped else "Diagnose"
        grouped[g].append(c)

    for g in GROUP_ORDER:
        if not grouped[g]:
            continue
        st.markdown(f"**{g}**")
        _render_chip_row(grouped[g], key_prefix=f"{key_prefix}:{g}", columns=3)


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


# ============================
# Chip rendering (taxonomy + deterministic)
# ============================

def render_action_chips(report, sel: SelectionLike) -> None:
    # Pill styling (kept from your original)
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

    # 1) Report-defined chips (author intent)
    report_chips_raw = report.build_action_chips(sel, st.session_state.filters) or []

    report_chips: List[Chip] = []
    for idx, rc in enumerate(report_chips_raw):
        rc_id = getattr(rc, "id", None)
        stable_id = rc_id or f"report:{_safe_slug(report.key)}:{_safe_slug(sel.entity_type)}:{_safe_slug(sel.entity_id)}:{idx}"

        # If report chips don’t specify a group, default to Diagnose (safe middle)
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

    # 2) Deterministic baseline chips (always available)
    core_chips = _default_chips_for_selection(report.name, sel)

    # 3) Combine with deterministic ordering + de-dupe by id
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


# ============================
# App UI
# ============================

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

# src/reports/compute_type_cost.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import sqlite3

from src.reports.base import ActionChip, ReportSpec, default_focus_for_selection


@dataclass(frozen=True)
class Selection:
    entity_type: str
    entity_id: str
    label: str
    payload: Dict[str, Any]


COMPUTE_TYPE_COST_SQL = """
SELECT
  u.compute_type              AS compute_type,
  SUM(u.cost_usd)             AS cost_total_usd,
  SUM(u.dbus_consumed)        AS dbus_total,
  COUNT(*)                    AS usage_records
FROM compute_usage u
GROUP BY u.compute_type
ORDER BY cost_total_usd DESC;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(COMPUTE_TYPE_COST_SQL, conn)
    finally:
        conn.close()
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df is None or df.empty:
        st.warning("'No compute type cost data returned.'")
        st.caption("Turn on Debug mode to see the SQL, then run it directly in sqlite to inspect results.")
        return

    df = df.sort_values("cost_total_usd", ascending=True)

    fig = px.bar(
        df,
        y="compute_type",
        x="cost_total_usd",
        orientation="h",
        title="Total Cost by Compute Type (USD)",
        labels={"cost_total_usd": "Cost (USD)", "compute_type": "Compute Type"},
        hover_data={
            "cost_total_usd": ":.2f",
            "dbus_total": ":.2f",
            "usage_records": True,
        },
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show underlying data", expanded=False):
        st.dataframe(df, use_container_width=True)


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df is None or df.empty:
        return []

    selections: List[Selection] = []
    for row in df.itertuples(index=False):
        compute_type = str(row.compute_type)
        selections.append(
            Selection(
                entity_type="compute_type",
                entity_id=compute_type,
                label=compute_type,
                payload={
                    "cost_total_usd": float(row.cost_total_usd),
                    "dbus_total": float(row.dbus_total) if row.dbus_total is not None else None,
                    "usage_records": int(row.usage_records) if row.usage_records is not None else None,
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    ct = sel.entity_id
    focus = default_focus_for_selection(sel)

    if ct == "JOB_RUN":
        followups = [
            ("What drives this cost?", "Explain what drives JOB_RUN cost in this environment. Reference job_id=... examples."),
            ("Where is it concentrated?", "Identify which job_id=... contribute most to JOB_RUN cost and why."),
            ("Optimization checklist", "Give a short optimization checklist for JOB_RUN spend (capacity, scheduling, utilization, spot ratio)."),
        ]
    elif ct == "SQL_WAREHOUSE":
        followups = [
            ("What drives this cost?", "Explain what drives SQL_WAREHOUSE cost in this environment. Reference query_id=... and warehouse usage patterns if available."),
            ("How to reduce spend", "Provide concrete ways to reduce SQL_WAREHOUSE spend (auto-stop, sizing, concurrency, query optimization)."),
            ("What to investigate", "Suggest what to investigate next (slow queries, high-frequency dashboards, long-running queries)."),
        ]
    elif ct == "APC_CLUSTER":
        followups = [
            ("What is this used for?", "Explain what APC_CLUSTER represents here and typical workload patterns."),
            ("Cost drivers", "Explain key cost drivers for APC_CLUSTER usage and how to control them."),
            ("What to investigate", "Suggest next drill-downs and what signals indicate waste or inefficiency."),
        ]
    else:
        followups = [
            ("Overview", f"Tell me more about compute usage of type {ct}. Explain typical workloads and primary cost drivers."),
            ("Optimization levers", f"Tell me more about compute usage of type {ct}. List common optimization levers and tradeoffs."),
            ("Where to look next", f"Tell me more about compute usage of type {ct}. Recommend the next drill-down analyses."),
        ]

    return [ActionChip(label=l, prompt=p, focus=focus) for (l, p) in followups]


REPORT = ReportSpec(
    key="compute_type_cost",
    name="Total Cost by Compute Type",
    description="Compare spend across JOB_RUN, SQL_WAREHOUSE, and other compute types.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=COMPUTE_TYPE_COST_SQL,
)

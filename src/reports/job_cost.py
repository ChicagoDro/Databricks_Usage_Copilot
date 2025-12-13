# src/reports/job_cost.py
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


JOB_COST_SQL = """
SELECT
  j.job_id                                   AS job_id,
  j.job_name                                 AS job_name,
  j.workspace_id                             AS workspace_id,

  SUM(u.cost_usd)                            AS cost_total_usd,

  SUM(u.cost_usd * r.spot_instance_ratio)    AS cost_spot_usd,
  SUM(u.cost_usd * (1 - r.spot_instance_ratio))
                                              AS cost_ondemand_usd,

  CASE WHEN SUM(u.cost_usd) > 0
       THEN SUM(u.cost_usd * r.spot_instance_ratio) / SUM(u.cost_usd)
       ELSE 0 END                             AS spot_pct,

  CASE WHEN SUM(u.cost_usd) > 0
       THEN SUM(u.cost_usd * (1 - r.spot_instance_ratio)) / SUM(u.cost_usd)
       ELSE 0 END                             AS ondemand_pct,

  AVG(u.avg_cpu_utilization)                  AS avg_cpu_utilization,
  MAX(u.max_memory_used_gb)                   AS max_memory_used_gb,
  SUM(u.dbus_consumed)                        AS total_dbus_consumed,
  COUNT(DISTINCT r.job_run_id)                AS run_count

FROM compute_usage u
JOIN job_runs r
  ON r.job_run_id = u.parent_id
JOIN jobs j
  ON j.job_id = r.job_id

WHERE u.compute_type = 'JOB_RUN'

GROUP BY j.job_id, j.job_name, j.workspace_id
ORDER BY cost_total_usd DESC;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(JOB_COST_SQL, conn)
    finally:
        conn.close()

    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df.empty:
        st.warning("No job cost data available.")
        return

    long_df = df.melt(
        id_vars=["job_id", "job_name", "cost_total_usd"],
        value_vars=["cost_spot_usd", "cost_ondemand_usd"],
        var_name="pricing",
        value_name="cost_usd",
    )

    long_df["pricing"] = long_df["pricing"].map(
        {"cost_spot_usd": "Spot", "cost_ondemand_usd": "On-Demand"}
    )

    job_order = (
        df.sort_values("cost_total_usd", ascending=True)["job_name"].tolist()
    )

    fig = px.bar(
        long_df,
        y="job_name",
        x="cost_usd",
        color="pricing",
        orientation="h",
        category_orders={"job_name": job_order},
        title="Job Cost (USD) — Spot vs On-Demand (Proportional)",
        labels={"cost_usd": "Cost (USD)", "job_name": "Job"},
        hover_data={
            "job_id": True,
            "cost_total_usd": ":.2f",
        },
    )

    fig.update_layout(
        barmode="stack",
        height=520,
        legend_title_text="Capacity Type",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show underlying data", expanded=False):
        st.dataframe(df, use_container_width=True)


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df.empty:
        return []

    top = df.sort_values("cost_total_usd", ascending=False).head(12)
    selections: List[Selection] = []

    for row in top.itertuples(index=False):
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=f"{row.job_name} (job_id={row.job_id})",
                payload={
                    "cost_total_usd": float(row.cost_total_usd),
                    "spot_pct": float(row.spot_pct),
                    "ondemand_pct": float(row.ondemand_pct),
                    "avg_cpu_utilization": float(row.avg_cpu_utilization)
                    if row.avg_cpu_utilization is not None
                    else None,
                    "run_count": int(row.run_count),
                },
            )
        )

    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)

    return [
        ActionChip(
            label="Tell me about this job",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Include purpose, schedule, run cadence, and cost drivers."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Spot vs On-Demand Analysis",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Explain the spot vs on-demand split based on spot_instance_ratio "
                f"and discuss operational risk."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Compute Configuration",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Include compute configuration, autoscaling behavior, and node types."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Recent Failures & Evictions",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Summarize failures, retries, evictions, and instability signals."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Optimization Opportunities",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Provide concrete optimization opportunities grounded in utilization and cost."
            ),
            focus=focus,
        ),
    ]


REPORT = ReportSpec(
    key="job_cost",
    name="Job Cost",
    description="Where is cost going, and how much is driven by Spot vs On-Demand capacity?",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
)

# src/reports/job_cost_pareto.py
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


PARETO_SQL = """
WITH job_cost AS (
  SELECT
    j.job_id                         AS job_id,
    j.job_name                       AS job_name,
    SUM(u.cost_usd)                  AS cost_total_usd,
    COUNT(DISTINCT r.job_run_id)     AS run_count
  FROM compute_usage u
  JOIN job_runs r
    ON r.job_run_id = u.parent_id
  JOIN jobs j
    ON j.job_id = r.job_id
  WHERE u.compute_type = 'JOB_RUN'
  GROUP BY j.job_id, j.job_name
),
ranked AS (
  SELECT
    job_id,
    job_name,
    cost_total_usd,
    run_count,
    SUM(cost_total_usd) OVER () AS grand_total_usd,
    SUM(cost_total_usd) OVER (
      ORDER BY cost_total_usd DESC, job_id
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_cost_usd
  FROM job_cost
)
SELECT
  job_id,
  job_name,
  cost_total_usd,
  run_count,
  grand_total_usd,
  cumulative_cost_usd,
  CASE WHEN grand_total_usd > 0
       THEN 1.0 * cumulative_cost_usd / grand_total_usd
       ELSE 0 END AS cumulative_pct
FROM ranked
ORDER BY cost_total_usd DESC, job_id;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(PARETO_SQL, conn)
    finally:
        conn.close()
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """
    Pareto chart: bars for cost_total_usd (sorted), line for cumulative_pct.
    This avoids dual axes complexity by plotting cumulative percent on a 0–100 scale line.
    """
    import streamlit as st

    if df is None or df.empty:
        st.warning("No data returned for Pareto Job Cost Concentration.")
        return

    # Top N display (keep it readable)
    top_n = int(filters.get("pareto_top_n", 25))
    top = df.head(top_n).copy()

    # Bar chart for cost
    bar = px.bar(
        top.sort_values("cost_total_usd", ascending=True),
        y="job_name",
        x="cost_total_usd",
        orientation="h",
        title=f"Cost Concentration (Pareto) — Top {top_n} Jobs",
        labels={"cost_total_usd": "Total Cost (USD)", "job_name": "Job"},
        hover_data={"job_id": True, "run_count": True, "cost_total_usd": ":.2f"},
    )
    bar.update_layout(height=600)

    st.plotly_chart(bar, use_container_width=True)

    # Cumulative line (separate chart for clarity)
    top_line = top.copy()
    top_line["cumulative_pct_100"] = top_line["cumulative_pct"] * 100.0

    line = px.line(
        top_line,
        x=list(range(1, len(top_line) + 1)),
        y="cumulative_pct_100",
        markers=True,
        title=f"Cumulative Share of Cost — Top {top_n} Jobs",
        labels={"x": "Rank (1 = highest cost)", "cumulative_pct_100": "Cumulative % of Total Cost"},
        hover_data={"job_id": True, "job_name": True, "cumulative_pct_100": ":.2f"},
    )
    line.update_layout(height=360, yaxis_range=[0, 100])

    st.plotly_chart(line, use_container_width=True)

    with st.expander("Show underlying data", expanded=False):
        st.dataframe(df, use_container_width=True)


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    """
    Selection chips should be only the top jobs (by cost) to keep the list useful.
    """
    if df is None or df.empty:
        return []

    top_n = int(filters.get("pareto_top_n", 25))
    top = df.head(top_n)

    selections: List[Selection] = []
    for row in top.itertuples(index=False):
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=f"{row.job_name} (job_id={row.job_id})",
                payload={
                    "cost_total_usd": float(row.cost_total_usd),
                    "run_count": int(row.run_count),
                    "cumulative_pct": float(row.cumulative_pct),
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)

    return [
        ActionChip(
            label="Why is this job in the top spenders?",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Explain why it is a top cost contributor. "
                f"Include run frequency, duration, compute configuration, and utilization if available."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Optimization checklist",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Give an ordered optimization checklist (right-sizing, schedule changes, caching, cluster policies, spot ratio)."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Risk: spot & reliability",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Discuss reliability risk (spot interruptions/evictions if applicable) and mitigation options."
            ),
            focus=focus,
        ),
    ]


REPORT = ReportSpec(
    key="job_cost_pareto",
    name="Cost Concentration (Pareto)",
    description="Is spend concentrated in a few jobs? Prioritize the small set that drives most of the cost.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
)

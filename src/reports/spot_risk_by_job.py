# src/reports/spot_risk_by_job.py
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


SPOT_RISK_SQL = """
WITH usage_job AS (
  SELECT
    u.compute_usage_id,
    u.cost_usd,
    u.avg_cpu_utilization,
    u.max_memory_used_gb,
    r.job_run_id,
    r.job_id,
    r.spot_instance_ratio,
    j.job_name
  FROM compute_usage u
  JOIN job_runs r
    ON r.job_run_id = u.parent_id
  JOIN jobs j
    ON j.job_id = r.job_id
  WHERE u.compute_type = 'JOB_RUN'
),
evicts AS (
  SELECT
    e.compute_usage_id,
    COUNT(*) AS eviction_events
  FROM events e
  WHERE e.event_type = 'SPOT_EVICTION'
    AND e.compute_usage_id IS NOT NULL
  GROUP BY e.compute_usage_id
),
job_agg AS (
  SELECT
    uj.job_id AS job_id,
    uj.job_name AS job_name,

    SUM(uj.cost_usd) AS cost_total_usd,

    SUM(uj.cost_usd * uj.spot_instance_ratio) AS spot_cost_usd,
    SUM(uj.cost_usd * (1 - uj.spot_instance_ratio)) AS ondemand_cost_usd,

    CASE WHEN SUM(uj.cost_usd) > 0
         THEN 1.0 * SUM(uj.cost_usd * uj.spot_instance_ratio) / SUM(uj.cost_usd)
         ELSE 0 END AS spot_pct_weighted,

    AVG(uj.spot_instance_ratio) AS avg_spot_ratio_unweighted,

    AVG(uj.avg_cpu_utilization) AS avg_cpu_utilization,
    MAX(uj.max_memory_used_gb) AS max_memory_used_gb,

    SUM(COALESCE(ev.eviction_events, 0)) AS eviction_event_count,

    COUNT(DISTINCT uj.job_run_id) AS run_count,
    COUNT(DISTINCT uj.compute_usage_id) AS usage_records

  FROM usage_job uj
  LEFT JOIN evicts ev
    ON ev.compute_usage_id = uj.compute_usage_id
  GROUP BY uj.job_id, uj.job_name
)
SELECT
  job_id,
  job_name,
  cost_total_usd,
  spot_cost_usd,
  ondemand_cost_usd,
  spot_pct_weighted,
  avg_spot_ratio_unweighted,
  eviction_event_count,
  run_count,
  usage_records,
  avg_cpu_utilization,
  max_memory_used_gb,

  -- Simple, transparent prioritization score:
  -- heavy spot exposure + evictions bubble to the top
  (spot_cost_usd * (1 + eviction_event_count)) AS risk_score

FROM job_agg
ORDER BY risk_score DESC, spot_cost_usd DESC, cost_total_usd DESC;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(SPOT_RISK_SQL, conn)
    finally:
        conn.close()
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """
    Scatter:
      x = weighted spot %
      y = total cost
      size = eviction count (with +1 so zeros still render)
    """
    import streamlit as st

    if df is None or df.empty:
        st.warning("No data returned for Spot Risk Exposure by Job.")
        return

    view_n = int(filters.get("spot_risk_view_n", 40))
    view = df.head(view_n).copy()

    view["spot_pct_100"] = view["spot_pct_weighted"] * 100.0
    view["evict_size"] = view["eviction_event_count"].fillna(0).astype(int) + 1

    fig = px.scatter(
        view,
        x="spot_pct_100",
        y="cost_total_usd",
        size="evict_size",
        hover_name="job_name",
        title=f"Spot Risk Exposure by Job — Top {view_n} by risk score",
        labels={
            "spot_pct_100": "Weighted Spot % (cost-weighted)",
            "cost_total_usd": "Total Cost (USD)",
        },
        hover_data={
            "job_id": True,
            "risk_score": ":.2f",
            "spot_cost_usd": ":.2f",
            "ondemand_cost_usd": ":.2f",
            "eviction_event_count": True,
            "run_count": True,
            "avg_cpu_utilization": ":.3f",
            "max_memory_used_gb": ":.2f",
        },
    )
    fig.update_layout(height=540, xaxis_range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

    # Ranking table (deterministic)
    with st.expander("Top risky jobs (table)", expanded=True):
        show_cols = [
            "job_name",
            "job_id",
            "cost_total_usd",
            "spot_cost_usd",
            "spot_pct_weighted",
            "eviction_event_count",
            "run_count",
            "risk_score",
        ]
        st.dataframe(view[show_cols], use_container_width=True)


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    """
    Chip list = top jobs by risk score (deterministic).
    """
    if df is None or df.empty:
        return []

    top_n = int(filters.get("spot_risk_chip_n", 12))
    top = df.head(top_n)

    selections: List[Selection] = []
    for row in top.itertuples(index=False):
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=f"{row.job_name} (job_id={row.job_id})",
                payload={
                    "risk_score": float(row.risk_score),
                    "spot_pct_weighted": float(row.spot_pct_weighted),
                    "spot_cost_usd": float(row.spot_cost_usd),
                    "eviction_event_count": int(row.eviction_event_count or 0),
                    "run_count": int(row.run_count or 0),
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)

    return [
        ActionChip(
            label="Explain this job’s spot risk",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Explain spot exposure using spot_instance_ratio and why this job is flagged as risky."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Show eviction evidence",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Summarize any SPOT_EVICTION events tied to this job’s compute usage, including timing and impact."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Mitigation playbook",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Give mitigation options: adjust spot_instance_ratio, use pools, increase retries, "
                f"fallback to on-demand, and schedule changes."
            ),
            focus=focus,
        ),
        ActionChip(
            label="Cost vs reliability tradeoff",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Recommend a spot ratio policy for this job given its cost and reliability profile."
            ),
            focus=focus,
        ),
    ]


REPORT = ReportSpec(
    key="spot_risk_by_job",
    name="Spot Risk Exposure by Job",
    description="Find jobs with high spot exposure and evidence of spot eviction events.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
)

# src/reports/job_cost.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3

from src.reports.base import ActionChip, ReportSpec, default_focus_for_selection


@dataclass(frozen=True)
class Selection:
    entity_type: str
    entity_id: str
    label: str
    payload: Dict[str, Any]


# =============================================================================
# SQL Query 1: SUMMARY (for visualization and most analysis)
# =============================================================================

JOB_COST_SQL = """
WITH job_runs_enriched AS (
  SELECT
    j.job_id,
    j.job_name,
    j.workspace_id,
    r.job_run_id,
    r.start_time,
    r.run_status,
    r.duration_ms,
    r.spot_ratio,
    u.total_cost,
    u.dbus_consumed,
    u.avg_cpu_utilization,
    u.avg_memory_gb,
    DATE(r.start_time) AS run_date
  FROM compute_usage u
  JOIN job_runs r ON r.job_run_id = u.parent_id
  JOIN jobs j ON j.job_id = r.job_id
  WHERE u.parent_type = 'JOB_RUN'
),
job_summary AS (
  SELECT
    job_id,
    job_name,
    workspace_id,
    
    -- Cost metrics
    SUM(total_cost) AS cost_total_usd,
    SUM(total_cost * spot_ratio) AS cost_spot_usd,
    SUM(total_cost * (1 - spot_ratio)) AS cost_ondemand_usd,
    
    -- Reliability metrics
    COUNT(*) AS total_runs,
    SUM(CASE WHEN run_status = 'SUCCESS' THEN 1 ELSE 0 END) AS success_runs,
    SUM(CASE WHEN run_status = 'FAILED' THEN 1 ELSE 0 END) AS failed_runs,
    SUM(CASE WHEN run_status = 'SKIPPED' THEN 1 ELSE 0 END) AS skipped_runs,
    
    -- Performance metrics
    AVG(duration_ms) / 1000.0 / 60.0 AS avg_duration_mins,
    AVG(avg_cpu_utilization) AS avg_cpu_utilization,
    MAX(avg_memory_gb) AS max_memory_gb,
    
    -- Spot usage
    AVG(spot_ratio) AS avg_spot_ratio,
    
    -- DBU efficiency
    SUM(dbus_consumed) AS total_dbus,
    CASE WHEN SUM(dbus_consumed) > 0 
         THEN SUM(total_cost) / SUM(dbus_consumed)
         ELSE 0 END AS cost_per_dbu
         
  FROM job_runs_enriched
  GROUP BY job_id, job_name, workspace_id
),
job_trend AS (
  SELECT
    job_id,
    run_date,
    SUM(total_cost) AS daily_cost,
    COUNT(*) AS daily_runs
  FROM job_runs_enriched
  GROUP BY job_id, run_date
)
SELECT 
  s.*,
  CAST(100.0 * s.success_runs / NULLIF(s.total_runs, 0) AS REAL) AS success_rate_pct,
  CAST(100.0 * s.failed_runs / NULLIF(s.total_runs, 0) AS REAL) AS failure_rate_pct,
  
  -- Cost concentration (for alerting)
  SUM(s.cost_total_usd) OVER () AS grand_total_cost,
  CAST(100.0 * s.cost_total_usd / NULLIF(SUM(s.cost_total_usd) OVER (), 0) AS REAL) AS pct_of_total_cost
  
FROM job_summary s
ORDER BY cost_total_usd DESC;
"""


# =============================================================================
# SQL Query 2: FAILURE INVESTIGATION (for "Why Is This Failing?" chip)
# =============================================================================
# NOTE: This query is loaded from sql/queries/job_failure_investigation.sql
# by the chat orchestrator when query_mode="failure_investigation"
# We document it here for reference, but the orchestrator uses the external file.
#
# If you need to update the failure investigation query, edit:
# sql/queries/job_failure_investigation.sql
# =============================================================================


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    sql = JOB_COST_SQL
    
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df.empty:
        st.warning("No job cost data available.")
        return

    # Limit to top 12 by cost for readability
    df_viz = df.nlargest(12, 'cost_total_usd')

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # Stacked bar: Spot vs On-Demand cost
    fig.add_trace(
        go.Bar(
            name='Spot Cost',
            x=df_viz['job_name'],
            y=df_viz['cost_spot_usd'],
            marker_color='#2ecc71',
            hovertemplate='<b>%{x}</b><br>Spot Cost: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            name='On-Demand Cost',
            x=df_viz['job_name'],
            y=df_viz['cost_ondemand_usd'],
            marker_color='#e74c3c',
            hovertemplate='<b>%{x}</b><br>On-Demand Cost: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False,
    )

    # Line: Success rate (secondary axis)
    fig.add_trace(
        go.Scatter(
            name='Success Rate',
            x=df_viz['job_name'],
            y=df_viz['success_rate_pct'],
            mode='lines+markers',
            marker=dict(size=8, color='#3498db'),
            line=dict(width=2, color='#3498db'),
            hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode='stack',
        title='Job Cost & Reliability Overview',
        xaxis_title='Job',
        yaxis_title='Cost (USD)',
        yaxis2_title='Success Rate (%)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Cost (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    total_cost = df['cost_total_usd'].sum()
    avg_success_rate = df['success_rate_pct'].mean()
    high_failure_jobs = len(df[df['failure_rate_pct'] > 10])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col2:
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
    with col3:
        st.metric("High Failure Jobs (>10%)", f"{high_failure_jobs}")


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df is None or df.empty:
        return []

    # Select jobs that are either:
    # 1. Top cost drivers (top 8)
    # 2. Low reliability (<95% success, top 4 by failure rate)
    top_cost = df.nlargest(8, 'cost_total_usd')
    low_reliability = df[df['success_rate_pct'] < 95].nlargest(4, 'failure_rate_pct')
    
    combined = pd.concat([top_cost, low_reliability]).drop_duplicates(subset=['job_id'])
    combined = combined.sort_values('cost_total_usd', ascending=False).head(12)
    
    selections: List[Selection] = []
    for row in combined.itertuples(index=False):
        # Build descriptive label with context
        alerts = []
        if row.failure_rate_pct > 10:
            alerts.append(f"⚠️ {row.failure_rate_pct:.0f}% fail")
        if row.pct_of_total_cost > 15:
            alerts.append(f"💰 {row.pct_of_total_cost:.0f}% of cost")
        
        label = f"{row.job_name}"
        if alerts:
            label += f" ({', '.join(alerts)})"
        
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=label,
                payload={
                    "cost_total_usd": float(row.cost_total_usd),
                    "success_rate_pct": float(row.success_rate_pct),
                    "failure_rate_pct": float(row.failure_rate_pct),
                    "avg_spot_ratio": float(row.avg_spot_ratio),
                    "total_runs": int(row.total_runs),
                },
            )
        )
    
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    """
    Streamlined cost-focused chips with failure investigation support.
    
    The failure chip now includes query_mode signal to trigger specialized
    failure investigation query in the chat orchestrator.
    """
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)
    
    payload = sel.payload
    failure_rate = payload.get('failure_rate_pct', 0)
    cost_total = payload.get('cost_total_usd', 0)
    spot_ratio = payload.get('avg_spot_ratio', 0)
    total_runs = payload.get('total_runs', 0)
    
    chips = []
    
    # PRIMARY: Cost Breakdown - includes context + breakdown
    chips.append(ActionChip(
        label="💰 Cost Breakdown",
        prompt=(
            f"Analyze job_id={job_id} (total cost: ${cost_total:,.2f}). "
            f"First, provide 2-sentence context: what this job does and its schedule. "
            f"Then break down cost into components: "
            f"(1) Driver vs worker node costs "
            f"(2) Spot (currently {spot_ratio*100:.0f}%) vs on-demand costs "
            f"(3) Compute runtime vs cluster startup/idle time "
            f"(4) Impact of {total_runs} runs and any retries "
            f"(5) Instance type and cluster size impact. "
            f"Identify which component is the biggest cost driver."
        ),
        focus=focus,
    ))
    
    # CONDITIONAL: Reliability - UPDATED with query_mode signal
    if failure_rate > 10:
        # Create specialized focus with query_mode signal
        failure_focus = {
            **focus,  # Copy base focus (entity_type, entity_id)
            "query_mode": "failure_investigation",  # Signal for specialized query
        }
        
        chips.append(ActionChip(
            label=f"🔍 Why Is This Failing {failure_rate:.1f}%?",
            prompt=(
                f"Analyze why job_id={job_id} has a {failure_rate:.1f}% failure rate. "
                f"The CONTEXT includes detailed failure records with error messages, "
                f"retry patterns, and spot eviction correlation data. "
                f"\n\n"
                f"Provide a structured analysis:"
                f"\n\n"
                f"**1. Error Pattern Analysis**"
                f"\n- Group failures by failure_category and error_summary"
                f"\n- Identify the most common failure type (what % of failures?)"
                f"\n- Show example error messages for each category"
                f"\n\n"
                f"**2. Spot Eviction Correlation**"
                f"\n- How many failures have nearby_evictions > 0?"
                f"\n- What % of failures are SPOT_EVICTION category?"
                f"\n- Are evictions clustered in time (check failure_date, failure_hour)?"
                f"\n\n"
                f"**3. Retry Pattern Analysis**"
                f"\n- What's the distribution of retry_count?"
                f"\n- Estimate wasted cost from retries (failed retries * avg cost per run)"
                f"\n\n"
                f"**4. Temporal Patterns**"
                f"\n- When do failures cluster? (time of day, specific dates)"
                f"\n- Has failure rate changed over time? (compare recent vs baseline)"
                f"\n\n"
                f"**5. Root Cause Hypothesis**"
                f"\n- What's the PRIMARY root cause? (support with evidence from data)"
                f"\n- What are contributing factors?"
                f"\n\n"
                f"**6. Recommended Fixes**"
                f"\n- Prioritized list of fixes (most impactful first)"
                f"\n- For each fix: estimated cost impact, implementation effort, risks"
            ),
            focus=failure_focus,  # Use specialized focus with query_mode
        ))
    
    # OPTIMIZATION: How to reduce cost
    chips.append(ActionChip(
        label="💡 How to Optimize",
        prompt=(
            f"Provide 3-5 ranked optimization recommendations for job_id={job_id}: "
            f"(1) Compute sizing (right-size clusters) "
            f"(2) Scheduling (off-peak hours, consolidation) "
            f"(3) Spot strategy (current: {spot_ratio*100:.0f}% spot) "
            f"(4) Configuration tuning (autoscaling, caching) "
            f"(5) Code/query optimization opportunities. "
            f"For each: estimate cost savings, implementation effort (high/medium/low), "
            f"and potential risks. Rank by ROI."
        ),
        focus=focus,
    ))
    
    return chips


# RESULT AFTER COMBINING WITH APP.PY DEFAULTS:
#
# For normal jobs (<10% failure rate):
#   - 💰 Cost Breakdown (report)
#   - 💡 How to Optimize (report)
#   + ✅ Next steps (default)
#   Total: 3 chips
#
# For jobs with failures (>10% failure rate):
#   - 💰 Cost Breakdown (report)
#   - 🔍 Why Is This Failing? (report - CONDITIONAL, uses specialized query)
#   - 💡 How to Optimize (report)
#   + 📈 Why a spike? (default - CONDITIONAL if pct_of_total_cost > 15%)
#   + ✅ Next steps (default)
#   Total: 4-5 chips
#
# KEY FEATURE:
# The failure chip now includes query_mode="failure_investigation" in its focus,
# which triggers the chat orchestrator to use sql/queries/job_failure_investigation.sql
# instead of the summary query, providing granular failure data with error messages.


REPORT = ReportSpec(
    key="job_cost",
    name="Job Cost & Reliability",
    description="Where is cost going? Which jobs are unreliable? Spot the problems.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=JOB_COST_SQL,
)
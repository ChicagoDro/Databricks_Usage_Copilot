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


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    
    # Extract filters with defaults
    date_start = filters.get('date_start', '2000-01-01')
    date_end = filters.get('date_end', '2099-12-31')
    workspaces = filters.get('workspaces', [])
    
    # Build workspace filter clause
    workspace_clause = ""
    if workspaces:
        ws_list = ','.join(f"'{w}'" for w in workspaces)
        workspace_clause = f"AND j.workspace_id IN ({ws_list})"
    
    # Build SQL with filters injected
    sql = f"""
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
    AND u.usage_date >= '{date_start}'
    AND u.usage_date <= '{date_end}'
    {workspace_clause}
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

    # Summary metrics at top
    st.markdown("### 📊 Cost Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = df['cost_total_usd'].sum()
    total_jobs = len(df)
    avg_reliability = df['success_rate_pct'].mean()
    high_cost_jobs = len(df[df['pct_of_total_cost'] > 10])
    
    with col1:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    with col2:
        st.metric("Active Jobs", f"{total_jobs}")
    with col3:
        st.metric("Avg Reliability", f"{avg_reliability:.1f}%")
    with col4:
        st.metric("High-Cost Jobs (>10%)", f"{high_cost_jobs}")
    
    st.markdown("---")

    # Main visualization: Cost with reliability overlay
    top_n = int(filters.get("job_cost_top_n", 15))
    top = df.head(top_n).copy()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Sort for horizontal bar chart
    top = top.sort_values('cost_total_usd', ascending=True)
    
    # Stacked bars for spot vs on-demand
    fig.add_trace(
        go.Bar(
            name='On-Demand',
            y=top['job_name'],
            x=top['cost_ondemand_usd'],
            orientation='h',
            marker_color='#1f77b4',
            hovertemplate='<b>%{y}</b><br>On-Demand: $%{x:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            name='Spot',
            y=top['job_name'],
            x=top['cost_spot_usd'],
            orientation='h',
            marker_color='#ff7f0e',
            hovertemplate='<b>%{y}</b><br>Spot: $%{x:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Overlay reliability as scatter
    fig.add_trace(
        go.Scatter(
            name='Success Rate',
            y=top['job_name'],
            x=top['success_rate_pct'],
            mode='markers',
            marker=dict(
                size=12,
                color=top['success_rate_pct'],
                colorscale='RdYlGn',
                cmin=80,
                cmax=100,
                showscale=True,
                colorbar=dict(title="Success %", x=1.15)
            ),
            hovertemplate='<b>%{y}</b><br>Success Rate: %{x:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"Top {top_n} Jobs by Cost (with Reliability Overlay)",
        barmode='stack',
        height=max(500, top_n * 35),
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Cost (USD)")
    fig.update_yaxes(title_text="", secondary_y=False)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 105], secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert section: Flag problematic jobs
    st.markdown("### ⚠️ Jobs Requiring Attention")
    
    problematic = df[
        (df['failure_rate_pct'] > 10) | 
        (df['pct_of_total_cost'] > 15) |
        (df['avg_spot_ratio'] > 0.7)
    ].copy()
    
    if not problematic.empty:
        problematic['issue'] = problematic.apply(lambda row: ', '.join([
            '🔴 High failure rate' if row['failure_rate_pct'] > 10 else '',
            '💰 Cost hotspot' if row['pct_of_total_cost'] > 15 else '',
            '⚡ High spot risk' if row['avg_spot_ratio'] > 0.7 else ''
        ]).strip(', '), axis=1)
        
        display_cols = ['job_name', 'cost_total_usd', 'failure_rate_pct', 
                       'pct_of_total_cost', 'avg_spot_ratio', 'issue']
        st.dataframe(
            problematic[display_cols].sort_values('cost_total_usd', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ No jobs flagged as requiring immediate attention")
    
    # Expandable full data table
    with st.expander("📋 Show all job details", expanded=False):
        display_cols = [
            'job_name', 'cost_total_usd', 'cost_spot_usd', 'cost_ondemand_usd',
            'total_runs', 'success_rate_pct', 'failure_rate_pct', 
            'avg_duration_mins', 'avg_cpu_utilization', 'cost_per_dbu'
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df.empty:
        return []

    # Prioritize: high cost OR low reliability
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
    Streamlined cost-focused chips - zero redundancy.
    Each chip answers a fundamentally different question.
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
    
    # CONDITIONAL: Reliability - only if failure rate warrants it
    if failure_rate > 10:
        chips.append(ActionChip(
            label=f"🔍 Why Is This Failing {failure_rate:.1f}%?",
            prompt=(
                f"Analyze why job_id={job_id} has a {failure_rate:.1f}% failure rate. "
                f"Investigate: "
                f"(1) Most common error patterns from recent failures "
                f"(2) Whether failures correlate with spot evictions, retries, or data issues "
                f"(3) Root cause hypothesis with supporting evidence "
                f"(4) Recommended fixes with estimated cost impact. "
                f"Focus on reliability improvements that reduce retry costs."
            ),
            focus=focus,
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
# Diagnose:
#   - 💰 Cost Breakdown (includes job context)
#   - 🔍 Why Is This Failing? (conditional: only if failure_rate > 10%)
#   - 📈 Why a spike? (from app.py default)
#
# Optimize:
#   - 💡 How to Optimize
#
# Monitor:
#   - ✅ Next steps (from app.py default)
#
# TOTAL: 4-5 chips (down from 8)
# Zero redundancy between report and defaults

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
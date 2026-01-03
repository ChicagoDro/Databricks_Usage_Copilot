# src/reports/spot_risk_by_job.py
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


SPOT_RISK_SQL = """
WITH job_runs_detail AS (
  SELECT
    r.job_run_id,
    r.job_id,
    j.job_name,
    r.run_status,
    r.spot_ratio,
    r.duration_ms,
    DATE(r.start_time) AS run_date,
    u.total_cost,
    u.compute_usage_id
  FROM job_runs r
  JOIN jobs j ON j.job_id = r.job_id
  JOIN compute_usage u ON u.parent_id = r.job_run_id
  WHERE u.parent_type = 'JOB_RUN'
),
eviction_counts AS (
  SELECT
    jrd.job_id,
    COUNT(DISTINCT e.event_id) AS eviction_count,
    COUNT(DISTINCT DATE(e.event_time)) AS days_with_evictions
  FROM job_runs_detail jrd
  JOIN events e ON e.compute_usage_id = jrd.compute_usage_id
  WHERE e.event_type = 'SPOT_EVICTION'
  GROUP BY jrd.job_id
),
job_risk_profile AS (
  SELECT
    jrd.job_id,
    jrd.job_name,
    
    -- Cost metrics
    SUM(jrd.total_cost) AS cost_total_usd,
    SUM(jrd.total_cost * jrd.spot_ratio) AS cost_spot_usd,
    SUM(jrd.total_cost * (1 - jrd.spot_ratio)) AS cost_ondemand_usd,
    
    -- Spot metrics
    AVG(jrd.spot_ratio) AS avg_spot_ratio,
    CAST(100.0 * SUM(jrd.total_cost * jrd.spot_ratio) / NULLIF(SUM(jrd.total_cost), 0) AS REAL) AS spot_cost_pct,
    
    -- Reliability impact
    COUNT(*) AS total_runs,
    SUM(CASE WHEN jrd.run_status = 'FAILED' THEN 1 ELSE 0 END) AS failed_runs,
    CAST(100.0 * SUM(CASE WHEN jrd.run_status = 'FAILED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS REAL) AS failure_rate_pct,
    
    -- Eviction data
    COALESCE(ec.eviction_count, 0) AS eviction_count,
    COALESCE(ec.days_with_evictions, 0) AS days_with_evictions,
    
    -- Performance
    AVG(jrd.duration_ms) / 1000.0 / 60.0 AS avg_duration_mins,
    
    -- Calculated risk score
    -- Formula: (spot_cost * spot_ratio * (1 + evictions)) + (failure_penalty)
    (SUM(jrd.total_cost * jrd.spot_ratio) * (1 + COALESCE(ec.eviction_count, 0) * 0.1)) +
    (SUM(CASE WHEN jrd.run_status = 'FAILED' THEN jrd.total_cost ELSE 0 END) * 2)
    AS risk_score
    
  FROM job_runs_detail jrd
  LEFT JOIN eviction_counts ec ON ec.job_id = jrd.job_id
  GROUP BY jrd.job_id, jrd.job_name
),
potential_savings AS (
  SELECT
    job_id,
    -- If we moved from current spot_ratio to 0% spot, how much would cost increase?
    cost_spot_usd * 0.7 AS spot_discount_benefit,  -- Assume 70% discount
    -- If we moved to 100% on-demand due to reliability concerns
    cost_spot_usd * 1.43 AS cost_if_all_ondemand  -- 1/0.7 = 1.43x
  FROM job_risk_profile
)
SELECT
  jrp.*,
  ps.spot_discount_benefit,
  ps.cost_if_all_ondemand,
  ps.cost_if_all_ondemand - jrp.cost_total_usd AS reliability_premium
FROM job_risk_profile jrp
LEFT JOIN potential_savings ps ON ps.job_id = jrp.job_id
ORDER BY risk_score DESC, cost_spot_usd DESC;
"""

EVICTION_TIMELINE_SQL = """
SELECT
  j.job_id,
  j.job_name,
  DATE(e.event_time) AS eviction_date,
  COUNT(*) AS evictions_that_day
FROM events e
JOIN compute_usage u ON u.compute_usage_id = e.compute_usage_id
JOIN job_runs r ON r.job_run_id = u.parent_id
JOIN jobs j ON j.job_id = r.job_id
WHERE e.event_type = 'SPOT_EVICTION'
GROUP BY j.job_id, j.job_name, DATE(e.event_time)
ORDER BY eviction_date, j.job_id;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    
    # Extract filters
    date_start = filters.get('date_start', '2000-01-01')
    date_end = filters.get('date_end', '2099-12-31')
    workspaces = filters.get('workspaces', [])
    
    # Build workspace clause (used in both queries)
    workspace_clause = ""
    if workspaces:
        ws_list = ','.join(f"'{w}'" for w in workspaces)
        workspace_clause = f"AND j.workspace_id IN ({ws_list})"
    
    # Main risk SQL with filters
    risk_sql = f"""
WITH job_runs_detail AS (
  SELECT
    r.job_run_id,
    r.job_id,
    j.job_name,
    r.run_status,
    r.spot_ratio,
    r.duration_ms,
    DATE(r.start_time) AS run_date,
    u.total_cost,
    u.compute_usage_id
  FROM job_runs r
  JOIN jobs j ON j.job_id = r.job_id
  JOIN compute_usage u ON u.parent_id = r.job_run_id
  WHERE u.parent_type = 'JOB_RUN'
    AND u.usage_date >= '{date_start}'
    AND u.usage_date <= '{date_end}'
    {workspace_clause}
),
eviction_counts AS (
  SELECT
    jrd.job_id,
    COUNT(DISTINCT e.event_id) AS eviction_count,
    COUNT(DISTINCT DATE(e.event_time)) AS days_with_evictions
  FROM job_runs_detail jrd
  JOIN events e ON e.compute_usage_id = jrd.compute_usage_id
  WHERE e.event_type = 'SPOT_EVICTION'
  GROUP BY jrd.job_id
),
job_risk_profile AS (
  SELECT
    jrd.job_id,
    jrd.job_name,
    SUM(jrd.total_cost) AS cost_total_usd,
    SUM(jrd.total_cost * jrd.spot_ratio) AS cost_spot_usd,
    SUM(jrd.total_cost * (1 - jrd.spot_ratio)) AS cost_ondemand_usd,
    AVG(jrd.spot_ratio) AS avg_spot_ratio,
    CAST(100.0 * SUM(jrd.total_cost * jrd.spot_ratio) / NULLIF(SUM(jrd.total_cost), 0) AS REAL) AS spot_cost_pct,
    COUNT(*) AS total_runs,
    SUM(CASE WHEN jrd.run_status = 'FAILED' THEN 1 ELSE 0 END) AS failed_runs,
    CAST(100.0 * SUM(CASE WHEN jrd.run_status = 'FAILED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS REAL) AS failure_rate_pct,
    COALESCE(ec.eviction_count, 0) AS eviction_count,
    COALESCE(ec.days_with_evictions, 0) AS days_with_evictions,
    AVG(jrd.duration_ms) / 1000.0 / 60.0 AS avg_duration_mins,
    (SUM(jrd.total_cost * jrd.spot_ratio) * (1 + COALESCE(ec.eviction_count, 0) * 0.1)) +
    (SUM(CASE WHEN jrd.run_status = 'FAILED' THEN jrd.total_cost ELSE 0 END) * 2) AS risk_score
  FROM job_runs_detail jrd
  LEFT JOIN eviction_counts ec ON ec.job_id = jrd.job_id
  GROUP BY jrd.job_id, jrd.job_name
),
potential_savings AS (
  SELECT
    job_id,
    cost_spot_usd * 0.7 AS spot_discount_benefit,
    cost_spot_usd * 1.43 AS cost_if_all_ondemand
  FROM job_risk_profile
)
SELECT
  jrp.*,
  ps.spot_discount_benefit,
  ps.cost_if_all_ondemand,
  ps.cost_if_all_ondemand - jrp.cost_total_usd AS reliability_premium
FROM job_risk_profile jrp
LEFT JOIN potential_savings ps ON ps.job_id = jrp.job_id
ORDER BY risk_score DESC, cost_spot_usd DESC;
"""
    
    # Timeline SQL with filters
    timeline_sql = f"""
SELECT
  j.job_id,
  j.job_name,
  DATE(e.event_time) AS eviction_date,
  COUNT(*) AS evictions_that_day
FROM events e
JOIN compute_usage u ON u.compute_usage_id = e.compute_usage_id
JOIN job_runs r ON r.job_run_id = u.parent_id
JOIN jobs j ON j.job_id = r.job_id
WHERE e.event_type = 'SPOT_EVICTION'
  AND DATE(e.event_time) >= '{date_start}'
  AND DATE(e.event_time) <= '{date_end}'
  {workspace_clause}
GROUP BY j.job_id, j.job_name, DATE(e.event_time)
ORDER BY eviction_date, j.job_id;
"""
    
    try:
        df = pd.read_sql_query(risk_sql, conn)
        timeline_df = pd.read_sql_query(timeline_sql, conn)
        df._eviction_timeline = timeline_df
    finally:
        conn.close()
    
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df is None or df.empty:
        st.warning("No spot risk data available.")
        return

    # Top-level metrics
    st.markdown("### ⚡ Spot Risk Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_evictions = df['eviction_count'].sum()
    high_risk_jobs = len(df[df['spot_cost_pct'] > 50])
    total_spot_cost = df['cost_spot_usd'].sum()
    avg_failure_rate = df['failure_rate_pct'].mean()
    
    with col1:
        st.metric("Total Evictions", f"{int(total_evictions)}")
    with col2:
        st.metric("High-Risk Jobs (>50% spot)", f"{high_risk_jobs}")
    with col3:
        st.metric("Total Spot Cost", f"${total_spot_cost:,.0f}")
    with col4:
        st.metric("Avg Failure Rate", f"{avg_failure_rate:.1f}%")
    
    st.markdown("---")

    # Main scatter plot: Risk vs Reward
    view_n = int(filters.get("spot_risk_view_n", 20))
    view = df.head(view_n).copy()
    
    fig = go.Figure()
    
    # Quadrant backgrounds
    fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=view['cost_total_usd'].max(),
                  fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=view['cost_total_usd'].max(),
                  fillcolor="lightyellow", opacity=0.2, layer="below", line_width=0)
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=view['spot_cost_pct'],
        y=view['cost_total_usd'],
        mode='markers+text',
        marker=dict(
            size=view['eviction_count'] * 5 + 10,  # Size by evictions
            color=view['failure_rate_pct'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Failure %"),
            line=dict(width=1, color='black')
        ),
        text=view['job_name'].str[:15],  # Truncate for readability
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Spot Cost %: %{x:.1f}%<br>' +
            'Total Cost: $%{y:,.0f}<br>' +
            'Evictions: ' + view['eviction_count'].astype(str) + '<br>' +
            'Failure Rate: ' + view['failure_rate_pct'].round(1).astype(str) + '%<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=f"Spot Risk Matrix: Cost vs Exposure (Top {view_n})",
        xaxis_title="Spot Cost % (weighted)",
        yaxis_title="Total Cost (USD)",
        height=600,
        annotations=[
            dict(x=25, y=view['cost_total_usd'].max() * 0.95, 
                 text="✅ Low Risk<br>Low Spot Usage", showarrow=False, 
                 font=dict(size=10, color='green')),
            dict(x=75, y=view['cost_total_usd'].max() * 0.95, 
                 text="⚠️ High Risk<br>Heavy Spot Usage", showarrow=False, 
                 font=dict(size=10, color='red'))
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Eviction timeline heatmap
    if hasattr(df, '_eviction_timeline') and not df._eviction_timeline.empty:
        st.markdown("### 📅 Eviction Timeline")
        
        timeline = df._eviction_timeline
        
        # Pivot for heatmap
        pivot = timeline.pivot_table(
            index='job_name',
            columns='eviction_date',
            values='evictions_that_day',
            fill_value=0
        )
        
        if not pivot.empty:
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Reds',
                hovertemplate='Job: %{y}<br>Date: %{x}<br>Evictions: %{z}<extra></extra>'
            ))
            
            fig_heat.update_layout(
                title="Eviction Frequency by Job Over Time",
                xaxis_title="Date",
                yaxis_title="Job",
                height=max(300, len(pivot) * 25)
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
    
    # Cost-benefit analysis table
    st.markdown("### 💰 Spot vs Reliability Tradeoff")
    
    analysis_df = view[[
        'job_name', 'cost_total_usd', 'spot_cost_pct', 
        'spot_discount_benefit', 'reliability_premium',
        'eviction_count', 'failure_rate_pct'
    ]].copy()
    
    analysis_df['recommendation'] = analysis_df.apply(lambda row:
        '🔴 Move to on-demand' if row['eviction_count'] > 5 and row['failure_rate_pct'] > 15
        else '🟡 Reduce spot ratio' if row['eviction_count'] > 2
        else '🟢 Current spot OK',
        axis=1
    )
    
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
    
    # Actionable alerts
    critical = view[(view['eviction_count'] > 5) | (view['failure_rate_pct'] > 20)]
    if not critical.empty:
        st.error(
            f"🚨 **Critical: {len(critical)} jobs** have excessive evictions or failures. "
            f"Immediate action required to reduce spot ratio or move to on-demand."
        )


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df is None or df.empty:
        return []

    # Prioritize by risk score
    top_n = int(filters.get("spot_risk_chip_n", 12))
    top = df.head(top_n)

    selections: List[Selection] = []
    for row in top.itertuples(index=False):
        # Build contextual label
        alerts = []
        if row.eviction_count > 5:
            alerts.append(f"⚠️ {int(row.eviction_count)} evictions")
        if row.failure_rate_pct > 15:
            alerts.append(f"🔴 {row.failure_rate_pct:.0f}% fail")
        
        label = f"{row.job_name}"
        if alerts:
            label += f" ({', '.join(alerts)})"
        
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=label,
                payload={
                    "risk_score": float(row.risk_score),
                    "spot_cost_pct": float(row.spot_cost_pct),
                    "eviction_count": int(row.eviction_count or 0),
                    "failure_rate_pct": float(row.failure_rate_pct),
                    "reliability_premium": float(row.reliability_premium) if pd.notna(row.reliability_premium) else 0,
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)
    payload = sel.payload
    
    eviction_count = payload.get('eviction_count', 0)
    failure_rate = payload.get('failure_rate_pct', 0)
    spot_pct = payload.get('spot_cost_pct', 0)
    reliability_premium = payload.get('reliability_premium', 0)
    
    chips = []
    
    # PRIMARY: Spot economics calculation (unique to this report)
    chips.append(ActionChip(
        label="⚡ Spot Economics",
        prompt=(
            f"Calculate the spot/on-demand tradeoff for job_id={job_id}. "
            f"Current state: {spot_pct:.0f}% spot usage, {eviction_count} evictions, "
            f"{failure_rate:.1f}% failure rate. "
            f"Show me: (1) Monthly savings from current spot ratio vs 100% on-demand "
            f"(2) Hidden cost of evictions (retry overhead, delays, wasted compute) "
            f"(3) Optimal spot ratio that minimizes TOTAL cost (savings minus eviction cost) "
            f"(4) Break-even analysis: at what eviction rate does spot stop being cost-effective? "
            f"Provide specific dollar figures and recommended spot ratio."
        ),
        focus=focus,
    ))
    
    # SECONDARY: Spot-specific mitigation strategies
    chips.append(ActionChip(
        label="🛡️ Reduce Eviction Risk",
        prompt=(
            f"Recommend specific changes to reduce evictions for job_id={job_id}. "
            f"Current: {eviction_count} evictions. Evaluate these options: "
            f"(1) Switch to instance types with historically lower eviction rates "
            f"(2) Use instance pools with reserved capacity "
            f"(3) Adjust spot bid strategies or use diversified instance types "
            f"(4) Schedule during off-peak hours when eviction risk is lower "
            f"(5) Implement graceful degradation (fallback to on-demand). "
            f"For each option, estimate: impact on eviction rate, cost delta, "
            f"implementation complexity. Rank by ROI."
        ),
        focus=focus,
    ))
    
    # CONDITIONAL: Emergency action for high-risk jobs
    if eviction_count > 5 or failure_rate > 15:
        chips.append(ActionChip(
            label="🚨 Emergency Fix",
            prompt=(
                f"URGENT: job_id={job_id} has {eviction_count} evictions and "
                f"{failure_rate:.1f}% failure rate. This requires immediate action. "
                f"Provide emergency mitigation: "
                f"(1) Immediate spot ratio reduction (from {spot_pct:.0f}% to what?) "
                f"(2) Instance pool configuration to add stability "
                f"(3) Retry policy tuning to handle evictions gracefully "
                f"(4) Temporary fallback to 100% on-demand until root cause is fixed. "
                f"Include: exact configuration changes, expected cost impact, "
                f"and timeline to implement (hours, not days)."
            ),
            focus=focus,
        ))
    
    return chips


# RESULT:
# 
# Normal spot jobs (low evictions, <15% failures):
#   - ⚡ Spot Economics (report-specific cost/benefit)
#   - 🛡️ Reduce Eviction Risk (spot-specific fixes)
#   + 📈 Why a spike? (default - conditional)
#   + ✅ Next steps (default)
#   Total: 3-4 chips
#
# High-risk spot jobs (>5 evictions OR >15% failures):
#   - ⚡ Spot Economics
#   - 🛡️ Reduce Eviction Risk
#   - 🚨 Emergency Fix (conditional)
#   + 📈 Why a spike? (default - conditional)
#   + ✅ Next steps (default)
#   Total: 4-5 chips
#
# ELIMINATED REDUNDANCIES:
# ❌ "Root Cause Analysis" - too generic, same as job_cost's "Why Is This Failing?"
# ❌ "Cost vs Reliability" - now integrated into "Spot Economics" with actual numbers
# ❌ "Long-Term Strategy" - redundant with default "Next steps"
# ❌ "Immediate Mitigation" - renamed to "Emergency Fix" and made more urgent
#
# UNIQUE VALUE:
# ✅ Spot Economics - CALCULATE savings, eviction costs, optimal spot ratio
# ✅ Reduce Eviction Risk - SPOT-SPECIFIC actions (instance types, pools, scheduling)
# ✅ Emergency Fix - URGENT conditional action for high-risk situations


REPORT = ReportSpec(
    key="spot_risk_by_job",
    name="Spot Risk & Evictions",
    description="Which jobs are being evicted? Calculate the cost of reliability.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=SPOT_RISK_SQL,
)
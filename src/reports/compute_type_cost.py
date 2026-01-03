# src/reports/compute_type_cost.py
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


COMPUTE_TYPE_COST_SQL = """
WITH daily_usage AS (
  SELECT
    parent_type AS compute_type,
    usage_date,
    SUM(total_cost) AS daily_cost,
    SUM(dbus_consumed) AS daily_dbus,
    COUNT(*) AS usage_records,
    AVG(avg_cpu_utilization) AS avg_cpu,
    COUNT(DISTINCT parent_id) AS unique_entities
  FROM compute_usage
  GROUP BY parent_type, usage_date
),
type_summary AS (
  SELECT
    compute_type,
    SUM(daily_cost) AS cost_total_usd,
    SUM(daily_dbus) AS dbus_total,
    AVG(daily_cost) AS avg_daily_cost,
    MAX(daily_cost) AS max_daily_cost,
    MIN(daily_cost) AS min_daily_cost,
    AVG(avg_cpu) AS avg_cpu_utilization,
    SUM(usage_records) AS total_usage_records,
    COUNT(DISTINCT usage_date) AS days_active,
    AVG(unique_entities) AS avg_active_entities
  FROM daily_usage
  GROUP BY compute_type
)
SELECT
  compute_type,
  cost_total_usd,
  dbus_total,
  avg_daily_cost,
  max_daily_cost,
  min_daily_cost,
  CAST(100.0 * cost_total_usd / NULLIF(SUM(cost_total_usd) OVER (), 0) AS REAL) AS pct_of_total,
  avg_cpu_utilization,
  total_usage_records,
  days_active,
  avg_active_entities,
  
  -- Volatility indicator
  CAST((max_daily_cost - min_daily_cost) / NULLIF(avg_daily_cost, 0) AS REAL) AS cost_volatility_ratio
  
FROM type_summary
ORDER BY cost_total_usd DESC;
"""

DAILY_TREND_SQL = """
SELECT
  parent_type AS compute_type,
  usage_date,
  SUM(total_cost) AS daily_cost
FROM compute_usage
GROUP BY parent_type, usage_date
ORDER BY usage_date;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    
    # Extract filters (note: workspace filter less relevant for type-level view)
    date_start = filters.get('date_start', '2000-01-01')
    date_end = filters.get('date_end', '2099-12-31')
    
    # Main SQL with date filter
    main_sql = f"""
WITH daily_usage AS (
  SELECT
    parent_type AS compute_type,
    usage_date,
    SUM(total_cost) AS daily_cost,
    SUM(dbus_consumed) AS daily_dbus,
    COUNT(*) AS usage_records,
    AVG(avg_cpu_utilization) AS avg_cpu,
    COUNT(DISTINCT parent_id) AS unique_entities
  FROM compute_usage
  WHERE usage_date >= '{date_start}'
    AND usage_date <= '{date_end}'
  GROUP BY parent_type, usage_date
),
type_summary AS (
  SELECT
    compute_type,
    SUM(daily_cost) AS cost_total_usd,
    SUM(daily_dbus) AS dbus_total,
    AVG(daily_cost) AS avg_daily_cost,
    MAX(daily_cost) AS max_daily_cost,
    MIN(daily_cost) AS min_daily_cost,
    AVG(avg_cpu) AS avg_cpu_utilization,
    SUM(usage_records) AS total_usage_records,
    COUNT(DISTINCT usage_date) AS days_active,
    AVG(unique_entities) AS avg_active_entities
  FROM daily_usage
  GROUP BY compute_type
)
SELECT
  compute_type,
  cost_total_usd,
  dbus_total,
  avg_daily_cost,
  max_daily_cost,
  min_daily_cost,
  CAST(100.0 * cost_total_usd / NULLIF(SUM(cost_total_usd) OVER (), 0) AS REAL) AS pct_of_total,
  avg_cpu_utilization,
  total_usage_records,
  days_active,
  avg_active_entities,
  CAST((max_daily_cost - min_daily_cost) / NULLIF(avg_daily_cost, 0) AS REAL) AS cost_volatility_ratio
FROM type_summary
ORDER BY cost_total_usd DESC;
"""
    
    # Trend SQL with date filter
    trend_sql = f"""
SELECT
  parent_type AS compute_type,
  usage_date,
  SUM(total_cost) AS daily_cost
FROM compute_usage
WHERE usage_date >= '{date_start}'
  AND usage_date <= '{date_end}'
GROUP BY parent_type, usage_date
ORDER BY usage_date;
"""
    
    try:
        df = pd.read_sql_query(main_sql, conn)
        trend_df = pd.read_sql_query(trend_sql, conn)
        df._trend_data = trend_df
    finally:
        conn.close()
    
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df is None or df.empty:
        st.warning("No compute type cost data returned.")
        return

    # Top metrics
    st.markdown("### 📊 Compute Cost Breakdown")
    
    col1, col2, col3 = st.columns(3)
    total_cost = df['cost_total_usd'].sum()
    
    with col1:
        st.metric("Total Spend", f"${total_cost:,.0f}")
    with col2:
        job_pct = df[df['compute_type'] == 'JOB_RUN']['pct_of_total'].sum()
        st.metric("Job Run %", f"{job_pct:.1f}%")
    with col3:
        wh_pct = df[df['compute_type'] == 'SQL_WAREHOUSE']['pct_of_total'].sum()
        st.metric("Warehouse %", f"{wh_pct:.1f}%")
    
    st.markdown("---")

    # Main chart: Horizontal bar with CPU utilization overlay
    df_sorted = df.sort_values('cost_total_usd', ascending=True)
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Cost bars
    fig.add_trace(
        go.Bar(
            name='Cost',
            y=df_sorted['compute_type'],
            x=df_sorted['cost_total_usd'],
            orientation='h',
            marker_color='#2ca02c',
            text=[f"${x:,.0f} ({p:.0f}%)" 
                  for x, p in zip(df_sorted['cost_total_usd'], df_sorted['pct_of_total'])],
            textposition='auto',
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Cost: $%{x:,.2f}<br>'
                '<extra></extra>'
            )
        ),
        secondary_y=False
    )
    
    # CPU utilization overlay
    fig.add_trace(
        go.Scatter(
            name='Avg CPU',
            y=df_sorted['compute_type'],
            x=df_sorted['avg_cpu_utilization'] * 100,  # Convert to percentage
            mode='markers+text',
            marker=dict(size=16, color='#d62728', symbol='diamond'),
            text=[f"{x*100:.0f}%" if pd.notna(x) else 'N/A' 
                  for x in df_sorted['avg_cpu_utilization']],
            textposition='middle right',
            hovertemplate='<b>%{y}</b><br>Avg CPU: %{x:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Total Cost by Compute Type (with CPU Utilization)",
        height=420,
        showlegend=True,
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="Cost (USD)")
    fig.update_yaxes(title_text="CPU Utilization (%)", range=[0, 100], secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis if available
    if hasattr(df, '_trend_data'):
        st.markdown("### 📈 Cost Trend Over Time")
        
        trend_df = df._trend_data
        
        fig_trend = go.Figure()
        
        for compute_type in trend_df['compute_type'].unique():
            type_data = trend_df[trend_df['compute_type'] == compute_type]
            fig_trend.add_trace(go.Scatter(
                x=pd.to_datetime(type_data['usage_date']),
                y=type_data['daily_cost'],
                mode='lines+markers',
                name=compute_type,
                hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>'
            ))
        
        fig_trend.update_layout(
            title="Daily Cost by Compute Type",
            xaxis_title="Date",
            yaxis_title="Daily Cost (USD)",
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Insights table
    st.markdown("### 📋 Detailed Metrics")
    
    display_cols = [
        'compute_type', 'cost_total_usd', 'pct_of_total', 
        'avg_daily_cost', 'avg_cpu_utilization', 
        'cost_volatility_ratio', 'avg_active_entities'
    ]
    
    display_df = df[display_cols].copy()
    display_df['avg_cpu_utilization'] = (display_df['avg_cpu_utilization'] * 100).round(1)
    display_df['cost_volatility_ratio'] = display_df['cost_volatility_ratio'].round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Alert on high volatility
    high_vol = df[df['cost_volatility_ratio'] > 1.5]
    if not high_vol.empty:
        st.warning(
            f"⚠️ **High cost volatility detected:** "
            f"{', '.join(high_vol['compute_type'].tolist())} "
            f"show significant daily cost swings. This may indicate "
            f"irregular workload patterns or inefficient resource usage."
        )


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df is None or df.empty:
        return []

    selections: List[Selection] = []
    for row in df.itertuples(index=False):
        compute_type = str(row.compute_type)
        
        # Add contextual info to label
        label = f"{compute_type} (${row.cost_total_usd:,.0f}, {row.pct_of_total:.0f}%)"
        
        selections.append(
            Selection(
                entity_type="compute_type",
                entity_id=compute_type,
                label=label,
                payload={
                    "cost_total_usd": float(row.cost_total_usd),
                    "pct_of_total": float(row.pct_of_total),
                    "avg_cpu_utilization": float(row.avg_cpu_utilization) if pd.notna(row.avg_cpu_utilization) else None,
                    "cost_volatility_ratio": float(row.cost_volatility_ratio) if pd.notna(row.cost_volatility_ratio) else None,
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    ct = sel.entity_id
    focus = default_focus_for_selection(sel)
    payload = sel.payload
    
    chips = []

    if ct == "JOB_RUN":
        chips.extend([
            ActionChip(
                label="📊 Type Comparison",
                prompt=(
                    f"Compare JOB_RUN compute to SQL_WAREHOUSE and APC_CLUSTER. "
                    f"Show: (1) Cost per entity (average job cost vs avg warehouse cost vs avg cluster cost) "
                    f"(2) Efficiency comparison (DBU/$ vs work accomplished) "
                    f"(3) Workload fit: are there jobs that should be SQL queries instead? "
                    f"(4) Migration opportunities: which workloads are on the wrong compute type? "
                    f"Include specific examples with cost deltas and migration ROI."
                ),
                focus=focus
            ),
            ActionChip(
                label="💰 Cross-Job Patterns",
                prompt=(
                    f"Find optimization patterns across ALL JOB_RUN instances: "
                    f"(1) Common configuration anti-patterns (oversized clusters, low spot usage) "
                    f"(2) Cluster sizing trends (are most jobs using similar configs?) "
                    f"(3) Spot usage distribution (what % of jobs use spot? how much?) "
                    f"(4) Policy recommendations that could apply to multiple jobs. "
                    f"Focus on changes that affect 5+ jobs for maximum impact."
                ),
                focus=focus
            ),
        ])
        
    elif ct == "SQL_WAREHOUSE":
        chips.extend([
            ActionChip(
                label="📏 Warehouse Right-Sizing",
                prompt=(
                    f"Analyze SQL_WAREHOUSE sizing across all warehouses. "
                    f"Identify: (1) Over-provisioned warehouses (low utilization, high cost) "
                    f"(2) Under-provisioned warehouses (queuing, poor performance) "
                    f"(3) Consolidation opportunities (can we merge small warehouses?) "
                    f"(4) Serverless migration candidates (predictable workloads). "
                    f"For each issue, provide: affected warehouses, cost impact, recommended size/config."
                ),
                focus=focus
            ),
            ActionChip(
                label="⏰ Usage Pattern Analysis",
                prompt=(
                    f"Analyze WHEN SQL_WAREHOUSE instances are used: "
                    f"(1) Peak hours vs idle time (when is demand highest?) "
                    f"(2) Auto-stop/auto-resume efficiency (are warehouses idling?) "
                    f"(3) Team-based sharing opportunities (can teams consolidate?) "
                    f"(4) Cost impact of better scheduling (shift workloads to off-peak). "
                    f"Include hourly usage heatmap insights and cost savings estimates."
                ),
                focus=focus
            ),
        ])
        
    elif ct == "APC_CLUSTER":
        chips.extend([
            ActionChip(
                label="🔍 Cluster Governance",
                prompt=(
                    f"Analyze APC_CLUSTER usage and governance: "
                    f"(1) Who uses these clusters? (identify top users by cost) "
                    f"(2) What are they used for? (ad-hoc analysis vs production) "
                    f"(3) Idle cluster waste (clusters left running, no activity) "
                    f"(4) Migration candidates (workloads that should be scheduled jobs). "
                    f"Recommend governance policies: auto-termination, cluster policies, training."
                ),
                focus=focus
            ),
            ActionChip(
                label="💸 Cost Controls",
                prompt=(
                    f"Recommend APC_CLUSTER cost controls: "
                    f"(1) Auto-termination policies (aggressive timeouts for dev clusters) "
                    f"(2) Instance pool usage (pre-warm capacity for frequent users) "
                    f"(3) Cluster policies (size limits, spot requirements, tags) "
                    f"(4) Migration plan (move predictable workloads to scheduled jobs). "
                    f"For each, estimate implementation effort and monthly savings."
                ),
                focus=focus
            ),
        ])
    else:
        chips.extend([
            ActionChip(
                label="📋 Type Overview",
                prompt=(
                    f"Explain compute usage of type {ct}: "
                    f"(1) Typical workloads and use cases "
                    f"(2) Primary cost drivers "
                    f"(3) Efficiency metrics vs other compute types "
                    f"(4) Optimization opportunities."
                ),
                focus=focus
            ),
        ])

    return chips


# RESULT:
# 
# JOB_RUN compute type:
#   - 📊 Type Comparison (cross-type migration opportunities)
#   - 💰 Cross-Job Patterns (patterns affecting 5+ jobs)
#   + 📈 Why a spike? (default - conditional)
#   + ✅ Next steps (default)
#   Total: 3-4 chips
#
# SQL_WAREHOUSE compute type:
#   - 📏 Warehouse Right-Sizing (sizing across all warehouses)
#   - ⏰ Usage Pattern Analysis (temporal patterns, scheduling)
#   + 📈 Why a spike? (default - conditional)
#   + ✅ Next steps (default)
#   Total: 3-4 chips
#
# APC_CLUSTER compute type:
#   - 🔍 Cluster Governance (users, usage, idle waste)
#   - 💸 Cost Controls (policies, pools, migration)
#   + 📈 Why a spike? (default - conditional)
#   + ✅ Next steps (default)
#   Total: 3-4 chips
#
# ELIMINATED REDUNDANCIES:
# ❌ "Top Cost Drivers" - too generic, overlaps with job_cost report
# ❌ "Quick Wins" - redundant across all reports
# ❌ "Reliability Risks" - belongs in job_cost or spot_risk, not type-level
# ❌ "Usage Patterns" (warehouse) - kept but made more specific
# ❌ "Performance Issues" - too generic
# ❌ "What Is This Used For?" - renamed to "Cluster Governance" with actual analysis
# ❌ "User Behavior" - integrated into "Cluster Governance"
# ❌ "Overview" / "Optimization" - generic fallbacks, kept minimal
#
# UNIQUE VALUE:
# ✅ Type Comparison - migration opportunities ACROSS compute types
# ✅ Cross-Job Patterns - patterns affecting MULTIPLE jobs (type-level view)
# ✅ Warehouse Right-Sizing - sizing analysis ACROSS all warehouses
# ✅ Usage Pattern Analysis - temporal patterns, scheduling optimization
# ✅ Cluster Governance - who, what, idle waste (APC-specific issues)


REPORT = ReportSpec(
    key="compute_type_cost",
    name="Compute Type Analysis",
    description="Compare spend across compute types. Identify trends and optimization opportunities.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=COMPUTE_TYPE_COST_SQL,
)
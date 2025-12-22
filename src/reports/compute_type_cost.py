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
    try:
        df = pd.read_sql_query(COMPUTE_TYPE_COST_SQL, conn)
        
        # Also load trend data
        trend_df = pd.read_sql_query(DAILY_TREND_SQL, conn)
        df._trend_data = trend_df  # Attach to main df
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
                label="🔍 Top Cost Drivers",
                prompt=(
                    f"Analyze JOB_RUN compute usage. "
                    f"Identify the top 3 job_id values driving cost and explain why. "
                    f"Include: run frequency, duration, cluster size, and spot ratio."
                ),
                focus=focus
            ),
            ActionChip(
                label="💡 Quick Wins",
                prompt=(
                    f"Suggest 3 quick optimization wins for JOB_RUN spend: "
                    f"(1) jobs to right-size, (2) jobs to increase spot ratio, "
                    f"(3) jobs to reschedule or consolidate."
                ),
                focus=focus
            ),
            ActionChip(
                label="⚠️ Reliability Risks",
                prompt=(
                    f"Identify JOB_RUN reliability risks: "
                    f"jobs with high failure rates, frequent retries, or spot evictions. "
                    f"Recommend mitigations."
                ),
                focus=focus
            ),
        ])
        
    elif ct == "SQL_WAREHOUSE":
        chips.extend([
            ActionChip(
                label="📊 Usage Patterns",
                prompt=(
                    f"Analyze SQL_WAREHOUSE usage patterns. "
                    f"Identify: (1) peak usage times, (2) idle periods, "
                    f"(3) warehouses with low utilization or excessive auto-resume cycles."
                ),
                focus=focus
            ),
            ActionChip(
                label="💰 Cost Reduction",
                prompt=(
                    f"Recommend SQL_WAREHOUSE cost reductions: "
                    f"right-sizing opportunities, auto-stop tuning, "
                    f"query optimization candidates, and warehouse consolidation."
                ),
                focus=focus
            ),
            ActionChip(
                label="🐌 Performance Issues",
                prompt=(
                    f"Identify SQL_WAREHOUSE performance issues: "
                    f"slow queries, warehouse queuing, undersized warehouses. "
                    f"Suggest specific actions."
                ),
                focus=focus
            ),
        ])
        
    elif ct == "APC_CLUSTER":
        chips.extend([
            ActionChip(
                label="🔍 What Is This Used For?",
                prompt=(
                    f"Explain APC_CLUSTER usage in this environment. "
                    f"Identify: typical workloads, primary users, "
                    f"and whether this represents ad-hoc analysis or production pipelines."
                ),
                focus=focus
            ),
            ActionChip(
                label="💸 Cost Controls",
                prompt=(
                    f"Recommend APC_CLUSTER cost controls: "
                    f"auto-termination policies, instance pool usage, "
                    f"cluster policies, and migration to job-based workflows."
                ),
                focus=focus
            ),
            ActionChip(
                label="👤 User Behavior",
                prompt=(
                    f"Analyze APC_CLUSTER user behavior. "
                    f"Identify users with highest usage, idle clusters, "
                    f"and opportunities for training or governance."
                ),
                focus=focus
            ),
        ])
    else:
        chips.extend([
            ActionChip(
                label="📋 Overview",
                prompt=(
                    f"Tell me more about compute usage of type {ct}. "
                    f"Explain typical workloads and primary cost drivers."
                ),
                focus=focus
            ),
            ActionChip(
                label="💡 Optimization",
                prompt=(
                    f"Suggest optimization opportunities for {ct} usage. "
                    f"Include: configuration tuning, scheduling, and governance policies."
                ),
                focus=focus
            ),
        ])

    return chips


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
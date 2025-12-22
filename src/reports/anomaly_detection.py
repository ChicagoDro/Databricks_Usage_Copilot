"""
Anomaly Detection Report
Integrates with existing Databricks Usage Copilot report system
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reports.base import ActionChip, ReportSpec, default_focus_for_selection


@dataclass(frozen=True)
class Selection:
    entity_type: str
    entity_id: str
    label: str
    payload: Dict[str, Any]


# SQL to get daily cost aggregates
ANOMALY_BASE_SQL = """
WITH daily_job_cost AS (
    SELECT
        j.job_id AS entity_id,
        j.job_name AS entity_name,
        'job' AS entity_type,
        u.usage_date,
        SUM(u.total_cost) AS total_cost,
        COUNT(DISTINCT r.job_run_id) AS run_count,
        AVG(r.duration_ms) / 60000.0 AS avg_duration_mins,
        SUM(CASE WHEN r.run_status = 'FAILED' THEN 1 ELSE 0 END) AS failure_count,
        AVG(r.spot_ratio) AS avg_spot_ratio
    FROM compute_usage u
    JOIN job_runs r ON u.parent_id = r.job_run_id
    JOIN jobs j ON r.job_id = j.job_id
    WHERE u.parent_type = 'JOB_RUN'
      AND u.usage_date >= date('now', '-90 days')
    GROUP BY j.job_id, j.job_name, u.usage_date
),
daily_compute_cost AS (
    SELECT
        c.compute_id AS entity_id,
        c.compute_name AS entity_name,
        c.compute_type AS entity_type,
        u.usage_date,
        SUM(u.total_cost) AS total_cost,
        COUNT(*) AS usage_records,
        AVG(u.avg_cpu_utilization) AS avg_cpu,
        SUM(u.dbus_consumed) AS total_dbus
    FROM compute_usage u
    JOIN non_job_compute c ON u.parent_id = c.compute_id
    WHERE u.parent_type IN ('SQL_WAREHOUSE', 'APC_CLUSTER')
      AND u.usage_date >= date('now', '-90 days')
    GROUP BY c.compute_id, c.compute_name, c.compute_type, u.usage_date
)
SELECT * FROM daily_job_cost
UNION ALL
SELECT 
    entity_id,
    entity_name,
    entity_type,
    usage_date,
    total_cost,
    usage_records AS run_count,
    NULL AS avg_duration_mins,
    0 AS failure_count,
    NULL AS avg_spot_ratio
FROM daily_compute_cost
ORDER BY entity_id, usage_date
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    """Load data and run anomaly detection"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(ANOMALY_BASE_SQL, conn)
    finally:
        conn.close()
    
    if df.empty:
        return df
    
    # Import the detector (assumes anomaly_detector.py is in src/forecasting/)
    from src.ml_statistical_forecasting.anomaly_detector import StatisticalAnomalyDetector
    
    # Run anomaly detection
    detector = StatisticalAnomalyDetector(
        z_threshold=3.0,
        min_history_days=14,
        use_rolling_window=True,
        rolling_window_days=7
    )
    
    anomalies = detector.detect_cost_anomalies(
        df,
        entity_col='entity_id',
        entity_name_col='entity_name',
        date_col='usage_date',
        cost_col='total_cost'
    )
    
    # Convert anomalies to DataFrame
    if anomalies:
        anomaly_df = pd.DataFrame([
            {
                'usage_date': a.date,
                'entity_id': a.entity_id,
                'anomaly_value': a.value,
                'anomaly_expected': a.expected_value,
                'z_score': a.z_score,
                'severity': a.severity,
                'pct_deviation': a.pct_deviation,
                'expected_min': a.expected_range[0],
                'expected_max': a.expected_range[1],
            }
            for a in anomalies
        ])
        
        # Merge back into main dataframe
        df = df.merge(
            anomaly_df,
            on=['entity_id', 'usage_date'],
            how='left'
        )
    else:
        # Add empty anomaly columns
        df['anomaly_value'] = None
        df['anomaly_expected'] = None
        df['z_score'] = None
        df['severity'] = None
        df['pct_deviation'] = None
        df['expected_min'] = None
        df['expected_max'] = None
    
    df['is_anomaly'] = df['severity'].notna()
    
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render anomaly detection visualization"""
    import streamlit as st
    
    if df.empty:
        st.warning("No data available for anomaly detection.")
        return
    
    # Summary metrics
    st.markdown("### 🚨 Anomaly Detection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_anomalies = df['is_anomaly'].sum()
    critical_count = len(df[df['severity'] == 'critical'])
    entities_affected = df[df['is_anomaly']]['entity_id'].nunique()
    
    if total_anomalies > 0:
        avg_deviation = df[df['is_anomaly']]['pct_deviation'].abs().mean()
    else:
        avg_deviation = 0
    
    with col1:
        st.metric("Total Anomalies", f"{int(total_anomalies)}")
    with col2:
        st.metric("Critical", f"{int(critical_count)}", delta_color="inverse")
    with col3:
        st.metric("Entities Affected", f"{int(entities_affected)}")
    with col4:
        st.metric("Avg Deviation", f"{avg_deviation:.0f}%", delta_color="inverse")
    
    if total_anomalies == 0:
        st.success("✅ No anomalies detected in the last 90 days.")
        return
    
    st.markdown("---")
    
    # Get top entities by anomaly count
    top_entities = (
        df[df['is_anomaly']]
        .groupby(['entity_id', 'entity_name'])
        .size()
        .reset_index(name='anomaly_count')
        .nlargest(10, 'anomaly_count')
    )
    
    if not top_entities.empty:
        # Select an entity to visualize
        selected_entity = st.selectbox(
            "Select entity to visualize:",
            options=top_entities['entity_id'].tolist(),
            format_func=lambda x: top_entities[top_entities['entity_id'] == x]['entity_name'].iloc[0]
        )
        
        # Filter data for selected entity
        entity_df = df[df['entity_id'] == selected_entity].copy()
        entity_df = entity_df.sort_values('usage_date')
        
        # Create visualization
        st.markdown(f"### 📈 Cost Timeline: {entity_df['entity_name'].iloc[0]}")
        
        fig = go.Figure()
        
        # All cost data (line)
        fig.add_trace(go.Scatter(
            x=entity_df['usage_date'],
            y=entity_df['total_cost'],
            mode='lines',
            name='Actual Cost',
            line=dict(color='lightblue', width=2),
            hovertemplate='%{x}<br>Cost: $%{y:.2f}<extra></extra>'
        ))
        
        # Expected range (filled area) - only for anomalies
        anomaly_points = entity_df[entity_df['is_anomaly']].copy()
        if not anomaly_points.empty:
            # Create expected range visualization
            for _, row in anomaly_points.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['usage_date'], row['usage_date']],
                    y=[row['expected_min'], row['expected_max']],
                    mode='lines',
                    line=dict(color='green', width=0),
                    fillcolor='rgba(0,255,0,0.1)',
                    fill='toself',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Anomaly points (markers)
        if not anomaly_points.empty:
            # Color by severity
            severity_colors = {
                'low': 'yellow',
                'medium': 'orange',
                'high': 'red',
                'critical': 'darkred'
            }
            
            for severity in ['low', 'medium', 'high', 'critical']:
                severity_points = anomaly_points[anomaly_points['severity'] == severity]
                if not severity_points.empty:
                    fig.add_trace(go.Scatter(
                        x=severity_points['usage_date'],
                        y=severity_points['anomaly_value'],
                        mode='markers',
                        name=f'{severity.title()} Anomaly',
                        marker=dict(
                            size=12,
                            color=severity_colors[severity],
                            symbol='x',
                            line=dict(width=2, color='black')
                        ),
                        hovertemplate=(
                            '%{x}<br>' +
                            'Cost: $%{y:.2f}<br>' +
                            'Expected: $' + severity_points['anomaly_expected'].astype(str) + '<br>' +
                            'Z-score: ' + severity_points['z_score'].round(2).astype(str) + '<br>' +
                            '<extra></extra>'
                        )
                    ))
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Cost (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly details table
    st.markdown("### 📋 Recent Anomalies")
    
    anomaly_details = df[df['is_anomaly']].copy()
    anomaly_details = anomaly_details.sort_values(['severity', 'z_score'], ascending=[True, False])
    
    # Severity order for sorting
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    anomaly_details['severity_order'] = anomaly_details['severity'].map(severity_order)
    anomaly_details = anomaly_details.sort_values(['severity_order', 'usage_date'], ascending=[True, False])
    
    display_cols = [
        'usage_date', 'entity_name', 'entity_type', 'total_cost',
        'anomaly_expected', 'pct_deviation', 'z_score', 'severity'
    ]
    
    # Format for display
    display_df = anomaly_details[display_cols].head(20).copy()
    display_df['pct_deviation'] = display_df['pct_deviation'].round(1)
    display_df['z_score'] = display_df['z_score'].round(2)
    display_df['total_cost'] = display_df['total_cost'].round(2)
    display_df['anomaly_expected'] = display_df['anomaly_expected'].round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "usage_date": "Date",
            "entity_name": "Entity",
            "entity_type": "Type",
            "total_cost": "Actual Cost",
            "anomaly_expected": "Expected Cost",
            "pct_deviation": "% Deviation",
            "z_score": "Z-Score",
            "severity": "Severity"
        }
    )
    
    # Alert for critical anomalies
    if critical_count > 0:
        st.error(
            f"🚨 **{critical_count} CRITICAL anomalies detected!** "
            f"These represent extreme deviations (>5σ) from historical patterns."
        )


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    """Build selection chips for anomalies"""
    if df.empty or not df['is_anomaly'].any():
        return []
    
    anomalies = df[df['is_anomaly']].copy()
    
    # Sort by severity and recency
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    anomalies['severity_order'] = anomalies['severity'].map(severity_order)
    anomalies = anomalies.sort_values(['severity_order', 'usage_date'], ascending=[True, False])
    
    # Take top 12 anomalies
    top_anomalies = anomalies.head(12)
    
    selections = []
    for row in top_anomalies.itertuples(index=False):
        # Build contextual label
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        
        emoji = severity_emoji.get(row.severity, '⚪')
        label = f"{emoji} {row.entity_name} on {row.usage_date} (+{abs(row.pct_deviation):.0f}%)"
        
        selections.append(Selection(
            entity_type="anomaly",
            entity_id=f"{row.entity_id}:{row.usage_date}",
            label=label,
            payload={
                "entity_id": row.entity_id,
                "entity_name": row.entity_name,
                "date": row.usage_date,
                "cost": float(row.total_cost),
                "expected_cost": float(row.anomaly_expected),
                "z_score": float(row.z_score),
                "severity": row.severity,
                "pct_deviation": float(row.pct_deviation),
                "run_count": int(row.run_count) if pd.notna(row.run_count) else 0,
                "failure_count": int(row.failure_count) if pd.notna(row.failure_count) else 0,
            }
        ))
    
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    """Build action chips for selected anomaly"""
    payload = sel.payload
    entity_id = payload['entity_id']
    entity_name = payload['entity_name']
    date = payload['date']
    severity = payload['severity']
    pct_dev = payload['pct_deviation']
    
    focus = default_focus_for_selection(sel)
    
    chips = [
        ActionChip(
            label="🔍 Root Cause Analysis",
            prompt=(
                f"Analyze the {severity} severity cost anomaly for {entity_name} (ID: {entity_id}) on {date}. "
                f"The cost was {abs(pct_dev):.0f}% {'above' if pct_dev > 0 else 'below'} expected. "
                f"Investigate: "
                f"(1) What specific events or changes occurred that day? "
                f"(2) Were there related failures, retries, or configuration changes? "
                f"(3) What is the most likely root cause? "
                f"Use the usage graph to find related events and compute usage patterns."
            ),
            focus=focus
        ),
        ActionChip(
            label="📊 Pattern Analysis",
            prompt=(
                f"Analyze historical patterns for {entity_name} to understand this anomaly on {date}. "
                f"Show: "
                f"(1) Normal cost baseline and variance "
                f"(2) Similar anomalies in the past "
                f"(3) Whether this is part of a trend or one-time spike "
                f"(4) Day-of-week or temporal patterns that might explain it"
            ),
            focus=focus
        ),
        ActionChip(
            label="⚡ Immediate Actions",
            prompt=(
                f"What immediate actions should be taken for {entity_name} after this {severity} anomaly on {date}? "
                f"Recommend: "
                f"(1) Whether to alert stakeholders immediately "
                f"(2) Quick checks to validate the anomaly isn't a data issue "
                f"(3) Emergency mitigation steps if still occurring "
                f"(4) What to monitor closely in the next 24-48 hours"
            ),
            focus=focus
        ),
        ActionChip(
            label="🛡️ Prevention Strategy",
            prompt=(
                f"Create a prevention strategy for {entity_name} to avoid similar anomalies. "
                f"Include: "
                f"(1) Configuration changes to improve stability "
                f"(2) Monitoring and alerting thresholds "
                f"(3) Resource constraints or guardrails to add "
                f"(4) Documentation or runbook updates needed"
            ),
            focus=focus
        ),
    ]
    
    return chips


# Report specification
REPORT = ReportSpec(
    key="anomaly_detection",
    name="Cost Anomaly Detection",
    description="Detect and explain unusual cost spikes using statistical analysis.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=ANOMALY_BASE_SQL,
)
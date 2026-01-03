# src/reports/job_cost_pareto.py
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


PARETO_SQL = """
WITH job_cost AS (
  SELECT
    j.job_id,
    j.job_name,
    SUM(u.total_cost) AS cost_total_usd,
    COUNT(DISTINCT r.job_run_id) AS run_count,
    AVG(r.duration_ms) / 1000.0 / 60.0 AS avg_duration_mins,
    CAST(100.0 * SUM(CASE WHEN r.run_status = 'FAILED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS REAL) AS failure_rate_pct,
    AVG(r.spot_ratio) AS avg_spot_ratio
  FROM compute_usage u
  JOIN job_runs r ON r.job_run_id = u.parent_id
  JOIN jobs j ON j.job_id = r.job_id
  WHERE u.parent_type = 'JOB_RUN'
  GROUP BY j.job_id, j.job_name
),
ranked AS (
  SELECT
    job_id,
    job_name,
    cost_total_usd,
    run_count,
    avg_duration_mins,
    failure_rate_pct,
    avg_spot_ratio,
    SUM(cost_total_usd) OVER () AS grand_total_usd,
    ROW_NUMBER() OVER (ORDER BY cost_total_usd DESC) AS rank,
    SUM(cost_total_usd) OVER (
      ORDER BY cost_total_usd DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_cost_usd
  FROM job_cost
)
SELECT
  rank,
  job_id,
  job_name,
  cost_total_usd,
  run_count,
  avg_duration_mins,
  failure_rate_pct,
  avg_spot_ratio,
  grand_total_usd,
  cumulative_cost_usd,
  CAST(100.0 * cumulative_cost_usd / NULLIF(grand_total_usd, 0) AS REAL) AS cumulative_pct,
  CAST(100.0 * cost_total_usd / NULLIF(grand_total_usd, 0) AS REAL) AS pct_of_total
FROM ranked
ORDER BY rank;
"""


def load_df(db_path: str, filters: Dict[str, Any]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    
    # Extract filters
    date_start = filters.get('date_start', '2000-01-01')
    date_end = filters.get('date_end', '2099-12-31')
    workspaces = filters.get('workspaces', [])
    
    # Build workspace clause
    workspace_clause = ""
    if workspaces:
        ws_list = ','.join(f"'{w}'" for w in workspaces)
        workspace_clause = f"AND j.workspace_id IN ({ws_list})"
    
    # Build SQL with filters
    sql = f"""
WITH job_cost AS (
  SELECT
    j.job_id,
    j.job_name,
    SUM(u.total_cost) AS cost_total_usd,
    COUNT(DISTINCT r.job_run_id) AS run_count,
    AVG(r.duration_ms) / 1000.0 / 60.0 AS avg_duration_mins,
    CAST(100.0 * SUM(CASE WHEN r.run_status = 'FAILED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS REAL) AS failure_rate_pct,
    AVG(r.spot_ratio) AS avg_spot_ratio
  FROM compute_usage u
  JOIN job_runs r ON r.job_run_id = u.parent_id
  JOIN jobs j ON j.job_id = r.job_id
  WHERE u.parent_type = 'JOB_RUN'
    AND u.usage_date >= '{date_start}'
    AND u.usage_date <= '{date_end}'
    {workspace_clause}
  GROUP BY j.job_id, j.job_name
),
ranked AS (
  SELECT
    job_id,
    job_name,
    cost_total_usd,
    run_count,
    avg_duration_mins,
    failure_rate_pct,
    avg_spot_ratio,
    SUM(cost_total_usd) OVER () AS grand_total_usd,
    ROW_NUMBER() OVER (ORDER BY cost_total_usd DESC) AS rank,
    SUM(cost_total_usd) OVER (
      ORDER BY cost_total_usd DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_cost_usd
  FROM job_cost
)
SELECT
  rank,
  job_id,
  job_name,
  cost_total_usd,
  run_count,
  avg_duration_mins,
  failure_rate_pct,
  avg_spot_ratio,
  grand_total_usd,
  cumulative_cost_usd,
  CAST(100.0 * cumulative_cost_usd / NULLIF(grand_total_usd, 0) AS REAL) AS cumulative_pct,
  CAST(100.0 * cost_total_usd / NULLIF(grand_total_usd, 0) AS REAL) AS pct_of_total
FROM ranked
ORDER BY rank;
"""
    
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    
    return df


def render_viz(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    import streamlit as st

    if df is None or df.empty:
        st.warning("No cost data for Pareto analysis.")
        return

    # Calculate Pareto insights
    jobs_for_80_pct = len(df[df['cumulative_pct'] <= 80])
    jobs_for_50_pct = len(df[df['cumulative_pct'] <= 50])
    total_jobs = len(df)
    concentration_ratio = (jobs_for_80_pct / total_jobs * 100) if total_jobs > 0 else 0
    
    # Top metrics
    st.markdown("### 📊 Cost Concentration Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", f"{total_jobs}")
    with col2:
        st.metric("Jobs for 50% of Cost", f"{jobs_for_50_pct}")
    with col3:
        st.metric("Jobs for 80% of Cost", f"{jobs_for_80_pct}")
    with col4:
        pct = (jobs_for_80_pct / total_jobs * 100) if total_jobs > 0 else 0
        st.metric("Concentration", f"{pct:.0f}%")
    
    # Insight box
    if jobs_for_80_pct <= total_jobs * 0.2:
        st.success(
            f"✅ **Classic Pareto Distribution**: {jobs_for_80_pct} jobs ({concentration_ratio:.0f}%) "
            f"drive 80% of cost. Focus optimization efforts here for maximum impact."
        )
    elif jobs_for_80_pct <= total_jobs * 0.4:
        st.info(
            f"ℹ️ **Moderate Concentration**: {jobs_for_80_pct} jobs ({concentration_ratio:.0f}%) "
            f"drive 80% of cost. Cost is somewhat distributed."
        )
    else:
        st.warning(
            f"⚠️ **High Fragmentation**: {jobs_for_80_pct} jobs ({concentration_ratio:.0f}%) "
            f"needed for 80% of cost. Many small jobs may indicate inefficient workload organization."
        )
    
    st.markdown("---")

    # Combined Pareto chart
    top_n = int(filters.get("pareto_top_n", 25))
    top = df.head(top_n).copy()
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Top {top_n} Jobs by Cost",
            "Cumulative Cost Contribution"
        )
    )
    
    # Top chart: Cost bars with failure rate overlay
    fig.add_trace(
        go.Bar(
            name='Cost',
            x=top['rank'],
            y=top['cost_total_usd'],
            marker_color='#1f77b4',
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'Rank: %{x}<br>' +
                'Cost: $%{y:,.2f}<br>' +
                'Runs: %{customdata[1]}<br>' +
                '<extra></extra>'
            ),
            customdata=top[['job_name', 'run_count']].values
        ),
        row=1, col=1
    )
    
    # Add failure rate as markers
    fig.add_trace(
        go.Scatter(
            name='Failure Rate',
            x=top['rank'],
            y=top['cost_total_usd'],
            mode='markers',
            marker=dict(
                size=10,
                color=top['failure_rate_pct'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Fail %", y=0.7, len=0.4),
                line=dict(width=1, color='black')
            ),
            hovertemplate=(
                '<b>%{customdata}</b><br>' +
                'Failure Rate: ' + top['failure_rate_pct'].round(1).astype(str) + '%<br>' +
                '<extra></extra>'
            ),
            customdata=top['job_name'].values
        ),
        row=1, col=1
    )
    
    # Bottom chart: Cumulative line with 80% marker
    fig.add_trace(
        go.Scatter(
            name='Cumulative %',
            x=top['rank'],
            y=top['cumulative_pct'],
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            hovertemplate=(
                'Rank: %{x}<br>' +
                'Cumulative: %{y:.1f}%<br>' +
                '<extra></extra>'
            )
        ),
        row=2, col=1
    )
    
    # Add 80% threshold line
    fig.add_hline(
        y=80, line_dash="dash", line_color="red",
        annotation_text="80% threshold",
        annotation_position="right",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Job Rank (1 = highest cost)", row=2, col=1)
    fig.update_yaxes(title_text="Cost (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown table
    st.markdown("### 📋 Top Jobs Breakdown")
    
    # Highlight the jobs that get us to 80%
    display_df = top.copy()
    display_df['priority'] = display_df.apply(
        lambda row: '🔥 Critical' if row['cumulative_pct'] <= 50
        else '⚠️ High' if row['cumulative_pct'] <= 80
        else '📊 Monitor',
        axis=1
    )
    
    display_cols = [
        'rank', 'job_name', 'cost_total_usd', 'pct_of_total',
        'cumulative_pct', 'failure_rate_pct', 'run_count', 'priority'
    ]
    
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "cost_total_usd": st.column_config.NumberColumn(
                "Total Cost",
                format="$%.2f"
            ),
            "pct_of_total": st.column_config.NumberColumn(
                "% of Total",
                format="%.1f%%"
            ),
            "cumulative_pct": st.column_config.NumberColumn(
                "Cumulative %",
                format="%.1f%%"
            ),
            "failure_rate_pct": st.column_config.NumberColumn(
                "Failure %",
                format="%.1f%%"
            ),
        }
    )
    
    # Optimization priorities
    st.markdown("### 💡 Optimization Priorities")
    
    st.markdown("""
    **Recommended approach:**
    1. **🔥 Critical (50% of cost):** Deep-dive optimization - these jobs have outsized impact
    2. **⚠️ High (next 30%):** Targeted improvements - quick wins with measurable impact  
    3. **📊 Monitor (remaining 20%):** Track for accumulation - many small jobs can add up
    """)
    
    # Find specific recommendations
    critical_jobs = df[df['cumulative_pct'] <= 50]
    high_failure_in_top = critical_jobs[critical_jobs['failure_rate_pct'] > 10]
    
    if not high_failure_in_top.empty:
        st.error(
            f"🚨 **Critical:** {len(high_failure_in_top)} high-cost jobs have reliability issues. "
            f"These jobs drive cost AND have failures - highest priority for intervention."
        )


def build_selections(df: pd.DataFrame, filters: Dict[str, Any]) -> List[Selection]:
    if df is None or df.empty:
        return []

    # Focus on jobs up to 80% cumulative
    top_80 = df[df['cumulative_pct'] <= 80].copy()
    
    # If fewer than 5 jobs reach 80%, take at least top 10
    if len(top_80) < 10:
        top_80 = df.head(10)

    selections: List[Selection] = []
    for row in top_80.itertuples(index=False):
        # Contextual label
        priority = (
            "🔥 Critical" if row.cumulative_pct <= 50
            else "⚠️ High" if row.cumulative_pct <= 80
            else "📊 Monitor"
        )
        
        label = f"#{int(row.rank)} {row.job_name} ({priority}, {row.pct_of_total:.1f}%)"
        
        selections.append(
            Selection(
                entity_type="job",
                entity_id=str(row.job_id),
                label=label,
                payload={
                    "rank": int(row.rank),
                    "cost_total_usd": float(row.cost_total_usd),
                    "pct_of_total": float(row.pct_of_total),
                    "cumulative_pct": float(row.cumulative_pct),
                    "failure_rate_pct": float(row.failure_rate_pct),
                },
            )
        )
    return selections


def build_action_chips(sel: Selection, filters: Dict[str, Any]) -> List[ActionChip]:
    job_id = sel.entity_id
    focus = default_focus_for_selection(sel)
    payload = sel.payload
    
    rank = payload.get('rank', 999)
    pct_of_total = payload.get('pct_of_total', 0)
    cumulative_pct = payload.get('cumulative_pct', 0)
    
    chips = []
    
    # PRIMARY: Pareto-specific positioning analysis
    chips.append(ActionChip(
        label="📊 Pareto Impact",
        prompt=(
            f"Analyze job_id={job_id}'s position in the cost distribution. "
            f"Rank #{rank}, representing {pct_of_total:.1f}% of total cost, "
            f"cumulative {cumulative_pct:.1f}%. "
            f"Explain: (1) Why this job ranks where it does (run frequency, "
            f"duration, cluster size, data volume) "
            f"(2) What % of total cost would be saved by optimizing this job "
            f"(3) How this job compares to jobs immediately above/below it in ranking "
            f"(4) Whether this ranking is stable or changing over time. "
            f"Include specific cost figures and efficiency metrics."
        ),
        focus=focus,
    ))
    
    # SECONDARY: Peer benchmarking (unique to Pareto view)
    chips.append(ActionChip(
        label="📈 Benchmark vs Peers",
        prompt=(
            f"Compare job_id={job_id} efficiency to similar jobs in the cost ranking. "
            f"Analyze: (1) Cost per run vs jobs with similar workloads "
            f"(2) DBU efficiency (cost per DBU vs peers) "
            f"(3) Runtime variance (is this job predictable?) "
            f"(4) Resource utilization patterns. "
            f"Identify if this job is inefficient relative to its peer group, "
            f"or if high cost is justified by workload characteristics."
        ),
        focus=focus,
    ))
    
    # CONDITIONAL: Only for top-tier cost drivers
    if pct_of_total > 10:
        chips.append(ActionChip(
            label="🎯 ROI Analysis",
            prompt=(
                f"This job represents {pct_of_total:.1f}% of total cost. "
                f"Calculate optimization ROI for job_id={job_id}: "
                f"(1) If we reduce cost by 20%, what's the dollar impact? "
                f"(2) What's the implementation effort for top 3 optimizations? "
                f"(3) Compare ROI of optimizing THIS job vs the next-ranked job "
                f"(4) Should we prioritize this job or focus elsewhere? "
                f"Provide concrete dollar figures and timeline estimates."
            ),
            focus=focus,
        ))
    
    return chips


# RESULT:
# 
# Normal jobs (rank >3, <10% of cost):
#   - 📊 Pareto Impact (report-specific)
#   - 📈 Benchmark vs Peers (report-specific)
#   + 📈 Why a spike? (default - conditional if pct_of_total > 15%)
#   + ✅ Next steps (default)
#   Total: 3-4 chips
#
# High-cost jobs (>10% of total cost):
#   - 📊 Pareto Impact
#   - 📈 Benchmark vs Peers
#   - 🎯 ROI Analysis (conditional)
#   + 📈 Why a spike? (default - conditional, triggers at >15%)
#   + ✅ Next steps (default)
#   Total: 4-5 chips
#
# ELIMINATED REDUNDANCIES:
# ❌ "Deep-Dive Analysis" - too generic, replaced by Pareto Impact
# ❌ "Cost Driver Analysis" - same as Deep-Dive, now in Pareto Impact
# ❌ "Quick Optimization Wins" - redundant with job_cost's "How to Optimize"
# ❌ "Strategic Roadmap" - redundant with default "Next steps"
#
# UNIQUE VALUE:
# ✅ Pareto Impact - WHERE in the distribution (rank, cumulative %)
# ✅ Benchmark vs Peers - efficiency RELATIVE to similar jobs
# ✅ ROI Analysis - should you PRIORITIZE this vs other jobs?


REPORT = ReportSpec(
    key="job_cost_pareto",
    name="Cost Concentration (Pareto)",
    description="Find the 20% of jobs driving 80% of cost. Focus your optimization efforts.",
    load_df=load_df,
    render_viz=render_viz,
    build_selections=build_selections,
    build_action_chips=build_action_chips,
    debug_sql=PARETO_SQL,
)
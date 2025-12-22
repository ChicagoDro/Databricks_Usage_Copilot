# src/reports/job_cost_pareto.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3

from src.rag_context_aware_prompts.reports.base import ActionChip, ReportSpec, default_focus_for_selection


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
    try:
        df = pd.read_sql_query(PARETO_SQL, conn)
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
    
    chips = []
    
    # Context-aware first chip
    if rank <= 3:
        chips.append(ActionChip(
            label="🎯 Deep-Dive Analysis",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"This is the #{rank} cost driver ({pct_of_total:.1f}% of total). "
                f"Provide comprehensive analysis: (1) cost breakdown by component, "
                f"(2) efficiency metrics vs similar jobs, (3) historical trends, "
                f"(4) specific optimization opportunities with ROI estimates."
            ),
            focus=focus,
        ))
    else:
        chips.append(ActionChip(
            label="📊 Cost Driver Analysis",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"Explain why this job ranks #{rank} in cost. "
                f"Include: run frequency, duration, cluster configuration, and data volume processed."
            ),
            focus=focus,
        ))
    
    # Quick wins
    chips.append(ActionChip(
        label="⚡ Quick Optimization Wins",
        prompt=(
            f"Tell me more about job_id={job_id}. "
            f"Identify 3 quick wins that could reduce cost by 15-30%: "
            f"right-sizing, spot ratio adjustments, scheduling changes, caching opportunities."
        ),
        focus=focus,
    ))
    
    # Comparative analysis
    chips.append(ActionChip(
        label="📈 Benchmark Against Peers",
        prompt=(
            f"Tell me more about job_id={job_id}. "
            f"How does this job's efficiency compare to similar jobs? "
            f"Analyze: cost per run, DBU efficiency, runtime variance, resource utilization."
        ),
        focus=focus,
    ))
    
    # Strategic recommendation
    if pct_of_total > 10:
        chips.append(ActionChip(
            label="🎯 Strategic Roadmap",
            prompt=(
                f"Tell me more about job_id={job_id}. "
                f"This job represents {pct_of_total:.1f}% of total cost. "
                f"Create a 90-day optimization roadmap: Phase 1 (quick wins), "
                f"Phase 2 (architectural improvements), Phase 3 (ongoing monitoring)."
            ),
            focus=focus,
        ))
    
    return chips


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
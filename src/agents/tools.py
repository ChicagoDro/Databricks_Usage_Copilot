"""
Agent Tools

Tools that agents can use to investigate Databricks usage data.
Each tool is a specific query capability exposed to the agent.
"""

import sqlite3
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


class AgentTool:
    """Base class for agent tools"""
    
    def __init__(self, name: str, description: str, db_path: str):
        self.name = name
        self.description = description
        self.db_path = db_path
    
    def _query_db(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dicts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def run(self, **kwargs) -> str:
        """Execute the tool. Must be implemented by subclasses."""
        raise NotImplementedError


class GetAnomalyDetailsTool(AgentTool):
    """Get detailed information about a specific anomaly"""
    
    def __init__(self, db_path: str):
        super().__init__(
            name="get_anomaly_details",
            description="Get detailed information about an anomaly (date, entity, cost, expected cost)",
            db_path=db_path
        )
    
    def run(self, entity_id: str, date: str) -> str:
        """
        Args:
            entity_id: The job or compute ID
            date: The date of the anomaly (YYYY-MM-DD)
        """
        sql = """
        SELECT 
            u.usage_date,
            u.entity_id,
            COALESCE(j.job_name, c.compute_name) as entity_name,
            u.parent_type as entity_type,
            u.total_cost,
            u.run_count,
            u.avg_duration_mins,
            u.failure_count,
            u.avg_spot_ratio
        FROM (
            SELECT 
                j.job_id as entity_id,
                u.parent_type,
                u.usage_date,
                SUM(u.total_cost) as total_cost,
                COUNT(DISTINCT r.job_run_id) as run_count,
                AVG(r.duration_ms) / 60000.0 as avg_duration_mins,
                SUM(CASE WHEN r.run_status = 'FAILED' THEN 1 ELSE 0 END) as failure_count,
                AVG(r.spot_ratio) as avg_spot_ratio
            FROM compute_usage u
            JOIN job_runs r ON u.parent_id = r.job_run_id
            JOIN jobs j ON r.job_id = j.job_id
            WHERE u.parent_type = 'JOB_RUN'
            GROUP BY j.job_id, u.usage_date
        ) u
        LEFT JOIN jobs j ON u.entity_id = j.job_id
        LEFT JOIN non_job_compute c ON u.entity_id = c.compute_id
        WHERE u.entity_id = ? AND u.usage_date = ?
        """
        
        results = self._query_db(sql, (entity_id, date))
        
        if not results:
            return f"No data found for {entity_id} on {date}"
        
        row = results[0]
        return f"""Anomaly Details:
- Date: {row['usage_date']}
- Entity: {row['entity_name']} ({row['entity_id']})
- Type: {row['entity_type']}
- Cost: ${row['total_cost']:.2f}
- Runs: {row['run_count']}
- Avg Duration: {row['avg_duration_mins']:.1f} minutes
- Failures: {row['failure_count']}
- Spot Ratio: {row['avg_spot_ratio']:.1%}"""


class GetHistoricalBaselineTool(AgentTool):
    """Get historical baseline for comparison"""
    
    def __init__(self, db_path: str):
        super().__init__(
            name="get_historical_baseline",
            description="Get historical cost baseline for an entity (avg, min, max over past 30 days)",
            db_path=db_path
        )
    
    def run(self, entity_id: str, date: str, days_back: int = 30) -> str:
        """
        Args:
            entity_id: The job or compute ID
            date: Reference date (will look back from here)
            days_back: How many days of history to analyze
        """
        sql = """
        WITH historical_costs AS (
            SELECT 
                usage_date,
                SUM(total_cost) as daily_cost
            FROM compute_usage u
            JOIN job_runs r ON u.parent_id = r.job_run_id
            WHERE r.job_id = ?
              AND u.parent_type = 'JOB_RUN'
              AND usage_date < ?
              AND usage_date >= date(?, '-' || ? || ' days')
            GROUP BY usage_date
        )
        SELECT 
            COUNT(*) as days_with_data,
            AVG(daily_cost) as avg_cost,
            MIN(daily_cost) as min_cost,
            MAX(daily_cost) as max_cost,
            (MAX(daily_cost) - MIN(daily_cost)) / NULLIF(AVG(daily_cost), 0) as volatility
        FROM historical_costs
        """
        
        results = self._query_db(sql, (entity_id, date, date, days_back))
        
        if not results or results[0]['days_with_data'] == 0:
            return f"No historical data found for {entity_id}"
        
        row = results[0]
        return f"""Historical Baseline ({days_back} days before {date}):
- Days with data: {row['days_with_data']}
- Average cost: ${row['avg_cost']:.2f}
- Min cost: ${row['min_cost']:.2f}
- Max cost: ${row['max_cost']:.2f}
- Volatility: {row['volatility']:.2f}"""


class CheckEventsAroundDateTool(AgentTool):
    """Check for events (evictions, failures) around a specific date"""
    
    def __init__(self, db_path: str):
        super().__init__(
            name="check_events_around_date",
            description="Check for spot evictions, failures, or other events around a specific date",
            db_path=db_path
        )
    
    def run(self, entity_id: str, date: str, days_window: int = 1) -> str:
        """
        Args:
            entity_id: The job or compute ID
            date: Center date to check
            days_window: Look +/- this many days
        """
        sql = """
        SELECT 
            e.event_type,
            e.event_time,
            e.details,
            ed.eviction_reason,
            ed.cloud_provider_message,
            r.job_run_id,
            r.run_status
        FROM events e
        JOIN compute_usage u ON e.compute_usage_id = u.compute_usage_id
        JOIN job_runs r ON u.parent_id = r.job_run_id
        LEFT JOIN eviction_details ed ON e.eviction_id = ed.eviction_id
        WHERE r.job_id = ?
          AND date(e.event_time) >= date(?, '-' || ? || ' days')
          AND date(e.event_time) <= date(?, '+' || ? || ' days')
        ORDER BY e.event_time
        """
        
        results = self._query_db(sql, (entity_id, date, days_window, date, days_window))
        
        if not results:
            return f"No events found for {entity_id} around {date}"
        
        # Summarize events
        event_counts = {}
        evictions = []
        failures = []
        
        for row in results:
            event_type = row['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event_type == 'SPOT_EVICTION':
                evictions.append({
                    'time': row['event_time'],
                    'reason': row['eviction_reason'],
                    'message': row['cloud_provider_message']
                })
            
            if row['run_status'] == 'FAILED':
                failures.append({
                    'run_id': row['job_run_id'],
                    'time': row['event_time']
                })
        
        summary = f"Events around {date} (±{days_window} days):\n"
        for event_type, count in event_counts.items():
            summary += f"- {event_type}: {count}\n"
        
        if evictions:
            summary += f"\nSpot Evictions ({len(evictions)}):\n"
            for ev in evictions[:3]:  # Show first 3
                summary += f"  - {ev['time']}: {ev['reason']} - {ev['message']}\n"
        
        if failures:
            summary += f"\nFailures ({len(failures)}):\n"
            for fail in failures[:3]:
                summary += f"  - {fail['time']}: {fail['run_id']}\n"
        
        return summary


class CheckConfigChangesTool(AgentTool):
    """Check if job configuration changed recently"""
    
    def __init__(self, db_path: str):
        super().__init__(
            name="check_config_changes",
            description="Check if job configuration (cluster size, spot ratio, instance type) changed recently",
            db_path=db_path
        )
    
    def run(self, entity_id: str, date: str, days_back: int = 7) -> str:
        """
        Args:
            entity_id: The job ID
            date: Reference date
            days_back: How far back to look for changes
        """
        sql = """
        SELECT 
            r.start_time,
            r.worker_instance_type,
            r.min_nodes,
            r.max_nodes,
            r.fixed_nodes,
            r.is_autoscaling_enabled,
            r.spot_ratio,
            r.instance_pool_id
        FROM job_runs r
        WHERE r.job_id = ?
          AND date(r.start_time) >= date(?, '-' || ? || ' days')
          AND date(r.start_time) <= date(?)
        ORDER BY r.start_time
        """
        
        results = self._query_db(sql, (entity_id, date, days_back, date))
        
        if not results:
            return f"No run data found for {entity_id} around {date}"
        
        # Check for config changes
        prev_config = None
        changes = []
        
        for row in results:
            current_config = {
                'instance_type': row['worker_instance_type'],
                'nodes': row['fixed_nodes'] or f"{row['min_nodes']}-{row['max_nodes']}",
                'autoscaling': row['is_autoscaling_enabled'],
                'spot_ratio': row['spot_ratio'],
                'pool': row['instance_pool_id']
            }
            
            if prev_config:
                # Check what changed
                for key in current_config:
                    if current_config[key] != prev_config[key]:
                        changes.append({
                            'time': row['start_time'],
                            'field': key,
                            'old': prev_config[key],
                            'new': current_config[key]
                        })
            
            prev_config = current_config
        
        if not changes:
            return f"No configuration changes detected for {entity_id} in the {days_back} days before {date}"
        
        summary = f"Configuration changes detected:\n"
        for change in changes:
            summary += f"- {change['time']}: {change['field']} changed from {change['old']} to {change['new']}\n"
        
        return summary


class CompareToSimilarJobsTool(AgentTool):
    """Compare performance to similar jobs"""
    
    def __init__(self, db_path: str):
        super().__init__(
            name="compare_to_similar_jobs",
            description="Compare cost/performance metrics to other jobs with similar characteristics",
            db_path=db_path
        )
    
    def run(self, entity_id: str, date: str) -> str:
        """
        Args:
            entity_id: The job ID to compare
            date: The date to analyze
        """
        # First, get the job's characteristics
        job_sql = """
        SELECT 
            j.job_name,
            j.workspace_id,
            r.worker_instance_type,
            AVG(u.total_cost) as avg_cost,
            AVG(r.duration_ms) / 60000.0 as avg_duration_mins
        FROM compute_usage u
        JOIN job_runs r ON u.parent_id = r.job_run_id
        JOIN jobs j ON r.job_id = j.job_id
        WHERE j.job_id = ?
          AND u.usage_date = ?
        GROUP BY j.job_name, j.workspace_id, r.worker_instance_type
        """
        
        job_info = self._query_db(job_sql, (entity_id, date))
        if not job_info:
            return f"No data found for {entity_id} on {date}"
        
        job = job_info[0]
        
        # Find similar jobs (same workspace, similar instance type)
        similar_sql = """
        SELECT 
            j.job_id,
            j.job_name,
            AVG(u.total_cost) as avg_cost,
            AVG(r.duration_ms) / 60000.0 as avg_duration_mins,
            AVG(r.spot_ratio) as avg_spot_ratio
        FROM compute_usage u
        JOIN job_runs r ON u.parent_id = r.job_run_id
        JOIN jobs j ON r.job_id = j.job_id
        WHERE j.workspace_id = ?
          AND r.worker_instance_type = ?
          AND j.job_id != ?
          AND u.usage_date >= date(?, '-30 days')
          AND u.usage_date <= date(?)
        GROUP BY j.job_id, j.job_name
        HAVING AVG(u.total_cost) > 0
        ORDER BY AVG(u.total_cost) DESC
        LIMIT 5
        """
        
        similar_jobs = self._query_db(
            similar_sql, 
            (job['workspace_id'], job['worker_instance_type'], entity_id, date, date)
        )
        
        if not similar_jobs:
            return f"No similar jobs found for comparison"
        
        summary = f"Comparison to similar jobs:\n"
        summary += f"This job: ${job['avg_cost']:.2f}, {job['avg_duration_mins']:.1f} min\n\n"
        summary += f"Similar jobs (same workspace, instance type):\n"
        
        for sj in similar_jobs:
            cost_diff = ((job['avg_cost'] - sj['avg_cost']) / sj['avg_cost'] * 100) if sj['avg_cost'] > 0 else 0
            summary += f"- {sj['job_name']}: ${sj['avg_cost']:.2f} ({cost_diff:+.0f}% vs this job)\n"
        
        return summary


def create_investigation_tools(db_path: str) -> List[AgentTool]:
    """Create all investigation tools"""
    return [
        GetAnomalyDetailsTool(db_path),
        GetHistoricalBaselineTool(db_path),
        CheckEventsAroundDateTool(db_path),
        CheckConfigChangesTool(db_path),
        CompareToSimilarJobsTool(db_path),
    ]


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Assuming db is in data/usage_rag_data.db
    db_path = Path(__file__).parent.parent.parent / "data" / "usage_rag_data.db"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        sys.exit(1)
    
    # Test the tools
    tools = create_investigation_tools(str(db_path))
    
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Test anomaly details tool
    print("\n=== Testing GetAnomalyDetailsTool ===")
    tool = tools[0]
    result = tool.run(entity_id="J-FIN-DAILY", date="2025-12-05")
    print(result)
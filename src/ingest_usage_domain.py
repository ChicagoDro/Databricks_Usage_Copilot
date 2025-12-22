# src/ingest_usage_domain.py

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List

from src.config import USAGE_DB_PATH


# ---------------------------------------------------------------------------
# Core document structure for ingestion
# ---------------------------------------------------------------------------

@dataclass
class RagDoc:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# WORKSPACE DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_workspace_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT workspace_id, workspace_name, account_id, description
            FROM workspace
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"workspace::{r['workspace_id']}",
            text=(
                f"Workspace: {r['workspace_name']} (ID: {r['workspace_id']})\n"
                f"Account ID: {r['account_id']}\n\n"
                f"Description:\n{r['description']}"
            ),
            metadata={
                "type": "workspace",
                "workspace_id": r["workspace_id"],
                "workspace_name": r["workspace_name"],
                "account_id": r["account_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# USER DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_user_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT user_id, name, department, workspace_id
            FROM users_lookup
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"user::{r['user_id']}",
            text=(
                f"User: {r['name']} (ID: {r['user_id']})\n"
                f"Department: {r['department']}\n"
                f"Workspace: {r['workspace_id']}"
            ),
            metadata={
                "type": "user",
                "user_id": r["user_id"],
                "name": r["name"],
                "department": r["department"],
                "workspace_id": r["workspace_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# JOB DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_job_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                j.job_id,
                j.job_name,
                j.description,
                j.tags,
                j.workspace_id,
                w.workspace_name,
                w.account_id
            FROM jobs j
            LEFT JOIN workspace w ON j.workspace_id = w.workspace_id
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"job::{r['job_id']}",
            text=(
                f"Job: {r['job_name']} (ID: {r['job_id']})\n"
                f"Workspace: {r['workspace_name']} ({r['workspace_id']})\n"
                f"Account: {r['account_id']}\n\n"
                f"Description:\n{r['description']}\n\n"
                f"Tags: {r['tags']}"
            ),
            metadata={
                "type": "job",
                "job_id": r["job_id"],
                "job_name": r["job_name"],
                "workspace_id": r["workspace_id"],
                "workspace_name": r["workspace_name"],
                "account_id": r["account_id"],
                "tags_json": r["tags"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# COMPUTE RESOURCE DOCUMENTS (APC + SQL WAREHOUSE)
# ---------------------------------------------------------------------------

def _fetch_compute_resource_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                c.compute_id,
                c.compute_name,
                c.compute_type,
                c.workspace_id,
                w.workspace_name,
                w.account_id
            FROM non_job_compute c
            LEFT JOIN workspace w ON c.workspace_id = w.workspace_id
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"compute::{r['compute_id']}",
            text=(
                f"Compute Resource: {r['compute_name']} (ID: {r['compute_id']})\n"
                f"Type: {r['compute_type']}\n"
                f"Workspace: {r['workspace_name']} ({r['workspace_id']})\n"
                f"Account: {r['account_id']}"
            ),
            metadata={
                "type": "compute_resource",
                "compute_id": r["compute_id"],
                "compute_name": r["compute_name"],
                "compute_type": r["compute_type"],
                "workspace_id": r["workspace_id"],
                "workspace_name": r["workspace_name"],
                "account_id": r["account_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# JOB RUN DOCUMENTS (SUMMARIES)
# ---------------------------------------------------------------------------

def _fetch_job_run_docs(limit=200) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                job_run_id,
                job_id,
                start_time,
                end_time,
                duration_ms,
                run_status,
                error_summary,
                worker_instance_type,
                fixed_nodes,
                min_nodes,
                max_nodes,
                is_autoscaling_enabled,
                spot_ratio
            FROM job_runs
            ORDER BY start_time DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        text = (
            f"Job Run: {r['job_run_id']}\n"
            f"Job ID: {r['job_id']}\n"
            f"Status: {r['run_status']}\n"
            f"Started: {r['start_time']}\n"
            f"Duration: {r['duration_ms']} ms\n"
            f"Instance Type: {r['worker_instance_type']}\n"
            f"Nodes: {r['fixed_nodes'] or f'{r['min_nodes']}-{r['max_nodes']}'}\n"
            f"Autoscaling: {'Yes' if r['is_autoscaling_enabled'] else 'No'}\n"
            f"Spot Ratio: {r['spot_ratio']:.0%}\n"
        )
        if r['error_summary']:
            text += f"Error: {r['error_summary']}\n"

        docs.append(RagDoc(
            doc_id=f"run::{r['job_run_id']}",
            text=text,
            metadata={
                "type": "job_run",
                "job_run_id": r["job_run_id"],
                "job_id": r["job_id"],
                "run_status": r["run_status"],
                "start_time": r["start_time"],
                "duration_ms": r["duration_ms"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# COMPUTE USAGE DOCUMENTS (AGGREGATED)
# ---------------------------------------------------------------------------

def _fetch_compute_usage_docs(limit=100) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                compute_usage_id,
                parent_id,
                parent_type,
                compute_sku,
                dbus_consumed,
                total_cost,
                avg_cpu_utilization,
                avg_memory_gb,
                is_production,
                usage_date
            FROM compute_usage
            ORDER BY total_cost DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"usage::{r['compute_usage_id']}",
            text=(
                f"Compute Usage: {r['compute_usage_id']}\n"
                f"Parent: {r['parent_id']} ({r['parent_type']})\n"
                f"Date: {r['usage_date']}\n"
                f"SKU: {r['compute_sku']}\n"
                f"DBUs: {r['dbus_consumed']:.2f}\n"
                f"Cost: ${r['total_cost']:.2f}\n"
                f"CPU: {r['avg_cpu_utilization']:.1%}\n"
                f"Memory: {r['avg_memory_gb']:.1f} GB\n"
                f"Production: {'Yes' if r['is_production'] else 'No'}"
            ),
            metadata={
                "type": "compute_usage",
                "compute_usage_id": r["compute_usage_id"],
                "parent_id": r["parent_id"],
                "parent_type": r["parent_type"],
                "usage_date": r["usage_date"],
                "total_cost": r["total_cost"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# EVICTION DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_eviction_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                eviction_id,
                cloud_instance_id,
                eviction_time,
                cloud_provider_message,
                eviction_reason,
                spot_price,
                was_retried
            FROM eviction_details
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"eviction::{r['eviction_id']}",
            text=(
                f"Spot Eviction: {r['eviction_id']}\n"
                f"Instance: {r['cloud_instance_id']}\n"
                f"Time: {r['eviction_time']}\n"
                f"Reason: {r['eviction_reason']}\n"
                f"Spot Price: ${r['spot_price']:.3f}\n"
                f"Provider Message: {r['cloud_provider_message']}\n"
                f"Retried: {'Yes' if r['was_retried'] else 'No'}"
            ),
            metadata={
                "type": "eviction",
                "eviction_id": r["eviction_id"],
                "eviction_time": r["eviction_time"],
                "eviction_reason": r["eviction_reason"],
                "was_retried": bool(r["was_retried"]),
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# EVENT DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_event_docs(limit=200) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                event_id,
                compute_usage_id,
                event_time,
                event_type,
                user_id,
                details,
                eviction_id
            FROM events
            ORDER BY event_time DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        text = (
            f"Event: {r['event_id']}\n"
            f"Type: {r['event_type']}\n"
            f"Time: {r['event_time']}\n"
            f"Compute: {r['compute_usage_id']}\n"
        )
        if r['user_id']:
            text += f"User: {r['user_id']}\n"
        if r['details']:
            text += f"Details: {r['details']}\n"
        if r['eviction_id']:
            text += f"Related Eviction: {r['eviction_id']}\n"

        docs.append(RagDoc(
            doc_id=f"event::{r['event_id']}",
            text=text,
            metadata={
                "type": "event",
                "event_id": r["event_id"],
                "event_type": r["event_type"],
                "event_time": r["event_time"],
                "compute_usage_id": r["compute_usage_id"],
                "eviction_id": r["eviction_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# SQL QUERY HISTORY DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_sql_query_docs(limit=100) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                query_id,
                parent_id,
                user_id,
                start_time,
                duration_ms,
                warehouse_sku,
                sql_text,
                error_message
            FROM sql_query_history
            ORDER BY start_time DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        text = (
            f"SQL Query: {r['query_id']}\n"
            f"Warehouse: {r['parent_id']}\n"
            f"User: {r['user_id']}\n"
            f"Start: {r['start_time']}\n"
            f"Duration: {r['duration_ms']} ms\n"
            f"SKU: {r['warehouse_sku']}\n\n"
            f"SQL:\n{r['sql_text'][:500]}\n"  # Truncate long queries
        )
        if r['error_message']:
            text += f"\nError: {r['error_message']}\n"

        docs.append(RagDoc(
            doc_id=f"query::{r['query_id']}",
            text=text,
            metadata={
                "type": "sql_query",
                "query_id": r["query_id"],
                "user_id": r["user_id"],
                "start_time": r["start_time"],
                "duration_ms": r["duration_ms"],
                "has_error": bool(r["error_message"]),
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# MAIN BUILDER
# ---------------------------------------------------------------------------

def build_usage_rag_docs() -> List[RagDoc]:
    """
    Fetch and build all RAG documents from the usage domain.
    
    Returns a list of RagDoc objects ready for embedding and indexing.
    """
    print("[ingest_usage_domain] Building workspace docs...")
    workspace_docs = _fetch_workspace_docs()
    
    print("[ingest_usage_domain] Building user docs...")
    user_docs = _fetch_user_docs()
    
    print("[ingest_usage_domain] Building job docs...")
    job_docs = _fetch_job_docs()
    
    print("[ingest_usage_domain] Building compute resource docs...")
    compute_docs = _fetch_compute_resource_docs()
    
    print("[ingest_usage_domain] Building job run docs...")
    run_docs = _fetch_job_run_docs(limit=200)
    
    print("[ingest_usage_domain] Building compute usage docs...")
    usage_docs = _fetch_compute_usage_docs(limit=100)
    
    print("[ingest_usage_domain] Building eviction docs...")
    eviction_docs = _fetch_eviction_docs()
    
    print("[ingest_usage_domain] Building event docs...")
    event_docs = _fetch_event_docs(limit=200)
    
    print("[ingest_usage_domain] Building SQL query docs...")
    query_docs = _fetch_sql_query_docs(limit=100)
    
    all_docs = (
        workspace_docs +
        user_docs +
        job_docs +
        compute_docs +
        run_docs +
        usage_docs +
        eviction_docs +
        event_docs +
        query_docs
    )
    
    print(f"[ingest_usage_domain] Total docs created: {len(all_docs)}")
    return all_docs
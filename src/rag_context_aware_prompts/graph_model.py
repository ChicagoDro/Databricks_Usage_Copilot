# src/graph_model.py

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.rag_context_aware_prompts.config import USAGE_DB_PATH


# ---------------------------------------------------------------------------
# Core graph data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str            # e.g. "job::J-FIN-DLY"
    type: str          # e.g. "job", "user", "workspace"
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    src: str           # node id
    dst: str           # node id
    type: str          # e.g. "OWNS", "RUN_OF", "EXECUTES_ON"
    properties: Dict[str, Any]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _add_node(nodes: Dict[str, GraphNode], node_id: str, node_type: str, props: Dict[str, Any]) -> None:
    if node_id not in nodes:
        nodes[node_id] = GraphNode(id=node_id, type=node_type, properties=props)


def _add_edge(
    edges: List[GraphEdge],
    src: str,
    dst: str,
    edge_type: str,
    props: Dict[str, Any] | None = None
) -> None:
    edges.append(GraphEdge(src=src, dst=dst, type=edge_type, properties=props or {}))


# ---------------------------------------------------------------------------
# Build nodes & edges from each table
# ---------------------------------------------------------------------------

def _build_workspace(nodes: Dict[str, GraphNode]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT workspace_id, workspace_name, account_id, description
            FROM workspace
        """).fetchall()

    for r in rows:
        nid = f"workspace::{r['workspace_id']}"
        _add_node(
            nodes,
            nid,
            "workspace",
            {
                "workspace_id": r["workspace_id"],
                "workspace_name": r["workspace_name"],
                "account_id": r["account_id"],
                "description": r["description"],
            },
        )


def _build_users(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT user_id, name, workspace_id, department
            FROM users_lookup
        """).fetchall()

    for r in rows:
        user_id = f"user::{r['user_id']}"
        workspace_id = f"workspace::{r['workspace_id']}"

        _add_node(
            nodes,
            user_id,
            "user",
            {
                "user_id": r["user_id"],
                "name": r["name"],
                "workspace_id": r["workspace_id"],
                "department": r["department"],
            },
        )
        # USER -> WORKSPACE (BELONGS_TO)
        _add_edge(edges, user_id, workspace_id, "BELONGS_TO")
        # WORKSPACE -> USER (HAS_USER)
        _add_edge(edges, workspace_id, user_id, "HAS_USER")


def _build_jobs(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                job_id,
                job_name,
                description,
                tags,
                workspace_id
            FROM jobs
        """).fetchall()

    for r in rows:
        job_id = f"job::{r['job_id']}"
        workspace_id = f"workspace::{r['workspace_id']}"

        _add_node(
            nodes,
            job_id,
            "job",
            {
                "job_id": r["job_id"],
                "job_name": r["job_name"],
                "description": r["description"],
                "tags": r["tags"],
                "workspace_id": r["workspace_id"],
            },
        )
        # JOB -> WORKSPACE (OWNED_BY)
        _add_edge(edges, job_id, workspace_id, "OWNED_BY")
        # WORKSPACE -> JOB (HAS_JOB)
        _add_edge(edges, workspace_id, job_id, "HAS_JOB")


def _build_compute_resources(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                compute_id,
                compute_name,
                compute_type,
                workspace_id
            FROM non_job_compute
        """).fetchall()

    for r in rows:
        cid = f"compute::{r['compute_id']}"
        workspace_id = f"workspace::{r['workspace_id']}"

        _add_node(
            nodes,
            cid,
            "compute_resource",
            {
                "compute_id": r["compute_id"],
                "compute_name": r["compute_name"],
                "compute_type": r["compute_type"],
                "workspace_id": r["workspace_id"],
            },
        )
        # COMPUTE_RESOURCE -> WORKSPACE (OWNED_BY)
        _add_edge(edges, cid, workspace_id, "OWNED_BY")
        # WORKSPACE -> COMPUTE_RESOURCE (HAS_COMPUTE)
        _add_edge(edges, workspace_id, cid, "HAS_COMPUTE")


def _build_job_runs(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
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
        """).fetchall()

    for r in rows:
        rid = f"jobrun::{r['job_run_id']}"
        job_id = f"job::{r['job_id']}"

        _add_node(
            nodes,
            rid,
            "job_run",
            {
                "job_run_id": r["job_run_id"],
                "job_id": r["job_id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration_ms": r["duration_ms"],
                "run_status": r["run_status"],
                "error_summary": r["error_summary"],
                "worker_instance_type": r["worker_instance_type"],
                "fixed_nodes": r["fixed_nodes"],
                "min_nodes": r["min_nodes"],
                "max_nodes": r["max_nodes"],
                "is_autoscaling_enabled": r["is_autoscaling_enabled"],
                "spot_ratio": r["spot_ratio"],
            },
        )
        # JOB_RUN -> JOB (RUN_OF)
        _add_edge(edges, rid, job_id, "RUN_OF")
        # JOB -> JOB_RUN (HAS_RUN)
        _add_edge(edges, job_id, rid, "HAS_RUN")


def _build_compute_usage(
    nodes: Dict[str, GraphNode],
    edges: List[GraphEdge],
) -> None:
    """
    compute_usage.parent_id can be either:
      - job_runs.job_run_id  (JOB_RUN)
      - non_job_compute.compute_id (APC_CLUSTER, SQL_WAREHOUSE)
    """
    with _get_conn() as conn:
        # Preload job_run_ids and compute_ids to classify parents
        job_run_ids = {
            row["job_run_id"] for row in conn.execute("SELECT job_run_id FROM job_runs").fetchall()
        }
        compute_ids = {
            row["compute_id"] for row in conn.execute("SELECT compute_id FROM non_job_compute").fetchall()
        }

        rows = conn.execute("""
            SELECT
                compute_usage_id,
                parent_id,
                parent_type,
                compute_sku,
                dbus_consumed,
                cluster_id,
                cluster_instance_type,
                total_cost,
                avg_cpu_utilization,
                avg_memory_gb,
                peak_concurrent_users,
                is_production,
                usage_date
            FROM compute_usage
        """).fetchall()

    for r in rows:
        uid = f"usage::{r['compute_usage_id']}"
        parent_key = r["parent_id"]

        _add_node(
            nodes,
            uid,
            "compute_usage",
            {
                "compute_usage_id": r["compute_usage_id"],
                "parent_id": r["parent_id"],
                "parent_type": r["parent_type"],
                "compute_sku": r["compute_sku"],
                "dbus_consumed": r["dbus_consumed"],
                "cluster_id": r["cluster_id"],
                "cluster_instance_type": r["cluster_instance_type"],
                "total_cost": r["total_cost"],
                "avg_cpu_utilization": r["avg_cpu_utilization"],
                "avg_memory_gb": r["avg_memory_gb"],
                "peak_concurrent_users": r["peak_concurrent_users"],
                "is_production": r["is_production"],
                "usage_date": r["usage_date"],
            },
        )

        # Edge: USAGE -> (JOB_RUN | COMPUTE_RESOURCE)
        if parent_key in job_run_ids:
            parent_node_id = f"jobrun::{parent_key}"
            _add_edge(edges, uid, parent_node_id, "USAGE_OF_JOB_RUN")
            _add_edge(edges, parent_node_id, uid, "HAS_USAGE")
        elif parent_key in compute_ids:
            parent_node_id = f"compute::{parent_key}"
            _add_edge(edges, uid, parent_node_id, "USAGE_OF_COMPUTE")
            _add_edge(edges, parent_node_id, uid, "HAS_USAGE")


def _build_events(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                event_id,
                compute_usage_id,
                event_time,
                event_type,
                user_id,
                details,
                eviction_id
            FROM events
        """).fetchall()

    for r in rows:
        eid = f"event::{r['event_id']}"
        usage_id = f"usage::{r['compute_usage_id']}"

        _add_node(
            nodes,
            eid,
            "event",
            {
                "event_id": r["event_id"],
                "compute_usage_id": r["compute_usage_id"],
                "event_time": r["event_time"],
                "event_type": r["event_type"],
                "user_id": r["user_id"],
                "details": r["details"],
                "eviction_id": r["eviction_id"],
            },
        )

        # EVENT -> USAGE (ON_USAGE)
        _add_edge(edges, eid, usage_id, "ON_USAGE")
        # USAGE -> EVENT (HAS_EVENT)
        _add_edge(edges, usage_id, eid, "HAS_EVENT")

        # EVENT -> USER (TRIGGERED_BY), if present
        if r["user_id"]:
            user_node_id = f"user::{r['user_id']}"
            _add_edge(edges, eid, user_node_id, "TRIGGERED_BY")
            _add_edge(edges, user_node_id, eid, "INITIATED_EVENT")

        # EVENT -> EVICTION (ASSOCIATED_WITH)
        if r["eviction_id"]:
            evict_node_id = f"eviction::{r['eviction_id']}"
            _add_edge(edges, eid, evict_node_id, "ASSOCIATED_EVICTION")
            _add_edge(edges, evict_node_id, eid, "HAS_EVENT")


def _build_evictions(nodes: Dict[str, GraphNode]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                eviction_id,
                cloud_instance_id,
                eviction_time,
                cloud_provider_message,
                eviction_reason,
                spot_price,
                eviction_action,
                was_retried
            FROM eviction_details
        """).fetchall()

    for r in rows:
        eid = f"eviction::{r['eviction_id']}"
        _add_node(
            nodes,
            eid,
            "eviction",
            {
                "eviction_id": r["eviction_id"],
                "cloud_instance_id": r["cloud_instance_id"],
                "eviction_time": r["eviction_time"],
                "cloud_provider_message": r["cloud_provider_message"],
                "eviction_reason": r["eviction_reason"],
                "spot_price": r["spot_price"],
                "eviction_action": r["eviction_action"],
                "was_retried": r["was_retried"],
            },
        )


def _build_sql_queries(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
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
        """).fetchall()

    for r in rows:
        qid = f"query::{r['query_id']}"
        parent_compute_id = f"compute::{r['parent_id']}"

        _add_node(
            nodes,
            qid,
            "sql_query",
            {
                "query_id": r["query_id"],
                "parent_id": r["parent_id"],
                "user_id": r["user_id"],
                "start_time": r["start_time"],
                "duration_ms": r["duration_ms"],
                "warehouse_sku": r["warehouse_sku"],
                "sql_text": r["sql_text"][:500] if r["sql_text"] else None,  # Truncate for storage
                "error_message": r["error_message"],
            },
        )

        # QUERY -> COMPUTE_RESOURCE (EXECUTED_ON)
        _add_edge(edges, qid, parent_compute_id, "EXECUTED_ON")
        _add_edge(edges, parent_compute_id, qid, "RAN_QUERY")

        # QUERY -> USER (SUBMITTED_BY)
        if r["user_id"]:
            user_node_id = f"user::{r['user_id']}"
            _add_edge(edges, qid, user_node_id, "SUBMITTED_BY")
            _add_edge(edges, user_node_id, qid, "SUBMITTED_QUERY")


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_usage_graph() -> Tuple[Dict[str, GraphNode], List[GraphEdge]]:
    """
    Build the complete usage graph from SQLite database.
    
    Returns:
        (nodes, edges) where:
        - nodes: dict of node_id -> GraphNode
        - edges: list of GraphEdge objects
    """
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    print("[graph] Building workspace nodes...")
    _build_workspace(nodes)

    print("[graph] Building user nodes + edges...")
    _build_users(nodes, edges)

    print("[graph] Building job nodes + edges...")
    _build_jobs(nodes, edges)

    print("[graph] Building compute resource nodes + edges...")
    _build_compute_resources(nodes, edges)

    print("[graph] Building job run nodes + edges...")
    _build_job_runs(nodes, edges)

    print("[graph] Building compute usage nodes + edges...")
    _build_compute_usage(nodes, edges)

    print("[graph] Building event nodes + edges...")
    _build_events(nodes, edges)

    print("[graph] Building eviction nodes...")
    _build_evictions(nodes)

    print("[graph] Building SQL query nodes + edges...")
    _build_sql_queries(nodes, edges)

    print(f"[graph] Graph built: {len(nodes)} nodes, {len(edges)} edges")

    return nodes, edges


# ---------------------------------------------------------------------------
# Debug / CLI entry point
# ---------------------------------------------------------------------------

def _print_graph_summary():
    """Simple CLI to inspect the graph structure."""
    nodes, edges = build_usage_graph()
    
    print("\n=== NODE TYPE COUNTS ===")
    type_counts = {}
    for node in nodes.values():
        type_counts[node.type] = type_counts.get(node.type, 0) + 1
    
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type}: {count}")
    
    print("\n=== EDGE TYPE COUNTS ===")
    edge_counts = {}
    for edge in edges:
        edge_counts[edge.type] = edge_counts.get(edge.type, 0) + 1
    
    for edge_type, count in sorted(edge_counts.items()):
        print(f"  {edge_type}: {count}")


if __name__ == "__main__":
    _print_graph_summary()
# Databricks System Tables → Usage Copilot Schema Mapping

## Executive Summary

This document maps your Databricks Usage Copilot schema to Databricks System Tables, identifying:
- ✅ **Available**: Direct mappings from system tables
- ⚠️ **Partial**: Available with transformations or heuristics  
- ❌ **Not Available**: Requires Databricks API, manual input, or synthetic data

**Coverage Summary:**
- **Fully Available**: ~65% of fields
- **Partially Available**: ~20% of fields  
- **Not Available**: ~15% of fields

---

## Table-by-Table Mapping

### 1. workspace

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `workspace_id` | `system.billing.usage.workspace_id` | ✅ Available | Extract DISTINCT workspace IDs |
| `workspace_name` | ❌ Not in system tables | ❌ Not Available | **Requires**: Workspace API `/api/2.0/workspace/list` or manual mapping |
| `account_id` | `system.billing.usage.account_id` | ✅ Available | Available in billing records |
| `description` | ❌ Not in system tables | ❌ Not Available | Manual entry required |

**Recommendation**: Use Databricks Workspace API to populate workspace names, or maintain manual mapping table.

---

### 2. users_lookup

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `user_id` | `system.billing.usage.identity_metadata.run_as` | ✅ Available | Email addresses from billing records |
| `name` | Derived from `user_id` | ⚠️ Partial | Extract from email prefix (e.g., `john.doe@company.com` → `john.doe`) |
| `workspace_id` | `system.billing.usage.workspace_id` | ✅ Available | From billing context |
| `department` | ❌ Not in system tables | ❌ Not Available | **Requires**: HR system integration or SCIM API |

**Recommendation**: Parse names from email addresses, integrate with HR system for departments.

---

### 3. jobs

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `job_id` | `system.lakeflow.jobs.job_id` | ✅ Available | Primary key |
| `workspace_id` | `system.lakeflow.jobs.workspace_id` | ✅ Available | Workspace association |
| `job_name` | `system.lakeflow.jobs.name` | ✅ Available | Job name |
| `description` | ❌ Not in system tables | ❌ Not Available | **Requires**: Jobs API `/api/2.1/jobs/get` |
| `tags` | `system.lakeflow.jobs.tags` | ✅ Available | Struct → JSON conversion |

**Recommendation**: Use `system.lakeflow.jobs` with ROW_NUMBER() to get latest version (SCD2 table).

---

### 4. job_runs

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `job_run_id` | `system.lakeflow.job_run_timeline.run_id` | ✅ Available | Run identifier |
| `job_id` | `system.lakeflow.job_run_timeline.job_id` | ✅ Available | Parent job |
| `start_time` | `system.lakeflow.job_run_timeline.start_time` | ✅ Available | ISO 8601 timestamp |
| `end_time` | `system.lakeflow.job_run_timeline.end_time` | ✅ Available | ISO 8601 timestamp |
| `duration_ms` | Calculated: `end_time - start_time` | ✅ Available | Milliseconds |
| `run_status` | `system.lakeflow.job_run_timeline.result_state` | ✅ Available | Map: `SUCCESS`, `FAILED`, `CANCELED` → `SKIPPED` |
| `error_summary` | `system.lakeflow.job_run_timeline.state_message` | ✅ Available | Error details for failed runs |
| `driver_instance_type` | `system.compute.clusters.driver_node_type_id` | ⚠️ Partial | Requires join via `job_task_run_timeline.compute_ids` |
| `worker_instance_type` | `system.compute.clusters.worker_node_type_id` | ⚠️ Partial | Requires join via `job_task_run_timeline.compute_ids` |
| `is_fleet_cluster` | ❌ Not in system tables | ❌ Not Available | Fleet cluster metadata not exposed |
| `instance_pool_id` | `system.compute.clusters.instance_pool_id` | ⚠️ Partial | Via cluster join |
| `min_nodes` | `system.compute.clusters.autoscale.min_workers` | ⚠️ Partial | Via cluster join; fallback to `num_workers` |
| `max_nodes` | `system.compute.clusters.autoscale.max_workers` | ⚠️ Partial | Via cluster join; fallback to `num_workers` |
| `fixed_nodes` | `system.compute.clusters.num_workers` | ⚠️ Partial | Via cluster join |
| `is_autoscaling_enabled` | `system.compute.clusters.autoscale IS NOT NULL` | ⚠️ Partial | Boolean check |
| `spot_ratio` | ❌ Not directly available | ⚠️ Partial | **Complex**: Calculate from `system.compute.node_timeline` instance patterns or pricing |

**Recommendation**: 
- Join `job_run_timeline` → `job_task_run_timeline` → `compute.clusters`
- Spot ratio requires advanced analysis of `node_timeline` (see supplemental query)

---

### 5. compute_usage

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `compute_usage_id` | Generate: `MD5(record_id + timestamp)` | ✅ Available | Synthetic unique ID |
| `parent_id` | `usage_metadata.{job_run_id, cluster_id, warehouse_id}` | ✅ Available | Conditional on billing type |
| `parent_type` | Derived from `billing_origin_product` | ✅ Available | Map: `JOBS` → `JOB_RUN`, `SQL` → `SQL_WAREHOUSE`, `INTERACTIVE` → `APC_CLUSTER` |
| `compute_sku` | `system.billing.usage.sku_name` | ✅ Available | SKU identifier |
| `dbus_consumed` | `system.billing.usage.usage_quantity` | ✅ Available | DBU consumption |
| `cluster_id` | `usage_metadata.cluster_id` | ✅ Available | Cluster identifier |
| `cluster_instance_type` | ❌ Not in billing table | ⚠️ Partial | Requires join to `compute.clusters` |
| `total_cost` | `usage_quantity * list_prices.pricing.default` | ✅ Available | Join with `system.billing.list_prices` |
| `avg_cpu_utilization` | `system.compute.node_timeline.cpu_utilization_percent` | ⚠️ Partial | Aggregate from node-level data |
| `avg_memory_gb` | `system.compute.node_timeline.used_memory_mb` | ⚠️ Partial | Aggregate from node-level data, convert MB → GB |
| `peak_concurrent_users` | ❌ Not in system tables | ❌ Not Available | Requires Workspace API or audit logs |
| `is_production` | Heuristic from SKU or tags | ⚠️ Partial | Infer from `PREMIUM`/`ENTERPRISE` SKUs or custom tags |
| `usage_date` | `system.billing.usage.usage_date` | ✅ Available | Date partition |

**Recommendation**:
- Cost calculation requires joining `billing.usage` with `billing.list_prices` (see query)
- CPU/memory requires aggregating `node_timeline` (minute-level granularity)

---

### 6. non_job_compute

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `compute_id` | `system.compute.clusters.cluster_id` OR `usage_metadata.warehouse_id` | ✅ Available | Cluster or warehouse ID |
| `compute_name` | `system.compute.clusters.cluster_name` | ⚠️ Partial | Cluster names available; warehouse names NOT available |
| `compute_type` | Derived from `cluster_source` or `billing_origin_product` | ✅ Available | `SQL_WAREHOUSE` or `APC_CLUSTER` |
| `workspace_id` | `system.compute.clusters.workspace_id` | ✅ Available | Workspace association |

**Recommendation**: 
- All-purpose clusters: Use `system.compute.clusters WHERE cluster_source IN ('UI', 'API')`
- SQL Warehouses: Extract from `billing.usage WHERE billing_origin_product = 'SQL'` (no dedicated warehouse table)

---

### 7. instance_pools

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `instance_pool_id` | `system.compute.clusters.instance_pool_id` | ✅ Available | Pool identifier from clusters |
| `pool_name` | ❌ Not in system tables | ❌ Not Available | **Requires**: Instance Pools API `/api/2.0/instance-pools/get` |
| `pool_instance_type` | Infer from `worker_node_type_id` of clusters using pool | ⚠️ Partial | Heuristic from cluster configs |
| `min_size` | ❌ Not in system tables | ❌ Not Available | Pool configuration not exposed |
| `max_size` | ❌ Not in system tables | ❌ Not Available | Pool configuration not exposed |
| `auto_termination_mins` | ❌ Not in system tables | ❌ Not Available | Pool configuration not exposed |

**Recommendation**: Use Instance Pools API for full configuration, or set reasonable defaults.

---

### 8. events

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `event_id` | Generate: `MD5(run_id + timestamp + type)` | ✅ Available | Synthetic ID |
| `run_id` | `system.lakeflow.job_run_timeline.run_id` | ✅ Available | Job run identifier |
| `job_id` | `system.lakeflow.job_run_timeline.job_id` | ✅ Available | Parent job |
| `event_time` | `start_time` or `end_time` | ✅ Available | Derived from state changes |
| `event_type` | Derived from `result_state` | ✅ Available | Map: `SUCCESS` → `TERMINATED`, `FAILED` → `INTERNAL_ERROR` |
| `event_details` | `state_message` | ✅ Available | Error/status message |

**Recommendation**: Generate events from job run state transitions (start → end).

---

### 9. eviction_details

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `eviction_id` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `run_id` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `cluster_id` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `eviction_time` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `cloud_provider_message` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `eviction_reason` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `spot_price` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `eviction_action` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |
| `was_retried` | ❌ Not in system tables | ❌ Not Available | **Not exposed** |

**Limitation**: Spot instance eviction details are **NOT available in Databricks System Tables**.

**Workarounds**:
1. **Cluster Events API**: `/api/2.0/clusters/events` returns cluster lifecycle events including spot terminations
2. **Cloud Provider Logs**: Query CloudWatch (AWS), Azure Monitor, or Cloud Logging (GCP) for spot termination events
3. **Synthetic Data**: Use synthetic data generator for demos/development

---

### 10. sql_query_history

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `query_id` | `system.query.history.statement_id` | ⚠️ Partial | **Requires**: `query.history` schema enabled |
| `parent_id` | `system.query.history.warehouse_id` | ⚠️ Partial | Warehouse ID |
| `user_id` | `system.query.history.user_id` | ⚠️ Partial | User email |
| `start_time` | `system.query.history.start_time` | ⚠️ Partial | Query start timestamp |
| `duration_ms` | `system.query.history.execution_duration` | ⚠️ Partial | Execution time in milliseconds |
| `warehouse_sku` | Join to `billing.usage` via `warehouse_id` | ⚠️ Partial | Requires cross-table join |
| `sql_text` | `system.query.history.statement_text` | ⚠️ Partial | Full SQL query text |
| `error_message` | `system.query.history.error_message` | ⚠️ Partial | NULL if successful |

**Requirement**: `system.query.history` must be **explicitly enabled** by account admin.

**Enable via**:
```sql
-- Must be account admin
ALTER SYSTEM SET system_table.query.history.enabled = true;
```

If not enabled, this table will be empty or unavailable.

---

### 11. date_series

| Copilot Field | System Table Source | Status | Notes |
|--------------|-------------------|--------|-------|
| `date` | Generate via `sequence()` function | ✅ Available | Simple date range generation |

**Recommendation**: Use `sequence(start_date, end_date, INTERVAL 1 DAY)` in SQL.

---

## Additional Data Sources

### Available System Tables Not Currently Used

These system tables exist but aren't directly mapped to your current schema:

1. **`system.compute.node_types`**
   - **Content**: Hardware specifications for all available instance types
   - **Use Case**: Enrich cluster analysis with CPU/memory specs, GPU availability

2. **`system.access.audit`**
   - **Content**: Audit logs of user actions (login, permissions changes, API calls)
   - **Use Case**: Security analysis, user behavior tracking, compliance

3. **`system.lakeflow.pipelines`**
   - **Content**: Delta Live Tables pipeline configurations
   - **Use Case**: DLT-specific cost/performance analysis

4. **`system.lakeflow.pipeline_update_timeline`**
   - **Content**: DLT pipeline run history
   - **Use Case**: Pipeline observability similar to job_run_timeline

---

## Data Not Available in System Tables

### Requires Databricks REST API

| Data | API Endpoint | Auth Required |
|------|-------------|--------------|
| Workspace names | `GET /api/2.0/workspace/list` | Workspace token |
| Job descriptions | `GET /api/2.1/jobs/get` | Workspace token |
| Instance pool configs | `GET /api/2.0/instance-pools/get` | Workspace token |
| Cluster autoscaling details | `GET /api/2.0/clusters/get` | Workspace token |
| SQL warehouse names | `GET /api/2.0/sql/warehouses` | Workspace token |
| Spot eviction events | `GET /api/2.0/clusters/events` | Workspace token |
| User department info | SCIM API `/api/2.0/preview/scim/v2/Users` | Account token |

### Requires External Systems

| Data | Source |
|------|--------|
| User departments | HR system, Active Directory, Okta |
| Cost center attribution | Finance system, ERP |
| Peak concurrent users | Application logs, BI dashboards |
| Fleet cluster indicators | Custom tagging strategy |

---

## Implementation Strategy

### Phase 1: Core Data (Immediate)
✅ Load from system tables with no dependencies:
- `workspace` (IDs only)
- `users_lookup` (emails only)
- `jobs` (from `lakeflow.jobs`)
- `job_runs` (from `lakeflow.job_run_timeline`)
- `compute_usage` (from `billing.usage` + `list_prices`)
- `events` (derived from job state changes)
- `date_series` (generated)

### Phase 2: Enrichment via Joins (Week 1)
⚠️ Add joins for partial data:
- Cluster configurations (`compute.clusters`)
- Node-level metrics (`node_timeline` → CPU/memory)
- SQL query history (`query.history` if enabled)

### Phase 3: API Integration (Week 2)
Use MCP or direct API calls:
- Workspace names
- Job descriptions
- Instance pool configurations
- SQL warehouse details

### Phase 4: Advanced Analytics (Week 3+)
Calculate derived metrics:
- Spot ratio from `node_timeline` instance patterns
- Cost trends and anomaly detection
- Performance regression analysis

---

## SQL Execution Order

```sql
-- 1. Core entities (no dependencies)
CREATE TABLE copilot.workspace ...
CREATE TABLE copilot.users_lookup ...
CREATE TABLE copilot.date_series ...

-- 2. Jobs data
CREATE TABLE copilot.jobs ...
CREATE TABLE copilot.job_runs ...

-- 3. Compute data (requires jobs)
CREATE TABLE copilot.compute_usage ...
CREATE TABLE copilot.non_job_compute ...
CREATE TABLE copilot.instance_pools ...

-- 4. Events (requires job_runs)
CREATE TABLE copilot.events ...

-- 5. SQL queries (if available)
CREATE TABLE copilot.sql_query_history ...

-- 6. Evictions (manual or API)
-- NOT AVAILABLE - use synthetic data or Cluster Events API

-- 7. Supplemental calculations
CREATE TABLE copilot.job_runs_spot_ratio ...  -- Advanced
```

---

## Data Quality Expectations

### Expected Row Counts (90-day window)

| Table | Typical Count | Notes |
|-------|--------------|-------|
| `workspace` | 5-50 | One per workspace |
| `users_lookup` | 50-500 | Active users in billing |
| `jobs` | 100-1000 | Active jobs |
| `job_runs` | 10K-100K | Daily job executions |
| `compute_usage` | 50K-500K | Hourly billing records |
| `non_job_compute` | 10-100 | All-purpose clusters + warehouses |
| `instance_pools` | 5-20 | Instance pools in use |
| `events` | 20K-200K | 2× job_runs (start + end) |
| `eviction_details` | **0** | Not available |
| `sql_query_history` | 100K-1M | If enabled; ad-hoc queries |
| `date_series` | 90 | One per day |

### Data Freshness

| System Table | Update Frequency | Lag |
|-------------|-----------------|-----|
| `billing.usage` | Hourly | 1-2 hours |
| `lakeflow.job_run_timeline` | Real-time | Minutes |
| `compute.clusters` | On change (SCD2) | Minutes |
| `compute.node_timeline` | Minute-level | Minutes |
| `query.history` | Real-time | Minutes |

---

## Gaps Summary

### Critical Gaps (❌ Not Available)
1. **Workspace names** - Requires API
2. **Job descriptions** - Requires API
3. **Spot eviction details** - Requires Cluster Events API or cloud logs
4. **User departments** - Requires HR/SCIM integration
5. **Instance pool configurations** - Requires API
6. **SQL warehouse names** - Requires API
7. **Peak concurrent users** - Not tracked
8. **Fleet cluster indicators** - Not exposed

### Workarounds
1. **For Development/Demos**: Use synthetic data generator (already implemented)
2. **For Production**: Implement MCP Server integration (Phase 2 roadmap)
3. **For Enrichment**: Batch API calls nightly to populate missing fields

---

## Next Steps

1. ✅ **Run SQL script** - Execute `load_from_system_tables.sql` in Databricks SQL Warehouse
2. ✅ **Validate data** - Run quality checks at end of script
3. ⚠️ **Enable query.history** - If SQL analytics needed (account admin required)
4. 📊 **Export to local DB** - Use one of 4 export methods (CSV, Delta, JDBC, Python connector)
5. 🔧 **Implement MCP** - Add real-time API verification for missing fields (Phase 2)
6. 🧪 **Test with real data** - Validate Copilot reports work with production data

---

## Questions & Considerations

1. **Which export method will you use?** (CSV, Delta, JDBC, Python connector)
2. **Is `system.query.history` enabled?** (Check with account admin)
3. **Do you need spot eviction analysis?** (Requires additional API integration)
4. **How frequently will you refresh?** (Daily, weekly, real-time via MCP?)
5. **Which workspaces are in scope?** (All regions or specific workspaces?)
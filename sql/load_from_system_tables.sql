-- =====================================================================================
-- Databricks Usage Copilot: Load from System Tables
-- =====================================================================================
-- Purpose: Transform Databricks System Tables into the Usage Copilot schema
-- Source: system.billing.usage, system.lakeflow.*, system.compute.*
-- Target: Your SQLite/PostgreSQL Copilot schema
--
-- Usage: Run this in a Databricks SQL Warehouse or Notebook
-- Output: Tables ready for export to your Copilot database
-- =====================================================================================

-- =====================================================================================
-- 1. WORKSPACE TABLE
-- =====================================================================================
-- Source: Derived from billing.usage workspace_id values
-- Note: System Tables don't have a dedicated workspace metadata table
-- Workaround: Extract unique workspaces from usage data

CREATE OR REPLACE TABLE copilot.workspace AS
SELECT DISTINCT
    CAST(workspace_id AS STRING) AS workspace_id,
    -- Workspace name not available in system tables - will need to populate separately
    CAST(NULL AS STRING) AS workspace_name,
    -- Account ID available in some system tables
    CAST(account_id AS STRING) AS account_id,
    CAST(NULL AS STRING) AS description
FROM system.billing.usage
WHERE workspace_id IS NOT NULL;

-- MANUAL STEP REQUIRED: Update workspace names via Databricks API or manual mapping
-- Example:
-- UPDATE copilot.workspace SET workspace_name = 'Production' WHERE workspace_id = '12345';


-- =====================================================================================
-- 2. USERS_LOOKUP TABLE
-- =====================================================================================
-- Source: system.billing.usage.identity_metadata
-- Note: Extracts unique users from billing records

CREATE OR REPLACE TABLE copilot.users_lookup AS
WITH unique_users AS (
    SELECT DISTINCT
        identity_metadata.run_as AS user_id,
        workspace_id
    FROM system.billing.usage
    WHERE identity_metadata.run_as IS NOT NULL
        AND identity_metadata.run_as NOT LIKE '%@databricks.com%' -- Filter service accounts
)
SELECT
    user_id,
    -- Name not available - extract from email or set manually
    COALESCE(
        SPLIT(user_id, '@')[0],  -- Use email prefix as name
        user_id
    ) AS name,
    CAST(workspace_id AS STRING) AS workspace_id,
    CAST(NULL AS STRING) AS department  -- Not available in system tables
FROM unique_users;

-- MANUAL STEP REQUIRED: Populate department field via HR system or manual mapping


-- =====================================================================================
-- 3. JOBS TABLE
-- =====================================================================================
-- Source: system.lakeflow.jobs (SCD2 table - need latest version)

CREATE OR REPLACE TABLE copilot.jobs AS
WITH latest_jobs AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY workspace_id, job_id 
            ORDER BY change_time DESC
        ) AS rn
    FROM system.lakeflow.jobs
    WHERE deleted_time IS NULL  -- Exclude deleted jobs
)
SELECT
    CAST(job_id AS STRING) AS job_id,
    CAST(workspace_id AS STRING) AS workspace_id,
    name AS job_name,
    -- Description not directly available
    CAST(NULL AS STRING) AS description,
    -- Tags: Convert struct to JSON string
    TO_JSON(tags) AS tags
FROM latest_jobs
WHERE rn = 1;


-- =====================================================================================
-- 4. JOB_RUNS TABLE
-- =====================================================================================
-- Source: system.lakeflow.job_run_timeline

CREATE OR REPLACE TABLE copilot.job_runs AS
WITH cluster_configs AS (
    -- Get latest cluster config for each cluster used in job runs
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY workspace_id, cluster_id 
            ORDER BY change_time DESC
        ) AS rn
    FROM system.compute.clusters
),
latest_clusters AS (
    SELECT * FROM cluster_configs WHERE rn = 1
),
job_runs_with_clusters AS (
    SELECT
        jr.*,
        c.driver_node_type_id,
        c.worker_node_type_id,
        c.autoscale,
        c.num_workers,
        c.instance_pool_id
    FROM system.lakeflow.job_run_timeline jr
    LEFT JOIN system.lakeflow.job_task_run_timeline jt
        ON jr.workspace_id = jt.workspace_id
        AND jr.job_id = jt.job_id
        AND jr.run_id = jt.run_id
    LEFT JOIN latest_clusters c
        ON jr.workspace_id = c.workspace_id
        AND ARRAY_CONTAINS(jt.compute_ids, c.cluster_id)
)
SELECT
    CAST(run_id AS STRING) AS job_run_id,
    CAST(job_id AS STRING) AS job_id,
    start_time AS start_time,
    end_time AS end_time,
    CAST((end_time - start_time) AS BIGINT) AS duration_ms,  -- In milliseconds
    CASE 
        WHEN result_state = 'SUCCESS' THEN 'SUCCESS'
        WHEN result_state = 'FAILED' THEN 'FAILED'
        WHEN result_state = 'CANCELED' THEN 'SKIPPED'
        WHEN result_state = 'SKIPPED' THEN 'SKIPPED'
        ELSE 'RUNNING'
    END AS run_status,
    state_message AS error_summary,
    driver_node_type_id AS driver_instance_type,
    worker_node_type_id AS worker_instance_type,
    -- Fleet cluster detection (not directly available - heuristic)
    0 AS is_fleet_cluster,
    instance_pool_id,
    -- Autoscaling detection
    CASE 
        WHEN autoscale.min_workers IS NOT NULL THEN autoscale.min_workers
        ELSE num_workers
    END AS min_nodes,
    CASE 
        WHEN autoscale.max_workers IS NOT NULL THEN autoscale.max_workers
        ELSE num_workers
    END AS max_nodes,
    num_workers AS fixed_nodes,
    CASE 
        WHEN autoscale IS NOT NULL THEN 1 
        ELSE 0 
    END AS is_autoscaling_enabled,
    -- Spot ratio: Not directly available - would need to calculate from node_timeline
    0.0 AS spot_ratio
FROM job_runs_with_clusters
WHERE result_state IS NOT NULL;  -- Only completed runs

-- LIMITATION: Spot ratio requires joining with system.compute.node_timeline
-- See separate query below for spot ratio calculation


-- =====================================================================================
-- 5. COMPUTE_USAGE TABLE
-- =====================================================================================
-- Source: system.billing.usage + system.billing.list_prices

CREATE OR REPLACE TABLE copilot.compute_usage AS
WITH usage_with_cost AS (
    SELECT
        u.record_id,
        u.workspace_id,
        u.usage_metadata.job_run_id,
        u.usage_metadata.cluster_id,
        u.usage_metadata.warehouse_id,
        u.sku_name,
        u.usage_quantity AS dbus_consumed,
        u.usage_date,
        u.usage_start_time,
        u.usage_end_time,
        -- Calculate cost by joining with list_prices
        u.usage_quantity * lp.pricing.default AS total_cost,
        u.billing_origin_product,
        u.identity_metadata.run_as
    FROM system.billing.usage u
    INNER JOIN system.billing.list_prices lp
        ON u.cloud = lp.cloud
        AND u.sku_name = lp.sku_name
        AND u.usage_start_time >= lp.price_start_time
        AND (u.usage_end_time <= lp.price_end_time OR lp.price_end_time IS NULL)
    WHERE u.usage_date >= CURRENT_DATE - INTERVAL 90 DAYS
),
-- Aggregate node-level metrics (CPU, memory) from node_timeline
node_metrics AS (
    SELECT
        workspace_id,
        cluster_id,
        DATE(timestamp) AS usage_date,
        AVG(cpu_utilization_percent) AS avg_cpu_utilization,
        AVG(used_memory_mb / 1024.0) AS avg_memory_gb
    FROM system.compute.node_timeline
    WHERE timestamp >= CURRENT_DATE - INTERVAL 90 DAYS
    GROUP BY workspace_id, cluster_id, DATE(timestamp)
)
SELECT
    -- Generate unique ID
    MD5(CONCAT(
        CAST(record_id AS STRING),
        CAST(usage_start_time AS STRING)
    )) AS compute_usage_id,
    -- Determine parent_id and parent_type
    COALESCE(
        CAST(job_run_id AS STRING),
        CAST(cluster_id AS STRING),
        CAST(warehouse_id AS STRING)
    ) AS parent_id,
    CASE
        WHEN job_run_id IS NOT NULL THEN 'JOB_RUN'
        WHEN warehouse_id IS NOT NULL THEN 'SQL_WAREHOUSE'
        WHEN cluster_id IS NOT NULL AND billing_origin_product = 'INTERACTIVE' THEN 'APC_CLUSTER'
        ELSE 'UNKNOWN'
    END AS parent_type,
    sku_name AS compute_sku,
    dbus_consumed,
    CAST(cluster_id AS STRING) AS cluster_id,
    -- Instance type not directly available in usage table
    CAST(NULL AS STRING) AS cluster_instance_type,
    total_cost,
    -- Join with node metrics
    COALESCE(nm.avg_cpu_utilization, 0.0) AS avg_cpu_utilization,
    COALESCE(nm.avg_memory_gb, 0.0) AS avg_memory_gb,
    -- Concurrent users: Not available in system tables
    0 AS peak_concurrent_users,
    -- Production flag: Heuristic based on SKU or tags
    CASE 
        WHEN sku_name LIKE '%PREMIUM%' OR sku_name LIKE '%ENTERPRISE%' THEN 1
        ELSE 0
    END AS is_production,
    CAST(usage_date AS STRING) AS usage_date
FROM usage_with_cost u
LEFT JOIN node_metrics nm
    ON u.workspace_id = nm.workspace_id
    AND u.cluster_id = nm.cluster_id
    AND u.usage_date = nm.usage_date;


-- =====================================================================================
-- 6. NON_JOB_COMPUTE TABLE
-- =====================================================================================
-- Source: system.compute.clusters (for all-purpose) + SQL warehouses

CREATE OR REPLACE TABLE copilot.non_job_compute AS
WITH latest_clusters AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY workspace_id, cluster_id 
            ORDER BY change_time DESC
        ) AS rn
    FROM system.compute.clusters
    WHERE cluster_source IN ('UI', 'API')  -- All-purpose clusters
        AND deleted_time IS NULL
),
all_purpose_clusters AS (
    SELECT
        CAST(cluster_id AS STRING) AS compute_id,
        cluster_name AS compute_name,
        'APC_CLUSTER' AS compute_type,
        CAST(workspace_id AS STRING) AS workspace_id
    FROM latest_clusters
    WHERE rn = 1
),
-- SQL Warehouses: Extract from billing.usage (no dedicated warehouse table)
sql_warehouses AS (
    SELECT DISTINCT
        CAST(usage_metadata.warehouse_id AS STRING) AS compute_id,
        -- Warehouse name not available - use ID
        CONCAT('warehouse-', usage_metadata.warehouse_id) AS compute_name,
        'SQL_WAREHOUSE' AS compute_type,
        CAST(workspace_id AS STRING) AS workspace_id
    FROM system.billing.usage
    WHERE usage_metadata.warehouse_id IS NOT NULL
        AND billing_origin_product = 'SQL'
)
SELECT * FROM all_purpose_clusters
UNION ALL
SELECT * FROM sql_warehouses;


-- =====================================================================================
-- 7. INSTANCE_POOLS TABLE
-- =====================================================================================
-- Source: system.compute.clusters.instance_pool_id
-- Note: No dedicated instance pool table in system tables
-- Workaround: Extract unique pool IDs

CREATE OR REPLACE TABLE copilot.instance_pools AS
SELECT DISTINCT
    CAST(instance_pool_id AS STRING) AS instance_pool_id,
    -- Pool details not available - use ID as name
    CONCAT('pool-', instance_pool_id) AS pool_name,
    -- Instance type from cluster using the pool
    FIRST_VALUE(worker_node_type_id) OVER (
        PARTITION BY instance_pool_id 
        ORDER BY change_time DESC
    ) AS pool_instance_type,
    -- Min/max size not available
    0 AS min_size,
    10 AS max_size,
    30 AS auto_termination_mins
FROM system.compute.clusters
WHERE instance_pool_id IS NOT NULL
    AND deleted_time IS NULL;

-- LIMITATION: Pool configuration details not available in system tables


-- =====================================================================================
-- 8. EVENTS TABLE
-- =====================================================================================
-- Source: Derived from job_run_timeline state changes and cluster lifecycle
-- Note: System tables don't have a dedicated events table

CREATE OR REPLACE TABLE copilot.events AS
WITH job_state_changes AS (
    SELECT
        MD5(CONCAT(
            CAST(run_id AS STRING),
            CAST(start_time AS STRING),
            'JOB_START'
        )) AS event_id,
        CAST(run_id AS STRING) AS run_id,
        CAST(job_id AS STRING) AS job_id,
        start_time AS event_time,
        'RUNNING' AS event_type,
        NULL AS event_details
    FROM system.lakeflow.job_run_timeline
    WHERE start_time IS NOT NULL
    
    UNION ALL
    
    SELECT
        MD5(CONCAT(
            CAST(run_id AS STRING),
            CAST(end_time AS STRING),
            'JOB_END'
        )) AS event_id,
        CAST(run_id AS STRING) AS run_id,
        CAST(job_id AS STRING) AS job_id,
        end_time AS event_time,
        CASE result_state
            WHEN 'SUCCESS' THEN 'TERMINATED'
            WHEN 'FAILED' THEN 'INTERNAL_ERROR'
            ELSE 'TERMINATED'
        END AS event_type,
        state_message AS event_details
    FROM system.lakeflow.job_run_timeline
    WHERE end_time IS NOT NULL
        AND result_state IS NOT NULL
)
SELECT
    event_id,
    run_id,
    job_id,
    event_time,
    event_type,
    event_details
FROM job_state_changes;


-- =====================================================================================
-- 9. EVICTION_DETAILS TABLE
-- =====================================================================================
-- Source: NOT AVAILABLE in Databricks System Tables
-- Note: Spot instance eviction details are not exposed in system tables

CREATE OR REPLACE TABLE copilot.eviction_details AS
SELECT
    CAST(NULL AS STRING) AS eviction_id,
    CAST(NULL AS STRING) AS run_id,
    CAST(NULL AS STRING) AS cluster_id,
    CAST(NULL AS TIMESTAMP) AS eviction_time,
    CAST(NULL AS STRING) AS cloud_provider_message,
    CAST(NULL AS STRING) AS eviction_reason,
    CAST(NULL AS DOUBLE) AS spot_price,
    CAST(NULL AS STRING) AS eviction_action,
    CAST(NULL AS INT) AS was_retried
WHERE 1=0;  -- Empty table

-- LIMITATION: Spot eviction data not available
-- Workaround: Use Databricks Cluster Events API or CloudWatch/Cloud Monitoring


-- =====================================================================================
-- 10. SQL_QUERY_HISTORY TABLE
-- =====================================================================================
-- Source: system.query.history (if enabled)
-- Note: This is a separate system table that may require additional permissions

CREATE OR REPLACE TABLE copilot.sql_query_history AS
SELECT
    CAST(statement_id AS STRING) AS query_id,
    CAST(warehouse_id AS STRING) AS parent_id,
    user_id,
    start_time,
    CAST(execution_duration AS BIGINT) AS duration_ms,
    -- Warehouse SKU from billing.usage
    NULL AS warehouse_sku,
    statement_text AS sql_text,
    error_message
FROM system.query.history
WHERE start_time >= CURRENT_DATE - INTERVAL 90 DAYS;

-- NOTE: system.query.history must be enabled separately
-- If not available, create empty table:
-- WHERE 1=0;


-- =====================================================================================
-- 11. DATE_SERIES TABLE
-- =====================================================================================
-- Helper table for continuous date ranges

CREATE OR REPLACE TABLE copilot.date_series AS
SELECT CAST(date AS STRING) AS date
FROM (
    SELECT sequence(
        CURRENT_DATE - INTERVAL 90 DAYS,
        CURRENT_DATE,
        INTERVAL 1 DAY
    ) AS date_array
)
LATERAL VIEW EXPLODE(date_array) AS date;


-- =====================================================================================
-- SUPPLEMENTAL: Spot Ratio Calculation (Advanced)
-- =====================================================================================
-- Calculate actual spot ratio from node_timeline

CREATE OR REPLACE TABLE copilot.job_runs_spot_ratio AS
WITH node_usage AS (
    SELECT
        nt.workspace_id,
        nt.cluster_id,
        DATE(nt.timestamp) AS usage_date,
        -- Spot detection: instance_id patterns or spot pricing
        AVG(CASE 
            WHEN nt.instance_id LIKE '%spot%' THEN 1.0
            ELSE 0.0
        END) AS spot_ratio
    FROM system.compute.node_timeline nt
    WHERE nt.timestamp >= CURRENT_DATE - INTERVAL 90 DAYS
    GROUP BY nt.workspace_id, nt.cluster_id, DATE(nt.timestamp)
),
job_cluster_mapping AS (
    SELECT DISTINCT
        jr.workspace_id,
        jr.run_id,
        jt.compute_ids
    FROM system.lakeflow.job_run_timeline jr
    INNER JOIN system.lakeflow.job_task_run_timeline jt
        ON jr.workspace_id = jt.workspace_id
        AND jr.job_id = jt.job_id
        AND jr.run_id = jt.run_id
)
SELECT
    CAST(jcm.run_id AS STRING) AS job_run_id,
    AVG(nu.spot_ratio) AS calculated_spot_ratio
FROM job_cluster_mapping jcm
LATERAL VIEW EXPLODE(jcm.compute_ids) AS cluster_id
LEFT JOIN node_usage nu
    ON jcm.workspace_id = nu.workspace_id
    AND cluster_id = nu.cluster_id
GROUP BY jcm.run_id;

-- Use this to update job_runs table:
-- UPDATE copilot.job_runs jr
-- SET spot_ratio = sr.calculated_spot_ratio
-- FROM copilot.job_runs_spot_ratio sr
-- WHERE jr.job_run_id = sr.job_run_id;


-- =====================================================================================
-- EXPORT INSTRUCTIONS
-- =====================================================================================

/*
To export these tables to your local Copilot database:

METHOD 1: Direct Export to CSV
------------------------------
For each table, run:

    COPY (SELECT * FROM copilot.workspace) 
    TO '/dbfs/tmp/copilot_export/workspace.csv' 
    WITH (FORMAT CSV, HEADER TRUE);

Then download from DBFS using Databricks CLI or UI.


METHOD 2: Export to Delta Lake
------------------------------
-- Save as Delta tables for incremental updates

    CREATE OR REPLACE TABLE copilot_delta.workspace
    USING DELTA
    LOCATION '/mnt/copilot/workspace'
    AS SELECT * FROM copilot.workspace;


METHOD 3: JDBC Connection
------------------------------
Connect your local PostgreSQL/SQLite directly via JDBC:

    from pyspark.sql import SparkSession
    
    spark.read.table("copilot.workspace") \
        .write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/copilot") \
        .option("dbtable", "workspace") \
        .option("user", "copilot_user") \
        .option("password", "***") \
        .mode("overwrite") \
        .save()


METHOD 4: Databricks SQL Connector (Python)
------------------------------
    from databricks import sql
    import pandas as pd
    
    connection = sql.connect(
        server_hostname="<workspace>.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/<warehouse_id>",
        access_token="<token>"
    )
    
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM copilot.workspace")
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    
    # Load into SQLite
    import sqlite3
    conn = sqlite3.connect('data/usage_rag_data.db')
    df.to_sql('workspace', conn, if_exists='replace', index=False)
*/


-- =====================================================================================
-- DATA QUALITY CHECKS
-- =====================================================================================

-- Run these queries to validate data completeness

SELECT 'workspace' AS table_name, COUNT(*) AS row_count FROM copilot.workspace
UNION ALL
SELECT 'users_lookup', COUNT(*) FROM copilot.users_lookup
UNION ALL
SELECT 'jobs', COUNT(*) FROM copilot.jobs
UNION ALL
SELECT 'job_runs', COUNT(*) FROM copilot.job_runs
UNION ALL
SELECT 'compute_usage', COUNT(*) FROM copilot.compute_usage
UNION ALL
SELECT 'non_job_compute', COUNT(*) FROM copilot.non_job_compute
UNION ALL
SELECT 'instance_pools', COUNT(*) FROM copilot.instance_pools
UNION ALL
SELECT 'events', COUNT(*) FROM copilot.events
UNION ALL
SELECT 'sql_query_history', COUNT(*) FROM copilot.sql_query_history
UNION ALL
SELECT 'date_series', COUNT(*) FROM copilot.date_series;


-- Check for missing critical fields
SELECT 
    'Jobs without workspace' AS issue,
    COUNT(*) AS count
FROM copilot.jobs
WHERE workspace_id IS NULL

UNION ALL

SELECT 
    'Job runs without duration' AS issue,
    COUNT(*) AS count
FROM copilot.job_runs
WHERE duration_ms IS NULL

UNION ALL

SELECT 
    'Compute usage without cost' AS issue,
    COUNT(*) AS count
FROM copilot.compute_usage
WHERE total_cost IS NULL;
-- Enhanced SQLite Seed Script with Realistic Operational Scenarios
-- Includes: cost spikes, cascading failures, spot eviction clusters, 
-- performance regressions, configuration drift, and messy tagging

-- --------------------------------------------------------------------------------------
-- 1. Define Constants and Date Series
-- --------------------------------------------------------------------------------------

CREATE TEMP TABLE IF NOT EXISTS constants (
    start_date      TEXT,
    end_date        TEXT,
    workspace_id    TEXT,
    cost_per_dbu    REAL
);

DELETE FROM constants;
INSERT INTO constants (start_date, end_date, workspace_id, cost_per_dbu)
VALUES ('2025-11-01', '2025-12-15', 'OU-CHICOLA', 0.40);

CREATE TEMP TABLE IF NOT EXISTS date_series (date TEXT PRIMARY KEY);
DELETE FROM date_series;

WITH RECURSIVE dates (date) AS (
    SELECT start_date FROM constants
    UNION ALL
    SELECT date(date, '+1 day')
    FROM dates
    WHERE date < (SELECT end_date FROM constants)
)
INSERT INTO date_series (date)
SELECT date FROM dates;

-- --------------------------------------------------------------------------------------
-- 2. Insert Core Entities
-- --------------------------------------------------------------------------------------

DELETE FROM workspace;
DELETE FROM users_lookup;
DELETE FROM instance_pools;
DELETE FROM non_job_compute;
DELETE FROM jobs;

INSERT INTO workspace
SELECT workspace_id, 'Chicago Cola Co.', 'CCOL-1234',
       'Leading beverage distributor, heavily focused on optimizing supply chain and finance reporting systems.'
FROM constants;

INSERT INTO users_lookup (user_id, name, workspace_id, department)
SELECT 'U-ALICE',   'Alice Chen',      (SELECT workspace_id FROM constants), 'Finance'      UNION ALL 
SELECT 'U-BOB',     'Bob Martinez',    (SELECT workspace_id FROM constants), 'Supply Chain' UNION ALL 
SELECT 'U-CHARLIE', 'Charlie Kumar',   (SELECT workspace_id FROM constants), 'HR'           UNION ALL 
SELECT 'U-DAVID',   'David Thompson',  (SELECT workspace_id FROM constants), 'Logistics'    UNION ALL 
SELECT 'U-EMILY',   'Emily Rodriguez', (SELECT workspace_id FROM constants), 'Data Science' UNION ALL
SELECT 'U-FRANK',   'Frank Lee',       (SELECT workspace_id FROM constants), 'Supply Chain' UNION ALL
SELECT 'U-SYSTEM',  'System Account',  (SELECT workspace_id FROM constants), 'System';

INSERT INTO instance_pools (instance_pool_id, pool_name, pool_instance_type, min_size, max_size, auto_termination_mins) VALUES
    ('POOL-WORKER-STD', 'Standard Worker Pool', 'i3.xlarge', 2, 8, 30),
    ('POOL-WORKER-GPU', 'GPU Worker Pool', 'g4dn.xlarge', 1, 4, 15);

-- Insert non-job compute (some with missing tags to show messy data)
INSERT INTO non_job_compute (compute_id, compute_name, compute_type, workspace_id)
SELECT 'APC-LOGISTICS', 'Logistics Analytics Cluster', 'APC_CLUSTER',   (SELECT workspace_id FROM constants) UNION ALL
SELECT 'APC-FINANCE',   'Finance Ad-Hoc Cluster',      'APC_CLUSTER',   (SELECT workspace_id FROM constants) UNION ALL
SELECT 'APC-LEGACY',    'Legacy ETL Cluster',          'APC_CLUSTER',   (SELECT workspace_id FROM constants) UNION ALL
SELECT 'WH-FINANCE',    'Finance SQL Warehouse',       'SQL_WAREHOUSE', (SELECT workspace_id FROM constants) UNION ALL
SELECT 'WH-SUPPLY',     'Supply Chain Warehouse',      'SQL_WAREHOUSE', (SELECT workspace_id FROM constants) UNION ALL
SELECT 'WH-HR',         'HR Analytics Warehouse',      'SQL_WAREHOUSE', (SELECT workspace_id FROM constants) UNION ALL
SELECT 'WH-ML',         'ML Experimentation WH',       'SQL_WAREHOUSE', (SELECT workspace_id FROM constants);

-- Insert jobs with varied configuration and tagging quality
INSERT INTO jobs (job_id, workspace_id, job_name, description, tags)
SELECT 'J-FIN-DAILY',    (SELECT workspace_id FROM constants), 
       'Daily Financial Aggregation',
       'Critical end-of-day revenue aggregation. High priority SLA.',
       json_array('Finance', 'Critical', 'Daily', 'SLA-4hr') UNION ALL

SELECT 'J-FIN-MONTHLY',  (SELECT workspace_id FROM constants),
       'Monthly Revenue Rollup',
       'Month-end financial reporting pipeline.',
       json_array('Finance', 'Monthly') UNION ALL

SELECT 'J-SUPPLY-RAW',   (SELECT workspace_id FROM constants),
       'Supply Chain Raw Ingestion',
       'Ingests sensor data from distribution centers. Upstream to transformation jobs.',
       json_array('Supply Chain', 'Ingestion', 'Upstream') UNION ALL

SELECT 'J-SUPPLY-TRANSFORM', (SELECT workspace_id FROM constants),
       'Supply Chain Transformation',
       'Transforms raw supply chain data. DEPENDS ON: J-SUPPLY-RAW',
       json_array('Supply Chain', 'Transformation', 'Downstream') UNION ALL

SELECT 'J-SUPPLY-DASHBOARD', (SELECT workspace_id FROM constants),
       'Supply Chain Dashboard Refresh',
       'Updates operational dashboard. DEPENDS ON: J-SUPPLY-TRANSFORM',
       json_array('Supply Chain', 'Dashboard', 'Downstream') UNION ALL

SELECT 'J-HR-WEEKLY',    (SELECT workspace_id FROM constants),
       'Weekly HR Metrics',
       'Calculates employee metrics for leadership review.',
       json_array('HR') UNION ALL  -- Minimal tags (messy tagging example)

SELECT 'J-ROUTE-OPT',    (SELECT workspace_id FROM constants),
       'Logistics Route Optimizer',
       'ML-based route optimization. High spot usage, prone to evictions.',
       json_array('Logistics', 'Optimization', 'Autoscaling', 'SpotHeavy') UNION ALL

SELECT 'J-ML-TRAINING',  (SELECT workspace_id FROM constants),
       'Product Recommendation Model Training',
       'Trains recommendation model. GPU-heavy, expensive.',
       json_array('DataScience', 'ML', 'GPU', 'Autoscaling', 'Critical') UNION ALL

SELECT 'J-LEGACY-ETL',   (SELECT workspace_id FROM constants),
       'Legacy Customer Data ETL',
       'Old ETL pipeline with inefficient queries. Needs refactor.',
       NULL; -- No tags at all (messy data example)


-- --------------------------------------------------------------------------------------
-- 3. Generate Realistic Job Runs with Specific Scenarios
-- --------------------------------------------------------------------------------------

DELETE FROM job_runs;
DELETE FROM eviction_details;
DELETE FROM compute_usage;
DELETE FROM events;
DELETE FROM sql_query_history;

-- Job configuration with varied characteristics
CREATE TEMP VIEW job_config AS
SELECT 'J-FIN-DAILY' AS job_id, 
       'r5.xlarge' AS worker_instance_type,
       8 AS fixed_nodes, 8 AS min_nodes, 8 AS max_nodes,
       0 AS is_autoscaling_enabled, 
       0 AS is_fleet_cluster,
       NULL AS instance_pool_id, 
       0.65 AS avg_cpu_base, 
       1.2 AS dbus_per_minute,
       30 AS base_runtime_mins,
       0.10 AS base_spot_ratio,
       0.98 AS base_success_rate

UNION ALL SELECT 'J-FIN-MONTHLY', 'r5.2xlarge', 12, 12, 12, 0, 0, NULL, 0.70, 2.0, 90, 0.0, 0.99
UNION ALL SELECT 'J-SUPPLY-RAW', 'i3.xlarge', 5, 5, 5, 0, 1, NULL, 0.40, 0.8, 20, 0.25, 0.95
UNION ALL SELECT 'J-SUPPLY-TRANSFORM', 'r5.xlarge', 6, 6, 12, 1, 0, NULL, 0.55, 1.0, 35, 0.30, 0.93
UNION ALL SELECT 'J-SUPPLY-DASHBOARD', 'i3.xlarge', 4, 4, 4, 0, 0, 'POOL-WORKER-STD', 0.45, 0.7, 15, 0.20, 0.97
UNION ALL SELECT 'J-HR-WEEKLY', 'i3.xlarge', 3, 3, 3, 0, 0, NULL, 0.35, 0.6, 25, 0.15, 0.99
UNION ALL SELECT 'J-ROUTE-OPT', 'r5.2xlarge', 4, 4, 12, 1, 0, NULL, 0.30, 1.5, 45, 0.70, 0.85
UNION ALL SELECT 'J-ML-TRAINING', 'g4dn.xlarge', 6, 6, 12, 1, 0, 'POOL-WORKER-GPU', 0.75, 3.5, 120, 0.50, 0.90
UNION ALL SELECT 'J-LEGACY-ETL', 'r5.xlarge', 10, 10, 10, 0, 0, NULL, 0.85, 2.2, 180, 0.05, 0.80;

-- Generate runs with realistic patterns
CREATE TEMP VIEW run_patterns AS
SELECT 
    ds.date,
    jc.*,
    'RUN-' || jc.job_id || '-' || strftime('%Y%m%d', ds.date) || '-' || 
        CASE 
            WHEN jc.job_id IN ('J-FIN-DAILY', 'J-SUPPLY-RAW', 'J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD') 
            THEN '01'  -- Daily jobs run once
            WHEN jc.job_id = 'J-HR-WEEKLY' AND CAST(strftime('%w', ds.date) AS INTEGER) = 1 
            THEN '01'  -- Weekly on Mondays
            WHEN jc.job_id = 'J-FIN-MONTHLY' AND CAST(strftime('%d', ds.date) AS INTEGER) = 1
            THEN '01'  -- Monthly on 1st
            WHEN jc.job_id IN ('J-ROUTE-OPT', 'J-ML-TRAINING')
            THEN printf('%02d', 1 + (abs(random()) % 3))  -- Multiple runs per day
            WHEN jc.job_id = 'J-LEGACY-ETL'
            THEN '01'  -- Daily legacy job
            ELSE NULL
        END AS run_num
FROM date_series ds
CROSS JOIN job_config jc
WHERE 
    -- Daily jobs
    (jc.job_id IN ('J-FIN-DAILY', 'J-SUPPLY-RAW', 'J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD', 'J-LEGACY-ETL'))
    OR 
    -- Weekly jobs (Mondays only)
    (jc.job_id = 'J-HR-WEEKLY' AND CAST(strftime('%w', ds.date) AS INTEGER) = 1)
    OR
    -- Monthly jobs (1st of month only)
    (jc.job_id = 'J-FIN-MONTHLY' AND CAST(strftime('%d', ds.date) AS INTEGER) = 1)
    OR
    -- Multiple-run-per-day jobs
    (jc.job_id IN ('J-ROUTE-OPT', 'J-ML-TRAINING'));

-- Add scenario-specific anomalies
CREATE TEMP VIEW run_calculations AS
SELECT 
    rp.*,
    rp.job_id || '-' || rp.date || '-R' || rp.run_num AS job_run_id,
    datetime(rp.date, 
        CASE rp.job_id
            WHEN 'J-FIN-DAILY' THEN '+02:00'
            WHEN 'J-FIN-MONTHLY' THEN '+00:00'
            WHEN 'J-SUPPLY-RAW' THEN '+01:00'
            WHEN 'J-SUPPLY-TRANSFORM' THEN '+02:30'
            WHEN 'J-SUPPLY-DASHBOARD' THEN '+03:30'
            WHEN 'J-HR-WEEKLY' THEN '+08:00'
            WHEN 'J-ROUTE-OPT' THEN '+' || printf('%02d', 6 + (abs(random()) % 12)) || ':00'
            WHEN 'J-ML-TRAINING' THEN '+' || printf('%02d', abs(random()) % 24) || ':00'
            WHEN 'J-LEGACY-ETL' THEN '+04:00'
        END,
        '+' || CAST(abs(random() % 30) AS TEXT) || ' minutes'
    ) AS start_time_str,
    
    -- Scenario 1: COST SPIKE on Dec 5-7 for J-FIN-DAILY (initial run fails, will be retried)
    CASE 
        WHEN rp.job_id = 'J-FIN-DAILY' AND rp.date BETWEEN '2025-12-05' AND '2025-12-07'
        THEN 'FAILED'
        
        -- Scenario 2: CASCADING FAILURE on Nov 15 (upstream failure propagates)
        WHEN rp.job_id = 'J-SUPPLY-RAW' AND rp.date = '2025-11-15'
        THEN 'FAILED'
        WHEN rp.job_id IN ('J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD') AND rp.date = '2025-11-15'
        THEN 'SKIPPED'
        
        -- Scenario 3: SPOT EVICTION CLUSTER on Nov 28-30 for route optimizer
        -- Half the runs fail during this period (using combination of date and run_num)
        WHEN rp.job_id = 'J-ROUTE-OPT' AND rp.date BETWEEN '2025-11-28' AND '2025-11-30'
             AND ((julianday(rp.date) * 100 + CAST(substr(rp.run_num, 1, 2) AS INTEGER)) % 2) = 0
        THEN 'FAILED'
        
        -- Scenario 4: ML Training performance degradation starting Dec 1 (config change)
        -- One third of runs fail after this date
        WHEN rp.job_id = 'J-ML-TRAINING' AND rp.date >= '2025-12-01'
             AND ((julianday(rp.date) * 100 + CAST(substr(rp.run_num, 1, 2) AS INTEGER)) % 3) = 0
        THEN 'FAILED'
        
        -- Scenario 5: Legacy job chronic unreliability
        WHEN rp.job_id = 'J-LEGACY-ETL' AND (julianday(rp.date) - julianday('2025-11-01')) % 5 < 1  -- 20% failure rate
        THEN 'FAILED'
        
        -- Normal occasional failures for other jobs
        WHEN rp.job_id NOT IN ('J-FIN-DAILY', 'J-SUPPLY-RAW', 'J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD', 
                                'J-ROUTE-OPT', 'J-ML-TRAINING', 'J-LEGACY-ETL')
             AND (julianday(rp.date) - julianday('2025-11-01')) % 20 < 1  -- 5% baseline failure rate
        THEN 'FAILED'
        
        ELSE 'SUCCESS'
    END AS run_status,
    
    -- Runtime calculation with realistic variance and anomalies
    CAST(
        rp.base_runtime_mins * 60 * 1000 *  -- Convert to milliseconds
        CASE
            -- Performance regression for ML training after Dec 1
            WHEN rp.job_id = 'J-ML-TRAINING' AND rp.date >= '2025-12-01'
            THEN 1.4  -- 40% slower
            
            -- Legacy job gets slower over time (data growth)
            WHEN rp.job_id = 'J-LEGACY-ETL'
            THEN 1.0 + (julianday(rp.date) - julianday('2025-11-01')) * 0.01
            
            -- Normal variance
            ELSE 1.0
        END *
        (0.85 + (abs(random()) % 30) / 100.0)  -- ±15% random variance
        AS INTEGER
    ) AS base_duration_ms,
    
    -- Spot ratio scenarios
    CASE
        WHEN rp.job_id = 'J-ROUTE-OPT' AND rp.date BETWEEN '2025-11-28' AND '2025-11-30'
        THEN 0.85  -- High spot during eviction cluster
        WHEN rp.job_id = 'J-ML-TRAINING' AND rp.date >= '2025-12-01'
        THEN 0.65  -- Increased spot after config change
        ELSE rp.base_spot_ratio
    END AS actual_spot_ratio,
    
    -- Error summary for failed runs (matches run_status conditions)
    CASE 
        -- Finance daily spot evictions during spike period (Dec 5-7)
        WHEN (rp.job_id = 'J-FIN-DAILY' AND rp.date BETWEEN '2025-12-05' AND '2025-12-07')
        THEN 'SpotInstanceTerminated: Instance reclaimed during peak usage'
        
        -- Supply chain upstream failure
        WHEN (rp.job_id = 'J-SUPPLY-RAW' AND rp.date = '2025-11-15')
        THEN 'IOException: Connection timeout to source database'
        
        -- Downstream skipped jobs
        WHEN (rp.job_id IN ('J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD') AND rp.date = '2025-11-15')
        THEN 'Upstream dependency failed - run skipped'
        
        -- Route optimizer spot evictions during cluster period (Nov 28-30)
        WHEN (rp.job_id = 'J-ROUTE-OPT' AND rp.date BETWEEN '2025-11-28' AND '2025-11-30'
              AND ((julianday(rp.date) * 100 + CAST(substr(rp.run_num, 1, 2) AS INTEGER)) % 2) = 0)
        THEN 'SpotInstanceTerminated: AWS EC2 capacity insufficient'
        
        -- ML training failures after Dec 1 (performance regression)
        WHEN (rp.job_id = 'J-ML-TRAINING' AND rp.date >= '2025-12-01'
              AND ((julianday(rp.date) * 100 + CAST(substr(rp.run_num, 1, 2) AS INTEGER)) % 3) = 0)
        THEN 'SpotInstanceTerminated: Insufficient GPU capacity'
        
        -- Legacy job chronic failures
        WHEN (rp.job_id = 'J-LEGACY-ETL' AND (julianday(rp.date) - julianday('2025-11-01')) % 5 < 1)
        THEN 'QueryExecutionException: Query timeout after 3 hours'
        
        -- Other occasional failures
        WHEN (rp.job_id NOT IN ('J-FIN-DAILY', 'J-SUPPLY-RAW', 'J-SUPPLY-TRANSFORM', 'J-SUPPLY-DASHBOARD', 
                                'J-ROUTE-OPT', 'J-ML-TRAINING', 'J-LEGACY-ETL')
             AND (julianday(rp.date) - julianday('2025-11-01')) % 20 < 1)
        THEN 'Unexpected job failure - see error logs'
        
        ELSE NULL
    END AS error_summary

FROM run_patterns rp;

-- Generate retry runs for failures
CREATE TEMP VIEW retry_runs AS
SELECT 
    rc.*,
    retry_num
FROM run_calculations rc
CROSS JOIN (SELECT '02' AS retry_num UNION ALL SELECT '03' UNION ALL SELECT '04') retries
WHERE rc.run_status = 'FAILED'
  AND rc.job_id IN ('J-FIN-DAILY', 'J-ML-TRAINING', 'J-LEGACY-ETL');  -- Only certain jobs retry

-- Insert all runs (initial + retries)
INSERT INTO job_runs
SELECT 
    job_run_id,
    job_id,
    start_time_str AS start_time,
    datetime(start_time_str, '+' || CAST(
        CASE WHEN run_status = 'SUCCESS' THEN base_duration_ms
             WHEN run_status = 'SKIPPED' THEN 0
             ELSE CAST(base_duration_ms * 0.3 AS INTEGER)  -- Failures fail faster
        END / 1000 AS TEXT
    ) || ' seconds') AS end_time,
    CASE WHEN run_status = 'SUCCESS' THEN base_duration_ms
         WHEN run_status = 'SKIPPED' THEN 0
         ELSE CAST(base_duration_ms * 0.3 AS INTEGER)
    END AS duration_ms,
    run_status,
    error_summary,  -- Now comes from run_calculations
    'm5.xlarge' AS driver_instance_type,
    worker_instance_type,
    is_fleet_cluster,
    instance_pool_id,
    min_nodes,
    CASE 
        WHEN is_autoscaling_enabled = 1 
        THEN min_nodes + CAST((max_nodes - min_nodes) * (0.3 + (abs(random()) % 50) / 100.0) AS INTEGER)
        ELSE max_nodes
    END AS max_nodes,
    fixed_nodes,
    is_autoscaling_enabled,
    actual_spot_ratio AS spot_ratio
FROM run_calculations

UNION ALL

-- Add retry runs
SELECT 
    job_id || '-' || date || '-R' || retry_num AS job_run_id,
    job_id,
    datetime(start_time_str, '+' || CAST((base_duration_ms / 1000 + 300) * (CAST(retry_num AS INTEGER) - 1) AS TEXT) || ' seconds') AS start_time,
    datetime(start_time_str, '+' || CAST((base_duration_ms / 1000 + 300) * CAST(retry_num AS INTEGER) AS TEXT) || ' seconds') AS end_time,
    CASE WHEN retry_num = '04' THEN base_duration_ms 
         ELSE CAST(base_duration_ms * 0.3 AS INTEGER) 
    END AS duration_ms,
    CASE WHEN retry_num = '04' THEN 'SUCCESS' ELSE 'FAILED' END AS run_status,
    CASE WHEN retry_num = '04' THEN NULL ELSE error_summary END AS error_summary,
    'm5.xlarge' AS driver_instance_type,
    worker_instance_type,
    is_fleet_cluster,
    instance_pool_id,
    min_nodes,
    max_nodes,
    fixed_nodes,
    is_autoscaling_enabled,
    actual_spot_ratio AS spot_ratio
FROM retry_runs;

-- --------------------------------------------------------------------------------------
-- 4. Insert Eviction Details
-- --------------------------------------------------------------------------------------

INSERT INTO eviction_details
SELECT 
    'EVICT-' || jr.job_run_id AS eviction_id,
    jr.worker_instance_type || '-i-' || substr(lower(hex(randomblob(8))), 1, 12) AS cloud_instance_id,
    datetime(jr.start_time, '+' || CAST(jr.duration_ms * 0.6 / 1000 AS INTEGER) || ' seconds') AS eviction_time,
    CASE 
        WHEN jr.job_id IN ('J-ROUTE-OPT', 'J-ML-TRAINING') AND jr.start_time BETWEEN '2025-11-28' AND '2025-11-30 23:59:59'
        THEN 'Server.InsufficientInstanceCapacity: Insufficient capacity in availability zone'
        WHEN jr.job_id = 'J-FIN-DAILY' AND jr.start_time BETWEEN '2025-12-05' AND '2025-12-07 23:59:59'
        THEN 'Server.SpotInstanceTermination: Spot price exceeded your maximum bid price'
        ELSE 'Server.SpotInstanceInterruption: Spot instance interrupted for capacity requirements'
    END AS cloud_provider_message,
    CASE
        WHEN abs(random()) % 100 < 70 THEN 'CAPACITY_CHANGE'
        ELSE 'PRICE_CHANGE'
    END AS eviction_reason,
    0.08 + (abs(random()) % 50) / 1000.0 AS spot_price,
    'STOP_DEALLOCATE' AS eviction_action,
    CASE WHEN jr.job_id IN ('J-FIN-DAILY', 'J-ML-TRAINING') THEN 1 ELSE 0 END AS was_retried
FROM job_runs jr
WHERE jr.run_status = 'FAILED' 
  AND jr.error_summary LIKE '%SpotInstance%';

-- --------------------------------------------------------------------------------------
-- 5. Insert Compute Usage (with cost calculations)
-- --------------------------------------------------------------------------------------

INSERT INTO compute_usage
SELECT 
    jr.job_run_id AS compute_usage_id,
    jr.job_run_id AS parent_id,
    'JOB_RUN' AS parent_type,
    'STANDARD_JOB_COMPUTE' AS compute_sku,
    CAST(
        jc.dbus_per_minute * (jr.duration_ms / 60000.0) * 
        ((jr.min_nodes + jr.max_nodes) / 2.0) *
        CASE 
            -- Retries consume resources inefficiently
            WHEN jr.job_run_id LIKE '%-R02' OR jr.job_run_id LIKE '%-R03' THEN 1.2
            ELSE 1.0
        END
        AS REAL
    ) AS dbus_consumed,
    jr.worker_instance_type || '-cluster-' || substr(lower(hex(randomblob(4))), 1, 8) AS cluster_id,
    jr.worker_instance_type AS cluster_instance_type,
    CAST(
        jc.dbus_per_minute * (jr.duration_ms / 60000.0) * 
        ((jr.min_nodes + jr.max_nodes) / 2.0) *
        CASE 
            WHEN jr.job_run_id LIKE '%-R02' OR jr.job_run_id LIKE '%-R03' THEN 1.2
            ELSE 1.0
        END *
        (SELECT cost_per_dbu FROM constants)
        AS REAL
    ) AS total_cost,
    jc.avg_cpu_base + (abs(random()) % 20 - 10) / 100.0 AS avg_cpu_utilization,
    CASE jr.worker_instance_type
        WHEN 'g4dn.xlarge' THEN 256.0
        WHEN 'r5.2xlarge' THEN 128.0
        ELSE 64.0
    END AS avg_memory_gb,
    1 + abs(random()) % 5 AS peak_concurrent_users,
    CASE WHEN jr.job_id IN ('J-FIN-DAILY', 'J-FIN-MONTHLY', 'J-ML-TRAINING') THEN 1 ELSE 0 END AS is_production,
    date(jr.start_time) AS usage_date
FROM job_runs jr
JOIN job_config jc ON jr.job_id = jc.job_id
WHERE jr.run_status != 'SKIPPED';

-- Insert APC cluster usage
INSERT INTO compute_usage
SELECT 
    nc.compute_id || '-' || strftime('%Y%m%d', ds.date) AS compute_usage_id,
    nc.compute_id AS parent_id,
    nc.compute_type AS parent_type,
    'STANDARD_ALL_PURPOSE_COMPUTE' AS compute_sku,
    CAST(
        (300 + abs(random()) % 200) *
        CASE 
            -- Legacy cluster shows high usage (inefficient)
            WHEN nc.compute_id = 'APC-LEGACY' THEN 1.5
            -- Finance cluster has spike on month-end
            WHEN nc.compute_id = 'APC-FINANCE' AND CAST(strftime('%d', ds.date) AS INTEGER) = 1 THEN 2.0
            ELSE 1.0
        END
        AS REAL
    ) AS dbus_consumed,
    nc.compute_id AS cluster_id,
    CASE nc.compute_id
        WHEN 'APC-LEGACY' THEN 'i3.2xlarge'
        ELSE 'r5.xlarge'
    END AS cluster_instance_type,
    CAST(
        (300 + abs(random()) % 200) *
        CASE 
            WHEN nc.compute_id = 'APC-LEGACY' THEN 1.5
            WHEN nc.compute_id = 'APC-FINANCE' AND CAST(strftime('%d', ds.date) AS INTEGER) = 1 THEN 2.0
            ELSE 1.0
        END *
        (SELECT cost_per_dbu FROM constants)
        AS REAL
    ) AS total_cost,
    0.45 + (abs(random()) % 40) / 100.0 AS avg_cpu_utilization,
    128.0 AS avg_memory_gb,
    2 + abs(random()) % 4 AS peak_concurrent_users,
    CASE WHEN nc.compute_id = 'APC-FINANCE' THEN 1 ELSE 0 END AS is_production,
    ds.date AS usage_date
FROM date_series ds
CROSS JOIN non_job_compute nc
WHERE nc.compute_type = 'APC_CLUSTER';

-- Insert SQL Warehouse usage
INSERT INTO compute_usage
SELECT 
    nc.compute_id || '-' || strftime('%Y%m%d', ds.date) AS compute_usage_id,
    nc.compute_id AS parent_id,
    nc.compute_type AS parent_type,
    'STANDARD_SQL_WAREHOUSE_COMPUTE' AS compute_sku,
    CAST(250 + abs(random()) % 150 AS REAL) AS dbus_consumed,
    nc.compute_id AS cluster_id,
    CASE 
        WHEN nc.compute_id IN ('WH-FINANCE', 'WH-SUPPLY') THEN 'L'
        WHEN nc.compute_id = 'WH-ML' THEN 'XL'
        ELSE 'M'
    END AS cluster_instance_type,
    CAST((250 + abs(random()) % 150) * (SELECT cost_per_dbu FROM constants) AS REAL) AS total_cost,
    0.35 + (abs(random()) % 30) / 100.0 AS avg_cpu_utilization,
    CASE 
        WHEN nc.compute_id IN ('WH-FINANCE', 'WH-SUPPLY') THEN 256.0
        WHEN nc.compute_id = 'WH-ML' THEN 512.0
        ELSE 128.0
    END AS avg_memory_gb,
    3 + abs(random()) % 8 AS peak_concurrent_users,
    CASE WHEN nc.compute_id IN ('WH-FINANCE', 'WH-SUPPLY') THEN 1 ELSE 0 END AS is_production,
    ds.date AS usage_date
FROM date_series ds
CROSS JOIN non_job_compute nc
WHERE nc.compute_type = 'SQL_WAREHOUSE';

-- --------------------------------------------------------------------------------------
-- 6. Insert Events
-- --------------------------------------------------------------------------------------

INSERT INTO events
-- Cluster start events
SELECT
    'EVT-START-' || jr.job_run_id AS event_id,
    jr.job_run_id AS compute_usage_id,
    jr.start_time AS event_time,
    'CLUSTER_START' AS event_type,
    'U-SYSTEM' AS user_id,
    'Cluster starting for job: ' || jr.job_id || ' with ' || 
        jr.min_nodes || ' to ' || jr.max_nodes || ' nodes' AS details,
    NULL AS eviction_id
FROM job_runs jr

UNION ALL

-- Cluster terminate events (successful runs)
SELECT
    'EVT-END-' || jr.job_run_id,
    jr.job_run_id,
    jr.end_time,
    'CLUSTER_TERMINATE',
    'U-SYSTEM',
    'Cluster terminated successfully after ' || 
        CAST(jr.duration_ms / 60000.0 AS INTEGER) || ' minutes',
    NULL
FROM job_runs jr
WHERE jr.run_status = 'SUCCESS'

UNION ALL

-- Spot eviction events
SELECT
    'EVT-EVICT-' || jr.job_run_id,
    jr.job_run_id,
    ed.eviction_time,
    'SPOT_EVICTION',
    'U-SYSTEM',
    'Spot instance evicted: ' || ed.cloud_provider_message,
    ed.eviction_id
FROM job_runs jr
JOIN eviction_details ed ON 'EVICT-' || jr.job_run_id = ed.eviction_id

UNION ALL

-- Autoscaling events for autoscaling jobs
SELECT
    'EVT-SCALE-' || jr.job_run_id || '-' || CAST(abs(random()) % 10 AS TEXT),
    jr.job_run_id,
    datetime(jr.start_time, '+' || CAST(abs(random()) % (jr.duration_ms / 1000) AS TEXT) || ' seconds'),
    'AUTOSCALING',
    'U-SYSTEM',
    'Cluster scaled from ' || jr.min_nodes || ' to ' || 
        CAST(jr.min_nodes + abs(random()) % (jr.max_nodes - jr.min_nodes + 1) AS TEXT) || ' nodes',
    NULL
FROM job_runs jr
WHERE jr.is_autoscaling_enabled = 1 
  AND jr.run_status = 'SUCCESS'
  AND abs(random()) % 100 < 60;  -- 60% of autoscaling runs have scaling events

-- --------------------------------------------------------------------------------------
-- 7. Insert SQL Query History
-- --------------------------------------------------------------------------------------

CREATE TEMP VIEW adhoc_queries AS
SELECT 1 AS q_id, 
       'WH-FINANCE' AS wh,
       'SELECT revenue, region, product FROM sales_daily WHERE date >= ''2025-11-01'' ORDER BY revenue DESC LIMIT 100' AS sql_text,
       'U-ALICE' AS user_id, 
       'SQLWH-SMALL' AS warehouse_sku, 
       450 AS duration_ms, 
       NULL AS error_message
UNION ALL SELECT 2, 'WH-FINANCE',
       'SELECT SUM(revenue) as total, date FROM sales_daily WHERE region IN (''NORTH'', ''SOUTH'', ''EAST'', ''WEST'') GROUP BY date',
       'U-ALICE', 'SQLWH-SMALL', 720, NULL
UNION ALL SELECT 3, 'WH-SUPPLY',
       'SELECT * FROM logistics_sensor_data WHERE temperature > 80 AND humidity > 70',
       'U-BOB', 'SQLWH-MEDIUM', 1200, NULL
UNION ALL SELECT 4, 'WH-SUPPLY',
       'CREATE TABLE supply_chain_analysis AS SELECT depot_id, AVG(delivery_time) as avg_time FROM shipments GROUP BY depot_id',
       'U-FRANK', 'SQLWH-MEDIUM', 2400, NULL
UNION ALL SELECT 5, 'WH-HR',
       'SELECT employee_id, department, salary, tenure FROM employees WHERE tenure > 5 AND salary > 75000',
       'U-CHARLIE', 'SQLWH-LARGE', 900, NULL
UNION ALL SELECT 6, 'WH-HR',
       'SELECT * FROM employee_metrics WHERE date >= ''2025-11-01''',
       'U-CHARLIE', 'SQLWH-SMALL', 8500, 'Query exceeded warehouse timeout (180s)'
UNION ALL SELECT 7, 'WH-ML',
       'SELECT feature_vector, label FROM ml_training_data WHERE split = ''train'' ORDER BY RANDOM() LIMIT 10000000',
       'U-EMILY', 'SQLWH-XL', 15000, NULL
UNION ALL SELECT 8, 'WH-FINANCE',
       'SELECT * FROM (SELECT * FROM transactions WHERE amount > 10000) t1 JOIN (SELECT * FROM customers) t2 ON t1.customer_id = t2.id',
       'U-ALICE', 'SQLWH-SMALL', 12000, 'Query killed: Result set exceeded 10GB limit';

INSERT INTO sql_query_history (query_id, parent_id, user_id, start_time, duration_ms, warehouse_sku, sql_text, error_message)
SELECT
    'Q-ADHOC-' || printf('%04d', 
        (julianday(ds.date) - julianday((SELECT start_date FROM constants))) * 8 + q_id
    ) AS query_id,
    wh AS parent_id,
    user_id,
    datetime(ds.date, '+08:00', '+' || CAST((q_id * 15) % 480 AS TEXT) || ' minutes') AS start_time,
    duration_ms + abs(random()) % 500 AS duration_ms,
    warehouse_sku,
    sql_text,
    error_message
FROM date_series ds
CROSS JOIN adhoc_queries
WHERE abs(random()) % 100 < 15;  -- ~15% of days have ad-hoc queries

-- --------------------------------------------------------------------------------------
-- Summary of Scenarios Embedded in Data
-- --------------------------------------------------------------------------------------
-- 1. COST SPIKE: J-FIN-DAILY has 3 retries/day on Dec 5-7 due to spot evictions
-- 2. CASCADING FAILURE: J-SUPPLY-RAW fails on Nov 15, blocking downstream jobs
-- 3. SPOT EVICTION CLUSTER: J-ROUTE-OPT has 40% eviction rate Nov 28-30
-- 4. PERFORMANCE REGRESSION: J-ML-TRAINING becomes 40% slower + 15% failure rate after Dec 1
-- 5. CHRONIC UNRELIABILITY: J-LEGACY-ETL has 20% baseline failure rate + increasing runtime
-- 6. MESSY TAGGING: Some jobs have minimal/no tags
-- 7. CONFIGURATION DRIFT: Autoscaling behavior varies across jobs
-- 8. AD-HOC QUERY ISSUES: Some queries timeout or exceed limits

-- --------------------------------------------------------------------------------------
-- DATA VALIDATION: Row counts for each table
-- --------------------------------------------------------------------------------------

.print ''
.print '========================================='
.print 'DATA VALIDATION - ROW COUNTS'
.print '========================================='
.print ''

.print 'Core Entities:'
SELECT 'workspace: ' || COUNT(*) || ' rows' FROM workspace;
SELECT 'users_lookup: ' || COUNT(*) || ' rows' FROM users_lookup;
SELECT 'instance_pools: ' || COUNT(*) || ' rows' FROM instance_pools;
SELECT 'non_job_compute: ' || COUNT(*) || ' rows' FROM non_job_compute;
SELECT 'jobs: ' || COUNT(*) || ' rows' FROM jobs;

.print ''
.print 'Job Execution Data:'
SELECT 'job_runs: ' || COUNT(*) || ' rows' FROM job_runs;
SELECT 'eviction_details: ' || COUNT(*) || ' rows' FROM eviction_details;
SELECT 'compute_usage: ' || COUNT(*) || ' rows' FROM compute_usage;
SELECT 'events: ' || COUNT(*) || ' rows' FROM events;
SELECT 'sql_query_history: ' || COUNT(*) || ' rows' FROM sql_query_history;

.print ''
.print 'Helper Tables:'
SELECT 'date_series: ' || COUNT(*) || ' rows (should be 45 days)' FROM date_series;

.print ''
.print '========================================='
.print 'JOB RUNS BREAKDOWN BY JOB'
.print '========================================='
SELECT job_id, COUNT(*) as run_count, 
       SUM(CASE WHEN run_status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
       SUM(CASE WHEN run_status = 'FAILED' THEN 1 ELSE 0 END) as failed,
       SUM(CASE WHEN run_status = 'SKIPPED' THEN 1 ELSE 0 END) as skipped
FROM job_runs 
GROUP BY job_id
ORDER BY run_count DESC;

.print ''
.print '========================================='
.print 'COMPUTE USAGE BY TYPE'
.print '========================================='
SELECT parent_type, COUNT(*) as usage_records, 
       ROUND(SUM(total_cost), 2) as total_cost,
       ROUND(SUM(dbus_consumed), 2) as total_dbus
FROM compute_usage 
GROUP BY parent_type;

.print ''
.print '========================================='
.print 'EVICTION DETAILS'
.print '========================================='
SELECT COUNT(*) as total_evictions,
       COUNT(DISTINCT DATE(eviction_time)) as days_with_evictions
FROM eviction_details;

.print ''
.print '========================================='
.print 'VALIDATION COMPLETE'
.print '========================================='
.print ''
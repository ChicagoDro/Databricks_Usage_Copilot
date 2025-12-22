-- Enhanced SQLite DDL for Databricks Usage Copilot
-- Supports realistic operational scenarios with proper schema

-- --------------------------------------------------------------------------------------
-- 1. Workspace / Organizational Context
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS workspace (
    workspace_id        TEXT PRIMARY KEY,
    workspace_name      TEXT NOT NULL,
    account_id          TEXT,
    description         TEXT
);

CREATE TABLE IF NOT EXISTS users_lookup (
    user_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    workspace_id        TEXT NOT NULL,
    department          TEXT,
    FOREIGN KEY (workspace_id) REFERENCES workspace(workspace_id)
);

-- --------------------------------------------------------------------------------------
-- 2. Compute Resources
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS instance_pools (
    instance_pool_id            TEXT PRIMARY KEY,
    pool_name                   TEXT NOT NULL,
    pool_instance_type          TEXT NOT NULL,
    min_size                    INTEGER DEFAULT 0,
    max_size                    INTEGER DEFAULT 10,
    auto_termination_mins       INTEGER DEFAULT 30
);

CREATE TABLE IF NOT EXISTS non_job_compute (
    compute_id          TEXT PRIMARY KEY,
    compute_name        TEXT NOT NULL,
    compute_type        TEXT NOT NULL CHECK (compute_type IN ('SQL_WAREHOUSE', 'APC_CLUSTER')),
    workspace_id        TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspace(workspace_id)
);

-- --------------------------------------------------------------------------------------
-- 3. Jobs and Job Runs
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS jobs (
    job_id              TEXT PRIMARY KEY,
    workspace_id        TEXT NOT NULL,
    job_name            TEXT NOT NULL,
    description         TEXT,
    tags                TEXT,  -- JSON array of tags
    FOREIGN KEY (workspace_id) REFERENCES workspace(workspace_id)
);

CREATE TABLE IF NOT EXISTS job_runs (
    job_run_id                  TEXT PRIMARY KEY,
    job_id                      TEXT NOT NULL,
    start_time                  TEXT NOT NULL,  -- ISO 8601 datetime
    end_time                    TEXT,           -- ISO 8601 datetime
    duration_ms                 INTEGER,
    run_status                  TEXT NOT NULL CHECK (run_status IN ('SUCCESS', 'FAILED', 'SKIPPED', 'RUNNING')),
    error_summary               TEXT,           -- Error message for failed runs
    driver_instance_type        TEXT,
    worker_instance_type        TEXT,
    is_fleet_cluster            INTEGER DEFAULT 0,
    instance_pool_id            TEXT,
    min_nodes                   INTEGER,
    max_nodes                   INTEGER,
    fixed_nodes                 INTEGER,
    is_autoscaling_enabled      INTEGER DEFAULT 0,
    spot_ratio                  REAL DEFAULT 0.0,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (instance_pool_id) REFERENCES instance_pools(instance_pool_id)
);

-- --------------------------------------------------------------------------------------
-- 4. Compute Usage (DBUs and Cost)
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS compute_usage (
    compute_usage_id            TEXT PRIMARY KEY,
    parent_id                   TEXT NOT NULL,  -- job_run_id or compute_id
    parent_type                 TEXT NOT NULL CHECK (parent_type IN ('JOB_RUN', 'SQL_WAREHOUSE', 'APC_CLUSTER')),
    compute_sku                 TEXT NOT NULL,
    dbus_consumed               REAL NOT NULL,
    cluster_id                  TEXT,
    cluster_instance_type       TEXT,
    total_cost                  REAL,
    avg_cpu_utilization         REAL,
    avg_memory_gb               REAL,
    peak_concurrent_users       INTEGER,
    is_production               INTEGER DEFAULT 0,
    usage_date                  TEXT NOT NULL  -- YYYY-MM-DD
);

CREATE INDEX IF NOT EXISTS idx_compute_usage_parent ON compute_usage(parent_id, parent_type);
CREATE INDEX IF NOT EXISTS idx_compute_usage_date ON compute_usage(usage_date);
CREATE INDEX IF NOT EXISTS idx_compute_usage_cost ON compute_usage(total_cost DESC);

-- --------------------------------------------------------------------------------------
-- 5. Events (Lifecycle, Evictions, Autoscaling)
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS events (
    event_id                    TEXT PRIMARY KEY,
    compute_usage_id            TEXT NOT NULL,
    event_time                  TEXT NOT NULL,  -- ISO 8601 datetime
    event_type                  TEXT NOT NULL CHECK (event_type IN (
        'CLUSTER_START', 
        'CLUSTER_TERMINATE', 
        'SPOT_EVICTION', 
        'AUTOSCALING',
        'CONFIG_CHANGE'
    )),
    user_id                     TEXT,
    details                     TEXT,
    eviction_id                 TEXT,  -- Reference to eviction_details if applicable
    FOREIGN KEY (user_id) REFERENCES users_lookup(user_id)
);

CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

-- --------------------------------------------------------------------------------------
-- 6. Eviction Details (Spot Instance Terminations)
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eviction_details (
    eviction_id                 TEXT PRIMARY KEY,
    cloud_instance_id           TEXT NOT NULL,
    eviction_time               TEXT NOT NULL,  -- ISO 8601 datetime
    cloud_provider_message      TEXT,
    eviction_reason             TEXT CHECK (eviction_reason IN (
        'CAPACITY_CHANGE',
        'PRICE_CHANGE',
        'INSTANCE_FAILURE',
        'MAINTENANCE'
    )),
    spot_price                  REAL,
    eviction_action             TEXT CHECK (eviction_action IN ('STOP_DEALLOCATE', 'DELETE')),
    was_retried                 INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eviction_time ON eviction_details(eviction_time);

-- --------------------------------------------------------------------------------------
-- 7. SQL Query History (Ad-hoc Warehouse Usage)
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sql_query_history (
    query_id                    TEXT PRIMARY KEY,
    parent_id                   TEXT NOT NULL,  -- warehouse compute_id
    user_id                     TEXT NOT NULL,
    start_time                  TEXT NOT NULL,  -- ISO 8601 datetime
    duration_ms                 INTEGER,
    warehouse_sku               TEXT,
    sql_text                    TEXT,
    error_message               TEXT,
    FOREIGN KEY (parent_id) REFERENCES non_job_compute(compute_id),
    FOREIGN KEY (user_id) REFERENCES users_lookup(user_id)
);

CREATE INDEX IF NOT EXISTS idx_sql_history_user ON sql_query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_sql_history_time ON sql_query_history(start_time);

-- --------------------------------------------------------------------------------------
-- 8. Helper Table: Date Series (for generating continuous date ranges)
-- --------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS date_series (
    date TEXT PRIMARY KEY  -- YYYY-MM-DD format
);

-- --------------------------------------------------------------------------------------
-- Schema Validation
-- --------------------------------------------------------------------------------------

select '';
select  '=========================================';
select  'SCHEMA CREATED SUCCESSFULLY';
select  '=========================================';
select  '';
select  'Tables created:';
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

select  '';
select  'Indexes created:';
SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name;

select  '';
select  '=========================================';
-- sql/queries/job_failure_investigation.sql
-- =====================================================
-- Job Failure Investigation Query
-- =====================================================
-- Purpose: Returns granular details of failed runs for root cause analysis
-- Used by: "Why Is This Failing?" action chip in Job Cost & Reliability report
-- Parameters: :job_id (string) - The job to investigate
-- Returns: Up to 50 most recent failures with diagnostic context

WITH failed_runs AS (
  SELECT
    r.job_run_id,
    r.job_id,
    r.start_time,
    r.end_time,
    r.duration_ms,
    r.run_status,
    r.error_summary,
    r.spot_ratio,
    r.worker_instance_type,
    r.min_nodes,
    r.max_nodes,
    r.is_autoscaling_enabled,
    DATE(r.start_time) AS failure_date,
    strftime('%H', r.start_time) AS failure_hour
  FROM job_runs r
  WHERE r.run_status = 'FAILED'
    AND r.job_id = :job_id
),

-- Detect retry patterns by looking for runs close together
retry_analysis AS (
  SELECT
    fr1.job_run_id,
    COUNT(DISTINCT fr2.job_run_id) AS retry_count,
    MAX(fr2.start_time) AS last_retry_time,
    GROUP_CONCAT(fr2.job_run_id, ', ') AS retry_run_ids
  FROM failed_runs fr1
  LEFT JOIN failed_runs fr2 
    ON fr1.failure_date = fr2.failure_date
    AND fr2.start_time > fr1.start_time
    AND (julianday(fr2.start_time) - julianday(fr1.start_time)) * 24 < 1  -- Within 1 hour
  GROUP BY fr1.job_run_id
),

-- Find evictions that happened around the same time as failures
spot_eviction_correlation AS (
  SELECT
    fr.job_run_id,
    COUNT(ed.eviction_id) AS evictions_within_window,
    GROUP_CONCAT(ed.eviction_reason, '; ') AS eviction_reasons,
    MIN(ed.spot_price) AS min_spot_price,
    MAX(ed.spot_price) AS max_spot_price
  FROM failed_runs fr
  LEFT JOIN eviction_details ed
    ON ABS(julianday(ed.eviction_time) - julianday(fr.start_time)) * 24 < 2  -- Within 2 hours
  GROUP BY fr.job_run_id
),

-- Calculate baseline metrics for comparison
baseline_metrics AS (
  SELECT
    job_id,
    AVG(duration_ms) AS avg_duration_ms,
    AVG(spot_ratio) AS avg_spot_ratio,
    COUNT(*) AS total_runs,
    SUM(CASE WHEN run_status = 'FAILED' THEN 1 ELSE 0 END) AS total_failures
  FROM job_runs
  WHERE job_id = :job_id
)

SELECT
  -- Run identifiers
  fr.job_run_id,
  fr.job_id,
  fr.start_time,
  fr.end_time,
  fr.failure_date,
  fr.failure_hour,
  
  -- Duration metrics
  fr.duration_ms / 1000.0 / 60.0 AS duration_mins,
  bm.avg_duration_ms / 1000.0 / 60.0 AS avg_duration_mins,
  CAST(100.0 * (fr.duration_ms - bm.avg_duration_ms) / bm.avg_duration_ms AS REAL) AS pct_duration_deviation,
  
  -- Error details
  fr.error_summary,
  CASE
    WHEN fr.error_summary LIKE '%SpotInstanceTerminated%' THEN 'SPOT_EVICTION'
    WHEN fr.error_summary LIKE '%timeout%' OR fr.error_summary LIKE '%Timeout%' THEN 'TIMEOUT'
    WHEN fr.error_summary LIKE '%Connection%' OR fr.error_summary LIKE '%IOException%' THEN 'CONNECTIVITY'
    WHEN fr.error_summary LIKE '%Upstream dependency%' THEN 'UPSTREAM_DEPENDENCY'
    WHEN fr.error_summary LIKE '%QueryExecutionException%' THEN 'QUERY_ERROR'
    WHEN fr.error_summary LIKE '%MemoryError%' OR fr.error_summary LIKE '%OutOfMemory%' THEN 'OOM'
    ELSE 'OTHER'
  END AS failure_category,
  
  -- Cluster configuration
  fr.spot_ratio * 100 AS spot_pct,
  bm.avg_spot_ratio * 100 AS avg_spot_pct,
  fr.worker_instance_type,
  fr.min_nodes,
  fr.max_nodes,
  fr.is_autoscaling_enabled,
  
  -- Retry context
  COALESCE(ra.retry_count, 0) AS retry_count,
  ra.last_retry_time,
  ra.retry_run_ids,
  
  -- Spot eviction correlation
  COALESCE(sec.evictions_within_window, 0) AS nearby_evictions,
  sec.eviction_reasons,
  sec.min_spot_price,
  sec.max_spot_price,
  
  -- Baseline comparison
  bm.total_runs,
  bm.total_failures,
  CAST(100.0 * bm.total_failures / bm.total_runs AS REAL) AS overall_failure_rate_pct

FROM failed_runs fr
CROSS JOIN baseline_metrics bm
LEFT JOIN retry_analysis ra ON fr.job_run_id = ra.job_run_id
LEFT JOIN spot_eviction_correlation sec ON fr.job_run_id = sec.job_run_id

ORDER BY fr.start_time DESC
LIMIT 50;
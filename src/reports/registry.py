# src/reports/registry.py
from __future__ import annotations

from typing import Dict, List

from src.reports.base import ReportSpec
from src.reports.job_cost import REPORT as JOB_COST_REPORT
from src.reports.compute_type_cost import REPORT as COMPUTE_TYPE_COST_REPORT
from src.reports.job_cost_pareto import REPORT as JOB_COST_PARETO_REPORT
from src.reports.spot_risk_by_job import REPORT as SPOT_RISK_BY_JOB_REPORT
from src.reports.anomaly_detection import REPORT as ANOMALY_DETECTION_REPORT


def get_reports() -> List[ReportSpec]:
    """
    Ordered list used for sidebar navigation.
    The order here defines how reports appear in the UI.
    """
    return [
        JOB_COST_REPORT,
        COMPUTE_TYPE_COST_REPORT,
        JOB_COST_PARETO_REPORT,
        SPOT_RISK_BY_JOB_REPORT,
        ANOMALY_DETECTION_REPORT,
    ]


def get_report_map() -> Dict[str, ReportSpec]:
    """
    Keyed lookup by ReportSpec.key.
    """
    return {r.key: r for r in get_reports()}


def get_default_report_key() -> str:
    """
    First report is the default landing view.
    """
    reports = get_reports()
    return reports[0].key if reports else "job_cost"

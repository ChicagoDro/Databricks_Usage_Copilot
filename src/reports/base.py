# src/reports/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol


class SelectionLike(Protocol):
    entity_type: str
    entity_id: str
    label: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ActionChip:
    label: str
    prompt: str
    focus: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ReportSpec:
    key: str
    name: str
    description: str

    # Load the dataframe for this report (from db_path, filters, etc.)
    load_df: Callable[[str, Dict[str, Any]], Any]

    # Render the visualization (must render into Streamlit)
    render_viz: Callable[[Any, Dict[str, Any]], None]

    # Build "selection chips" shown below the chart (deterministic)
    build_selections: Callable[[Any, Dict[str, Any]], List[SelectionLike]]

    # Given a selection, return deterministic action chips for the commentary panel
    build_action_chips: Callable[[SelectionLike, Dict[str, Any]], List[ActionChip]]

    # Optional: Show this SQL in debug mode (copy/paste into sqlite3)
    debug_sql: Optional[str] = None


def default_focus_for_selection(sel: SelectionLike) -> Dict[str, str]:
    return {"entity_type": str(sel.entity_type), "entity_id": str(sel.entity_id)}

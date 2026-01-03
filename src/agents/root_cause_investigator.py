"""
Root Cause Investigator Agent

Autonomous agent that investigates cost anomalies by:
1. Gathering facts about the anomaly
2. Forming hypotheses about root causes
3. Testing hypotheses with data
4. Ranking causes by evidence
5. Providing actionable recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

from src.agents.base import BaseAgent, AgentResult, AgentStep
from src.agents.tools import create_investigation_tools


class RootCauseInvestigator(BaseAgent):
    """
    Agent that investigates root causes of cost anomalies.
    
    Uses a structured investigation process:
    1. Get anomaly details
    2. Get historical baseline
    3. Check for events (evictions, failures)
    4. Check for config changes
    5. Compare to similar jobs
    6. Synthesize findings into ranked hypotheses
    """
    
    def __init__(self, db_path: str, llm, verbose: bool = False):
        tools = create_investigation_tools(db_path)
        super().__init__(tools=tools, max_steps=8, verbose=verbose)
        self.llm = llm
    
    def plan(self, task: str, context: Dict[str, Any]) -> List[str]:
        """
        Create investigation plan.
        For root cause investigation, we follow a standard process.
        """
        return [
            "Get detailed information about the anomaly",
            "Establish historical baseline for comparison",
            "Check for spot evictions or failures around that date",
            "Check if configuration changed recently",
            "Compare to similar jobs for context",
            "Synthesize findings and rank hypotheses"
        ]
    
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute root cause investigation.
        
        Args:
            task: Description of what to investigate
            context: Must contain 'entity_id' and 'date'
        """
        self.steps = []
        
        entity_id = context.get('entity_id')
        date = context.get('date')
        
        if not entity_id or not date:
            return AgentResult(
                conclusion="Investigation failed: Missing entity_id or date in context",
                confidence=0.0,
                evidence=[],
                recommendations=[],
                steps=self.steps,
                metadata=context
            )
        
        # Step 1: Get anomaly details
        details_tool = self.get_tool('get_anomaly_details')
        anomaly_details = details_tool.run(entity_id=entity_id, date=date)
        
        self.log_step(
            action="Get Anomaly Details",
            thought=f"Need to understand the specifics of the anomaly for {entity_id} on {date}",
            observation=anomaly_details,
            tool_used='get_anomaly_details',
            tool_input={'entity_id': entity_id, 'date': date},
            tool_output=anomaly_details
        )
        
        # Step 2: Get historical baseline
        baseline_tool = self.get_tool('get_historical_baseline')
        baseline_info = baseline_tool.run(entity_id=entity_id, date=date, days_back=30)
        
        self.log_step(
            action="Establish Baseline",
            thought="Compare anomaly to historical patterns",
            observation=baseline_info,
            tool_used='get_historical_baseline',
            tool_input={'entity_id': entity_id, 'date': date},
            tool_output=baseline_info
        )
        
        # Step 3: Check for events
        events_tool = self.get_tool('check_events_around_date')
        events_info = events_tool.run(entity_id=entity_id, date=date, days_window=1)
        
        self.log_step(
            action="Check Events",
            thought="Look for spot evictions, failures, or other incidents",
            observation=events_info,
            tool_used='check_events_around_date',
            tool_input={'entity_id': entity_id, 'date': date},
            tool_output=events_info
        )
        
        # Step 4: Check config changes
        config_tool = self.get_tool('check_config_changes')
        config_info = config_tool.run(entity_id=entity_id, date=date, days_back=7)
        
        self.log_step(
            action="Check Configuration",
            thought="See if cluster configuration changed recently",
            observation=config_info,
            tool_used='check_config_changes',
            tool_input={'entity_id': entity_id, 'date': date},
            tool_output=config_info
        )
        
        # Step 5: Compare to similar jobs
        compare_tool = self.get_tool('compare_to_similar_jobs')
        comparison_info = compare_tool.run(entity_id=entity_id, date=date)
        
        self.log_step(
            action="Compare to Peers",
            thought="Understand if this is unique or part of a broader pattern",
            observation=comparison_info,
            tool_used='compare_to_similar_jobs',
            tool_input={'entity_id': entity_id, 'date': date},
            tool_output=comparison_info
        )
        
        # Step 6: Synthesize findings
        evidence = [
            anomaly_details,
            baseline_info,
            events_info,
            config_info,
            comparison_info
        ]
        
        synthesis_prompt = f"""You are a data platform engineer investigating a cost anomaly.

Task: {task}

Evidence gathered:

1. ANOMALY DETAILS:
{anomaly_details}

2. HISTORICAL BASELINE:
{baseline_info}

3. EVENTS AROUND DATE:
{events_info}

4. CONFIGURATION CHANGES:
{config_info}

5. COMPARISON TO SIMILAR JOBS:
{comparison_info}

Based on this evidence, provide your analysis in this exact format:

ROOT CAUSE:
<1-2 sentence explanation of the most likely root cause>

CONFIDENCE: <number between 0-100>

SUPPORTING EVIDENCE:
- <evidence point 1>
- <evidence point 2>
- <evidence point 3>

ALTERNATIVE HYPOTHESES:
- <alternative 1> (likelihood: low/medium/high)
- <alternative 2> (likelihood: low/medium/high)

RECOMMENDATIONS:
IMMEDIATE:
- <immediate action 1>
- <immediate action 2>

SHORT-TERM (1-2 weeks):
- <short-term action 1>
- <short-term action 2>

LONG-TERM (1-3 months):
- <long-term action 1>
- <long-term action 2>

VERIFICATION STEPS:
- <how to verify root cause>
- <how to confirm fix worked>
"""
        
        synthesis_response = self.llm.invoke(synthesis_prompt)
        
        # Parse the structured response
        root_cause = self._extract_section(synthesis_response, "ROOT CAUSE")
        confidence_str = self._extract_section(synthesis_response, "CONFIDENCE")
        confidence = float(confidence_str.strip('%')) / 100 if confidence_str else 0.5
        
        supporting_evidence = self._extract_bullet_list(synthesis_response, "SUPPORTING EVIDENCE")
        
        # Extract recommendations
        recs = []
        immediate = self._extract_bullet_list(synthesis_response, "IMMEDIATE")
        recs.extend([f"[IMMEDIATE] {r}" for r in immediate])
        
        short_term = self._extract_bullet_list(synthesis_response, "SHORT-TERM")
        recs.extend([f"[SHORT-TERM] {r}" for r in short_term])
        
        long_term = self._extract_bullet_list(synthesis_response, "LONG-TERM")
        recs.extend([f"[LONG-TERM] {r}" for r in long_term])
        
        self.log_step(
            action="Synthesize Findings",
            thought="Analyze all evidence to determine root cause",
            observation=f"Root cause identified with {confidence:.0%} confidence: {root_cause}"
        )
        
        return AgentResult(
            conclusion=root_cause,
            confidence=confidence,
            evidence=supporting_evidence,
            recommendations=recs,
            steps=self.steps,
            metadata={
                "task": task,
                "context": context,
                "synthesis": synthesis_response
            }
        )
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from structured text"""
        if f"{section_name}:" not in text:
            return ""
        
        parts = text.split(f"{section_name}:")
        if len(parts) < 2:
            return ""
        
        # Get everything until next section (all caps word followed by colon) or end
        content = parts[1]
        
        # Find next section
        import re
        next_section = re.search(r'\n[A-Z][A-Z\s-]+:', content)
        if next_section:
            content = content[:next_section.start()]
        
        return content.strip()
    
    def _extract_bullet_list(self, text: str, section_name: str) -> List[str]:
        """Extract bullet points from a section"""
        section_content = self._extract_section(text, section_name)
        if not section_content:
            return []
        
        lines = section_content.split('\n')
        bullets = []
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                bullet = line.lstrip('-•').strip()
                if bullet:
                    bullets.append(bullet)
        
        return bullets


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Mock LLM for testing
    class MockLLM:
        def invoke(self, prompt: str) -> str:
            return """ROOT CAUSE:
The cost spike was caused by multiple spot instance evictions during peak hours, forcing the job to retry 3 times with fallback to on-demand instances.

CONFIDENCE: 85

SUPPORTING EVIDENCE:
- 3 spot evictions detected between 14:00-16:00 on the anomaly date
- Job failed twice, succeeded on third attempt
- Run duration was 3x normal due to retries and instance startup time
- Spot ratio was 70%, exposing job to eviction risk

ALTERNATIVE HYPOTHESES:
- Data volume spike (likelihood: low) - no evidence of increased data volume
- Configuration change (likelihood: low) - no config changes detected in past 7 days
- Upstream dependency delay (likelihood: medium) - possible but no direct evidence

RECOMMENDATIONS:
IMMEDIATE:
- Review spot instance pricing history for that time period
- Check if other jobs were affected by same eviction event
- Verify current job status and cost

SHORT-TERM (1-2 weeks):
- Reduce spot ratio from 70% to 50% for this job
- Implement exponential backoff for retries
- Add monitoring for spot eviction frequency

LONG-TERM (1-3 months):
- Schedule job during off-peak hours to avoid spot contention
- Implement instance pool with reserved capacity
- Consider dedicated cluster for critical jobs

VERIFICATION STEPS:
- Monitor spot eviction rate over next 2 weeks
- Compare cost after spot ratio reduction
- Track job reliability metrics (failure rate, duration variance)"""
    
    # Test the investigator
    db_path = Path(__file__).parent.parent.parent / "data" / "usage_rag_data.db"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Run 'make db' first to create the database")
        sys.exit(1)
    
    investigator = RootCauseInvestigator(
        db_path=str(db_path),
        llm=MockLLM(),
        verbose=True
    )
    
    result = investigator.execute(
        task="Investigate cost spike for J-FIN-DAILY on 2025-12-05",
        context={
            "entity_id": "J-FIN-DAILY",
            "date": "2025-12-05",
            "anomaly_severity": "critical",
            "cost_increase_pct": 275
        }
    )
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)
    print(f"\nRoot Cause: {result.conclusion}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nEvidence:")
    for e in result.evidence:
        print(f"  - {e}")
    print(f"\nRecommendations:")
    for r in result.recommendations:
        print(f"  - {r}")
"""
Base Agent Framework

Provides foundation for building LangChain agents that can:
- Use tools to query data
- Maintain conversation state
- Show reasoning steps (explainability)
- Integrate with existing RAG system
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod


@dataclass
class AgentStep:
    """
    Represents one step in an agent's reasoning process.
    Used for explainability and debugging.
    """
    step_number: int
    action: str
    thought: str
    observation: str
    tool_used: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None


@dataclass
class AgentResult:
    """
    Final result from an agent investigation.
    """
    conclusion: str
    confidence: float  # 0-1
    evidence: List[str]
    recommendations: List[str]
    steps: List[AgentStep]
    metadata: Dict[str, Any]


class Tool(Protocol):
    """
    Protocol for agent tools.
    Tools are functions agents can call to gather information.
    """
    name: str
    description: str
    
    def run(self, **kwargs) -> str:
        """Execute the tool and return results as string"""
        ...


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    
    Agents coordinate multiple tools to solve complex problems.
    They show their reasoning and can be interrupted/debugged.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        max_steps: int = 10,
        verbose: bool = False
    ):
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.steps: List[AgentStep] = []
    
    @abstractmethod
    def plan(self, task: str, context: Dict[str, Any]) -> List[str]:
        """
        Create a plan to solve the task.
        Returns list of actions to take.
        """
        pass
    
    @abstractmethod
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's investigation.
        Returns final result with evidence and recommendations.
        """
        pass
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def log_step(
        self,
        action: str,
        thought: str,
        observation: str,
        tool_used: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[str] = None
    ):
        """Log a reasoning step for explainability"""
        step = AgentStep(
            step_number=len(self.steps) + 1,
            action=action,
            thought=thought,
            observation=observation,
            tool_used=tool_used,
            tool_input=tool_input,
            tool_output=tool_output
        )
        self.steps.append(step)
        
        if self.verbose:
            print(f"\n--- Step {step.step_number} ---")
            print(f"Action: {action}")
            print(f"Thought: {thought}")
            if tool_used:
                print(f"Tool: {tool_used}")
            print(f"Observation: {observation}")
    
    def format_steps_for_display(self) -> str:
        """Format reasoning steps for UI display"""
        lines = []
        for step in self.steps:
            lines.append(f"**Step {step.step_number}: {step.action}**")
            lines.append(f"*{step.thought}*")
            if step.tool_used:
                lines.append(f"🔧 Used tool: `{step.tool_used}`")
            lines.append(f"📊 Observation: {step.observation}")
            lines.append("")
        return "\n".join(lines)


class SimpleReActAgent(BaseAgent):
    """
    Simple ReAct (Reasoning + Acting) agent implementation.
    
    Uses a thought → action → observation loop to solve problems.
    Good for teaching agent concepts before using full LangChain agents.
    """
    
    def __init__(self, tools: List[Tool], llm, max_steps: int = 10, verbose: bool = False):
        super().__init__(tools, max_steps, verbose)
        self.llm = llm
    
    def plan(self, task: str, context: Dict[str, Any]) -> List[str]:
        """
        Create a high-level plan.
        In ReAct, planning happens dynamically as we go.
        """
        # For simple cases, we can ask LLM for a plan upfront
        prompt = f"""Given this task: {task}
        
Context: {context}

Available tools:
{self._format_tools()}

Create a 3-5 step plan to solve this task. Be specific about which tools to use.
Format as a numbered list."""

        plan_text = self.llm.invoke(prompt)
        # Parse into steps (simple split on newlines with numbers)
        steps = [line.strip() for line in plan_text.split('\n') if line.strip() and line[0].isdigit()]
        return steps
    
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute ReAct loop: Think → Act → Observe → Repeat
        """
        self.steps = []  # Reset steps
        
        # Initial thought
        thought = f"I need to investigate: {task}"
        self.log_step("Initialize", thought, "Starting investigation")
        
        # Get initial plan
        plan = self.plan(task, context)
        
        evidence = []
        current_findings = ""
        
        for step_num, planned_action in enumerate(plan):
            if step_num >= self.max_steps:
                break
            
            # Decide what to do next based on current findings
            next_action_prompt = f"""Current task: {task}
            
Planned action: {planned_action}

Findings so far:
{current_findings}

Available tools:
{self._format_tools()}

What should I do next? Respond with:
1. Tool to use (name)
2. Tool input (JSON)
3. Why this will help

Format:
TOOL: <tool_name>
INPUT: <json>
REASONING: <explanation>
"""
            
            response = self.llm.invoke(next_action_prompt)
            
            # Parse response (simple parsing)
            tool_name = self._extract_field(response, "TOOL")
            tool_input_str = self._extract_field(response, "INPUT")
            reasoning = self._extract_field(response, "REASONING")
            
            # Execute tool
            tool = self.get_tool(tool_name)
            if not tool:
                observation = f"Tool {tool_name} not found"
                self.log_step(f"Error", reasoning, observation)
                continue
            
            try:
                import json
                tool_input = json.loads(tool_input_str) if tool_input_str else {}
                tool_output = tool.run(**tool_input)
                
                observation = f"Tool returned: {tool_output[:200]}..."
                evidence.append(tool_output)
                current_findings += f"\n{planned_action}: {tool_output}\n"
                
                self.log_step(
                    planned_action,
                    reasoning,
                    observation,
                    tool_used=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output
                )
            except Exception as e:
                observation = f"Tool execution failed: {str(e)}"
                self.log_step(f"Error: {planned_action}", reasoning, observation)
        
        # Generate conclusion
        conclusion_prompt = f"""Based on this investigation:

Task: {task}
Evidence gathered:
{chr(10).join(evidence)}

Provide:
1. Main conclusion (2-3 sentences)
2. Confidence level (0-100%)
3. Top 3 recommendations

Format:
CONCLUSION: <conclusion>
CONFIDENCE: <number>
RECOMMENDATIONS:
- <rec1>
- <rec2>
- <rec3>
"""
        
        final_response = self.llm.invoke(conclusion_prompt)
        
        conclusion = self._extract_field(final_response, "CONCLUSION")
        confidence_str = self._extract_field(final_response, "CONFIDENCE")
        confidence = float(confidence_str.strip('%')) / 100 if confidence_str else 0.5
        
        # Extract recommendations
        recs_section = final_response.split("RECOMMENDATIONS:")[-1] if "RECOMMENDATIONS:" in final_response else ""
        recommendations = [
            line.strip().lstrip('-').strip() 
            for line in recs_section.split('\n') 
            if line.strip() and line.strip().startswith('-')
        ]
        
        return AgentResult(
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            steps=self.steps,
            metadata={"task": task, "context": context}
        )
    
    def _format_tools(self) -> str:
        """Format tools for prompt"""
        lines = []
        for tool in self.tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract a field from structured text"""
        if f"{field}:" not in text:
            return ""
        parts = text.split(f"{field}:")
        if len(parts) < 2:
            return ""
        # Get everything until next field or end
        value = parts[1].split("\n")[0].strip()
        return value


# Example usage
if __name__ == "__main__":
    # This would be in a separate file, but shown here for completeness
    
    class DummyTool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
        
        def run(self, **kwargs) -> str:
            return f"Dummy result from {self.name} with input {kwargs}"
    
    class DummyLLM:
        def invoke(self, prompt: str) -> str:
            return "This is a dummy response"
    
    # Create agent
    tools = [
        DummyTool("query_runs", "Query job run history"),
        DummyTool("query_events", "Query events around a date"),
    ]
    
    agent = SimpleReActAgent(tools=tools, llm=DummyLLM(), verbose=True)
    
    # Run investigation
    result = agent.execute(
        task="Investigate cost spike on 2024-12-15 for job J-FIN-DAILY",
        context={"anomaly_cost": 1500, "expected_cost": 400}
    )
    
    print("\n=== Final Result ===")
    print(f"Conclusion: {result.conclusion}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Recommendations: {result.recommendations}")
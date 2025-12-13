# src/chat_orchestrator.py

"""
Main orchestration for Databricks Usage Copilot.

This module is now "dashboard-friendly":
- The UI (Streamlit) owns deterministic chips/actions.
- The orchestrator owns: retrieval (GraphRAG) + LLM answer + optional debug artifacts.

Public API:
    assistant = DatabricksUsageAssistant.from_local()
    result = assistant.answer("Tell me more about job_id=J-123", focus={"entity_type":"job","entity_id":"J-123"})
    print(result.answer)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.config import DEFAULT_TEMPERATURE, LLM_PROVIDER, get_chat_model_name
from src.graph_retriever import GraphRAGRetriever


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    answer: str
    context_docs: Optional[List[Document]] = None
    graph_explanation: Optional[str] = None
    llm_prompt: Optional[str] = None
    llm_context: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm():
    provider = (LLM_PROVIDER or "openai").lower()
    model_name = get_chat_model_name()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    if provider == "grok":
        raise NotImplementedError(
            "Grok chat model not wired up yet. "
            "Update src/chat_orchestrator.get_llm() with your Grok client."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ASSISTANT_SYSTEM_PROMPT = """You are an enterprise Databricks usage copilot.

You answer questions using the provided CONTEXT snippets (GraphRAG + vector retrieval).
Be concise, grounded, and reference entity IDs when useful, such as:
job_id=..., run_id=..., query_id=..., user_id=..., workspace_id=..., ou_id=...,
warehouse_id=..., compute_id=..., usage_id=..., event_id=..., eviction_id=...

Rules:
- Do NOT invent facts not supported by context. If you infer, say so explicitly.
- If asked for "what to do next", propose a short, ordered checklist.
"""

ASSISTANT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_PROMPT),
        ("system", "FOCUS (if provided):\n{focus}\n"),
        ("system", "CONTEXT:\n{context}\n\nUse this context to answer the user's question."),
        ("human", "{question}"),
    ]
)

QUESTION_CLASSIFIER_PROMPT = PromptTemplate.from_template(
    """You are classifying a user question about Databricks usage.

Return a JSON object with:
- "intent": one of ["global_aggregate", "global_topn", "local_explanation", "entity_lookup", "other"]
- "entity_type": one of ["job", "user", "warehouse", "org_unit", "query", "other", null]

Now classify this question:

Q: "{question}"
JSON:
"""
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_top_n(question: str, default: int = 3, max_n: int = 20) -> int:
    nums = re.findall(r"\b(\d+)\b", question)
    if not nums:
        return default
    try:
        n = int(nums[0])
        if n < 1:
            return default
        return min(n, max_n)
    except Exception:
        return default


def _looks_like_job_count_question(question: str) -> bool:
    q = question.lower()
    patterns = ["how many jobs", "number of jobs", "count of jobs", "jobs are there", "jobs do we have"]
    return any(p in q for p in patterns)


def _looks_like_usage_overview_question(question: str) -> bool:
    q = question.lower()
    patterns = [
        "tell me about my databricks usage",
        "tell me about our databricks usage",
        "overview of my databricks usage",
        "overview of our databricks usage",
        "summarize our databricks usage",
        "summarize my databricks usage",
        "overall databricks usage",
        "summary of job usage",
        "job usage summary",
        "summarize my job usage",
        "summarize our job usage",
    ]
    return any(p in q for p in patterns)


def _looks_like_jobs_optimization_question(question: str) -> bool:
    q = question.lower()
    patterns = [
        "which jobs need optimizing",
        "which jobs need optimization",
        "which jobs should we optimize",
        "jobs need optimizing",
        "jobs need optimization",
        "which jobs are inefficient",
        "which jobs are most expensive",
        "which jobs need optimizing first",
    ]
    return any(p in q for p in patterns)


def build_graph_explanation(node_ids: List[str], retriever: GraphRAGRetriever) -> str:
    if not node_ids:
        return "GraphRAG fallback: no anchor nodes – used plain vector search."

    nodes = retriever.adj.nodes  # type: ignore[attr-defined]
    by_type: Dict[str, int] = {}
    for nid in node_ids:
        n = nodes.get(nid)
        if not n:
            continue
        t = getattr(n, "type", "unknown") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1

    parts = [f"GraphRAG used a subgraph with {len(node_ids)} nodes."]
    if by_type:
        type_chunks = [f"{count} {node_type}" for node_type, count in sorted(by_type.items())]
        parts.append("Node types: " + ", ".join(type_chunks) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DatabricksUsageAssistant:
    def __init__(self, graph_retriever: GraphRAGRetriever) -> None:
        self.graph_retriever = graph_retriever
        self.llm = get_llm()
        self.chain = ASSISTANT_PROMPT_TEMPLATE | self.llm | StrOutputParser()
        self.classifier_chain = QUESTION_CLASSIFIER_PROMPT | self.llm | StrOutputParser()

    @classmethod
    def from_local(cls) -> "DatabricksUsageAssistant":
        retriever = GraphRAGRetriever.from_local_index()
        return cls(graph_retriever=retriever)

    def _classify_question(self, question: str) -> dict:
        raw = self.classifier_chain.invoke({"question": question})
        try:
            return json.loads(raw)
        except Exception:
            return {"intent": "other", "entity_type": None}

    def _render_context(self, docs: List[Document]) -> str:
        chunks: List[str] = []
        for d in docs or []:
            doc_id = d.metadata.get("doc_id", "unknown")
            doc_type = d.metadata.get("type", d.metadata.get("source", "doc"))
            chunks.append(f"[{doc_type} | {doc_id}]\n{d.page_content}")
        return "\n\n---\n\n".join(chunks) if chunks else "No relevant documents retrieved."

    # ---- Optional deterministic “global” answers (kept for convenience) ----

    def _answer_global_aggregate(self, entity_type: str) -> ChatResult:
        # Minimal generic aggregate (counts nodes by type in your in-memory graph)
        nodes = self.graph_retriever.adj.nodes  # type: ignore[attr-defined]
        count = 0
        for _, n in nodes.items():
            if getattr(n, "type", None) == entity_type:
                count += 1
        answer = f"Count of {entity_type} entities in the graph: {count}."
        return ChatResult(answer=answer, context_docs=[], graph_explanation="Answered via in-memory graph node count.")

    def _answer_global_topn_jobs(self, top_n: int) -> ChatResult:
        # If you have a proper cost aggregation somewhere else, wire it here.
        answer = (
            "Top-N job cost ranking is not yet implemented in this dashboard-first branch. "
            "Use the Job Cost report for ranking and drill-down."
        )
        return ChatResult(answer=answer, context_docs=[], graph_explanation="Top-N handler stub.")

    def _answer_global_usage_overview(self) -> ChatResult:
        answer = (
            "Usage overview is best handled by the dashboard reports now (Job Cost, Compute Type Cost, etc.). "
            "Pick a report and click a data point to generate grounded commentary."
        )
        return ChatResult(answer=answer, context_docs=[], graph_explanation="Overview handler stub.")

    def _answer_jobs_needing_optimization(self) -> ChatResult:
        answer = (
            "Optimization candidates are best identified from the Job Cost report (sorted by total cost). "
            "Select a job and use the Optimization Opportunities chip."
        )
        return ChatResult(answer=answer, context_docs=[], graph_explanation="Optimization handler stub.")

    # -------------------- Public API --------------------

    def answer(self, question: str, focus: Optional[dict] = None) -> ChatResult:
        """
        question: user prompt
        focus: {"entity_type": "...", "entity_id": "..."} from dashboard selection (optional)

        Focus biases retrieval and is also passed to the LLM as a separate field.
        """
        question = (question or "").strip()
        focus = focus or {}

        focus_note = ""
        if focus.get("entity_type") and focus.get("entity_id"):
            focus_note = f"{focus['entity_type']} {focus['entity_id']}"

        # Keep old chat routes as lightweight conveniences
        if _looks_like_job_count_question(question):
            return self._answer_global_aggregate("job")

        if _looks_like_usage_overview_question(question):
            return self._answer_global_usage_overview()

        if _looks_like_jobs_optimization_question(question):
            return self._answer_jobs_needing_optimization()

        classification = self._classify_question(question)
        intent = classification.get("intent", "other")
        entity_type = classification.get("entity_type")

        if intent == "global_aggregate" and entity_type:
            return self._answer_global_aggregate(str(entity_type))

        if intent == "global_topn" and entity_type == "job":
            top_n = _extract_top_n(question, default=3)
            return self._answer_global_topn_jobs(top_n)

        # Retrieval query is biased with focus (but the user question stays clean)
        retrieval_query = question
        if focus_note:
            retrieval_query = f"{question}\n\nCurrent focus: {focus_note}"

        docs, node_ids = self.graph_retriever.get_subgraph_for_query(
            query=retrieval_query,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )

        context_str = self._render_context(docs)
        graph_explanation = build_graph_explanation(node_ids=node_ids, retriever=self.graph_retriever)

        # Debug-friendly raw prompt
        llm_prompt_text = (
            ASSISTANT_SYSTEM_PROMPT
            + ("\n\nFOCUS:\n" + focus_note if focus_note else "\n\nFOCUS:\n(none)")
            + "\n\nCONTEXT:\n"
            + context_str
            + "\n\nQUESTION:\n"
            + question
        )

        answer_text = self.chain.invoke(
            {
                "focus": (focus_note or "(none)"),
                "context": context_str,
                "question": question,
            }
        )

        return ChatResult(
            answer=answer_text,
            context_docs=docs,
            graph_explanation=graph_explanation,
            llm_prompt=llm_prompt_text,
            llm_context=context_str,
        )

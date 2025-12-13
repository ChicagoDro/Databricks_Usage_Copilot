# src/chat_orchestrator.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    LLM_PROVIDER,
    get_chat_model_name,
    DEFAULT_TEMPERATURE,
)
from src.graph_retriever import GraphRAGRetriever


def get_llm():
    provider = LLM_PROVIDER.lower()
    model_name = get_chat_model_name()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    if provider == "grok":
        raise NotImplementedError("Grok not wired up yet in get_llm().")

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


ASSISTANT_SYSTEM_PROMPT = """You are an enterprise Databricks usage copilot.

You answer questions using the provided CONTEXT snippets (GraphRAG + vector retrieval).
Be concise, grounded, and reference entity IDs when useful, such as:
job_id=..., run_id=..., query_id=..., user_id=..., workspace_id=..., ou_id=...,
warehouse_id=..., compute_id=..., usage_id=..., compute_usage_id=..., event_id=..., eviction_id=...
If the context does not support a claim, say so.
"""

ASSISTANT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_PROMPT),
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


def build_graph_explanation(node_ids: List[str], retriever: GraphRAGRetriever) -> str:
    if not node_ids:
        return "GraphRAG fallback: no anchor nodes – used plain vector search."

    nodes = retriever.adj.nodes  # type: ignore[attr-defined]
    by_type: Dict[str, int] = {}
    for nid in node_ids:
        n = nodes.get(nid)
        if not n:
            continue
        by_type[n.type] = by_type.get(n.type, 0) + 1

    parts = [f"GraphRAG used a subgraph with {len(node_ids)} nodes."]
    if by_type:
        type_chunks = [f"{count} {node_type}" for node_type, count in sorted(by_type.items())]
        parts.append("Node types: " + ", ".join(type_chunks) + ".")

    sample_ids = node_ids[:5]
    sample_descriptions = []
    for nid in sample_ids:
        n = nodes.get(nid)
        if not n:
            continue
        label = (
            n.properties.get("job_name")
            or n.properties.get("name")
            or n.properties.get("compute_name")
            or n.id
        )
        sample_descriptions.append(f"{n.id} ({n.type}, label={label})")

    if sample_descriptions:
        parts.append("Example nodes: " + "; ".join(sample_descriptions) + ".")

    return " ".join(parts)


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
    ]
    return any(p in q for p in patterns)


@dataclass
class ChatResult:
    answer: str
    context_docs: Optional[List[Document]] = None
    graph_explanation: Optional[str] = None
    llm_prompt: Optional[str] = None
    llm_context: Optional[str] = None
    # For UI chips/buttons
    entities: List[dict] = field(default_factory=list)


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

    # ---------- Entities for UI ----------

    @staticmethod
    def _normalize_node_id(node_id: str) -> Tuple[Optional[str], str]:
        if "::" in node_id:
            prefix, raw_id = node_id.split("::", 1)
            return prefix, raw_id
        return None, node_id

    def _entities_from_node_ids(self, node_ids: List[str]) -> List[dict]:
        nodes = self.graph_retriever.adj.nodes  # type: ignore[attr-defined]
        out: List[dict] = []
        seen: set[Tuple[str, str]] = set()

        for nid in node_ids or []:
            n = nodes.get(nid)
            if not n:
                continue

            prefix, raw_id = self._normalize_node_id(nid)
            entity_type = prefix or n.type or "entity"
            entity_id = raw_id

            label = (
                n.properties.get("job_name")
                or n.properties.get("name")
                or n.properties.get("compute_name")
                or entity_id
            )

            key = (str(entity_type), str(entity_id))
            if key in seen:
                continue
            seen.add(key)

            out.append({"entity_type": str(entity_type), "entity_id": str(entity_id), "label": str(label)})

        return out

    @staticmethod
    def _entities_from_answer_text(text: str) -> List[dict]:
        """
        Anchor chips are created ONLY when an entity id is explicitly mentioned in the answer,
        using patterns like job_id=..., run_id=..., query_id=..., etc.
        """
        if not text:
            return []

        out: List[dict] = []
        seen: set[Tuple[str, str]] = set()

        def add(et: str, eid: str, label: str):
            key = (et, eid)
            if not eid or key in seen:
                return
            seen.add(key)
            out.append({"entity_type": et, "entity_id": eid, "label": label})

        # Jobs
        for job_id in re.findall(r"\bjob_id=([A-Za-z0-9_-]+)\b", text):
            add("job", job_id, f"job_id={job_id}")

        # Job runs (support run_id=... and job_run_id=...)
        for run_id in re.findall(r"\b(?:run_id|job_run_id)=([A-Za-z0-9_-]+)\b", text):
            add("job_run", run_id, f"run_id={run_id}")

        # SQL queries
        for query_id in re.findall(r"\bquery_id=([A-Za-z0-9_-]+)\b", text):
            add("query", query_id, f"query_id={query_id}")

        # Users
        for user_id in re.findall(r"\buser_id=([A-Za-z0-9_-]+)\b", text):
            add("user", user_id, f"user_id={user_id}")

        # Workspaces / org units (you sometimes call it workspace_id or ou_id)
        for workspace_id in re.findall(r"\b(?:workspace_id|ou_id)=([A-Za-z0-9_-]+)\b", text):
            add("ou", workspace_id, f"workspace_id={workspace_id}")

        # Compute resources (warehouses, clusters)
        for compute_id in re.findall(r"\b(?:warehouse_id|compute_id)=([A-Za-z0-9_-]+)\b", text):
            # keep entity_type as "compute" so your UI prompt becomes "tell me more about this compute X"
            add("compute", compute_id, f"compute_id={compute_id}")

        # Compute usage
        for usage_id in re.findall(r"\b(?:compute_usage_id|usage_id)=([A-Za-z0-9_-]+)\b", text):
            add("usage", usage_id, f"usage_id={usage_id}")

        # Events
        for event_id in re.findall(r"\bevent_id=([A-Za-z0-9_-]+)\b", text):
            add("event", event_id, f"event_id={event_id}")

        # Evictions
        for eviction_id in re.findall(r"\beviction_id=([A-Za-z0-9_-]+)\b", text):
            add("eviction", eviction_id, f"eviction_id={eviction_id}")

        return out

    @staticmethod
    def _merge_entities(*lists: List[dict]) -> List[dict]:
        merged: List[dict] = []
        seen: set[Tuple[str, str]] = set()
        for lst in lists:
            for e in lst or []:
                et = str(e.get("entity_type") or "entity")
                eid = str(e.get("entity_id") or "")
                if not eid:
                    continue
                key = (et, eid)
                if key in seen:
                    continue
                seen.add(key)
                merged.append({"entity_type": et, "entity_id": eid, "label": e.get("label") or eid})
        return merged

    # ---------- Global helpers ----------

    def _compute_job_costs(self) -> Dict[str, Dict[str, float]]:
        nodes = self.graph_retriever.adj.nodes  # type: ignore[attr-defined]
        neighbors = self.graph_retriever.adj.neighbors  # type: ignore[attr-defined]
        job_costs: Dict[str, Dict[str, float]] = {}

        for n in nodes.values():
            if n.type != "compute_usage":
                continue
            cost = n.properties.get("cost_usd")
            if cost is None:
                continue

            for nbr_id in neighbors.get(n.id, set()):
                nbr = nodes.get(nbr_id)
                if not nbr or nbr.type != "job_run":
                    continue

                job_id = nbr.properties.get("job_id")
                if not job_id:
                    continue

                job_node_id = f"job::{job_id}"
                job_node = nodes.get(job_node_id)
                job_name = job_node.properties.get("job_name") if job_node else job_id

                entry = job_costs.setdefault(job_id, {"name": job_name, "cost": 0.0})
                entry["cost"] += float(cost)

        return job_costs

    def _answer_global_aggregate(self, entity_type: str) -> ChatResult:
        nodes = self.graph_retriever.adj.nodes  # type: ignore[attr-defined]

        target_type = None
        label_field = None

        if entity_type == "job":
            target_type = "job"
            label_field = "job_name"
        elif entity_type == "user":
            target_type = "user"
            label_field = "name"
        elif entity_type == "warehouse":
            target_type = "compute_resource"
        elif entity_type == "org_unit":
            target_type = "company_ou"
            label_field = "name"

        if target_type is None:
            answer = "I couldn't recognize the entity type for this global aggregate question."
            return ChatResult(answer=answer, entities=self._entities_from_answer_text(answer))

        matched = []
        for n in nodes.values():
            if n.type != target_type:
                continue
            if entity_type == "warehouse" and n.properties.get("compute_type") != "SQL_WAREHOUSE":
                continue
            matched.append(n)

        count = len(matched)
        lines = [f"There are {count} {entity_type}(s) in this environment."]

        answer = "\n".join(lines)
        graph_explanation = f"Answered via direct graph inspection over '{target_type}' nodes."

        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation=graph_explanation,
            entities=self._entities_from_answer_text(answer),
        )

    def _answer_global_topn_jobs(self, top_n: int) -> ChatResult:
        job_costs = self._compute_job_costs()
        if not job_costs:
            answer = "I couldn't find any cost data for jobs in the graph."
            return ChatResult(answer=answer, entities=self._entities_from_answer_text(answer))

        ranked = sorted(job_costs.items(), key=lambda kv: kv[1]["cost"], reverse=True)
        top_n = min(top_n, len(ranked))
        top_slice = ranked[:top_n]

        lines = [f"Top {top_n} most expensive jobs by total cost:"]
        for rank, (job_id, info) in enumerate(top_slice, start=1):
            lines.append(f"{rank}. {info['name']} (job_id={job_id}) — ${info['cost']:,.2f}")

        answer = "\n".join(lines)
        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation="Answered via direct graph aggregation (compute_usage.cost_usd rolled up to jobs).",
            entities=self._entities_from_answer_text(answer),
        )

    def _answer_global_usage_overview(self) -> ChatResult:
        job_costs = self._compute_job_costs()
        if not job_costs:
            answer = "I couldn't find any usage data in the graph to summarize."
            return ChatResult(answer=answer, entities=self._entities_from_answer_text(answer))

        ranked = sorted(job_costs.items(), key=lambda kv: kv[1]["cost"], reverse=True)
        total_cost = sum(info["cost"] for _, info in ranked)
        num_jobs = len(ranked)

        lines = [
            f"You currently have {num_jobs} jobs with recorded compute usage.",
            f"Total observed cost across all jobs: ${total_cost:,.2f}.",
            "",
            "Breakdown by job (sorted by total cost):",
        ]
        for job_id, info in ranked:
            share = (info["cost"] / total_cost * 100.0) if total_cost > 0 else 0.0
            lines.append(f"- {info['name']} (job_id={job_id}) — ${info['cost']:,.2f} ({share:.1f}% of total)")

        answer = "\n".join(lines)
        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation="Answered via global usage overview (aggregated compute_usage.cost_usd across jobs).",
            entities=self._entities_from_answer_text(answer),
        )

    def _answer_jobs_needing_optimization(self) -> ChatResult:
        job_costs = self._compute_job_costs()
        if not job_costs:
            answer = "I couldn't find any cost data for jobs to decide which need optimization."
            return ChatResult(answer=answer, entities=self._entities_from_answer_text(answer))

        ranked = sorted(job_costs.items(), key=lambda kv: kv[1]["cost"], reverse=True)
        total_cost = sum(info["cost"] for _, info in ranked)
        top_k = min(3, len(ranked))
        top_slice = ranked[:top_k]

        lines = [f"Highest-impact optimization candidates (top {top_k} by total cost):"]
        for rank, (job_id, info) in enumerate(top_slice, start=1):
            share = (info["cost"] / total_cost * 100.0) if total_cost > 0 else 0.0
            lines.append(f"{rank}. {info['name']} (job_id={job_id}) — ${info['cost']:,.2f} ({share:.1f}% of total)")

        answer = "\n".join(lines)
        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation="Identified optimization candidates by ranking jobs by total compute cost.",
            entities=self._entities_from_answer_text(answer),
        )

    # ---------- Context rendering ----------

    def _render_context(self, docs: List[Document]) -> str:
        chunks = []
        for d in docs:
            doc_id = d.metadata.get("doc_id", "unknown")
            doc_type = d.metadata.get("type", d.metadata.get("source", "doc"))
            chunks.append(f"[{doc_type} | {doc_id}]\n{d.page_content}")
        return "\n\n---\n\n".join(chunks) if chunks else "No relevant documents retrieved."

    # ---------- Public API ----------

    def answer(self, question: str, focus: Optional[dict] = None) -> ChatResult:
        focus_note = ""
        retrieval_query = question
        if focus and focus.get("entity_type") and focus.get("entity_id"):
            focus_note = f"Current focus: {focus['entity_type']} {focus['entity_id']}"
            retrieval_query = f"{question}\n\n{focus_note}"

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
            return self._answer_global_aggregate(entity_type)

        if intent == "global_topn" and entity_type:
            top_n = _extract_top_n(question, default=3)
            if entity_type == "job":
                return self._answer_global_topn_jobs(top_n)

        docs, node_ids = self.graph_retriever.get_subgraph_for_query(
            query=retrieval_query,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )

        context_str = self._render_context(docs)
        graph_explanation = build_graph_explanation(node_ids=node_ids, retriever=self.graph_retriever)

        llm_prompt_text = (
            ASSISTANT_SYSTEM_PROMPT
            + ("\n\nFOCUS:\n" + focus_note if focus_note else "")
            + "\n\nCONTEXT:\n"
            + context_str
            + "\n\nQUESTION:\n"
            + question
        )

        answer_text = self.chain.invoke(
            {"context": context_str, "question": (question + ("\n\n" + focus_note if focus_note else ""))}
        )

        entities = self._merge_entities(
            self._entities_from_node_ids(node_ids),
            self._entities_from_answer_text(answer_text),
        )

        return ChatResult(
            answer=answer_text,
            context_docs=docs,
            graph_explanation=graph_explanation,
            llm_prompt=llm_prompt_text,
            llm_context=context_str,
            entities=entities,
        )

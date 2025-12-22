# src/chat_orchestrator.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    LLM_PROVIDER,
    get_chat_model_name,
    get_embed_model_name,
    DEFAULT_TEMPERATURE,
    DOCS_FAISS_INDEX_PATH,
    DOCS_RETRIEVER_K,
)
from src.rag_context_aware_prompts.graph_retriever import GraphRAGRetriever


# ---------------------------------------------------------------------------
# Vendor docs routing heuristics
# ---------------------------------------------------------------------------

_DOCS_INTENT_PATTERNS = [
    r"\bwhat is\b",
    r"\bwhat does\b",
    r"\bhow do i\b",
    r"\bhow to\b",
    r"\bhow does\b",
    r"\bconfigure\b",
    r"\bsetting(s)?\b",
    r"\bbest practice(s)?\b",
    r"\blimit(s)?\b",
]

_DOCS_TOPIC_KEYWORDS = [
    "compute",
    "cluster",
    "clusters",
    "autoscaling",
    "auto scaling",
    "node type",
    "instance type",
    "spot",
    "on-demand",
    "ondemand",
    "photon",
    "sql warehouse",
    "warehouse",
    "serverless",
    "dbu",
    "pools",
    "cluster policy",
    "policies",
    "job cluster",
    "all-purpose",
    "all purpose",
]

def _looks_like_docs_question(q: str) -> bool:
    ql = q.lower().strip()
    if any(k in ql for k in _DOCS_TOPIC_KEYWORDS):
        return True
    return any(re.search(p, ql) for p in _DOCS_INTENT_PATTERNS)


def _has_entity_anchor(q: str) -> bool:
    ql = q.lower()
    return any(
        token in ql
        for token in [
            "job_id=",
            "run_id=",
            "query_id=",
            "warehouse_id=",
            "cluster_id=",
            "user_id=",
            "compute_type=",
        ]
    )


def _get_embeddings_for_docs():
    provider = (LLM_PROVIDER or "openai").lower()
    model_name = get_embed_model_name()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model_name)

    if provider == "grok":
        raise NotImplementedError(
            "Grok embeddings are not wired up yet. "
            "Set LLM_PROVIDER=openai or gemini to use the Databricks docs corpus."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER for embeddings: {provider}")


class DatabricksDocsRetriever:
    def __init__(self, index_path):
        self.index_path = index_path
        self._vs = None

    def is_available(self) -> bool:
        try:
            return self.index_path.exists()
        except Exception:
            return False

    def _load(self):
        if self._vs is not None:
            return
        embeddings = _get_embeddings_for_docs()
        self._vs = FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = DOCS_RETRIEVER_K) -> List[Document]:
        if not self.is_available():
            return []
        self._load()
        return self._vs.similarity_search(query, k=k)


def _extract_doc_sources(docs: List[Document]) -> List[Tuple[str, str]]:
    """
    Return unique (title, url) pairs for a set of docs, preserving order.
    """
    results: List[Tuple[str, str]] = []
    seen = set()
    for d in docs:
        meta = d.metadata or {}
        url = meta.get("url") or meta.get("source_url")
        title = meta.get("title") or "Databricks Docs"
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        results.append((title, url))
    return results


def _format_docs_sources_block(docs: List[Document]) -> str:
    """
    For debug mode: show a neat list with title + URL.
    """
    pairs = _extract_doc_sources(docs)
    if not pairs:
        return ""
    lines = []
    for title, url in pairs:
        lines.append(f"- {title}\n  {url}")
    return "\n".join(lines)


def _append_sources_to_answer(answer_text: str, vendor_docs: List[Document]) -> str:
    """
    Add a deterministic "Sources" section to the answer if vendor docs were used.
    """
    pairs = _extract_doc_sources(vendor_docs)
    if not pairs:
        return answer_text

    lines = ["", "Sources (Databricks Docs):"]
    for title, url in pairs:
        lines.append(f"- {title} — {url}")

    # Ensure clean separation even if model already ended with a newline
    return answer_text.rstrip() + "\n" + "\n".join(lines) + "\n"


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
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ASSISTANT_SYSTEM_PROMPT = """You are a Databricks Usage Copilot.
You help answer questions about Databricks jobs, runs, compute usage, cost, SQL queries, and spot/eviction risk.

Rules:
- Use the provided CONTEXT as the primary source of truth.
- If something is missing from context, say what is missing and suggest what to check.
- Keep answers concise and practical.
"""

ASSISTANT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_PROMPT),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]
)

QUESTION_CLASSIFIER_PROMPT = PromptTemplate.from_template(
    """You are a classifier. Return JSON only.

Classify the user's question into:
- intent: one of ["global_aggregate", "global_topn", "other"]
- entity_type: one of ["job", "warehouse", "user", "query", "run", null]

Guidance:
- "How many X are there?" -> global_aggregate, entity_type=X
- "Top N most expensive jobs" -> global_topn, entity_type=job
- Otherwise -> other, entity_type=null

Question: {question}

Return JSON with keys intent and entity_type.
"""
)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _looks_like_job_count_question(q: str) -> bool:
    ql = q.lower()
    return ("how many jobs" in ql) or ("number of jobs" in ql)


def _looks_like_usage_overview_question(q: str) -> bool:
    ql = q.lower()
    return ("tell me about my databricks usage" in ql) or ("summarize my databricks usage" in ql)


def _looks_like_jobs_optimization_question(q: str) -> bool:
    ql = q.lower()
    return ("jobs need optimizing" in ql) or ("which jobs should i optimize" in ql)


def _extract_top_n(q: str, default: int = 3) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q.lower())
    if m:
        return int(m.group(1))
    return default


# ---------------------------------------------------------------------------
# Graph explanation helper (your GraphRAGRetriever structure)
# ---------------------------------------------------------------------------

def build_graph_explanation(node_ids: List[str], retriever: GraphRAGRetriever) -> str:
    if not node_ids:
        return "No graph nodes were retrieved."

    nodes = retriever.adj.nodes
    lines = [f"Retrieved {len(node_ids)} graph nodes (showing up to 12):"]

    for nid in node_ids[:12]:
        n = nodes.get(nid)
        if not n:
            lines.append(f"- {nid} | (missing node)")
            continue

        ntype = getattr(n, "type", "unknown")
        props = getattr(n, "properties", {}) or {}
        name = (
            props.get("job_name")
            or props.get("user_name")
            or props.get("compute_name")
            or props.get("job_id")
            or props.get("run_id")
            or ""
        )
        lines.append(f"- {nid} | type={ntype} | name={name}")

    if len(node_ids) > 12:
        lines.append(f"... ({len(node_ids) - 12} more)")

    return "\n".join(lines)


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
# Main Orchestrator
# ---------------------------------------------------------------------------

class DatabricksUsageAssistant:
    def __init__(
        self,
        graph_retriever: GraphRAGRetriever,
        docs_retriever: Optional[DatabricksDocsRetriever] = None,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.docs_retriever = docs_retriever
        self.llm = get_llm()
        self.chain = ASSISTANT_PROMPT_TEMPLATE | self.llm | StrOutputParser()
        self.classifier_chain = QUESTION_CLASSIFIER_PROMPT | self.llm | StrOutputParser()

    @classmethod
    def from_local(cls) -> "DatabricksUsageAssistant":
        retriever = GraphRAGRetriever.from_local_index()
        docs = DatabricksDocsRetriever(DOCS_FAISS_INDEX_PATH)
        docs_retriever = docs if docs.is_available() else None
        return cls(graph_retriever=retriever, docs_retriever=docs_retriever)

    def _classify_question(self, question: str) -> dict:
        raw = self.classifier_chain.invoke({"question": question})
        try:
            return json.loads(raw)
        except Exception:
            return {"intent": "other", "entity_type": None}

    # -----------------------------------------------------------------------
    # Deterministic aggregations (using your adjacency graph)
    # -----------------------------------------------------------------------

    def _compute_job_costs(self) -> Dict[str, Dict[str, float]]:
        nodes = self.graph_retriever.adj.nodes
        nbrs = self.graph_retriever.adj.neighbors

        job_costs: Dict[str, Dict[str, float]] = {}

        for node_id, node in nodes.items():
            if getattr(node, "type", None) != "compute_usage":
                continue

            props = getattr(node, "properties", {}) or {}
            cost = float(props.get("cost_usd", 0.0))

            job_run_id = None
            for nb in nbrs.get(node_id, set()):
                nb_node = nodes.get(nb)
                if nb_node and getattr(nb_node, "type", None) == "job_run":
                    job_run_id = nb
                    break
            if not job_run_id:
                continue

            job_node_id = None
            for nb in nbrs.get(job_run_id, set()):
                nb_node = nodes.get(nb)
                if nb_node and getattr(nb_node, "type", None) == "job":
                    job_node_id = nb
                    break
            if not job_node_id:
                continue

            job_node = nodes.get(job_node_id)
            job_props = getattr(job_node, "properties", {}) or {}

            job_key = job_props.get("job_id") or job_node_id
            job_name = job_props.get("job_name") or job_key

            if job_key not in job_costs:
                job_costs[job_key] = {"name": job_name, "cost": 0.0}
            job_costs[job_key]["cost"] += cost

        return job_costs

    def _answer_global_aggregate(self, entity_type: str) -> ChatResult:
        entity_type = entity_type.lower()
        count = 0
        for node in self.graph_retriever.adj.nodes.values():
            if getattr(node, "type", None) == entity_type:
                count += 1
        return ChatResult(
            answer=f"There are {count} {entity_type}(s) in the dataset.",
            graph_explanation=f"[deterministic] Counted nodes of type '{entity_type}'.",
        )

    def _answer_global_usage_overview(self) -> ChatResult:
        counts: Dict[str, int] = {}
        for node in self.graph_retriever.adj.nodes.values():
            t = getattr(node, "type", "unknown")
            counts[t] = counts.get(t, 0) + 1

        keys = ["workspace", "user", "job", "job_run", "compute_usage", "sql_query", "event", "eviction"]
        lines = ["Here’s a high-level overview of your Databricks usage dataset:"]
        for k in keys:
            if k in counts:
                lines.append(f"- {k}: {counts[k]}")

        return ChatResult(
            answer="\n".join(lines),
            graph_explanation="[deterministic] Summarized entity counts by node.type",
        )

    def _answer_global_topn_jobs(self, top_n: int) -> ChatResult:
        job_costs = self._compute_job_costs()
        ranked = sorted(job_costs.items(), key=lambda kv: kv[1]["cost"], reverse=True)[:top_n]

        lines = [f"Top {top_n} most expensive jobs:"]
        for job_id, info in ranked:
            lines.append(f"- {info['name']} (job_id={job_id}): ${info['cost']:.2f}")

        return ChatResult(
            answer="\n".join(lines),
            graph_explanation="[deterministic] Aggregated cost_usd from compute_usage → job_run → job",
        )

    def _answer_jobs_needing_optimization(self) -> ChatResult:
        job_costs = self._compute_job_costs()
        ranked = sorted(job_costs.items(), key=lambda kv: kv[1]["cost"], reverse=True)[:5]

        lines = ["Jobs that likely need optimization (highest cost drivers):"]
        for job_id, info in ranked:
            lines.append(f"- {info['name']} (job_id={job_id}): ${info['cost']:.2f}")

        return ChatResult(
            answer="\n".join(lines),
            graph_explanation="[deterministic] Top-5 jobs by cost_usd aggregation",
        )

    # -----------------------------------------------------------------------
    # Context rendering
    # -----------------------------------------------------------------------

    def _render_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No context retrieved."

        parts = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            doc_type = meta.get("doc_type") or meta.get("document_type") or ""
            title = meta.get("title") or ""
            url = meta.get("url") or meta.get("source_url") or ""

            header_bits = [f"[{i}] source={src}"]
            if doc_type:
                header_bits.append(f"type={doc_type}")
            if title:
                header_bits.append(f"title={title}")
            if url:
                header_bits.append(f"url={url}")

            parts.append(" | ".join(header_bits))
            parts.append(d.page_content.strip())

        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # Main entrypoint
    # -----------------------------------------------------------------------

    def answer(self, question: str, focus: Optional[dict] = None) -> ChatResult:
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
            query=question,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )

        vendor_docs: List[Document] = []
        if self.docs_retriever is not None:
            use_docs = _looks_like_docs_question(question)
            if _has_entity_anchor(question) and not use_docs:
                use_docs = False
            if use_docs:
                vendor_docs = self.docs_retriever.retrieve(question, k=DOCS_RETRIEVER_K)

        context_str = self._render_context(docs)
        if vendor_docs:
            vendor_context = self._render_context(vendor_docs)
            context_str = (
                "=== TELEMETRY CONTEXT (your usage data) ===\n"
                + context_str
                + "\n\n=== DATABRICKS DOCUMENTATION CONTEXT (vendor docs) ===\n"
                + vendor_context
            )

        graph_explanation = build_graph_explanation(node_ids=node_ids, retriever=self.graph_retriever)

        if vendor_docs:
            debug_sources = _format_docs_sources_block(vendor_docs)
            if debug_sources:
                graph_explanation = (graph_explanation or "") + "\n\n[docs_sources]\n" + debug_sources

        if focus:
            graph_explanation = (graph_explanation or "") + "\n\n[focus]\n" + json.dumps(focus, indent=2)

        llm_prompt_text = (
            ASSISTANT_SYSTEM_PROMPT
            + "\n\nCONTEXT:\n"
            + context_str
            + "\n\nQUESTION:\n"
            + question
        )

        answer_text = self.chain.invoke({"context": context_str, "question": question})

        # ✅ Add deterministic Sources section to the answer if vendor docs were used
        if vendor_docs:
            answer_text = _append_sources_to_answer(answer_text, vendor_docs)

        return ChatResult(
            answer=answer_text,
            context_docs=docs,
            graph_explanation=graph_explanation,
            llm_prompt=llm_prompt_text,
            llm_context=context_str,
        )

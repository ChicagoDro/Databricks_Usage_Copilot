# src/chat_orchestrator.py

"""
Main chat orchestration for the Databricks Usage domain.

- Uses GraphRAGRetriever for graph-aware retrieval.
- Uses your configured LLM provider (OpenAI / Gemini / Grok stub).
- Adds a lightweight classifier to detect:
    * "global_aggregate" questions (e.g., 'how many jobs are there?')
    * "global_topn" questions (e.g., 'top 3 most expensive jobs')
- Adds heuristic routes for:
    * 'tell me about my Databricks usage' (global usage overview)
    * 'which jobs need optimizing?' (high-cost jobs)

Returns:
    - The LLM (or deterministic) answer.
    - A graph explanation string that your UI can display or log.
    - Optionally, the context documents used for the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from src.config import (
    LLM_PROVIDER,
    get_chat_model_name,
    DEFAULT_TEMPERATURE,
)
from src.graph_retriever import GraphRAGRetriever


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm():
    """
    Return a LangChain ChatModel based on LLM_PROVIDER + model name.

    Supported:
      - openai  -> langchain_openai.ChatOpenAI
      - gemini  -> langchain_google_genai.ChatGoogleGenerativeAI
      - grok    -> (placeholder) raise for now or plug in your client later.
    """
    provider = LLM_PROVIDER.lower()
    model_name = get_chat_model_name()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "grok":
        raise NotImplementedError(
            "Grok chat model not wired up yet. "
            "Update chat_orchestrator.get_llm() with your Grok client."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ASSISTANT_SYSTEM_PROMPT = """You are an enterprise Databricks usage copilot.

You are answering questions for:
- Platform teams,
- FinOps teams,
- Data engineering leads,
- Analytics / ML practitioners.

Your knowledge comes from:
- Databricks jobs and their metadata,
- Job runs and cluster configs,
- Compute usage (DBUs, cost, instance types),
- Ad-hoc SQL queries,
- Events and spot evictions,
- Org units, users, and their departments.

Guidelines:
- Always ground your answers in the provided CONTEXT snippets.
- When explaining cost or reliability, reference specific jobs, compute resources,
  or queries by name/ID when possible.
- If multiple root causes are possible, say so and explain the tradeoffs.
- If the answer is not fully supported by the context, say you are inferring
  and call that out explicitly.
"""

ASSISTANT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_PROMPT),
        (
            "system",
            "CONTEXT:\n{context}\n\nUse this context to answer the user's question.",
        ),
        ("human", "{question}"),
    ]
)

# Small classifier prompt to detect intent + entity_type
# NOTE: JSON examples are escaped with {{ }} so LangChain treats them as literals.
QUESTION_CLASSIFIER_PROMPT = PromptTemplate.from_template(
    """You are classifying a user question about Databricks usage.

Return a JSON object with:
- "intent": one of ["global_aggregate", "global_topn", "local_explanation", "entity_lookup", "other"]
- "entity_type": one of ["job", "user", "warehouse", "org_unit", "query", "other", null]

Examples:
Q: "How many jobs are there?"
{{"intent": "global_aggregate", "entity_type": "job"}}

Q: "How many users do we have?"
{{"intent": "global_aggregate", "entity_type": "user"}}

Q: "What are the top 3 most expensive jobs?"
{{"intent": "global_topn", "entity_type": "job"}}

Q: "Show me the top 5 most costly SQL warehouses."
{{"intent": "global_topn", "entity_type": "warehouse"}}

Q: "Why did the Logistics Optimizer job fail yesterday?"
{{"intent": "local_explanation", "entity_type": "job"}}

Q: "Tell me about the Production ML Training job."
{{"intent": "entity_lookup", "entity_type": "job"}}

Q: "What's the org structure for Finance?"
{{"intent": "entity_lookup", "entity_type": "org_unit"}}

Q: "What is Databricks?"
{{"intent": "other", "entity_type": null}}

Now classify this question:

Q: "{question}"
JSON:
"""
)


# ---------------------------------------------------------------------------
# Graph explanation builder
# ---------------------------------------------------------------------------


def build_graph_explanation(
    node_ids: List[str],
    retriever: GraphRAGRetriever,
) -> str:
    """
    Build a human-readable explanation string about the subgraph that was used.

    Example:
      "Looked at 18 nodes across types: 1 org units, 3 users, 5 jobs, 4 job runs,
       3 compute_usage, 2 events."
    """
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
        type_chunks = [
            f"{count} {node_type}" for node_type, count in sorted(by_type.items())
        ]
        parts.append("Node types: " + ", ".join(type_chunks) + ".")
    else:
        parts.append("Node type breakdown unavailable.")

    # Optional: show a couple of example nodes to make it feel concrete
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_top_n(question: str, default: int = 3, max_n: int = 20) -> int:
    """
    Extract an integer N from a 'top N' style question.
    E.g., "top 3 most expensive jobs" -> 3.
    If none found, fall back to default.
    """
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
    """
    Heuristic to detect questions that are asking about the total number of jobs.
    This is a backup in case the classifier doesn't label it as global_aggregate.
    """
    q = question.lower()
    patterns = [
        "how many jobs",
        "number of jobs",
        "count of jobs",
        "jobs are there",
        "jobs do we have",
    ]
    return any(p in q for p in patterns)


def _looks_like_usage_overview_question(question: str) -> bool:
    """
    Detect 'tell me about my Databricks usage' / 'give me an overview of usage'
    / 'summary of job usage' style questions that should look across all jobs,
    not a single neighborhood.
    """
    q = question.lower()
    patterns = [
        # existing
        "tell me about my databricks usage",
        "tell me about our databricks usage",
        "overview of my databricks usage",
        "overview of our databricks usage",
        "what does our databricks usage look like",
        "what does my databricks usage look like",
        "summarize our databricks usage",
        "summarize my databricks usage",
        "overall databricks usage",
        "overall usage on databricks",
        # NEW: job-usage phrasing
        "summary of my job usage",
        "summary of our job usage",
        "summary of job usage",
        "job usage summary",
        "summary of jobs usage",
        "summarize my job usage",
        "summarize our job usage",
    ]
    return any(p in q for p in patterns)


def _looks_like_jobs_optimization_question(question: str) -> bool:
    """
    Detect 'which jobs need optimizing' / 'which jobs should we optimize'
    style questions that should surface the most expensive jobs.
    """
    q = question.lower()
    patterns = [
        "which jobs need optimizing",
        "which jobs need optimization",
        "which jobs should we optimize",
        "which jobs should i optimize",
        "jobs need optimizing",
        "jobs need optimization",
        "jobs should we optimize",
        "jobs should i optimize",
        "which jobs are inefficient",
        "which jobs are most expensive to run",
    ]
    return any(p in q for p in patterns)


# ---------------------------------------------------------------------------
# Core result type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core result type
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    answer: str
    # original behavior: what docs were used to build the context
    context_docs: Optional[List[Document]] = None
    # explanation of which graph nodes / subgraph were used
    graph_explanation: Optional[str] = None
    # NEW: raw-ish prompt that was sent to the LLM (for UI debugging)
    llm_prompt: Optional[str] = None
    # NEW: the rendered context string we passed into the prompt
    llm_context: Optional[str] = None



# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class DatabricksUsageAssistant:
    """
    Main chat orchestrator for Databricks usage questions.

    Usage:

        assistant = DatabricksUsageAssistant.from_local()
        result = assistant.answer("Why is my logistics job so expensive?")
        print(result.answer)
        print(result.graph_explanation)
    """

    def __init__(
        self,
        graph_retriever: GraphRAGRetriever,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.llm = get_llm()
        self.chain = ASSISTANT_PROMPT_TEMPLATE | self.llm | StrOutputParser()
        # classifier chain reusing the same LLM
        self.classifier_chain = QUESTION_CLASSIFIER_PROMPT | self.llm | StrOutputParser()

    @classmethod
    def from_local(cls) -> "DatabricksUsageAssistant":
        retriever = GraphRAGRetriever.from_local_index()
        return cls(graph_retriever=retriever)

    # --------------------- Classification helpers --------------------- #

    def _classify_question(self, question: str) -> dict:
        """
        Classify (intent, entity_type) for routing.
        Falls back to {"intent": "other", "entity_type": None} on parse failures.
        """
        raw = self.classifier_chain.invoke({"question": question})
        try:
            return json.loads(raw)
        except Exception:
            return {"intent": "other", "entity_type": None}

    # --------------------- Graph aggregation helpers ------------------ #

    def _compute_job_costs(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate total cost per job_id from compute_usage nodes.

        Returns:
            job_costs: dict like:
                {
                  "J-HR-DASH": {"name": "HR Dashboard Prep", "cost": 123.45},
                  ...
                }
        """
        nodes = self.graph_retriever.adj.nodes  # type: ignore[attr-defined]
        neighbors = self.graph_retriever.adj.neighbors  # type: ignore[attr-defined]

        job_costs: Dict[str, Dict[str, float]] = {}

        for n in nodes.values():
            if n.type != "compute_usage":
                continue
            cost = n.properties.get("cost_usd")
            if cost is None:
                continue

            # Find neighboring job_run nodes
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

                entry = job_costs.setdefault(
                    job_id, {"name": job_name, "cost": 0.0}
                )
                entry["cost"] += float(cost)

        return job_costs

    # --------------------- Global aggregate helper -------------------- #

    def _answer_global_aggregate(self, entity_type: str) -> ChatResult:
        """
        Answer global aggregate questions (e.g., 'how many jobs/users/warehouses are there?')
        by inspecting the graph directly instead of going through GraphRAG.
        """
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
            # In this synthetic data, warehouses are compute_resource with compute_type='SQL_WAREHOUSE'
            target_type = "compute_resource"
            # We'll filter by compute_type in the loop
        elif entity_type == "org_unit":
            target_type = "workspace"
            label_field = "name"

        # If we don't recognize the type, just bail out with a simple message
        if target_type is None:
            return ChatResult(
                answer="I couldn't recognize the entity type for this global aggregate question.",
                context_docs=[],
                graph_explanation=(
                    f"Classification chose global_aggregate but entity_type '{entity_type}' is unsupported."
                ),
            )

        # Filter nodes
        matched = []
        for n in nodes.values():
            if n.type != target_type:
                continue
            if entity_type == "warehouse":
                if n.properties.get("compute_type") != "SQL_WAREHOUSE":
                    continue
            matched.append(n)

        count = len(matched)
        names = []
        for n in matched:
            if label_field:
                names.append(n.properties.get(label_field, n.id))
            else:
                names.append(n.id)

        lines = [f"There are {count} {entity_type}(s) in this environment."]
        if names:
            lines.append("")
            lines.append(f"{entity_type.title()}s:")
            for name in names:
                lines.append(f"- {name}")
        answer = "\n".join(lines)

        graph_explanation = (
            f"Answered via direct graph inspection over nodes of type '{target_type}' "
            f"for entity_type '{entity_type}'. Found {count} matching nodes."
        )

        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation=graph_explanation,
        )

    # --------------------- Global top-N helper (jobs) ----------------- #

    def _answer_global_topn_jobs(self, top_n: int) -> ChatResult:
        """
        Compute the top N most expensive jobs by aggregating cost_usd from
        compute_usage nodes associated with job runs.
        """
        job_costs = self._compute_job_costs()

        if not job_costs:
            return ChatResult(
                answer="I couldn't find any cost data for jobs in the graph.",
                context_docs=[],
                graph_explanation=(
                    "Attempted to compute top-N jobs by aggregating compute_usage.cost_usd, "
                    "but found no relevant nodes."
                ),
            )

        ranked = sorted(
            job_costs.items(),
            key=lambda kv: kv[1]["cost"],
            reverse=True,
        )

        top_n = min(top_n, len(ranked))
        top_slice = ranked[:top_n]

        lines = [f"Top {top_n} most expensive jobs by total cost:"]
        for rank, (job_id, info) in enumerate(top_slice, start=1):
            name = info["name"]
            cost = info["cost"]
            lines.append(f"{rank}. {name} (job_id={job_id}) — ${cost:,.2f}")

        answer = "\n".join(lines)

        graph_explanation = (
            f"Answered via direct graph aggregation: summed compute_usage.cost_usd "
            f"for each job reachable from compute_usage -> job_run -> job. "
            f"Computed totals for {len(ranked)} jobs and returned the top {top_n}."
        )

        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation=graph_explanation,
        )

    # --------------------- Global usage overview ---------------------- #

    def _answer_global_usage_overview(self) -> ChatResult:
        """
        Answer 'tell me about my Databricks usage' style questions
        by looking across ALL jobs and summarizing cost distribution.
        """
        job_costs = self._compute_job_costs()
        if not job_costs:
            return ChatResult(
                answer="I couldn't find any usage data in the graph to summarize.",
                context_docs=[],
                graph_explanation=(
                    "Attempted a global usage overview by aggregating job costs, "
                    "but found no compute_usage nodes tied to jobs."
                ),
            )

        ranked = sorted(
            job_costs.items(),
            key=lambda kv: kv[1]["cost"],
            reverse=True,
        )

        total_cost = sum(info["cost"] for _, info in ranked)
        num_jobs = len(ranked)

        lines = []
        lines.append(f"You currently have {num_jobs} jobs with recorded compute usage.")
        lines.append(f"Total observed cost across all jobs: ${total_cost:,.2f}.")
        lines.append("")
        lines.append("Here is a breakdown by job (sorted by total cost):")

        for job_id, info in ranked:
            name = info["name"]
            cost = info["cost"]
            share = (cost / total_cost * 100.0) if total_cost > 0 else 0.0
            lines.append(f"- {name} (job_id={job_id}) — ${cost:,.2f} ({share:.1f}% of total)")

        lines.append("")
        lines.append(
            "You can ask follow-ups like 'which jobs need optimizing?' or "
            "'why is [job name] so expensive?' for deeper analysis."
        )

        answer = "\n".join(lines)

        graph_explanation = (
            "Answered via global usage overview: aggregated compute_usage.cost_usd for every job "
            "reachable from compute_usage -> job_run -> job, then summarized totals and shares."
        )

        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation=graph_explanation,
        )

    # --------------------- Jobs needing optimization ------------------ #

    def _answer_jobs_needing_optimization(self) -> ChatResult:
        """
        Heuristic: jobs that contribute the largest share of total cost
        are the best candidates for optimization.

        We surface the top few high-cost jobs and explain why.
        """
        job_costs = self._compute_job_costs()
        if not job_costs:
            return ChatResult(
                answer="I couldn't find any cost data for jobs to decide which need optimization.",
                context_docs=[],
                graph_explanation=(
                    "Attempted to identify high-cost jobs via compute_usage.cost_usd, "
                    "but found no relevant nodes."
                ),
            )

        ranked = sorted(
            job_costs.items(),
            key=lambda kv: kv[1]["cost"],
            reverse=True,
        )

        total_cost = sum(info["cost"] for _, info in ranked)
        if total_cost <= 0:
            return ChatResult(
                answer="I found jobs, but their total cost appears to be zero; nothing stands out for optimization.",
                context_docs=[],
                graph_explanation=(
                    "Aggregated job costs, but total_cost <= 0, so there is no clear optimization target."
                ),
            )

        top_k = min(3, len(ranked))
        top_slice = ranked[:top_k]

        lines = []
        lines.append(
            f"Based on total compute cost, these {top_k} job(s) are the strongest candidates for optimization:"
        )
        for rank, (job_id, info) in enumerate(top_slice, start=1):
            name = info["name"]
            cost = info["cost"]
            share = (cost / total_cost * 100.0)
            lines.append(f"{rank}. {name} (job_id={job_id}) — ${cost:,.2f} ({share:.1f}% of total spend)")

        lines.append("")
        lines.append(
            "Consider focusing on these for cost optimization first. "
            "You can ask things like 'why is [job name] so expensive?' "
            "or 'how is [job name] configured?' for deeper analysis."
        )

        answer = "\n".join(lines)

        graph_explanation = (
            "Identified optimization candidates by ranking all jobs by total compute_usage.cost_usd "
            "and selecting the top few as the highest-impact targets."
        )

        return ChatResult(
            answer=answer,
            context_docs=[],
            graph_explanation=graph_explanation,
        )

    # --------------------- Context rendering -------------------------- #

    def _render_context(self, docs: List[Document]) -> str:
        """
        Render retrieved docs into a single context string.

        We group and sort by type so the LLM sees a more structured view:
        - jobs first,
        - then job_runs,
        - then compute_usage,
        - then events,
        - then queries,
        - then users / org units / everything else.
        """
        # Group docs by type
        docs_by_type: Dict[str, List[Document]] = {}
        for d in docs:
            doc_type = d.metadata.get("type", d.metadata.get("source", "doc"))
            docs_by_type.setdefault(doc_type, []).append(d)

        # Desired order for readability
        type_order = [
            "workspace",
            "job",
            "job_run",
            "compute_usage",
            "event",
            "query",
            "user",
        ]

        # Collect in order, then any unknown types at the end
        ordered_docs: List[Document] = []
        seen_types = set()

        for t in type_order:
            for d in docs_by_type.get(t, []):
                ordered_docs.append(d)
            if t in docs_by_type:
                seen_types.add(t)

        # Append any remaining types that weren't explicitly ordered
        for t, t_docs in docs_by_type.items():
            if t not in seen_types:
                ordered_docs.extend(t_docs)

        # Render
        chunks = []
        for d in ordered_docs:
            doc_id = d.metadata.get("doc_id", "unknown")
            doc_type = d.metadata.get("type", d.metadata.get("source", "doc"))
            header = f"[{doc_type} | {doc_id}]"
            body = d.page_content
            chunks.append(f"{header}\n{body}")

        return "\n\n---\n\n".join(chunks)



    # --------------------- Public API -------------------------------- #

    # --------------------- Public API -------------------------------- #

    def answer(self, question: str) -> ChatResult:
        """
        Full orchestration:
          0. If it looks like a special global question (job count, overview, optimization),
             route to the appropriate deterministic graph-based handler.
          1. Otherwise, classify the question (intent + entity_type).
          2. If intent is global_aggregate and entity_type is supported,
             answer directly via graph inspection.
          3. If intent is global_topn and entity_type is supported (e.g., job),
             answer via graph aggregation.
          4. Otherwise, use GraphRAG + LLM.
        """
        # 0a) Hard override for job-count questions
        if _looks_like_job_count_question(question):
            return self._answer_global_aggregate("job")

        # 0b) Hard override for global usage overview
        if _looks_like_usage_overview_question(question):
            return self._answer_global_usage_overview()

        # 0c) Hard override for "which jobs need optimizing?"
        if _looks_like_jobs_optimization_question(question):
            return self._answer_jobs_needing_optimization()

        # 1) Normal classification path
        classification = self._classify_question(question)
        intent = classification.get("intent", "other")
        entity_type = classification.get("entity_type")

        # 2) Global aggregate path: e.g. "How many jobs are there?"
        if intent == "global_aggregate" and entity_type:
            return self._answer_global_aggregate(entity_type)

        # 3) Global top-N path: e.g. "Top 3 most expensive jobs"
        if intent == "global_topn" and entity_type:
            top_n = _extract_top_n(question, default=3)
            if entity_type == "job":
                return self._answer_global_topn_jobs(top_n)
            # Other entity_types (warehouse, user, etc.) could be added later

        # 4) Default path: GraphRAG-based retrieval + LLM synthesis
        docs, node_ids = self.graph_retriever.get_subgraph_for_query(
            query=question,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )

        context_str = self._render_context(docs)
        graph_explanation = build_graph_explanation(
            node_ids=node_ids,
            retriever=self.graph_retriever,
        )

        # Build a debug-friendly representation of the full prompt
        # (approximation of what ChatPromptTemplate is doing)
        llm_prompt_text = (
            ASSISTANT_SYSTEM_PROMPT
            + "\n\nCONTEXT:\n"
            + context_str
            + "\n\nQUESTION:\n"
            + question
        )

        # Let the LangChain chain actually call the model
        answer_text = self.chain.invoke(
            {
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


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

def _interactive_cli() -> None:
    assistant = DatabricksUsageAssistant.from_local()
    print("[usage-assistant] Interactive mode. Type a question, or 'exit' to quit.")

    while True:
        try:
            q = input("\nYou> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        result = assistant.answer(q)
        print("\nAssistant>\n", result.answer)
        print("\n[graph_debug]", result.graph_explanation)


if __name__ == "__main__":
    _interactive_cli()

# Databricks Usage Copilot

**Pete Tamisin** – Technical GTM Leader • AI & Data Engineering Architect • Builder & Teacher
Chicago, IL

* 20+ years designing data & AI platforms (Director at Capital One, ex-Databricks, 2x Series A startup exits, x-Siemens, x-Motorola)
* Focused on **modern data platforms**, **context-aware RAG systems**, and **enterprise GenAI adoption**
* Passionate about **teaching** and helping teams ship real-world AI systems

📧 Email: `pete@tamisin.com`
🔗 LinkedIn: [https://www.linkedin.com/in/peter-tamisin-50a3233a/](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

---

## Overview

**Databricks Usage Copilot** is a deterministic, report-driven AI assistant for analyzing cost, reliability, performance, and operational risk in Data Engineering workloads.

Unlike chat-first copilots, this project is **report-driven and deterministic**:

* SQL defines the facts
* Reports define the question
* User selections define scope
* LLMs explain results instead of inventing them

The result is an **enterprise-grade AI copilot** that is explainable, debuggable, and trustworthy.

---

## Core Design Principle

> **Don't let the model guess what the user meant.**
> Use deterministic reports to define intent, and use the LLM to explain the result with context.

This project deliberately avoids "blank chat box" UX. Instead:

* **Reports** define what is being analyzed
* **Clicks** define what needs explanation
* **Prompts** are deterministic and repeatable
* **LLMs** provide narrative, root-cause hypotheses, and next actions

---

## Key Features

### ✅ Deterministic Reports
Each report is powered by explicit SQL, known semantics, and predefined drill actions:

* **Job Cost & Reliability** – Stacked bars showing spot vs on-demand ratio, reliability overlay
* **Cost Concentration (Pareto)** – Cumulative cost curve highlighting the 20% driving 80%
* **Compute Type Analysis** – Cross-type comparison (Jobs, Warehouses, Clusters)
* **Spot Risk & Evictions** – Risk matrix plotting cost vs spot exposure
* **Cost Anomaly Detection** – Statistical anomaly detection with agent-based investigation

Reports are the **interface**. AI is the **commentary layer**.

### ✅ Context-Aware Action Chips
Rather than free-form prompting, the UI presents **deterministic action chips** organized by taxonomy:

- **Understand** – What is this? What does "good" look like?
- **Diagnose** – Why is this happening? What changed?
- **Optimize** – What should I change to improve cost, reliability, or performance?
- **Monitor** – How do I validate improvements and prevent regressions?

Chips are:
- **Deterministic** – each chip maps to a fixed prompt template
- **Context-aware** – prompts parameterized by selected entity (job, cluster, warehouse)
- **Conditional** – only appear when relevant (e.g., "Why a spike?" only for high-deviation entities)
- **Stable** – chip identity doesn't change across runs
- **Explainable** – users can inspect the exact prompt executed

### ✅ Interactive Filtering & Export
* **Date range filters** with quick-select buttons (Last 7d, 30d, 90d)
* **Workspace filters** for multi-tenant analysis
* **CSV export** from any report
* **Loading indicators** for better UX
* **Auto-refresh** data cache

### ✅ Dual-Corpus RAG
* **Telemetry corpus** → Your usage data, reports, and graph context
* **Databricks docs corpus** → Official product documentation with real citations

When documentation is used, answers include deterministic **Sources** sections:
```
Sources (Databricks Docs):
- Spot Instances — https://docs.databricks.com/...
- Autoscaling Clusters — https://docs.databricks.com/...
```

---

## Report → Selection → Prompt → Answer

```text
┌─────────────────────────┐
│ Report (SQL + semantics)│
│ Chart / Table / KPI     │
└───────────┬─────────────┘
            │ click / select
            ▼
┌─────────────────────────┐
│ Selection Context       │
│ entity_type + entity_id │
└───────────┬─────────────┘
            │ deterministic template
            ▼
┌─────────────────────────┐
│ Prompt Builder          │
│ "Tell me more about…"   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LLM Commentary Answer   │
│ + optional debug panel  │
└─────────────────────────┘
```

---

## Primary Knowledge Corpus: Usage Graph

The first and most important knowledge corpus in Databricks Usage Copilot is a **usage graph** that models how Databricks workloads actually operate in practice.

Rather than treating usage data as flat tables or isolated logs, the Copilot represents the system as a graph of connected entities, including:

- Jobs
- Job runs
- Compute resources (clusters / warehouses)
- Compute usage records
- Lifecycle and failure events

This structure mirrors how platform operators reason about real systems:  
*jobs run on compute, produce usage, encounter events, and impact cost, reliability, and performance.*

### Why a Graph?

Most of the underlying data originates in relational tables and could be queried with SQL alone. SQL excels at computing metrics, aggregations, and time-series statistics.

The graph exists to make **relationships and causality explicit**.

Many operational questions are fundamentally relationship-driven:

- Which jobs are responsible for the majority of cost on a given cluster?
- What failures correlate with specific compute configurations?
- Is a cost spike caused by more runs, longer runtimes, retries, or infrastructure churn?
- What downstream workloads are affected when an upstream job degrades?

Answering these questions requires navigating relationships across multiple entities. While this is possible with SQL joins, the resulting logic is often brittle, hard-coded, and difficult to reuse.

The graph provides a stable, navigable representation of the system that supports dynamic traversal and contextual reasoning.

### What the Graph Powers

The usage graph directly enables:

- **Selection-aware AI commentary** grounded in actual system structure
- **Root-cause style explanations** that connect symptoms to causes
- **Cross-entity insights** spanning jobs, compute, and events
- **Deterministic action chips** whose prompts are parameterized by graph context

When a user selects an entity in a report, the Copilot extracts the relevant subgraph and uses it as structured context for reasoning and explanation.

### Graph as a Reasoning Layer

The graph is not a replacement for SQL.

Instead, the system follows a clear separation of responsibilities:

- **SQL** computes metrics and aggregates from raw telemetry
- **The graph** models how entities relate and interact
- **The LLM** uses that structure to explain *why* those metrics look the way they do

This intermediate graph representation acts as a semantic compression layer, turning large volumes of raw data into meaningful, explainable structure that language models can reason over reliably.

### Implementation and Evolution

The current implementation uses a lightweight, local graph representation built from SQLite-backed usage data. This keeps the project easy to run locally and focused on reasoning and UX rather than infrastructure.

The graph schema is intentionally designed to align with production graph databases. As the project evolves, the same model can be upgraded to a system like Neo4j to support larger datasets, deeper traversals, and multi-tenant views without changing the Copilot's reasoning model.

> **Design principle:** The graph is not an optimization — it is the model.

---

## Dataset Overview

All data lives in a local **SQLite database**:

```
data/usage_rag_data.db
```

Generated from:

* `create_usage_tables.sql`
* `seed_usage_tables.sql`
* `database_setup.py`

### Tables

| Table               | Description                 |
| ------------------- | --------------------------- |
| `workspace`         | Org units / cost centers    |
| `users_lookup`      | Users + departments         |
| `jobs`              | Scheduled jobs              |
| `job_runs`          | Job executions              |
| `compute_usage`     | DBUs, cost, utilization     |
| `non_job_compute`   | Warehouses / all-purpose    |
| `events`            | Lifecycle + eviction events |
| `eviction_details`  | Spot eviction telemetry     |
| `sql_query_history` | Ad-hoc SQL usage            |
| `date_series`       | Synthetic daily ranges      |

The schema is intentionally **relational and interconnected**, ideal for GraphRAG.

---

## Architecture Overview

```text
SQLite Usage DB
   ↓ SQL
Reports Registry
   ↓
Streamlit Dashboard
   - Visualization Pane
   - Commentary Pane (LLM)
   - Deterministic Chips
   - Debug Toggle
   - Filters & Export
   ↓
Prompt Builder + Context Assembler
   ↓
GraphRAG (usage graph)
   +
Docs RAG (Databricks docs)
   ↓
LLM (OpenAI / Gemini / Grok)
   ↓
MCP Tools (optional)
```

---

## Setup & Installation

### 1. Clone

```bash
git clone https://github.com/ChicagoDro/AI-Portfolio
cd AI-Portfolio
```

### 2. Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

---

## Running the System (Makefile)

### Available Targets

* `make db` – build & seed SQLite database
* `make index` – build telemetry FAISS index
* `make docs` – build Databricks docs FAISS index
* `make app` – launch Streamlit UI
* `make all` – db + index + docs + app
* `make clean` – remove generated artifacts

### First Run

```bash
make all
```

Streamlit will launch at:

```
http://localhost:8501
```

---

## Project Structure

```text
src/
  app.py                    # Streamlit UI with filters & export
  chat_orchestrator.py      # Routing + prompts + citations
  graph_model.py            # Nodes + edges
  graph_retriever.py        # GraphRAG traversal
  ingest_embed_index.py     # Telemetry embeddings
  ingest_databricks_docs.py # Databricks docs ingestion
  reports/
    base.py                 # Report specification interface
    registry.py             # Report catalog
    job_cost.py             # Job cost & reliability report
    job_cost_pareto.py      # Pareto concentration analysis
    spot_risk_by_job.py     # Spot eviction risk analysis
    compute_type_cost.py    # Cross-type cost comparison
    anomaly_detection.py    # Statistical anomaly detection
```

---

## Technical Highlights

### Statistical Anomaly Detection
The **Cost Anomaly Detection** report implements a custom statistical model for identifying unusual spend patterns:

- **Baseline calculation** using 30-day rolling averages with configurable lookback windows
- **Multi-factor deviation scoring** combining cost variance, frequency changes, and reliability signals
- **Severity classification** (Critical/High/Medium/Low) based on statistical thresholds
- **Temporal correlation** to distinguish one-time spikes from sustained pattern changes

This isn't just "show me expensive things" – it's detecting *changes* in behavior that require investigation.

### Agent-Based Root Cause Investigation
The **Auto-Investigate** capability demonstrates agentic AI workflows:

- **Multi-step reasoning** – agent breaks investigation into phases (characterization → temporal analysis → configuration changes → data volume → failures)
- **Hypothesis ranking** – generates ranked hypotheses with confidence levels based on evidence
- **Actionable next steps** – proposes specific verification queries and remediation paths
- **Graph-aware context** – leverages the usage graph to understand entity relationships during investigation

Unlike static prompts, the agent *reasons* through the investigation systematically.

### GraphRAG for Operational Telemetry
Traditional RAG works for documents. Operational telemetry requires **relationship-aware retrieval**:

- **Graph model** captures how jobs, runs, compute, and events relate
- **Subgraph extraction** around selected entities (BFS traversal with configurable hops)
- **Semantic compression** – graph structure encodes meaning that pure vectors miss
- **Causal reasoning** – "did evictions cause this failure?" requires traversing event→usage→run edges

The graph isn't an optimization – it's the reasoning model.

---

## Why This Matters (Portfolio Value)

This project demonstrates how to build **enterprise-ready AI copilots** that:

* Are deterministic instead of guess-driven
* Separate facts from explanations
* Support auditing and debugging
* Earn trust from engineers and FinOps teams

> **Chat-first copilots optimize for convenience.
> Report-driven copilots optimize for correctness, trust, and scale.**

---

## Roadmap: Depth Over Breadth

This project prioritizes **architectural depth and novel capabilities** over incremental feature expansion.

### Phase 1: Deterministic Foundation ✅ **Complete**

**Achievements:**
- 5 production-ready reports demonstrating the report-driven pattern
- GraphRAG implementation for operational telemetry
- Statistical anomaly detection with custom baseline models
- Agent-based investigation (Auto-Investigate capability)
- Dual-corpus RAG (telemetry + Databricks docs)

**Value Demonstrated:**
- Report-driven AI is more trustworthy than chat-first
- GraphRAG enables causal reasoning over operational data
- Deterministic action chips guide users to valuable questions

---

### Phase 2: MCP Integration 🔄 **In Progress**

**Goal:** Enable agentic workflows with real-world tool integration

**Planned MCP Servers:**

1. **Databricks Workspace MCP Server**
   - Query actual job configurations and cluster settings via Databricks API
   - Validate AI recommendations against real infrastructure state
   - Enable "show me the config" → actual API data, not hallucinated configs
   - **Value:** Grounds AI recommendations in reality

2. **Filesystem MCP Server**
   - Generate configuration files (job JSON, cluster policies, Terraform)
   - Create diffs showing before/after for recommendations
   - Export analysis reports and visualizations
   - **Value:** Makes recommendations actionable, not just advisory

3. **Web Search MCP Server**
   - Look up current Databricks pricing and product updates
   - Cross-reference recommendations with latest best practices
   - Validate assumptions against external knowledge
   - **Value:** Keeps recommendations current and fact-checked

**Architecture Impact:**
- Agents can now **verify** recommendations before presenting them
- AI can generate **actual configuration files**, not just suggestions
- Recommendations become **executable**, not just descriptive

**Conference Talking Points:**
- "How MCP bridges the gap between AI advice and production systems"
- "Making AI recommendations verifiable and actionable"
- "When to use agents vs prompts in production AI systems"

---

### Phase 3: Evaluation Framework 📋 **Next**

**Goal:** Treat AI behavior as a testable system, not a black box

**Planned Components:**

1. **Deterministic Test Suite**
   - Regression tests for action chip prompts (ensure stability)
   - Ground truth evaluation sets for key scenarios
   - SQL validation (reports return expected data for known inputs)

2. **Response Quality Metrics**
   - Groundedness checks (does answer reference actual telemetry?)
   - Citation accuracy (are Databricks docs links correct?)
   - Recommendation validity (can suggestions actually be implemented?)

3. **Performance Monitoring**
   - Token usage and cost tracking per interaction
   - Latency budgets for report loading and AI generation
   - Cache hit rates and retrieval quality

**Architecture Impact:**
- AI behavior becomes **repeatable and testable**
- Regressions are **caught before deployment**
- System performance is **measurable and improvable**

**Conference Talking Points:**
- "How to test AI systems that use LLMs"
- "Evaluation strategies for enterprise AI copilots"
- "Moving from demos to production: the testing gap"

---

### Phase 4: Multi-Agent Orchestration 📋 **Future**

**Goal:** Coordinate specialized agents for complex investigations

**Planned Capabilities:**

- **Investigation Agent** – Deep-dive root cause analysis (already prototyped in Auto-Investigate)
- **Configuration Agent** – Generate and validate config changes using MCP tools
- **Comparison Agent** – Evaluate alternative optimization strategies in parallel
- **Verification Agent** – Test recommendations in sandbox environments before applying

**Orchestration Patterns:**
- Sequential workflows (diagnose → optimize → verify)
- Parallel comparisons (evaluate multiple strategies simultaneously)
- Escalation logic (simple fixes vs complex migrations)

**Architecture Impact:**
- Move from single-shot AI to **multi-step reasoning**
- Enable **comparative analysis** (which optimization is best?)
- Support **validation loops** (test before recommending)

**Conference Talking Points:**
- "Designing multi-agent workflows for operational AI"
- "When agents should collaborate vs work independently"
- "Orchestration patterns for production AI systems"

---

## Why This Roadmap?

### The Production Gap

Current state: The copilot provides **accurate analysis and recommendations**, but stops short of production readiness in three critical areas:

**1. Recommendations aren't verifiable**
- AI suggests "increase cluster size to 8 nodes" 
- But what if the current config is already 8 nodes?
- Without MCP, the AI can't check actual state

**2. System behavior isn't testable**
- Prompts can drift over time
- No way to catch regressions before users see them
- Can't measure if improvements actually improve outcomes

**3. Complex workflows require manual orchestration**
- Users must run investigation → get recommendation → verify validity → apply fix
- Each step is a separate interaction
- No way to automate multi-step reasoning

### The Solution Path

**MCP Integration** solves the verification problem:
- Recommendations reference **actual infrastructure state**, not assumptions
- "Your cluster is currently 4 nodes; increasing to 8 would cost $X more per day"
- AI can verify claims before making them
- **Enterprise value:** Recommendations become trustworthy enough to act on

**Evaluation Framework** solves the reliability problem:
- Test suite catches prompt drift and regressions
- Metrics prove the system is getting better (or worse)
- Ground truth validation ensures answers match reality
- **Enterprise value:** System behavior becomes predictable and measurable

**Multi-Agent Orchestration** solves the workflow problem:
- Investigation agent finds root cause
- Configuration agent generates fix
- Verification agent validates fix against real workspace
- Comparison agent evaluates alternatives
- **Enterprise value:** End-to-end automation, not just advice

### Why This Order?

**Phase 2 (MCP) before Phase 3 (Evaluation):**
- Can't test recommendation validity without MCP tools to check ground truth
- Evaluation framework needs MCP to verify "did the AI get it right?"

**Phase 3 (Evaluation) before Phase 4 (Agents):**
- Multi-agent systems are complex; need testing foundation first
- Can't coordinate agents without measuring if they're working correctly

**Phase 4 (Agents) builds on both:**
- Agents use MCP tools to interact with real systems
- Evaluation framework measures agent success rates
- Together: testable, verifiable, automated workflows

### What This Demonstrates

- **MCP integration** → How to ground AI in reality
- **Evaluation** → How to test AI systems
- **Multi-agent** → How to orchestrate complex workflows

Each phase moves closer to **production-grade enterprise AI**, not just demos.

---

## Design Decisions

### Why Report-Driven Instead of Chat-First?

**Chat-first copilots** rely on the LLM to infer intent from natural language. This leads to:
- Ambiguous queries ("show me cost" → which time range? which jobs? what granularity?)
- Inconsistent results (same question, different SQL each time)
- Difficult debugging (what did the LLM decide to query?)
- Trust issues (is this number right?)

**Report-driven copilots** use deterministic reports to define intent:
- SQL is explicit and reviewable
- Results are reproducible
- Debugging is straightforward (view the SQL)
- Trust is earned through transparency

The LLM's role shifts from "guess what the user wants" to "explain what the data means."

### Why Action Chips Instead of Free-Form Prompts?

**Free-form prompts** are flexible but:
- Require users to know what to ask
- Lead to vague questions ("how can I optimize?")
- Produce generic answers
- Waste tokens on prompt engineering

**Action chips** are constrained but:
- Guide users toward valuable questions
- Generate specific, parameterized prompts
- Produce targeted, actionable answers
- Reduce wasted compute

The best of both worlds: chips for common patterns, free-form for exploration.

### Why GraphRAG Instead of Pure Vector Search?

**Vector search alone** works for documents but struggles with:
- Relationship queries ("which jobs share this cluster?")
- Causality ("did evictions cause this failure?")
- Multi-hop reasoning ("what's affected downstream?")

**GraphRAG** adds:
- Explicit relationships between entities
- Traversable structure for multi-hop queries
- Causal reasoning pathways
- Semantic compression (graph = meaning)

SQL computes metrics. Graph models relationships. LLM explains causality.

---

## Contributing

This is a portfolio project demonstrating enterprise AI copilot design patterns. While not open for external contributions, the code is structured to be educational and reusable.

**Key learnings shared:**
- Deterministic vs probabilistic AI interfaces
- GraphRAG for operational telemetry
- Action chip taxonomy for guided AI interactions
- MCP integration for agentic workflows
- Report-driven UX patterns

---

## License

MIT License - See LICENSE file for details.

---

## Contact

**Pete Tamisin**
📧 pete@tamisin.com
🔗 [LinkedIn](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

*Building AI systems that earn trust through transparency, not magic.*
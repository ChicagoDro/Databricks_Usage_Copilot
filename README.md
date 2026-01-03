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

## Recent Improvements (Jan 2026)

### Action Chip Optimization
- **Eliminated redundancies** across reports (42% reduction in chip count)
- **Conditional chips** only appear when justified by data (e.g., "Why a spike?" only for high-deviation entities)
- **Report-specific value** – each chip provides unique insights tied to the report's lens
- **Taxonomy organization** – chips grouped by Understand/Diagnose/Optimize/Monitor

### Interactive Filters & Export
- **Date range filters** with quick buttons (Last 7d, 30d, 90d)
- **Workspace filters** for multi-tenant analysis
- **CSV export** from every report
- **Loading indicators** during data fetch and AI generation
- **Refresh data** button to clear cache

### Report Quality
- **All reports respect filters** – SQL queries dynamically filtered by date range and workspace
- **Commentary auto-clears** when switching reports (prevents stale context confusion)
- **Consistent chip quality** across all 5 reports

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

## Roadmap

The Copilot is intentionally built as an extensible system. Upcoming work focuses on expanding coverage across the pillars of data engineering and introducing higher-level reasoning on top of deterministic foundations.

### MCP (Model Context Protocol) Integration (In Progress)

The next major enhancement is **MCP server integration** to enable agentic workflows with external tools.

**Planned MCP Capabilities:**

1. **Databricks Workspace MCP Server**
   - Query job configurations, cluster settings, and workspace metadata
   - Validate recommendations against actual infrastructure
   - Fetch real-time job run status and logs
   - Enable "show me the config" → actual API data, not hallucinated

2. **Filesystem MCP Server**
   - Read/write configuration files (job JSON, cluster policies)
   - Generate configuration diffs for recommendations
   - Export reports and analysis to local filesystem
   - Support "save this analysis" workflows

3. **Web Search MCP Server**
   - Look up current Databricks pricing
   - Find recent product updates and best practices
   - Validate assumptions against external knowledge
   - Enable "what's the current spot discount?" → real-time pricing

**MCP Integration Benefits:**
- **Grounded recommendations** – Validate against actual workspace state
- **Actionable outputs** – Generate configuration files, not just text
- **Current information** – Access real-time pricing and product updates
- **Verifiable claims** – Cross-reference recommendations with actual data

**Integration Timeline:**
1. ✅ Phase 1: Deterministic foundation (reports, chips, filters) – **Complete**
2. 🔄 Phase 2: MCP server integration – **Next**
3. 📋 Phase 3: Multi-agent orchestration
4. 📋 Phase 4: Evaluation framework

### Pillar-Based Reports (Planned)

Additional reports will be added under the following pillars:

- **Cost Management** *(5/8 complete)*
  - ✅ Cost anomalies vs baseline
  - ✅ Cost concentration (Pareto)
  - ✅ Spot risk analysis
  - 📋 Cost efficiency ($ per run, $ per GB processed)
  - 📋 Spot vs on-demand counterfactual analysis

- **Reliability** *(Planned)*
  - Job reliability scorecards (success rate, retries, SLA breaches)
  - Failure pattern analysis (error signatures, root causes)
  - Fragility detection (jobs that barely succeed)

- **Performance & Efficiency** *(Planned)*
  - Runtime regression detection (p50 / p95 drift)
  - Resource utilization efficiency (CPU/memory)
  - Shuffle and spill hotspots

- **Data Quality** *(Planned)*
  - Dataset freshness monitoring
  - Volume drift detection
  - Upstream/downstream blast radius analysis

- **Resilience** *(Planned)*
  - Recovery time metrics (MTTR)
  - Sensitivity to configuration or code changes
  - Single points of failure identification

Reports not yet implemented are visible in the UI as disabled placeholders to make the system's intended scope explicit.

### Agent-Based Capabilities (Future)

Once MCP tools are integrated, the Copilot will introduce **agents** that can:

- Execute multi-step investigations across reports
- Compare alternative optimization strategies using real workspace data
- Propose remediation plans with verification steps (via Databricks API)
- Generate configuration files for recommended changes
- Escalate from diagnosis → optimization → monitoring automatically

Agents will build on deterministic chips and MCP tools rather than replacing them.

### Evaluation & Testing (Future)

Planned work also includes:

- Prompt and response regression tests
- Deterministic evaluation sets for key scenarios
- Groundedness checks against report data
- Cost and latency tracking for AI interactions
- MCP tool call accuracy metrics

The goal is to treat AI behavior as a **testable system**, not a black box.

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
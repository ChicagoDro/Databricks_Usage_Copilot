# Data Reliability Copilot

**Pete Tamisin** – Technical GTM Leader • AI & Data Engineering Architect • Builder & Teacher
Chicago, IL

* 20+ years designing data & AI platforms (Director at Capital One, ex-Databricks, 2x Series A startup exits, x-Siemens, x-Motorola)
* Focused on **modern data platforms**, **context-aware RAG systems**, and **enterprise GenAI adoption**
* Passionate about **teaching** and helping teams ship real-world AI systems

📧 Email: `pete@tamisin.com`
🔗 LinkedIn: [https://www.linkedin.com/in/peter-tamisin-50a3233a/](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

---

## Overview

**Data Reliability Copilot** is a deterministic, report-driven AI assistant for analyzing cost, reliability, performance, and operational risk in Data Engineering workloads.


Unlike chat-first copilots, this project is **report-driven and deterministic**:

* SQL defines the facts
* Reports define the question
* User selections define scope
* LLMs explain results instead of inventing them

The result is an **enterprise-grade AI copilot** that is explainable, debuggable, and trustworthy.

---

## Core Design Principle

> **Don’t let the model guess what the user meant.**
> Use deterministic reports to define intent, and use the LLM to explain the result with context.

This project deliberately avoids “blank chat box” UX. Instead:

* **Reports** define what is being analyzed
* **Clicks** define what needs explanation
* **Prompts** are deterministic and repeatable
* **LLMs** provide narrative, root-cause hypotheses, and next actions

---

## Deterministic Reports (Not Chat Guessing)

Each report is powered by:

* Explicit SQL
* Known semantic meaning
* Defined entity mappings
* Predefined drill actions

Current reports include:

* **Job Cost**
  Stacked horizontal bars by job, segmented by spot vs on-demand ratio

* **Total Cost by Compute Type**
  Sorted bar chart (avoids misleading pie charts)

* **Pareto Job Cost Concentration**
  Cumulative cost contribution curve highlighting top drivers

* **Spot Risk Exposure by Job**
  Ranks jobs by spot ratio and eviction signals

Reports are the **interface**.
AI is the **commentary layer**.

---

## Deterministic Action Chips (Key Differentiator)

A core design principle of Databricks Usage Copilot is that **AI actions are deterministic, contextual, and intentional**.

Rather than free-form prompting, the UI presents **action chips** that are:

- **Deterministic** – each chip maps to a fixed prompt template
- **Context-aware** – prompts are parameterized by the selected entity (job, cluster, warehouse, etc.)
- **Stable** – chip identity and ordering do not change across runs
- **Explainable** – users can always inspect the exact prompt that was executed

### Chip Taxonomy

Action chips are organized into four conceptual lanes that mirror how platform operators think:

- **Understand** – What is this? What does “good” look like?
- **Diagnose** – Why is this happening? What changed?
- **Optimize** – What should I change to improve cost, reliability, or performance?
- **Monitor** – How do I validate improvements and prevent regressions?

This structure aligns directly with the pillars of data engineering (cost, reliability, performance, resilience, data quality) and provides a repeatable “operational playbook” for each report and selection.

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
│ “Tell me more about…”   │
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

The graph schema is intentionally designed to align with production graph databases. As the project evolves, the same model can be upgraded to a system like Neo4j to support larger datasets, deeper traversals, and multi-tenant views without changing the Copilot’s reasoning model.

> **Design principle:** The graph is not an optimization — it is the model.

---

## Databricks Documentation as a Second Corpus (With Citations)

The copilot ingests **official Databricks documentation** (AWS Compute section) as a **separate vector corpus**.

This allows the system to:

* Explain *what* a feature is (autoscaling, spot, DBUs, warehouses)
* Provide accurate configuration guidance
* Avoid generic or hallucinated advice

### Dual-Corpus Retrieval

* **Telemetry corpus** → your usage data, reports, and graph context
* **Docs corpus** → Databricks product documentation

Routing is intentional:

* Entity-anchored questions prioritize telemetry
* “What is / how does / how do I configure” questions retrieve docs
* Some answers use both

### Real Citations (Not Just Debug Info)

When documentation is used, answers include a deterministic **Sources** section, for example:

```
Sources (Databricks Docs):
- Spot Instances — https://docs.databricks.com/...
- Autoscaling Clusters — https://docs.databricks.com/...
```

Citations are appended **programmatically**, not left to the model to remember.

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

## Architecture Overview (Reports + Graph + Docs)

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
   ↓
Prompt Builder + Context Assembler
   ↓
GraphRAG (usage graph)
   +
Docs RAG (Databricks docs)
   ↓
LLM
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
  app.py                    # Streamlit UI
  chat_orchestrator.py      # Routing + prompts + citations
  graph_model.py            # Nodes + edges
  graph_retriever.py        # GraphRAG traversal
  ingest_embed_index.py     # Telemetry embeddings
  ingest_databricks_docs.py # Databricks docs ingestion
  reports/
    registry.py             # Report definitions
```

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

### Pillar-Based Reports (Planned)

Additional reports will be added under the following pillars:

- **Cost Management**
  - Cost anomalies vs baseline
  - Cost efficiency ($ per run, $ per GB processed)
  - Spot vs on-demand counterfactual analysis

- **Reliability**
  - Job reliability scorecards (success rate, retries, SLA breaches)
  - Failure pattern analysis (error signatures, root causes)
  - Fragility detection (jobs that barely succeed)

- **Performance & Efficiency**
  - Runtime regression detection (p50 / p95 drift)
  - Resource utilization efficiency (CPU/memory)
  - Shuffle and spill hotspots

- **Data Quality**
  - Dataset freshness monitoring
  - Volume drift detection
  - Upstream/downstream blast radius analysis

- **Resilience**
  - Recovery time metrics (MTTR)
  - Sensitivity to configuration or code changes
  - Single points of failure identification

Reports not yet implemented are visible in the UI as disabled placeholders to make the system’s intended scope explicit.

### Agent-Based Capabilities (Future)

Once deterministic reports and action chips are in place, the Copilot will introduce **agents** that can:

- Execute multi-step investigations across reports
- Compare alternative optimization strategies
- Propose remediation plans with verification steps
- Escalate from diagnosis → optimization → monitoring automatically

Agents will build on deterministic chips rather than replacing them.

### Evaluation & Testing (Future)

Planned work also includes:

- Prompt and response regression tests
- Deterministic evaluation sets for key scenarios
- Groundedness checks against report data
- Cost and latency tracking for AI interactions

The goal is to treat AI behavior as a **testable system**, not a black box.


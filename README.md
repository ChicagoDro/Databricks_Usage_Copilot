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

**Databricks Usage Copilot** is an AI-powered analytics system for exploring **Databricks usage, cost, and reliability**.

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

Every meaningful data point produces deterministic action chips such as:

* `Tell me more about job_id = J-123`
* `Explain spend for compute_type = SQL_WAREHOUSE`
* `Why is this job risky from a spot perspective?`

If charts cannot host clickable marks cleanly, chips are rendered below the visualization.

This ensures:

* Repeatable prompts
* Predictable behavior
* Auditable reasoning paths

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

## Future Enhancements

* Additional deterministic reports (utilization efficiency, cost drift)
* Evaluation harnesses (groundedness, completeness)
* Export graph to Neo4j for scale
* Role-based report views


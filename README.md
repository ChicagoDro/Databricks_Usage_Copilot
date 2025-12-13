# Who Am I?
**Pete Tamisin** – Technical GTM Leader • AI & Data Engineering Architect • Builder & Teacher
Based in Chicago, IL.

* 20+ years designing data & AI platforms (Dir. at Capital One, ex-Databricks, 2x series A startup exits, x-Siemens, x-Motorola)
* Focused on **modern data platforms**, **Context aware RAG systems**, and **enterprise GenAI adoption**
* Passionate about **teaching** and helping teams ship real-world AI systems

📧 Email: `pete@tamisin.com`
🔗 LinkedIn: [peter-tamisin-50a3233a](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

---

# Databricks Usage Copilot

An AI-powered analytics copilot for exploring Databricks usage, cost, and reliability.
This project combines **SQL-backed analytics, GraphRAG, and deterministic LLM prompts** to deliver explainable, decision-ready insights — without relying on guess-driven chat interactions.

This project ingests structured Databricks-like operational data into:

- A **SQLite database** with realistic usage tables  
- A **FAISS vector index** for semantic retrieval  
- An **in-memory graph** of your environment (org units → users → jobs → runs → usage → events → evictions → SQL queries)  
- A **Graph-aware orchestrator** that performs graph expansion + semantic retrieval  
- A **Streamlit UI** + **CLI** that show both the answer *and* “how the AI reasoned”

### 🎯 Project Goals

Provide clear visibility into Databricks usage and cost drivers
Enable reliable drill-downs into jobs, compute types, and execution behavior
Demonstrate how AI can explain data instead of inventing it
Showcase production-style patterns for enterprise AI copilots


---

## Core Design Principle

> **Don’t let the model guess what the user meant.**  
> Use **deterministic reports** to define the question, and use the LLM to explain the result with context.

This is a different (and more enterprise-friendly) UX than “chat-first RAG”:

- **Reports** define “what we’re looking at”
- **Clicks** define “what we want explained”
- **Prompts** are deterministic and repeatable
- **LLM** provides narrative, root-cause hypotheses, and next actions

---

## Deterministic Reports (Not Chat Guessing)

Each report is powered by explicit SQL and a known semantic meaning.  
The chart/table is the interface; the AI is the commentary layer.

Example reports (current + planned):

- **Job Cost** — stacked horizontal bars by job, segmented by spot vs on-demand ratio
- **Total Cost by Compute Type** — recommended as a **sorted bar chart** (not a pie chart)
- **Pareto Job Cost Concentration** — cumulative contribution curve / Pareto view
- **Spot Risk Exposure by Job** — rank jobs by spot ratio + eviction signals

---

## Deterministic Action Chips (Key Differentiator)

Every meaningful data point in a report produces deterministic “action chips” (buttons) that trigger a known prompt, for example:

- Clicking a job bar → `Tell me more about job_id = J-...`
- Clicking a compute type → `Explain spend for compute_type = ...`
- Clicking “top driver” → `Explain why this driver is expensive and what to optimize`

If a visualization itself can’t host clickable links cleanly, chips are rendered below the chart as “drill actions” for the visible marks.

---

## One-Diagram Overview: Report → Selection → Prompt → Answer

```text
┌───────────────────────┐
│  Report (SQL query)    │
│  Chart / Table / KPI   │
└───────────┬───────────┘
            │ click a mark / row / chip
            ▼
┌───────────────────────┐
│ Selection → Context     │
│ entity_type + entity_id │
└───────────┬───────────┘
            │ deterministic template
            ▼
┌───────────────────────┐
│ Prompt Builder          │
│ "Tell me more about ...│
│  include X, Y, Z"       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ LLM Commentary Answer   │
│ + optional debug panel  │
└───────────────────────┘
````

---

## 📦 Dataset Overview

All data is stored in a local **SQLite database**:

```text
data/Databricks_Usage/usage_rag_data.db
````

This database is generated from:

* `data/Databricks_Usage/create_usage_tables.sql` — table definitions
* `data/Databricks_Usage/seed_usage_tables.sql` — synthetic but realistic seed data
* `database_setup.py` — orchestration script that creates & seeds the database

### Tables & Concepts

| Table               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `workspace`        | Organizational units (cost centers)                          |
| `users_lookup`      | Users, departments, and OU membership                        |
| `jobs`              | Scheduled Databricks jobs with metadata + tags               |
| `job_runs`          | Daily executions of jobs with status + cluster settings      |
| `compute_usage`     | DBU usage, cost, instance type, CPU/memory metrics           |
| `non_job_compute`   | SQL Warehouses & All-Purpose Clusters                        |
| `events`            | Cluster lifecycle events and spot eviction-related events    |
| `eviction_details`  | Detailed spot eviction telemetry                             |
| `sql_query_history` | Ad-hoc SQL query executions (user, warehouse, duration, SQL) |
| `date_series`       | Synthetic daily range used to generate runs/usage            |

### Dataset Purpose

The dataset is intentionally **relational** and **interconnected** to mimic real telemetry:

* **Jobs** → **job runs** → **compute usage**
* **Usage** → **events** → **evictions**
* **Users** → **queries** and **org units**

This makes it ideal for demonstrating:

* **Graphs & relationships** (jobs → runs → usage → events → queries)
* **Context assembly via traversal**
* **Hybrid retrieval** (semantic + structural)
* **RAG systems for FinOps / observability / governance**

---

## 🕸️ Architecture Overview (Graph + Reports + UI)

At a high level:

1. **SQLite DB** holds structured Databricks usage data.
2. The **report registry** defines each report:

   * SQL query
   * visualization type
   * which columns become “entities”
   * chip templates for deterministic prompts
3. The **Streamlit dashboard** renders the chosen report in the visualization pane.
4. User clicks a mark/row/chip → the system builds a deterministic prompt and executes it.
5. The **LLM** returns commentary in the always-present commentary pane.
6. If debug mode is enabled, the UI shows the underlying SQL, prompt, and any additional reasoning artifacts.

### Consolidated System Diagram

```text
                 ┌──────────────────────────┐
                 │   SQLite Usage DB         │
                 │ (jobs, runs, usage, ...)  │
                 └───────────┬──────────────┘
                             │ SQL
                             ▼
                 ┌──────────────────────────┐
                 │  Reports Registry         │
                 │  - SQL per report         │
                 │  - viz config             │
                 │  - entity mapping         │
                 │  - chip templates         │
                 └───────────┬──────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                        │
│  Sidebar: Report links + Debug toggle                         │
│                                                              │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │ Visualization Pane   │   │ Commentary Pane (LLM)         │  │
│  │ (chart/table/KPI)    │   │ "Tell me more about ..."      │  │
│  │ click → context      │   │ + freeform prompt box         │  │
│  └──────────┬──────────┘   └───────────┬───────────────────┘  │
│             │ selection                 │ deterministic prompt  │
│             ▼                           ▼                       │
│      ┌───────────────────────────────────────────────┐         │
│      │ Prompt Builder + Context Assembler             │         │
│      └──────────────────────┬────────────────────────┘         │
│                             ▼                                  │
│                      ┌──────────────┐                          │
│                      │      LLM      │                          │
│                      └──────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup & Installation

> These instructions assume a working Python 3.10+ and `git`.

### 1. Clone the repository

```bash
git clone https://github.com/ChicagoDro/AI-Portfolio
cd AI-Portfolio
```

### 2. Create & activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with at least:

```env
LLM_PROVIDER=openai          # or gemini, etc.
OPENAI_API_KEY=sk-...        # if using OpenAI

# Optional: override models
# OPENAI_CHAT_MODEL=gpt-4.1-mini
# OPENAI_EMBED_MODEL=text-embedding-3-small
```

---

# 🚦 How to Run the System (Using the Makefile)

This project now includes a convenient **Makefile** to automate the entire workflow:

* Creating the SQLite database
* Building the FAISS vector index
* Launching the Streamlit UI
* Cleaning generated artifacts

You no longer need to manually run `database_setup.py`, `ingest_embed_index.py`, or `streamlit run …`.
Just use `make`.

---

## 🧰 Available Make Targets

### **`make db` — Create & Seed the SQLite Database**

This target:

1. Runs `database_setup.py`
2. Creates `data/usage_rag_data.db`
3. Executes:

   * `create_usage_tables.sql`
   * `seed_usage_tables.sql`

This gives you a **fully populated Databricks-like usage database**, including:

* Jobs
* Job runs
* Compute usage
* Events
* Evictions
* SQL query history
* Users & org units

---

### **`make index` — Build the FAISS Semantic Index**

This target:

1. Runs the full domain ingestion pipeline (`src/ingest_usage_domain.py`)
2. Embeds every usage document
3. Builds a FAISS vector index
4. Saves it to:

```
indexes/usage_faiss/
```

The index is what allows the assistant to:

* Perform semantic retrieval
* Pick anchor nodes
* Trigger graph expansion for GraphRAG

---

### **`make app` — Launch the Streamlit UI**

This target:

1. Sets the correct `PYTHONPATH`
2. Boots the Streamlit interface at:

```
http://localhost:8501
```

The UI includes:

* Report navigation in the sidebar
* Visualization pane (charts/tables)
* Commentary pane (LLM)
* Optional debug info (SQL/prompt/context, if enabled)

---

### **`make all` — Full Pipeline: DB → Index → UI**

This is the smoothest end-to-end experience.

Running:

```
make all
```

will:

1. Build / rebuild the SQLite database
2. Build / rebuild the FAISS index
3. Launch the Streamlit app immediately

Perfect for first-time setup or after making schema changes.

---

### **`make clean` — Remove All Generated Artifacts**

This target deletes:

* The SQLite database
* The FAISS index directory

Useful when:

* You want to regenerate everything from scratch
* You updated the schema or seed data
* You’re debugging ingestion or graph-building issues

---

## 🎯 Recommended Workflow

To set up the system for the first time:

```bash
make all
```

After that, when you update:

* Seed data → run `make db index`
* Embedding model or ingestion logic → run `make index`
* UI only → run `make app`
* Reset everything → run `make clean && make all`

---

## 📌 Under the Hood (What Each Step Actually Does)

| Make Target  | What Happens Internally                                                             |
| ------------ | ----------------------------------------------------------------------------------- |
| `make db`    | Executes Python schema builder → creates all tables → inserts all synthetic records |
| `make index` | Generates RAG docs → computes embeddings → builds FAISS index → stores metadata     |
| `make app`   | Loads report registry → runs report SQL → renders UI → wires click→prompt→LLM loop  |
| `make clean` | Removes SQLite DB + FAISS index folder                                              |
| `make all`   | `db` + `index` + `app`                                                              |

---

## 📁 Project Structure

```text
AI-Portfolio/
  database_setup.py               # SQLite setup + schema population
  .env                            # environment variables (ignored in git)
  requirements.txt                # dependencies

  data/
    create_usage_tables.sql       # Creates Databricks Usage schema
    seed_usage_tables.sql         # Loads sample data
    usage_rag_data.db             # Generated SQLite DB

  indexes/
    usage_faiss/                  # FAISS index (created at runtime)

  src/
    config.py                     # Paths + provider config
    ingest_usage_domain.py        # SQL → RAG docs
    ingest_embed_index.py         # Docs → embeddings → FAISS
    graph_model.py                # Nodes & edges & adjacency (HAS_* edges)
    graph_retriever.py            # Graph-aware retriever (GraphRAG)
    chat_orchestrator.py          # LLM orchestration + routing + debug
    app.py                        # Streamlit UI (reports + commentary)
    reports/                      # Report definitions (SQL + viz + chip mapping)
      registry.py                   # Report registry (navigation + metadata)
```

---

## 🔭 Future Enhancements

* Add richer drill paths (multi-hop exploration) while staying deterministic.
* Add evaluation harnesses (report accuracy checks, LLM groundedness checks).
* Export the in-memory graph into **Neo4j** for large-scale graph analytics.


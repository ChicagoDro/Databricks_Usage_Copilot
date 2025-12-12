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

A fully-featured **enterprise Retrieval-Augmented Generation (RAG)** system that turns Databricks-style platform usage telemetry into an interactive AI assistant.

This project ingests structured Databricks-like operational data into:

- A **SQLite database** with realistic usage tables  
- A **FAISS vector index** for semantic retrieval  
- An **in-memory graph** of your environment (org units → users → jobs → runs → usage → events → evictions → SQL queries)  
- A **Graph-aware orchestrator** that performs graph expansion + semantic retrieval  
- A **Streamlit UI** + **CLI** that show both the answer *and* “how the AI reasoned”

The result is an **AI copilot** capable of answering questions such as:

- “Why did this job cost so much yesterday?”  
- “Show me SQL queries contributing most to Finance warehouse spend.”  
- “What spot evictions have impacted ML workloads recently?”  
- “Which org unit owns the compute driving last week’s DBU spike?”  
- “Which jobs need optimizing based on total cost?”  

This repo demonstrates a **production-style architecture** for enterprise LLM applications built on **RAG + graphs + orchestration**.

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
| `company_ou`        | Organizational units (cost centers)                          |
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

## 🕸️ Architecture Overview (GraphRAG + UI)

At a high level:

1. **SQLite DB** holds structured Databricks usage data.
2. `ingest_usage_domain.py` reads the DB and turns rows into **RAG documents**.
3. `ingest_embed_index.py` embeds those docs and stores them in a **FAISS index**.
4. `graph_model.py` builds **nodes and edges** representing org structure and workload relationships.
5. `graph_retriever.py`:

   * Uses vector search to find **anchor documents** for a query
   * Expands a **subgraph** around those anchors using BFS
   * Collects all relevant docs for context
6. `chat_orchestrator.py`:

   * Classifies the **question type** (global aggregate, global top-N, local explanation, etc.)
   * Routes to **deterministic graph logic** when appropriate (e.g., “how many jobs?”, “which jobs need optimizing?”)
   * Otherwise calls the **GraphRAG retriever + LLM**
   * Returns an answer, a **graph explanation**, and the **LLM prompt + context** used
7. `app.py` (Streamlit UI):

   * Renders a chat interface
   * Shows an expandable **“How I reasoned”** panel with:

     * Graph subgraph summary
     * The prompt sent to the LLM
     * The context passed into the prompt

### Diagram

```text
                  ┌────────────────────────┐
                  │   SQLite Usage DB      │
                  │ (jobs, runs, usage,    │
                  │  events, queries, OU)  │
                  └───────────┬────────────┘
                              │ SQL (SELECT)
                 ┌────────────┴────────────┐
                 │ ingest_usage_domain.py  │
                 │  - Rows → RAG docs      │
                 └────────────┬────────────┘
                              │ Documents
                 ┌────────────┴────────────┐
                 │ ingest_embed_index.py   │
                 │  - Embeddings           │
                 │  - FAISS index          │
                 └────────────┬────────────┘
                              │ Vector search
                ┌─────────────┴─────────────┐
                │   graph_model.py          │
                │  - Nodes & edges          │
                │  - Adjacency (BFS)        │
                └─────────────┬─────────────┘
                              │ node_ids
                ┌─────────────┴─────────────┐
                │   graph_retriever.py      │
                │  - Vector anchors (FAISS) │
                │  - Graph expansion (BFS)  │
                │  - Context assembly       │
                └─────────────┬─────────────┘
                              │ context docs
                ┌─────────────┴──────────────┐
                │   chat_orchestrator.py     │
                │  - Question classifier     │
                │  - Global aggregates      │
                │  - GraphRAG + LLM          │
                │  - Debug prompt + context  │
                └─────────────┬──────────────┘
                              │ answer + debug
                ┌─────────────┴──────────────┐
                │   Streamlit UI (app.py)    │
                │  - Chat                    │
                │  - "How I reasoned" panel  │
                └────────────────────────────┘
```

---

## 🧭 Graph vs. Vector Routing

The assistant doesn’t treat every question the same.  
Some questions are best answered by **direct graph aggregation**, others by **GraphRAG** (vector + graph), and a few fall back to **plain vector RAG**.

At a high level:

- **Vector search** answers:  
  > “What are we talking about?”  
  (Find the most relevant nodes/docs.)

- **Graph traversal / aggregation** answers:  
  > “What else is related?” and “How do we compute totals, rankings, or coverage across the whole environment?”

### 🔀 Routing Strategies

The `DatabricksUsageAssistant` routes questions through three main paths:

1. **Global Graph Aggregates (Graph-only, no LLM reasoning needed)**
   - Examples:
     - “How many jobs are there?”
     - “How many users do we have?”
   - Behavior:
     - Skip vector search
     - Directly inspect the graph (count nodes by type)
     - Return a deterministic answer like:
       > “There are 5 jobs in this environment…”

2. **Global Usage & Top-N (Graph Aggregation + LLM Copyediting)**
   - Examples:
     - “Tell me about my Databricks usage.”
     - “Give me a summary of my job usage.”
     - “Which jobs need optimizing?”
     - “Top 3 most expensive jobs.”
   - Behavior:
     - Skip GraphRAG neighborhood
     - Traverse the graph:
       - `compute_usage` → `job_run` → `job`
     - Aggregate `cost_usd` per job
     - Rank, compute shares of total spend, etc.
     - Use the LLM to turn those numbers into a readable explanation.

3. **GraphRAG (Vector + Graph Expansion + LLM)**
   - Examples:
     - “Why is the HR Dashboard Prep job expensive?”
     - “What happened around the last eviction in Logistics?”
   - Behavior:
     - Use **vector search** over FAISS to find **anchor docs**
       (e.g., job J-HR-DASH, its runs, a usage record).
     - Expand a **subgraph** around those anchors with BFS:
       - job → runs → usage → events → evictions → user
     - Render that neighborhood into a context string.
     - Feed context + question into the LLM.
     - Return the answer and a “How I reasoned” explanation.

If the classifier can’t confidently categorize the question, the system defaults to the **GraphRAG** path.

---

### 🧠 Routing Flow Diagram

```text
               ┌────────────────────────────────────────┐
               │           User Question                │
               └────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Classifier + Heuristics│
                    │  (intent, entity_type)  │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
 ┌─────────────────┐   ┌──────────────────────┐  ┌──────────────────────┐
 │ Global Aggregate│   │ Global Usage / Top-N │  │   Local / Other      │
 │ (counts)        │   │ (cost, ranking)      │  │ (explanations, why?) │
 └──────┬──────────┘   └─────────┬────────────┘  └──────────┬───────────┘
        │                        │                         │
        ▼                        ▼                         ▼
 ┌───────────────┐       ┌────────────────┐      ┌────────────────────────┐
 │ Graph-only    │       │ Graph-only     │      │  GraphRAG              │
 │ (node counts) │       │ aggregates     │      │  1) Vector anchors     │
 │               │       │ (cost by job,  │      │  2) BFS subgraph       │
 │ e.g. jobs,    │       │  top-N jobs)   │      │  3) Context + LLM      │
 │ users, OUs    │       └──────┬─────────┘      └──────────┬─────────────┘
 └──────┬────────┘              │                           │
        │                        │                           │
        ▼                        ▼                           ▼
 ┌─────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
 │ Deterministic   │   │ Deterministic + LLM   │   │ LLM Answer          │
 │ answer string   │   │ narrative (optional)  │   │ (with graph context)│
 └────────┬────────┘   └───────────┬───────────┘   └─────────┬──────────┘
          │                        │                         │
          └──────────────┬─────────┴─────────────┬───────────┘
                         ▼                       ▼
             ┌───────────────────┐   ┌─────────────────────────┐
             │  Answer           │   │ "How I reasoned" panel  │
             │                   │   │ - Subgraph summary      │
             │                   │   │ - Prompt + context      │
             └───────────────────┘   └─────────────────────────┘
````

This routing lets the assistant:

* Use **graphs** where structure matters (counts, cost aggregation, relationships).
* Use **vector search + graph expansion** where semantics matter (“why did this job behave this way?”).
* Stay **transparent**, thanks to the “How I reasoned” panel exposing the graph, prompt, and context.

```
```

---

## 🧠 Why This Architecture?

### 1. Enterprise telemetry is a graph

Databricks usage is naturally modeled as:

* Org units → users
* Users → jobs & queries
* Jobs → job runs → compute usage
* Usage → events → evictions

Answering **“why”**, **“who”**, and **“what else is related”** requires following **relationships**, not just matching text.

### 2. Pure vector RAG struggles on structural questions

Example:

> “Why did job J-LOGI-OPT fail yesterday?”

Pure vector search might miss:

* The specific **job run** that failed
* The **eviction event** tied to that run
* The **compute usage** record that shows spot capacity
* The **queries** that ran shortly before

GraphRAG ensures those nodes are traversed and included in context.

### 3. Hybrid = semantic + structural power

* **Vector search** → finds *what the user is talking about*
* **Graph traversal** → finds *everything structurally related*
* **LLM** → synthesizes an answer with full context

This is the pattern you’d want for a real **FinOps / observability / governance copilot**.

### 4. Transparent reasoning (“How I reasoned”)

Each answer includes:

* A **graph explanation**: what node types were used, how many, and some example nodes
* The **exact prompt** sent to the LLM (system prompt + context + user question)
* The **context** (rendered docs from the graph neighborhood)

This is perfect for:

* Debugging “why did it only talk about one job?”
* Showing platform teams how the AI arrived at its answer
* Teaching others how GraphRAG flows work

---

## 🧩 Design Challenges & How We Solved Them

This project isn’t just a happy path — it documents some **real graph/RAG issues** and how we fixed them.

### 1. Parent–Child Edge Direction

**Problem:**
Initially, graph edges were modeled only **child → parent**:

* `job_run` → `job` (`RUN_OF`)
* `compute_usage` → `job_run` (`USAGE_OF_JOB_RUN`)
* `event` → `usage` (`ON_USAGE`)

This is natural from a “this thing belongs to that” perspective, but made it hard to answer questions like:

> “Summarize usage for each job.”

Because from the **job** node’s perspective, there were no outgoing edges to its runs/usages.

**Fix: Reverse “HAS_*” edges**

In `graph_model.py` we kept the original edges but added **reverse parent → child** edges:

* `job` → `job_run` (`HAS_RUN`)
* `job_run` → `compute_usage` (`HAS_USAGE`)
* `usage` → `event` (`HAS_EVENT`)
* `user` → `query` (`HAS_QUERY`)

Now we can easily traverse **from a job** down to all of its runs → usage → events without doing expensive reverse lookups.

> **Lesson:** For GraphRAG, it’s often worth maintaining **both directions** (semantic: “RUN_OF”, ergonomic: “HAS_RUN”).

---

### 2. “Why am I only seeing one job?” (Routing & Coverage)

**Problem:**
For global-sounding questions like:

> “give me a summary of my job usage”

the retriever was:

1. Doing a semantic search on that text.
2. Picking a single **anchor job** (e.g., `J-HR-DASH`) and its neighborhood.
3. Building context entirely around that one job.

So the LLM answer looked reasonable, but it only talked about **one job**, not **all five**.

**Fix: Question-type routing in `chat_orchestrator.py`**

We introduced a **classifier + heuristics** that route certain question types away from the pure GraphRAG path and into **deterministic graph-based logic**:

* **Global aggregates**

  * Intent: `"global_aggregate"`
  * Examples:

    * “How many jobs are there?”
    * “How many users do we have?”
  * Solution: `_answer_global_aggregate` → counts nodes by type directly from the graph.

* **Global top-N (jobs)**

  * Intent: `"global_topn"` + `entity_type=="job"`
  * Example:

    * “Top 3 most expensive jobs”
  * Solution: `_answer_global_topn_jobs` → aggregates `compute_usage.cost_usd` per job and ranks.

* **Global usage overview**

  * Heuristic: `_looks_like_usage_overview_question`
  * Examples:

    * “tell me about my databricks usage”
    * “summary of my job usage”
  * Solution: `_answer_global_usage_overview` → aggregates cost for **all jobs** and returns a full breakdown.

* **Jobs needing optimization**

  * Heuristic: `_looks_like_jobs_optimization_question`
  * Example:

    * “which jobs need optimizing?”
  * Solution: `_answer_jobs_needing_optimization` → surfaces the highest-cost jobs by share of total spend.

If a question matches one of these, it **never** goes down the “single anchor GraphRAG” path — it uses **all jobs** via graph aggregation.

> **Lesson:** Not every question should be answered via “retrieve a neighborhood + LLM.”
> Some are better served by **explicit graph computations**.

---

### 3. Debugging Context & Prompt (“How I reasoned” Panel)

**Problem:**
When debugging, it wasn’t clear:

* Which nodes were actually included in the subgraph
* Which docs were sent as context
* What prompt the LLM actually saw

This made it difficult to answer:
“Is this a retrieval issue, a graph issue, or a language-model issue?”

**Fix: Rich `ChatResult` + Streamlit UI**

`ChatResult` now includes:

* `answer`: final LLM (or deterministic) answer
* `context_docs`: the retrieved `Document` objects
* `graph_explanation`: summary of node counts and sample nodes
* `llm_prompt`: the assembled prompt text (system + context + question)
* `llm_context`: the rendered context string

In `app.py`, the Streamlit UI adds an expander:

> 🔍 **How I reasoned (GraphRAG explanation)**

Inside it you see:

* A human-readable description of the subgraph
* The **LLM prompt** (copy-pasteable for inspection)
* The **context** passed to the LLM

This made it immediately obvious when a “global” question was only seeing one job in the context — which pointed straight back to routing and retriever behavior instead of the DB or graph.

> **Lesson:** For serious RAG/GraphRAG, invest in **debug visibility**.
> Being able to see prompt + context + graph summary is huge.

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

* Chat window
* Graph-aware explanations ("How I Reasoned")
* Debug info (prompt + context, if enabled)

This is the interactive Databricks Usage Copilot experience.

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

| Make Target  | What Happens Internally                                                                 |
| ------------ | --------------------------------------------------------------------------------------- |
| `make db`    | Executes Python schema builder → creates all tables → inserts all synthetic records     |
| `make index` | Generates RAG docs → computes embeddings → builds FAISS index → stores metadata         |
| `make app`   | Loads the FAISS index + graph → initializes routing logic → launches Streamlit frontend |
| `make clean` | Removes SQLite DB + FAISS index folder                                                  |
| `make all`   | `db` + `index` + `app`                                                                  |

---

## 🧪 Tip: You Can Combine Targets

Make supports chaining:

```bash
make db index
```

or:

```bash
make index app
```

or even:

```bash
make db clean   # (not recommended—it deletes the DB right after!)
```

---

## 💬 Example Prompts to Try

### Cost / FinOps

* “How many jobs are there?”
* “Top 3 most expensive jobs.”
* “Which jobs need optimizing based on cost?”
* “Break down DBU consumption by org unit.”

### Reliability

* “What spot evictions have impacted ML jobs?”
* “Which runs of the Finance ETL job failed and why?”

### Usage Overview

* “Tell me about my Databricks usage.”
* “Give me a summary of my job usage.”

### Governance / Ownership

* “Which org unit owns the compute driving last week’s DBU spike?”
* “Which users issued long-running queries yesterday?”

---

## 📁 Project Structure

```text
AI-Portfolio/
  database_setup.py               # SQLite setup + schema population
  .env                            # environment variables (ignored in git)
  requirements.txt                # dependencies

  data/
    create_usage_tables.sql     # Creates Databricks Usage schema
    seed_usage_tables.sql       # Loads sample data
    usage_rag_data.db           # Generated SQLite DB

  indexes/
    usage_faiss/                  # FAISS index (created at runtime)

  src/
    config.py                     # Paths + provider config
    ingest_usage_domain.py        # SQL → RAG docs
    ingest_embed_index.py         # Docs → embeddings → FAISS
    graph_model.py                # Nodes & edges & adjacency (HAS_* edges)
    graph_retriever.py            # Graph-aware retriever (GraphRAG)
    chat_orchestrator.py          # LLM orchestration + routing + debug
    app.py                        # Streamlit UI ("How I reasoned" panel)
```

---

## 🚀 Portfolio Value Statement

You can honestly say:

> **“I designed and implemented a GraphRAG system that models Databricks usage telemetry as both a FAISS vector index and a graph of jobs, runs, compute usage, events, evictions, and SQL queries. I debugged real-world issues like edge direction (child→parent vs parent→child) and global vs local question routing, adding reverse ‘HAS_*’ edges, explicit global aggregation paths, and a ‘How I reasoned’ panel that surfaces the exact prompt and context sent to the LLM. The result is an explainable FinOps / observability copilot that combines vector search, graph traversal, and LLM reasoning.”**

---

## 🔭 Future Enhancements

* Add a **router** that sends Databricks Best Practice PDFs to hybrid RAG and usage questions to GraphRAG.
* Integrate **LangGraph** for multi-step workflows and tools.
* Add an **evaluation harness** (groundedness, answer quality, coverage).
* Export the graph into **Neo4j** for large-scale graph analytics.
* Add **graph visualization** in the UI (e.g., job → runs → usage → events view).

```
```

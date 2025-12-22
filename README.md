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

## 🎓 What This Project Teaches

This project serves as a **comprehensive learning platform** for modern AI/ML engineering, specifically designed for **SQL analysts and data engineers** transitioning to AI-powered systems.

### **Current Implementation (Core Foundation)**

The existing system demonstrates:

**1. Production RAG Architecture**
- Multi-corpus retrieval (usage telemetry + vendor documentation)
- Graph-based context retrieval (not just vector search)
- Deterministic prompt engineering (stable, testable, explainable)
- Citation management and source tracking

**2. Graph-Based Reasoning**
- Usage graph models real system relationships
- Semantic compression layer between raw data and LLM reasoning
- Graph traversal for contextual analysis
- Relationship-driven insights (not just metrics)

**3. Enterprise AI Patterns**
- Deterministic action system (no "blank chat box" guessing)
- Taxonomy-based interaction design (Understand/Diagnose/Optimize/Monitor)
- Explainable AI with debug transparency
- Separation of facts (SQL/graph) from explanations (LLM)

**4. Modern Data Engineering**
- Dual-corpus RAG (telemetry + documentation)
- Provider-agnostic LLM abstraction (OpenAI/Gemini/Grok)
- Embedding strategy and vector index management
- Report-driven analytics with programmatic commentary

---

## 🚀 ML/AI Roadmap

The project will expand to demonstrate **10 critical AI/ML capabilities** that production systems need, building from foundation to advanced topics.

### **Phase 1: Foundations (Months 1-2)**

#### **1. Time Series Forecasting & Anomaly Detection**
**What you'll learn:**
- Feature engineering from temporal data
- Prophet/ARIMA for forecasting
- Isolation Forest for anomaly detection
- Trend decomposition and seasonality

**New capabilities:**
- Cost forecasting (next 30 days)
- Automated spike detection with explanations
- Seasonal pattern recognition
- Alert generation with business context

**Module:** `src/forecasting/`

---

#### **2. Clustering & Workload Classification**
**What you'll learn:**
- Unsupervised learning fundamentals
- K-means, DBSCAN, hierarchical clustering
- Dimensionality reduction (PCA/UMAP)
- Feature engineering from structured data
- Cluster interpretation in business context

**New capabilities:**
- Job similarity analysis
- Workload archetype discovery (5-7 common patterns)
- User behavior segmentation
- "Find similar jobs" recommendations

**Module:** `src/clustering/`

**New Reports:**
- "Job Similarity Analysis"
- "Workload Archetypes"
- "User Behavior Segmentation"

---

### **Phase 2: Core ML (Months 3-4)**

#### **3. Predictive Failure Models**
**What you'll learn:**
- Binary classification fundamentals
- XGBoost, Random Forest, LightGBM
- Handling imbalanced datasets
- Feature importance interpretation
- Model evaluation beyond accuracy (precision/recall/F1)
- Threshold tuning for production

**New capabilities:**
- Failure risk prediction with confidence scores
- Proactive alerting before failures
- Root cause feature identification
- Risk scoring dashboard

**Module:** `src/prediction/`

**Features engineered:**
- Recent failure rate trends
- Spot ratio volatility
- Configuration drift signals
- Data volume growth patterns
- Temporal features (day/hour patterns)

---

#### **4. Recommendation Engine**
**What you'll learn:**
- Collaborative vs content-based filtering
- Hybrid recommendation systems
- Cold-start problem handling
- Recommendation quality metrics
- A/B testing frameworks

**New capabilities:**
- Optimization recommendations based on similar jobs
- Configuration advisor ("Jobs like this optimized by...")
- Personalized suggestions per user/team
- Success tracking and ROI measurement

**Module:** `src/recommendations/`

**New Features:**
- "Jobs Similar to This" panel
- "Teams who optimized X saw Y% reduction"
- Personalized optimization queue

---

### **Phase 3: Advanced Topics (Month 5)**

#### **5. Natural Language to SQL**
**What you'll learn:**
- LangChain SQL agents
- Schema-aware prompt engineering
- Query validation and safety patterns
- Structured output generation
- When to use RAG vs direct LLM queries

**New capabilities:**
- "Ask Data" natural language interface
- Auto-generated SQL with explanations
- Safe query execution (read-only, limits)
- Result interpretation and visualization

**Module:** `src/nl_to_sql/`

**Examples:**
- "Which jobs cost more than $1000 last week?"
- "Show me all failed runs for finance jobs"
- "What's the average spot ratio for ML training jobs?"

---

#### **6. Enhanced Embeddings & Semantic Search**
**What you'll learn:**
- Embedding fundamentals (how they capture meaning)
- Distance metrics (cosine, euclidean, dot product)
- Fine-tuning embeddings for domain-specific use
- Hybrid search (semantic + keyword)
- Embedding visualization techniques

**New capabilities:**
- Semantic job search ("Find ML training jobs")
- Error message similarity clustering
- Tag auto-suggestion using embeddings
- Documentation search improvements

**Module:** `src/embeddings/` (enhancement to existing `graph_retriever.py`)

---

#### **7. Multi-Agent Systems**
**What you'll learn:**
- Agent vs tool vs chain architecture
- State management across agent steps
- Agent coordination patterns
- When autonomy helps vs hurts
- Building explainable agent reasoning
- Tool selection and error handling

**New capabilities:**
- Root Cause Analyzer (multi-step investigation)
- Cost Reduction Planner (analyze → recommend → estimate ROI)
- Compliance Checker (automated policy verification)
- Alert Triage Agent (prioritize human attention)

**Module:** `src/agents/`

**Agents:**
1. **Investigation Agent**: Coordinates complex analysis
2. **Optimizer Agent**: Autonomous optimization proposals
3. **Compliance Agent**: Policy enforcement and reporting

---

### **Phase 4: MLOps & Production (Month 6)**

#### **8. Feature Store & Model Monitoring**
**What you'll learn:**
- Why feature stores matter at scale
- Feature versioning and lineage
- Model performance tracking
- Data drift detection
- Concept drift vs distribution shift
- When to retrain models

**New capabilities:**
- Centralized feature computation
- Model health dashboard
- Automated retraining triggers
- Feature importance tracking over time

**Module:** `src/ml_ops/`

**Components:**
- `feature_store.py` - Feature cache and versioning
- `model_monitor.py` - Track prediction quality
- `data_drift_detector.py` - Alert on distribution changes

---

#### **9. Causal Inference**
**What you'll learn:**
- Causal inference fundamentals
- Confounders and treatment effects
- Propensity score matching
- Difference-in-differences
- A/B test design principles
- Counterfactual reasoning

**New capabilities:**
- Impact analysis ("What was the real impact of autoscaling?")
- Counterfactual prediction ("What if we used on-demand?")
- Optimization simulator (predict before implementing)
- Policy change evaluation

**Module:** `src/causal/`

**Use cases:**
- Measure optimization ROI with causal rigor
- Compare alternative strategies
- Isolate variable effects
- Design experiments properly

---

#### **10. AutoML Pipeline**
**What you'll learn:**
- End-to-end ML workflow automation
- Feature engineering automation
- Algorithm selection strategies
- Hyperparameter tuning (grid/random/Bayesian)
- Pipeline composition and versioning
- When AutoML helps vs manual modeling

**New capabilities:**
- "Auto-optimize this job" button
- Multi-strategy testing and comparison
- Automated feature discovery
- Confidence-scored recommendations

**Module:** `src/automl/`

**Components:**
- `pipeline_builder.py` - Auto feature engineering
- `model_selector.py` - Algorithm comparison
- `hyperparameter_tuner.py` - Optimization search

---

## 📚 Six-Month Learning Curriculum

### **Months 1-2: Foundations**
**Focus:** Time series and unsupervised learning

**Week 1-2: Time Series Basics**
- Prophet fundamentals
- Anomaly detection (statistical → ML)
- Hands-on: Build cost forecasting model

**Week 3-4: Clustering**
- K-means implementation
- Feature engineering workshop
- Hands-on: Discover workload archetypes

**Deliverable:** Cost anomaly detector + job clustering report

---

### **Months 3-4: Core ML**
**Focus:** Supervised learning and recommendations

**Week 1-3: Classification**
- Binary classification theory
- XGBoost deep dive
- Model evaluation workshop
- Hands-on: Build failure predictor

**Week 4-6: Recommendation Systems**
- Collaborative filtering
- Content-based filtering
- Hybrid approaches
- Hands-on: Build optimization recommender

**Deliverable:** Failure prediction dashboard + recommendation engine

---

### **Month 5: Advanced Topics**
**Focus:** NLP and multi-agent systems

**Week 1-2: Text-to-SQL**
- LangChain SQL agents
- Schema awareness
- Safety patterns
- Hands-on: Build "Ask Data" interface

**Week 3-4: Multi-Agent Systems**
- Agent architecture
- State management
- Tool coordination
- Hands-on: Build root cause analyzer

**Deliverable:** Natural language query interface + investigation agent

---

### **Month 6: MLOps**
**Focus:** Production ML patterns

**Week 1-2: Feature Stores & Monitoring**
- Feature store design
- Model monitoring
- Drift detection
- Hands-on: Build model health dashboard

**Week 3-4: Causal Inference & AutoML**
- Causal inference basics
- A/B test design
- AutoML pipelines
- Hands-on: Build optimization simulator

**Deliverable:** Production ML pipeline with monitoring

---

## 🏗️ Project Structure Evolution

```
src/
├── forecasting/           # Time series models
│   ├── cost_forecasting.py
│   ├── anomaly_detector.py
│   └── trend_decomposition.py
│
├── clustering/            # Unsupervised learning
│   ├── job_similarity.py
│   ├── workload_profiles.py
│   └── usage_segmentation.py
│
├── prediction/            # Supervised learning
│   ├── failure_predictor.py
│   ├── feature_store.py
│   └── model_registry.py
│
├── recommendations/       # Recommendation engines
│   ├── optimization_recommender.py
│   ├── configuration_adviser.py
│   └── similar_jobs_recommender.py
│
├── nl_to_sql/            # Natural language interface
│   ├── query_generator.py
│   ├── query_validator.py
│   └── result_explainer.py
│
├── agents/               # Multi-agent systems
│   ├── investigation_agent.py
│   ├── cost_optimizer_agent.py
│   └── alert_triage_agent.py
│
├── ml_ops/               # Feature store, monitoring
│   ├── feature_store.py
│   ├── model_monitor.py
│   └── data_drift_detector.py
│
├── causal/               # Causal inference
│   ├── impact_analysis.py
│   ├── counterfactual.py
│   └── optimization_simulator.py
│
├── automl/               # AutoML pipelines
│   ├── pipeline_builder.py
│   ├── model_selector.py
│   └── hyperparameter_tuner.py
│
└── reports/              # Existing reports (enhanced with ML)
    ├── registry.py
    ├── base.py
    ├── job_cost.py
    ├── compute_type_cost.py
    ├── job_cost_pareto.py
    └── spot_risk_by_job.py
```

---

## 🎯 Quick Wins to Start With

### **1. Add Anomaly Detection (1-2 weeks)**
**Why start here:**
- Immediate business value (catch cost spikes early)
- Introduces ML without complex infrastructure
- Clear success metrics (false positives/negatives)

**Implementation:**
```python
from sklearn.ensemble import IsolationForest

# Train on historical cost patterns
detector = IsolationForest(contamination=0.05)
detector.fit(historical_costs)

# Detect anomalies in new data
anomalies = detector.predict(current_costs)
```

**New features:**
- 🚨 Anomaly Alert chip with explanation
- Shows similar past anomalies and resolutions
- Automated Slack/email alerts

---

### **2. Job Clustering (2-3 weeks)**
**Why next:**
- Reveals hidden patterns in workload behavior
- Introduces unsupervised learning
- Enables similarity-based recommendations

**Implementation:**
```python
from sklearn.cluster import KMeans

# Feature engineering
features = ['avg_duration', 'cost', 'failure_rate', 'spot_ratio']
X = jobs[features].values

# Clustering
kmeans = KMeans(n_clusters=5)
job_clusters = kmeans.fit_predict(X)
```

**New features:**
- "Job Archetypes" report
- "Show similar jobs" action chip
- Optimization patterns per cluster

---

### **3. Failure Prediction (3-4 weeks)**
**Why third:**
- High-impact proactive capability
- Introduces supervised learning
- Demonstrates feature importance

**Implementation:**
```python
from xgboost import XGBClassifier

# Features
X = [recent_failure_rate, spot_ratio, duration_trend, ...]
y = will_fail_next_run

# Train classifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict with confidence
predictions = model.predict_proba(X_new)
```

**New features:**
- "Failure Risk" score per job
- Proactive alerts before predicted failures
- Feature importance explanations

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

- **Understand** – What is this? What does "good" look like?
- **Diagnose** – Why is this happening? What changed?
- **Optimize** – What should I change to improve cost, reliability, or performance?
- **Monitor** – How do I validate improvements and prevent regressions?

This structure aligns directly with the pillars of data engineering (cost, reliability, performance, resilience, data quality) and provides a repeatable "operational playbook" for each report and selection.

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
* "What is / how does / how do I configure" questions retrieve docs
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

## Why This Matters (Portfolio Value)

This project demonstrates how to build **enterprise-ready AI copilots** that:

* Are deterministic instead of guess-driven
* Separate facts from explanations
* Support auditing and debugging
* Earn trust from engineers and FinOps teams

> **Chat-first copilots optimize for convenience.  
> Report-driven copilots optimize for correctness, trust, and scale.**

### As a Learning Platform

This project is **uniquely positioned** as both:

1. **Production-quality reference implementation** of modern AI patterns
2. **Hands-on learning curriculum** with progressive complexity

Unlike toy examples or academic exercises, every feature:
- Solves a real business problem
- Demonstrates a production AI/ML pattern
- Builds on prior knowledge incrementally
- Includes explainability and debugging tools

---

## For SQL Analysts and Data Engineers

### **Why This Approach Works**

**1. SQL Analysts**
- Already understand the data and queries
- ML shows patterns hidden from SQL alone
- Feature engineering builds on SQL skills
- Model outputs become new "computed columns"

**2. Data Engineers**
- Already understand pipelines and orchestration
- MLOps shows how to productionize ML at scale
- Graph modeling extends their data modeling skills
- Monitoring patterns mirror their existing practices

**3. Incremental Learning**
- Each module adds ONE new ML concept
- Builds on Python/SQL skills they already have
- Real business value at each step
- No "toy datasets" — actual usage data

**4. Explainable AI**
- Deterministic chip approach means ML predictions are always interpretable
- Debug mode shows exactly what the model saw and why
- Feature importance explains model decisions
- Trust through transparency

---

## License

MIT License - See LICENSE file for details

---

## Contributing

This project is a learning platform. Contributions that:
- Add new ML/AI capabilities from the roadmap
- Improve explainability and interpretability
- Enhance the learning experience
- Add production ML patterns

...are especially welcome.

See CONTRIBUTING.md for guidelines.

---

## Acknowledgments

Built as a teaching tool for data teams transitioning to AI/ML engineering. Designed to demonstrate that production AI systems can be both powerful AND explainable.

Special thanks to the Databricks community for inspiration on real-world usage patterns and optimization challenges.
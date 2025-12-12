# src/app.py

import streamlit as st

from src.chat_orchestrator import DatabricksUsageAssistant, ChatResult


def get_assistant() -> DatabricksUsageAssistant:
    """Singleton-style assistant stored in Streamlit session_state."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = DatabricksUsageAssistant.from_local()
    return st.session_state.assistant


def init_session_state():
    if "messages" not in st.session_state:
        # Each message:
        # {
        #   "role": "user"|"assistant",
        #   "content": str,
        #   "graph_explanation": Optional[str],
        #   "llm_prompt": Optional[str],
        #   "llm_context": Optional[str],
        # }
        st.session_state.messages = []


# -------------------- Page config -------------------- #

st.set_page_config(
    page_title="Databricks Usage Monitor - GraphRAG Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Databricks Usage GraphRAG Assistant")
st.caption("Ask questions about jobs, runs, compute usage, cost, evictions, and SQL queries.")

with st.sidebar:
    st.header("ℹ️ How this works")
    st.markdown(
        """
        This assistant uses:

        - **SQLite** Databricks usage telemetry  
        - **FAISS** for semantic vector search  
        - A **graph model** of org units, users, jobs, runs, usage, events, evictions & SQL queries  
        - A **GraphRAG orchestrator** to expand subgraphs and build context  

        Special capabilities:
        - Global counts (e.g., *"How many jobs are there?"*)  
        - Top-N queries (e.g., *"Top 3 most expensive jobs"*)  
        - Global usage overview (e.g., *"Tell me about my Databricks usage"*)  
        - Optimization hints (e.g., *"Which jobs need optimizing?"*)  
        """
    )
    st.markdown("---")
    st.markdown("Make sure you've run:")
    st.code(
        "python database_setup.py data/Databricks_Usage/usage_rag_data.db\n"
        "python -m src.ingest_embed_index"
    )

# -------------------- Main chat UI -------------------- #

init_session_state()
assistant = get_assistant()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and (
            msg.get("graph_explanation") or msg.get("llm_prompt") or msg.get("llm_context")
        ):
            with st.expander("🔍 How I reasoned (GraphRAG explanation)", expanded=False):
                # 1. Graph explanation (which jobs, nodes, edges, etc.)
                if msg.get("graph_explanation"):
                    st.markdown(msg["graph_explanation"])

                # 2. Raw LLM prompt
                if msg.get("llm_prompt"):
                    st.markdown("**LLM Prompt**")
                    st.code(msg["llm_prompt"], language="markdown")

                # 3. Context sent to LLM (graph + retrieved docs)
                if msg.get("llm_context"):
                    st.markdown("**Context sent to LLM**")
                    st.code(msg["llm_context"], language="markdown")

# Chat input
user_input = st.chat_input("Ask a question about Databricks usage…")

if user_input:
    # 1. Show user message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "graph_explanation": None,
            "llm_prompt": None,
            "llm_context": None,
        }
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Get assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking with GraphRAG…"):
            result: ChatResult = assistant.answer(user_input)

        # Display main answer
        st.markdown(result.answer)

        # Display graph explanation + debug prompt/context in an expander
        if result.graph_explanation or result.llm_prompt or result.llm_context:
            with st.expander("🔍 How I reasoned (GraphRAG explanation)", expanded=False):
                if result.graph_explanation:
                    st.markdown(result.graph_explanation)
                if getattr(result, "llm_prompt", None):
                    st.markdown("**LLM Prompt**")
                    st.code(result.llm_prompt, language="markdown")
                if getattr(result, "llm_context", None):
                    st.markdown("**Context sent to LLM**")
                    st.code(result.llm_context, language="markdown")

    # 3. Save assistant message to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "graph_explanation": result.graph_explanation,
            "llm_prompt": getattr(result, "llm_prompt", None),
            "llm_context": getattr(result, "llm_context", None),
        }
    )

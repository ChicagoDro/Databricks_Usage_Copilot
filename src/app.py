# src/app.py

import streamlit as st
from src.chat_orchestrator import DatabricksUsageAssistant, ChatResult


def get_assistant() -> DatabricksUsageAssistant:
    if "assistant" not in st.session_state:
        st.session_state.assistant = DatabricksUsageAssistant.from_local()
    return st.session_state.assistant


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "focus" not in st.session_state:
        st.session_state.focus = None  # {"entity_type": str, "entity_id": str}

    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message = None

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False


def send_message(user_input: str, assistant: DatabricksUsageAssistant):
    user_input = (user_input or "").strip()
    if not user_input:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "entities": [],
            "graph_explanation": None,
            "llm_prompt": None,
            "llm_context": None,
        }
    )

    result: ChatResult = assistant.answer(user_input, focus=st.session_state.focus)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "entities": getattr(result, "entities", []) or [],
            "graph_explanation": getattr(result, "graph_explanation", None),
            "llm_prompt": getattr(result, "llm_prompt", None),
            "llm_context": getattr(result, "llm_context", None),
        }
    )


def render_entity_chips(entities: list[dict], key_prefix: str, max_items: int = 14):
    if not entities:
        return

    entities = entities[:max_items]

    # Make buttons look like chips
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            border: 1px solid rgba(49, 51, 63, 0.25);
            font-size: 0.85rem;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Actions:**")
    cols = st.columns(min(4, len(entities)))

    for i, e in enumerate(entities):
        et = e.get("entity_type") or "entity"
        eid = e.get("entity_id")
        label = e.get("label") or f"{et}:{eid}"
        if not eid:
            continue

        with cols[i % len(cols)]:
            if st.button(label, key=f"{key_prefix}-chip-{i}"):
                # Always auto-send
                st.session_state.focus = {"entity_type": str(et), "entity_id": str(eid)}
                st.session_state.pending_user_message = f"tell me more about this {et} {eid}"
                st.rerun()


def maybe_render_debug(msg: dict):
    if not st.session_state.debug_mode:
        return

    if not (msg.get("graph_explanation") or msg.get("llm_prompt") or msg.get("llm_context")):
        return

    with st.expander("🔍 How I reasoned (debug)", expanded=False):
        if msg.get("graph_explanation"):
            st.markdown(msg["graph_explanation"])
        if msg.get("llm_prompt"):
            st.markdown("**LLM Prompt**")
            st.code(msg["llm_prompt"], language="markdown")
        if msg.get("llm_context"):
            st.markdown("**Context sent to LLM**")
            st.code(msg["llm_context"], language="markdown")


# -------------------- Page config -------------------- #

st.set_page_config(
    page_title="Databricks Usage Copilot",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Databricks Usage Copilot")
st.caption("Ask questions about jobs, runs, compute usage, cost, evictions, and SQL queries.")

init_session_state()
assistant = get_assistant()

with st.sidebar:
    st.header("Controls")
    st.checkbox("Debug mode", key="debug_mode")

    st.divider()

    if st.session_state.focus:
        f = st.session_state.focus
        st.info(f"Context: {f['entity_type']} {f['entity_id']}")
        if st.button("Clear context"):
            st.session_state.focus = None
            st.rerun()


# -------------------- Render chat history -------------------- #

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            entities = msg.get("entities") or []
            if entities:
                render_entity_chips(entities, key_prefix=f"hist-{idx}")

            maybe_render_debug(msg)


# -------------------- Auto-send from chip -------------------- #

if st.session_state.pending_user_message:
    to_send = st.session_state.pending_user_message
    st.session_state.pending_user_message = None
    send_message(to_send, assistant)
    st.rerun()


# -------------------- Manual input -------------------- #

user_input = st.chat_input("Ask a question about Databricks usage…")
if user_input:
    send_message(user_input, assistant)
    st.rerun()

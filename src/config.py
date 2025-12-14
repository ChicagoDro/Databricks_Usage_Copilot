# src/Databrick_Usage/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

# Root of the repo (…/AI-Portfolio)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# New Databricks usage domain (SQLite DB)
USAGE_DB_PATH = PROJECT_ROOT / "data" / "usage_rag_data.db"

# Where to store vector indexes
INDEX_DIR = PROJECT_ROOT / "indices"

USAGE_FAISS_INDEX_PATH = INDEX_DIR / "databricks_usage_faiss"


# --------------------------------------------------------------------------------------
# Databricks documentation corpus (vendor knowledge) — optional
# --------------------------------------------------------------------------------------

DOCS_SITEMAP_URL = os.getenv("DOCS_SITEMAP_URL", "https://docs.databricks.com/aws/en/sitemap.xml")
DOCS_URL_PREFIX = os.getenv("DOCS_URL_PREFIX", "https://docs.databricks.com/aws/en/compute/")

# Where to store the Databricks docs FAISS index (kept separate from telemetry index)
DOCS_FAISS_INDEX_PATH = INDEX_DIR / "databricks_compute_docs_faiss"

# How many doc chunks to retrieve (vendor docs corpus)
DOCS_RETRIEVER_K = int(os.getenv("DOCS_RETRIEVER_K", "4"))


# --------------------------------------------------------------------------------------
# LLM / embedding provider config
# ----------------------------------------------
# You can switch providers with an env var:
#   LLM_PROVIDER=openai | gemini | grok
# and set API keys in your .env / environment:
#   OPENAI_API_KEY=...
#   GOOGLE_API_KEY=...
#   XAI_API_KEY=...

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Model names (tweak as you like)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

GROK_CHAT_MODEL = os.getenv("GROK_CHAT_MODEL", "grok-beta")
GROK_EMBED_MODEL = os.getenv("GROK_EMBED_MODEL", "grok-embed")  # placeholder

# --------------------------------------------------------------------------------------
# Retrieval / chunking config
# --------------------------------------------------------------------------------------

# Chunking for PDFs / long text (used in your existing ingestion)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Top-k for retriever
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# Temperature / style knobs for the chat model
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


# --------------------------------------------------------------------------------------
# Small helpers for provider routing (optional but handy)
# --------------------------------------------------------------------------------------

def get_chat_model_name() -> str:
    if LLM_PROVIDER == "openai":
        return OPENAI_CHAT_MODEL
    if LLM_PROVIDER == "gemini":
        return GEMINI_CHAT_MODEL
    if LLM_PROVIDER == "grok":
        return GROK_CHAT_MODEL
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def get_embed_model_name() -> str:
    if LLM_PROVIDER == "openai":
        return OPENAI_EMBED_MODEL
    if LLM_PROVIDER == "gemini":
        return GEMINI_EMBED_MODEL
    if LLM_PROVIDER == "grok":
        return GROK_EMBED_MODEL
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

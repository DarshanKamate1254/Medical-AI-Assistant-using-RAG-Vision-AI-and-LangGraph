"""
config.py - Central configuration for Medical AI Assistant
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Pinecone ─────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_DIMENSION: int = 1536   # text-embedding-3-small

# ── LangSmith ────────────────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "medical-ai-assistant")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── App settings ─────────────────────────────────────────────────────────────
APP_TITLE: str = "🏥 Medical AI Assistant"
MAX_FILE_SIZE_MB: int = 20
ALLOWED_EXTENSIONS: list[str] = ["pdf", "jpg", "jpeg", "png"]
RAG_TOP_K: int = 5

# ── Medical disclaimer ────────────────────────────────────────────────────────
DISCLAIMER: str = (
    "⚠️ **Medical Disclaimer:** This AI system provides informational guidance only "
    "and does **not** replace professional medical advice. "
    "Please consult a certified doctor before taking any treatment."
)

# ── Severity levels ───────────────────────────────────────────────────────────
SEVERITY_LEVELS: dict = {
    "Mild": {"color": "🟢", "action": "Monitor symptoms. Schedule a routine check-up."},
    "Moderate": {"color": "🟡", "action": "Consult a doctor within 24-48 hours."},
    "Emergency": {"color": "🔴", "action": "Seek immediate medical attention or call emergency services!"},
}


def validate_config() -> list[str]:
    """Return list of missing required environment variables."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    return missing

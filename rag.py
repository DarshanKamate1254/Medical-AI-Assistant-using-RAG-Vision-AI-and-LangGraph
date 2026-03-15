"""
rag.py - Pinecone vector database RAG for medical knowledge retrieval.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Return OpenAI embedding vector for *text*."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pinecone_index():
    """Return a Pinecone Index object, creating the index if needed."""
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s'…", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until ready
        for _ in range(30):
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready"):
                break
            time.sleep(2)

    return pc.Index(PINECONE_INDEX_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Seed knowledge base
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_KNOWLEDGE: list[dict] = [
    # ── Diabetes ──────────────────────────────────────────────────────────
    {
        "id": "diabetes_001",
        "text": "Diabetes mellitus is a chronic metabolic disease characterised by elevated blood glucose levels. Type 1 is autoimmune; Type 2 is linked to insulin resistance. Key markers: fasting glucose >126 mg/dL, HbA1c ≥6.5%.",
        "metadata": {"category": "disease", "condition": "Diabetes", "specialization": "Endocrinologist"},
    },
    {
        "id": "diabetes_002",
        "text": "Diabetes symptoms: frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow-healing sores. Complications include neuropathy, nephropathy, and retinopathy.",
        "metadata": {"category": "symptoms", "condition": "Diabetes", "specialization": "Endocrinologist"},
    },
    {
        "id": "diabetes_003",
        "text": "Diabetes precautions: maintain a low-sugar diet, exercise regularly, monitor blood glucose daily, take medications as prescribed, attend regular check-ups, control blood pressure and cholesterol.",
        "metadata": {"category": "precautions", "condition": "Diabetes", "specialization": "Endocrinologist"},
    },
    # ── Hypertension ──────────────────────────────────────────────────────
    {
        "id": "hypertension_001",
        "text": "Hypertension (high blood pressure) is defined as persistent blood pressure ≥130/80 mmHg. It is a major risk factor for stroke, heart attack, and kidney disease.",
        "metadata": {"category": "disease", "condition": "Hypertension", "specialization": "Cardiologist"},
    },
    {
        "id": "hypertension_002",
        "text": "Hypertension precautions: reduce sodium intake (<2300 mg/day), adopt DASH diet, exercise 150 min/week, limit alcohol, quit smoking, manage stress, take antihypertensive medications as prescribed.",
        "metadata": {"category": "precautions", "condition": "Hypertension", "specialization": "Cardiologist"},
    },
    # ── Anaemia ───────────────────────────────────────────────────────────
    {
        "id": "anaemia_001",
        "text": "Anaemia is characterised by low haemoglobin: <13 g/dL in men, <12 g/dL in women. Iron-deficiency anaemia is the most common type globally. Symptoms include fatigue, pallor, shortness of breath, dizziness.",
        "metadata": {"category": "disease", "condition": "Anaemia", "specialization": "Hematologist"},
    },
    {
        "id": "anaemia_002",
        "text": "Anaemia precautions: eat iron-rich foods (leafy greens, red meat, legumes), take iron supplements if prescribed, consume vitamin C to improve absorption, treat underlying causes.",
        "metadata": {"category": "precautions", "condition": "Anaemia", "specialization": "Hematologist"},
    },
    # ── Thyroid disorders ─────────────────────────────────────────────────
    {
        "id": "thyroid_001",
        "text": "Hypothyroidism: TSH >4.5 mIU/L; symptoms include fatigue, weight gain, cold intolerance, constipation, depression. Hyperthyroidism: TSH <0.4 mIU/L; symptoms include weight loss, rapid heartbeat, anxiety, heat intolerance.",
        "metadata": {"category": "disease", "condition": "Thyroid Disorder", "specialization": "Endocrinologist"},
    },
    # ── Kidney disease ────────────────────────────────────────────────────
    {
        "id": "kidney_001",
        "text": "Chronic kidney disease (CKD) is diagnosed by reduced eGFR (<60 mL/min/1.73m²) or elevated creatinine (>1.2 mg/dL women, >1.4 mg/dL men) persisting >3 months. Causes: diabetes, hypertension.",
        "metadata": {"category": "disease", "condition": "Kidney Disease", "specialization": "Nephrologist"},
    },
    # ── Liver disease ─────────────────────────────────────────────────────
    {
        "id": "liver_001",
        "text": "Liver disease markers: elevated ALT (>40 U/L), AST (>40 U/L), ALP, or bilirubin. Common causes include hepatitis B/C, alcoholic liver disease, non-alcoholic fatty liver disease (NAFLD).",
        "metadata": {"category": "disease", "condition": "Liver Disease", "specialization": "Gastroenterologist"},
    },
    # ── Heart disease ─────────────────────────────────────────────────────
    {
        "id": "heart_001",
        "text": "Coronary artery disease (CAD) is characterised by plaque build-up in coronary arteries. Risk factors: high LDL cholesterol (>160 mg/dL), low HDL (<40 mg/dL men, <50 mg/dL women), smoking, hypertension, diabetes.",
        "metadata": {"category": "disease", "condition": "Heart Disease", "specialization": "Cardiologist"},
    },
    # ── Respiratory ───────────────────────────────────────────────────────
    {
        "id": "respiratory_001",
        "text": "Pneumonia and respiratory infections present with fever, productive cough, chest pain, and low SpO2. Chest X-ray shows consolidation. CBC may show elevated WBC (>11,000/μL) with neutrophilia.",
        "metadata": {"category": "disease", "condition": "Respiratory Infection", "specialization": "Pulmonologist"},
    },
    # ── Specialization mapping ─────────────────────────────────────────────
    {
        "id": "spec_map_001",
        "text": "Doctor specialization guide: Cardiologist for heart and blood pressure conditions; Endocrinologist for diabetes and thyroid; Nephrologist for kidney disease; Hematologist for blood disorders; Gastroenterologist for liver and digestive issues; Pulmonologist for lung conditions; Neurologist for brain and nerve conditions; Orthopedic Surgeon for bone and joint issues.",
        "metadata": {"category": "specialization_map", "condition": "General"},
    },
]


def seed_knowledge_base(force: bool = False) -> None:
    """Upsert medical knowledge into Pinecone (skip if already seeded)."""
    index = _get_pinecone_index()
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)

    if total >= len(MEDICAL_KNOWLEDGE) and not force:
        logger.info("Pinecone index already seeded (%d vectors). Skipping.", total)
        return

    logger.info("Seeding %d medical knowledge vectors into Pinecone…", len(MEDICAL_KNOWLEDGE))
    vectors = []
    for item in MEDICAL_KNOWLEDGE:
        embedding = embed_text(item["text"])
        vectors.append({"id": item["id"], "values": embedding, "metadata": {**item["metadata"], "text": item["text"]}})

    # Batch upsert
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    logger.info("Seeding complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_medical_context(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant medical knowledge chunks for *query*.

    Returns list of dicts with keys: text, score, metadata.
    """
    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY not set – returning empty context.")
        return []

    try:
        index = _get_pinecone_index()
        query_vector = embed_text(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        chunks = []
        for match in results.matches:
            chunks.append(
                {
                    "text": match.metadata.get("text", ""),
                    "score": round(match.score, 4),
                    "condition": match.metadata.get("condition", ""),
                    "category": match.metadata.get("category", ""),
                    "specialization": match.metadata.get("specialization", ""),
                }
            )
        logger.info("RAG retrieved %d chunks for query (top score: %.3f)", len(chunks), chunks[0]["score"] if chunks else 0)
        return chunks

    except Exception as exc:
        logger.exception("Pinecone retrieval failed: %s", exc)
        return []

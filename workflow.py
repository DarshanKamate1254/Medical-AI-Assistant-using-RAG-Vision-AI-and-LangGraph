"""
workflow.py - LangGraph decision workflow for Medical AI Assistant.

Nodes:
  1. report_reader
  2. scan_type_detector         ← NEW: routes X-ray/MRI/CT to vision branch
  3. ocr_extraction             ← used only for text reports
  4. xray_vision_analysis       ← NEW: GPT-4o Vision + image annotation
  5. medical_parameter_extractor
  6. rag_retrieval_pinecone
  7. llm_medical_reasoning
  8. severity_classifier
  9. doctor_specialization_selector
  10. hospital_recommender
  11. navigation_link_generator
  12. result_formatter
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph  # type: ignore

import llm as llm_module
import rag
from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2, SEVERITY_LEVELS
from hospital import Hospital, recommend_hospitals
from ocr import extract_text
from xray_analyzer import XRayReport, analyse_xray, is_medical_scan

logger = logging.getLogger(__name__)

# ── LangSmith tracing ────────────────────────────────────────────────────────
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"]      = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"]   = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_PROJECT"]      = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]     = "https://api.smith.langchain.com"
    logger.info("LangSmith tracing enabled for project: %s", LANGCHAIN_PROJECT)


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class MedicalState(TypedDict, total=False):
    # Input
    file_bytes: bytes
    filename: str
    user_lat: float | None
    user_lon: float | None
    patient_info: dict | None

    # Routing flag
    is_scan: bool
    report_type: str                     # "scan" | "text"

    # Intermediate
    raw_text: str
    rag_context: list[dict]
    llm_result: dict
    xray_report: dict

    # Output
    extracted_parameters: list[dict]
    detected_conditions: list[dict]
    severity: str
    severity_reason: str
    severity_info: dict
    doctor_specialization: str
    precautions: list[str]
    health_guidance: str
    summary: str
    hospitals: list[dict]
    annotated_image_bytes: bytes | None
    scan_type: str
    body_part: str
    scan_findings: list[dict]
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────────────────────

def report_reader(state: MedicalState) -> MedicalState:
    logger.info("[Node] report_reader")
    if not state.get("file_bytes") or not state.get("filename"):
        return {**state, "error": "No file provided."}
    return {**state, "error": None}


def scan_type_detector(state: MedicalState) -> MedicalState:
    """Detect whether upload is a medical scan (X-ray / MRI / CT)."""
    logger.info("[Node] scan_type_detector")
    if state.get("error"):
        return state
    filename   = state["filename"]
    file_bytes = state["file_bytes"]
    ext        = Path(filename).suffix.lower().lstrip(".")
    if ext == "pdf":
        return {**state, "is_scan": False, "report_type": "text"}
    is_scan = is_medical_scan(filename, file_bytes)
    logger.info("File '%s' -> is_scan=%s", filename, is_scan)
    return {**state, "is_scan": is_scan, "report_type": "scan" if is_scan else "text"}


def xray_vision_analysis(state: MedicalState) -> MedicalState:
    """GPT-4o Vision analysis + oval annotation of findings."""
    logger.info("[Node] xray_vision_analysis")
    if state.get("error"):
        return state
    try:
        report: XRayReport = analyse_xray(state["file_bytes"], state["filename"])
        findings_dicts = [
            {"label": f.label, "description": f.description, "severity": f.severity,
             "region": f.region, "confidence": f.confidence,
             "cx": f.cx, "cy": f.cy, "w": f.w, "h": f.h}
            for f in report.findings
        ]
        conditions = [
            {"condition": f.label, "confidence": f.confidence,
             "explanation": f"{f.description} (Region: {f.region})"}
            for f in report.findings
        ]
        return {
            **state,
            "scan_type":             report.scan_type,
            "body_part":             report.body_part,
            "scan_findings":         findings_dicts,
            "severity":              report.severity,
            "severity_reason":       report.overall_impression,
            "severity_info":         SEVERITY_LEVELS.get(report.severity,
                                         {"color": "⚪", "action": "Consult a doctor."}),
            "doctor_specialization": report.doctor_specialization,
            "precautions":           report.precautions,
            "health_guidance":       report.health_guidance,
            "summary":               report.overall_impression,
            "detected_conditions":   conditions,
            "extracted_parameters":  [],
            "annotated_image_bytes": report.annotated_image_bytes,
            "raw_text": (f"[{report.scan_type} – {report.body_part}] "
                         f"{len(report.findings)} finding(s) detected via vision analysis."),
        }
    except Exception as exc:
        logger.exception("X-ray vision analysis failed")
        return {**state, "error": f"Scan analysis failed: {exc}"}


def ocr_extraction(state: MedicalState) -> MedicalState:
    logger.info("[Node] ocr_extraction")
    if state.get("error") or state.get("is_scan"):
        return state
    try:
        text = extract_text(state["file_bytes"], state["filename"])
        return {**state, "raw_text": text}
    except Exception as exc:
        logger.exception("OCR failed")
        return {**state, "error": f"OCR failed: {exc}"}


def medical_parameter_extractor(state: MedicalState) -> MedicalState:
    logger.info("[Node] medical_parameter_extractor (pass-through)")
    return state


def rag_retrieval_pinecone(state: MedicalState) -> MedicalState:
    logger.info("[Node] rag_retrieval_pinecone")
    if state.get("error"):
        return {**state, "rag_context": []}
    if state.get("is_scan"):
        parts = [state.get("scan_type", ""), state.get("body_part", "")]
        for f in state.get("scan_findings", []):
            parts.append(f.get("label", ""))
        query = " ".join(filter(None, parts))
    else:
        query = state.get("raw_text", "")[:1000]
    try:
        context = rag.retrieve_medical_context(query)
        return {**state, "rag_context": context}
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return {**state, "rag_context": []}


def llm_medical_reasoning(state: MedicalState) -> MedicalState:
    logger.info("[Node] llm_medical_reasoning")
    if state.get("error"):
        return state
    if state.get("is_scan"):
        logger.info("Scan branch – skipping text LLM (Vision API already ran).")
        return state
    try:
        result = llm_module.analyse_medical_report(
            report_text=state.get("raw_text", ""),
            rag_context=state.get("rag_context", []),
            patient_info=state.get("patient_info"),
        )
        return {**state, "llm_result": result}
    except Exception as exc:
        logger.exception("LLM reasoning failed")
        return {**state, "error": f"LLM analysis failed: {exc}"}


def severity_classifier(state: MedicalState) -> MedicalState:
    logger.info("[Node] severity_classifier")
    if state.get("error") or state.get("is_scan"):
        return state
    result = state.get("llm_result", {})
    severity = result.get("severity", "Unknown")
    return {**state, "severity": severity,
            "severity_reason": result.get("severity_reason", ""),
            "severity_info": SEVERITY_LEVELS.get(severity, {"color": "⚪", "action": "Consult a doctor."})}


def doctor_specialization_selector(state: MedicalState) -> MedicalState:
    logger.info("[Node] doctor_specialization_selector")
    if state.get("error") or state.get("is_scan"):
        return state
    result = state.get("llm_result", {})
    return {
        **state,
        "doctor_specialization": result.get("doctor_specialization", "General Practitioner"),
        "extracted_parameters":  result.get("extracted_parameters", []),
        "detected_conditions":   result.get("detected_conditions", []),
        "precautions":           result.get("precautions", []),
        "health_guidance":       result.get("health_guidance", ""),
        "summary":               result.get("summary", ""),
    }


def hospital_recommender(state: MedicalState) -> MedicalState:
    logger.info("[Node] hospital_recommender")
    if state.get("error"):
        return {**state, "hospitals": []}
    try:
        hospitals: list[Hospital] = recommend_hospitals(
            specialization=state.get("doctor_specialization", "General Practitioner"),
            user_lat=state.get("user_lat"),
            user_lon=state.get("user_lon"),
            top_n=5,
        )
        return {**state, "hospitals": [h.__dict__ for h in hospitals]}
    except Exception as exc:
        logger.warning("Hospital recommendation failed: %s", exc)
        return {**state, "hospitals": []}


def navigation_link_generator(state: MedicalState) -> MedicalState:
    logger.info("[Node] navigation_link_generator")
    import urllib.parse
    hospitals = state.get("hospitals", [])
    for h in hospitals:
        if not h.get("maps_link"):
            encoded = urllib.parse.quote_plus(f"{h['name']}, {h['address']}, {h['city']}")
            h["maps_link"] = f"https://www.google.com/maps/dir/?api=1&destination={encoded}"
    return {**state, "hospitals": hospitals}


def result_formatter(state: MedicalState) -> MedicalState:
    logger.info("[Node] result_formatter – severity=%s report_type=%s",
                state.get("severity"), state.get("report_type"))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_scan_detector(state: MedicalState) -> str:
    if state.get("error"):
        return "result_formatter"
    return "xray_vision_analysis" if state.get("is_scan") else "ocr_extraction"


def _build_graph() -> Any:
    graph = StateGraph(MedicalState)

    graph.add_node("report_reader",               report_reader)
    graph.add_node("scan_type_detector",          scan_type_detector)
    graph.add_node("xray_vision_analysis",        xray_vision_analysis)
    graph.add_node("ocr_extraction",              ocr_extraction)
    graph.add_node("medical_parameter_extractor", medical_parameter_extractor)
    graph.add_node("rag_retrieval_pinecone",      rag_retrieval_pinecone)
    graph.add_node("llm_medical_reasoning",       llm_medical_reasoning)
    graph.add_node("severity_classifier",         severity_classifier)
    graph.add_node("doctor_specialization_selector", doctor_specialization_selector)
    graph.add_node("hospital_recommender",        hospital_recommender)
    graph.add_node("navigation_link_generator",   navigation_link_generator)
    graph.add_node("result_formatter",            result_formatter)

    graph.set_entry_point("report_reader")
    graph.add_edge("report_reader", "scan_type_detector")

    graph.add_conditional_edges(
        "scan_type_detector",
        _route_after_scan_detector,
        {
            "xray_vision_analysis": "xray_vision_analysis",
            "ocr_extraction":       "ocr_extraction",
            "result_formatter":     "result_formatter",
        },
    )

    graph.add_edge("xray_vision_analysis",        "rag_retrieval_pinecone")
    graph.add_edge("ocr_extraction",              "medical_parameter_extractor")
    graph.add_edge("medical_parameter_extractor", "rag_retrieval_pinecone")
    graph.add_edge("rag_retrieval_pinecone",          "llm_medical_reasoning")
    graph.add_edge("llm_medical_reasoning",           "severity_classifier")
    graph.add_edge("severity_classifier",             "doctor_specialization_selector")
    graph.add_edge("doctor_specialization_selector",  "hospital_recommender")
    graph.add_edge("hospital_recommender",            "navigation_link_generator")
    graph.add_edge("navigation_link_generator",       "result_formatter")
    graph.add_edge("result_formatter",                END)

    return graph.compile()


medical_graph = _build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public runner
# ─────────────────────────────────────────────────────────────────────────────

def run_medical_workflow(
    file_bytes: bytes,
    filename: str,
    user_lat: float | None = None,
    user_lon: float | None = None,
    patient_info: dict | None = None,
) -> MedicalState:
    initial: MedicalState = {
        "file_bytes":   file_bytes,
        "filename":     filename,
        "user_lat":     user_lat,
        "user_lon":     user_lon,
        "patient_info": patient_info,
    }
    return medical_graph.invoke(initial)

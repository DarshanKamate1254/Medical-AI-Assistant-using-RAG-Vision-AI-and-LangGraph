"""
app.py - Streamlit frontend for Medical AI Assistant.
Supports text reports AND medical scans (X-ray / MRI / CT).
Includes a post-analysis chat Q&A section powered by GPT-4o + RAG context.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from io import BytesIO

import streamlit as st

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    ALLOWED_EXTENSIONS,
    DISCLAIMER,
    MAX_FILE_SIZE_MB,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SEVERITY_LEVELS,
    validate_config,
)
from utils.helpers import format_parameter_table, geocode_city

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main-header {font-size:2.4rem;font-weight:800;color:#1a6eb5;margin-bottom:0.2rem;}
  .sub-header  {font-size:1.05rem;color:#555;margin-bottom:1.5rem;}
  .disclaimer-box {
      background:#b45309;border-left:5px solid #f59e0b;
      border-radius:6px;padding:1rem 1.2rem;margin-bottom:1.2rem;
      color:#ffffff !important;
  }
  .disclaimer-box * { color:#ffffff !important; }
  .severity-mild      {background:#166534;border-left:5px solid #22c55e;border-radius:6px;padding:.8rem 1rem;color:#fff !important;}
  .severity-mild *    {color:#fff !important;}
  .severity-moderate  {background:#92400e;border-left:5px solid #f59e0b;border-radius:6px;padding:.8rem 1rem;color:#fff !important;}
  .severity-moderate *{color:#fff !important;}
  .severity-emergency {background:#991b1b;border-left:5px solid #ef4444;border-radius:6px;padding:.8rem 1rem;color:#fff !important;}
  .severity-emergency *{color:#fff !important;}
  .condition-card {
      background:#1e3a5f;border:1px solid #3b82f6;
      border-radius:8px;padding:1rem;margin-bottom:.6rem;color:#fff !important;
  }
  .condition-card * {color:#fff !important;}
  .finding-card {
      background:#7c2d12;border-left:4px solid #f97316;
      border-radius:8px;padding:1rem;margin-bottom:.6rem;color:#fff !important;
  }
  .finding-card * {color:#fff !important;}
  .finding-card-emergency {
      background:#7f1d1d;border-left:4px solid #ef4444;
      border-radius:8px;padding:1rem;margin-bottom:.6rem;color:#fff !important;
  }
  .finding-card-emergency * {color:#fff !important;}
  .hospital-card {
      background:#1e3a5f;border:1px solid #3b82f6;
      border-radius:8px;padding:1rem;margin-bottom:.8rem;color:#fff !important;
  }
  .hospital-card * {color:#fff !important;}
  .hospital-card a {color:#93c5fd !important;}
  .scan-badge {
      display:inline-block;background:#6f42c1;color:white;
      border-radius:20px;padding:4px 14px;font-size:.85rem;font-weight:700;
  }
  .step-badge {
      display:inline-block;background:#1a6eb5;color:white;
      border-radius:50%;width:26px;height:26px;text-align:center;
      line-height:26px;font-size:.8rem;font-weight:700;margin-right:8px;
  }
  .chat-section {
      background:#0f1f3d;border:1px solid #2563eb;
      border-radius:12px;padding:1.2rem 1.4rem;margin-top:1.5rem;
  }
  .chat-header {
      font-size:1.1rem;font-weight:700;color:#93c5fd;margin-bottom:.8rem;
  }
  .chat-bubble-user {
      background:#1e40af;color:#fff;border-radius:12px 12px 2px 12px;
      padding:.7rem 1rem;margin:.4rem 0;max-width:85%;margin-left:auto;
      font-size:.92rem;
  }
  .chat-bubble-ai {
      background:#1e3a5f;color:#e0eaff;border-radius:12px 12px 12px 2px;
      padding:.7rem 1rem;margin:.4rem 0;max-width:92%;
      font-size:.92rem;border-left:3px solid #3b82f6;
  }
  .chat-bubble-user *, .chat-bubble-ai * {color:inherit !important;}
  .suggested-chip {
      display:inline-block;background:#1e3a5f;color:#93c5fd;
      border:1px solid #3b82f6;border-radius:20px;
      padding:4px 12px;font-size:.8rem;cursor:pointer;
      margin:3px 3px 3px 0;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_chat_system_prompt(result: dict) -> str:
    """Build a rich system prompt that gives GPT-4o full context of the report."""
    conditions = ", ".join(
        c.get("condition", "") for c in result.get("detected_conditions", [])
    ) or "None detected"
    params = "; ".join(
        f"{p.get('parameter')} = {p.get('value')} ({p.get('status')})"
        for p in result.get("extracted_parameters", [])
        if p.get("status", "").upper() != "NORMAL"
    ) or "No abnormal parameters"

    scan_section = ""
    if result.get("is_scan"):
        findings = "; ".join(
            f"{f.get('label')} in {f.get('region')} ({f.get('severity')})"
            for f in result.get("scan_findings", [])
        )
        scan_section = f"\nScan Findings: {findings or 'None'}"

    return f"""You are a friendly and knowledgeable medical AI assistant.
The user has just received a medical report analysis. Here is the full context:

Report Type: {"Medical Scan (" + result.get("scan_type","") + ")" if result.get("is_scan") else "Lab / Text Report"}
Detected Conditions: {conditions}
Severity: {result.get("severity", "Unknown")}
Abnormal Parameters: {params}{scan_section}
Recommended Specialist: {result.get("doctor_specialization", "General Practitioner")}
Summary: {result.get("summary", "")[:500]}
Health Guidance: {result.get("health_guidance", "")[:500]}

Your role:
- Answer the user's follow-up questions about their report clearly and simply
- Explain medical terms in plain language
- Refer back to their specific values and findings when relevant
- Never diagnose — always remind them to consult a doctor for final decisions
- Be empathetic, calm, and supportive
- Keep answers concise (3-5 sentences unless a detailed explanation is needed)

Always end with a gentle reminder to consult a certified doctor if the question is clinical."""


def _chat_with_gpt(system_prompt: str, messages: list[dict]) -> str:
    """Send conversation to GPT-4o and return assistant reply."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content or "Sorry, I couldn't generate a response."


def _suggested_questions(result: dict) -> list[str]:
    """Generate relevant suggested questions based on the analysis result."""
    questions = []
    conditions = [c.get("condition", "") for c in result.get("detected_conditions", [])]

    for cond in conditions[:2]:
        if cond:
            questions.append(f"What is {cond} in simple words?")
            questions.append(f"What foods should I avoid with {cond}?")

    if result.get("is_scan"):
        questions.append("Can you explain the findings in simple language?")
        questions.append("Is my condition reversible?")
    else:
        questions.append("Which of my values are most concerning?")
        questions.append("What lifestyle changes should I make?")

    questions.append(f"What does a {result.get('doctor_specialization','specialist')} do?")
    questions.append("How long will recovery take?")
    questions.append("Do I need to fast before my next test?")
    questions.append("Is my condition hereditary?")

    # Deduplicate and limit
    seen, unique = set(), []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Chat section renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_section(result: dict):
    """Render the full chat Q&A panel below the report results."""
    st.markdown("---")
    st.markdown('<div class="chat-header">💬 Ask a Question About Your Report</div>',
                unsafe_allow_html=True)
    st.caption("Ask anything about your results, conditions, medications, lifestyle, or next steps.")

    # Initialise session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_system_prompt" not in st.session_state:
        st.session_state.chat_system_prompt = _build_chat_system_prompt(result)

    # ── Suggested question chips ─────────────────────────────────────────────
    suggestions = _suggested_questions(result)
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                # Inject as if user typed it
                st.session_state.chat_messages.append({"role": "user", "content": q})
                with st.spinner("Thinking…"):
                    reply = _chat_with_gpt(
                        st.session_state.chat_system_prompt,
                        st.session_state.chat_messages,
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("")

    # ── Chat history display ─────────────────────────────────────────────────
    if st.session_state.chat_messages:
        st.markdown("**Conversation:**")
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")

    # ── Input box ────────────────────────────────────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Your question",
                placeholder="e.g. What does low haemoglobin mean? Can I exercise?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking…"):
            try:
                reply = _chat_with_gpt(
                    st.session_state.chat_system_prompt,
                    st.session_state.chat_messages,
                )
            except Exception as exc:
                reply = f"Sorry, I couldn't answer right now: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Clear chat button ────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Shared render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_class(severity: str) -> str:
    return {"Mild": "severity-mild", "Moderate": "severity-moderate",
            "Emergency": "severity-emergency"}.get(severity, "severity-mild")


def _render_disclaimer():
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


def _render_parameter_table(parameters: list[dict]):
    if not parameters:
        st.info("No numeric parameters extracted (scan report).")
        return
    rows = format_parameter_table(parameters)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_conditions(conditions: list[dict], is_scan: bool = False):
    if not conditions:
        st.success("✅ No abnormalities detected.")
        return
    for cond in conditions:
        conf_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(cond.get("confidence", ""), "⚪")
        st.markdown(f"""
<div class="condition-card">
  <strong>{cond.get('condition', 'Unknown')}</strong>
  &nbsp;{conf_color}&nbsp;<em>Confidence: {cond.get('confidence','–')}</em>
  <p style="margin:.4rem 0 0">{cond.get('explanation','')}</p>
</div>""", unsafe_allow_html=True)


def _render_scan_findings(findings: list[dict]):
    if not findings:
        st.success("✅ No findings — scan appears normal.")
        return
    for i, f in enumerate(findings, start=1):
        sev = f.get("severity", "Mild")
        card_cls = "finding-card-emergency" if sev == "Emergency" else "finding-card"
        sev_icon = {"Mild": "🟢", "Moderate": "🟡", "Emergency": "🔴"}.get(sev, "⚪")
        st.markdown(f"""
<div class="{card_cls}">
  <strong>Finding #{i}: {f.get('label','Anomaly')}</strong>
  &nbsp;{sev_icon} <em>{sev}</em>
  &nbsp;|&nbsp; 📍 <em>{f.get('region','unknown region')}</em>
  &nbsp;|&nbsp; Confidence: <strong>{f.get('confidence','–')}</strong>
  <p style="margin:.4rem 0 0">{f.get('description','')}</p>
</div>""", unsafe_allow_html=True)


def _render_hospitals(hospitals: list[dict]):
    if not hospitals:
        st.info("No hospital recommendations available.")
        return
    for h in hospitals:
        dist = f"  📍 {h['distance_km']} km away" if h.get("distance_km") else ""
        maps_link = h.get("maps_link", "#")
        st.markdown(f"""
<div class="hospital-card">
  <strong>🏥 {h['name']}</strong>{dist}<br>
  📬 {h.get('full_address') or h.get('address','')}, {h.get('city','')}<br>
  👨‍⚕️ <strong>{h.get('doctor_name','N/A')}</strong> – {h.get('doctor_qualification','')}<br>
  ⭐ Rating: {h.get('rating','N/A')} &nbsp;|&nbsp; 📞 {h.get('phone','N/A')}<br>
  <a href="{maps_link}" target="_blank">🗺️ Get Directions (Google Maps)</a>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> dict:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital.png", width=64)
        st.title("⚙️ Settings")

        st.subheader("👤 Patient Info (optional)")
        name   = st.text_input("Patient Name", placeholder="e.g. John Doe")
        age    = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])

        st.subheader("📍 Location (for hospital search)")
        loc_mode = st.radio("Location input", ["City name", "GPS coordinates"])
        user_lat = user_lon = None
        if loc_mode == "City name":
            city = st.text_input("City", placeholder="e.g. New Delhi")
            if city:
                coords = geocode_city(city)
                if coords:
                    user_lat, user_lon = coords
                    st.success(f"✅ {city} ({user_lat:.2f}, {user_lon:.2f})")
                else:
                    st.warning("City not in database. Sorting by rating.")
        else:
            user_lat = st.number_input("Latitude",  value=28.6139, format="%.4f")
            user_lon = st.number_input("Longitude", value=77.2090, format="%.4f")

        st.markdown("---")
        st.subheader("🔬 Scan Detection")
        st.caption(
            "Images are auto-detected as X-rays / MRI / CT scans "
            "based on filename and greyscale analysis. Include keywords like "
            "'xray', 'mri', 'ct', 'brain' in the filename to force scan mode."
        )
        st.markdown("---")
        st.caption("Medical AI Assistant v2.0\nStreamlit · LangGraph · GPT-4o Vision · Pinecone")

    patient_info = None
    if name or gender != "Not specified":
        patient_info = {"name": name, "age": age, "gender": gender}
    return {"patient_info": patient_info, "user_lat": user_lat, "user_lon": user_lon}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🏥 Medical AI Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a medical report or scan (X-ray / MRI / CT) '
        'and get AI-powered health insights with annotated findings.</p>',
        unsafe_allow_html=True,
    )
    _render_disclaimer()

    missing = validate_config()
    if missing:
        st.error(f"⚠️ Missing environment variables: {', '.join(missing)}. "
                 "Please add them to your `.env` file.")
        st.stop()

    sidebar_data = _sidebar()

    # ── Upload ───────────────────────────────────────────────────────────────
    st.subheader("📤 Upload Medical Report or Scan")
    st.caption(
        "**Text reports** (blood tests, lab reports) → OCR + LLM analysis  \n"
        "**Scan images** (X-ray, MRI, CT) → GPT-4o Vision + annotated findings image"
    )
    uploaded_file = st.file_uploader(
        f"Supported: {', '.join('.' + e for e in ALLOWED_EXTENSIONS)} (max {MAX_FILE_SIZE_MB} MB)",
        type=ALLOWED_EXTENSIONS,
    )

    if not uploaded_file:
        st.info("👆 Upload a report or scan to begin.")
        _how_it_works()
        return

    file_bytes = uploaded_file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
        return

    # Preview
    col1, col2 = st.columns([1, 2])
    with col1:
        if uploaded_file.type and uploaded_file.type.startswith("image"):
            st.image(BytesIO(file_bytes), caption=uploaded_file.name, use_container_width=True)
        else:
            st.success(f"✅ PDF: **{uploaded_file.name}**")
    with col2:
        st.markdown("**File details**")
        st.write(f"• Name: `{uploaded_file.name}`")
        st.write(f"• Size: `{len(file_bytes)/1024:.1f} KB`")
        st.write(f"• Type: `{uploaded_file.type}`")

    if st.button("🔬 Analyse Report / Scan", type="primary", use_container_width=True):
        # Clear previous chat when a new analysis starts
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("chat_system_prompt", None)
        _run_analysis(file_bytes, uploaded_file.name, sidebar_data)

    # Re-render results + chat if already analysed (session state)
    elif "last_result" in st.session_state:
        _render_results(st.session_state["last_result"])
        _render_chat_section(st.session_state["last_result"])


# ─────────────────────────────────────────────────────────────────────────────
# Analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis(file_bytes: bytes, filename: str, sidebar_data: dict):
    from workflow import run_medical_workflow

    progress = st.progress(0, text="Starting…")
    status   = st.empty()

    steps = [
        (10, "🔍 Detecting file type…"),
        (25, "🧠 Running AI analysis…"),
        (50, "📡 Retrieving medical knowledge…"),
        (70, "✍️ Generating findings report…"),
        (85, "🏥 Finding nearby hospitals…"),
        (95, "🖼️ Preparing annotated image…"),
    ]

    import time
    for pct, msg in steps:
        progress.progress(pct, text=msg)
        status.info(msg)
        time.sleep(0.35)

    try:
        result = run_medical_workflow(
            file_bytes=file_bytes,
            filename=filename,
            user_lat=sidebar_data.get("user_lat"),
            user_lon=sidebar_data.get("user_lon"),
            patient_info=sidebar_data.get("patient_info"),
        )
    except Exception as exc:
        progress.empty(); status.empty()
        st.error(f"❌ Analysis failed: {exc}")
        logger.exception("Workflow error")
        return

    progress.progress(100, text="✅ Done!")
    time.sleep(0.4)
    progress.empty(); status.empty()

    if result.get("error"):
        st.error(f"❌ {result['error']}")
        return

    # Save result to session state so chat persists across reruns
    st.session_state["last_result"] = result

    _render_results(result)
    _render_chat_section(result)


# ─────────────────────────────────────────────────────────────────────────────
# Results renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(result: dict):
    st.success("✅ Analysis Complete!")

    is_scan = result.get("is_scan", False)

    if is_scan:
        scan_type = result.get("scan_type", "Medical Scan")
        body_part = result.get("body_part", "")
        st.markdown(
            f'<span class="scan-badge">🩻 {scan_type}'
            f'{" – " + body_part if body_part else ""}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

    if result.get("summary"):
        st.subheader("📝 Overall Impression" if is_scan else "📝 Summary")
        st.write(result["summary"])

    severity      = result.get("severity", "Unknown")
    severity_info = result.get("severity_info", {})
    css_class     = _severity_class(severity)
    st.subheader("⚠️ Severity Level")
    st.markdown(f"""
<div class="{css_class}">
  <strong>{severity_info.get('color','')} {severity}</strong><br>
  {result.get('severity_reason','')}
  <br><em>{severity_info.get('action','')}</em>
</div>""", unsafe_allow_html=True)
    st.markdown("")

    if is_scan:
        _render_scan_results(result)
    else:
        _render_text_report_results(result)

    st.markdown("---")
    st.subheader("🏥 Nearby Hospital Recommendations")
    spec = result.get("doctor_specialization", "General Practitioner")
    st.markdown(
        f'<div style="background:#1e3a5f;border-left:5px solid #3b82f6;border-radius:6px;'
        f'padding:.8rem 1.2rem;color:#ffffff;font-size:1rem;font-weight:600;">'
        f'👨‍⚕️ Recommended Specialist: {spec}</div>',
        unsafe_allow_html=True,
    )
    _render_hospitals(result.get("hospitals", []))
    st.caption("*Hospital data is for demonstration. Always verify with current sources.*")

    st.markdown("---")
    _render_disclaimer()


def _render_scan_results(result: dict):
    findings        = result.get("scan_findings", [])
    annotated_bytes = result.get("annotated_image_bytes")

    col_img, col_findings = st.columns([3, 2])

    with col_img:
        st.subheader("🩻 Annotated Scan Image")
        if annotated_bytes:
            st.image(
                BytesIO(annotated_bytes),
                caption=(
                    f"{'🔴 ' if any(f.get('severity') == 'Emergency' for f in findings) else ''}"
                    f"{len(findings)} finding(s) marked with coloured ovals"
                    if findings else "No anomalies detected"
                ),
                use_container_width=True,
            )
            st.download_button(
                label="⬇️ Download Annotated Scan",
                data=annotated_bytes,
                file_name="annotated_scan.png",
                mime="image/png",
            )
        else:
            st.warning("Annotated image not available.")

    with col_findings:
        st.subheader(f"🔬 Findings ({len(findings)})")
        if findings:
            _render_scan_findings(findings)
        else:
            st.success("✅ No anomalies detected.")

        if result.get("precautions"):
            st.subheader("🛡️ Precautions")
            for p in result["precautions"]:
                st.markdown(f"• {p}")

    st.subheader("💊 Health Guidance")
    st.write(result.get("health_guidance", "Please consult a radiologist."))

    with st.expander("📄 Vision Analysis Summary (raw)"):
        st.text(result.get("raw_text", ""))


def _render_text_report_results(result: dict):
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Parameters", "🦠 Conditions", "💊 Guidance", "📄 Raw Text"
    ])

    with tab1:
        st.subheader("Extracted Medical Parameters")
        _render_parameter_table(result.get("extracted_parameters", []))

    with tab2:
        st.subheader("Detected Conditions")
        _render_conditions(result.get("detected_conditions", []))
        if result.get("precautions"):
            st.subheader("🛡️ Precautions")
            for p in result["precautions"]:
                st.markdown(f"• {p}")

    with tab3:
        st.subheader("💊 Health Guidance")
        st.write(result.get("health_guidance", "No guidance available."))

    with tab4:
        st.subheader("📄 Extracted OCR Text")
        raw = result.get("raw_text", "")
        st.text_area("OCR Output", raw, height=300)
        st.caption(f"Characters extracted: {len(raw)}")


# ─────────────────────────────────────────────────────────────────────────────
# How it works
# ─────────────────────────────────────────────────────────────────────────────

def _how_it_works():
    st.markdown("---")
    st.subheader("🔄 How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🩻 For Scans (X-ray / MRI / CT)")
        st.markdown("""
1. Upload your scan image (JPG/PNG)
2. AI auto-detects it as a medical scan
3. **GPT-4o Vision** analyses the image directly
4. Anomalies are marked with **coloured oval annotations**
5. Download the annotated image + ask follow-up questions
""")
    with col2:
        st.markdown("#### 📋 For Lab Reports / PDFs")
        st.markdown("""
1. Upload blood test / lab report PDF or image
2. **EasyOCR** extracts all text
3. **Pinecone RAG** retrieves relevant medical knowledge
4. **GPT-4o** analyses parameters and detects conditions
5. Get severity, precautions, hospitals + ask follow-up questions
""")
    st.markdown("---")
    st.caption("💡 **Tip:** Include keywords like `xray`, `mri`, `brain`, `ct` in your "
               "filename to ensure scan mode (e.g. `chest_xray.jpg`, `brain_mri.png`).")


if __name__ == "__main__":
    main()

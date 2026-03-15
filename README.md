# 🏥 Medical AI Assistant

> Upload a medical report or scan → AI analyses it → Get findings, severity, doctor recommendations & nearby hospitals.

---

## 📖 Table of Contents

1. [What Does This App Do?](#-what-does-this-app-do)
2. [File Structure](#-file-structure)
3. [What Each File Does](#-what-each-file-does)
4. [How Files Connect to Each Other](#-how-files-connect-to-each-other)
5. [Full Workflow — Step by Step](#-full-workflow--step-by-step)
6. [LangGraph Workflow Explained](#-langgraph-workflow-explained)
7. [Technology Explained Simply](#-technology-explained-simply)
8. [Setup & Installation](#-setup--installation)
9. [Environment Variables](#-environment-variables)
10. [Running the App](#-running-the-app)
11. [How to Use the App](#-how-to-use-the-app)
12. [Common Errors & Fixes](#-common-errors--fixes)

---

## 🤔 What Does This App Do?

In simple words:

```
You upload a medical report (PDF or image)
         ↓
The app reads it (OCR or Vision AI)
         ↓
AI analyses the content
         ↓
You get:
  ✅ What conditions might be present
  ✅ How serious it is (Mild / Moderate / Emergency)
  ✅ What doctor to see
  ✅ What precautions to take
  ✅ Nearby hospitals with Google Maps links
  ✅ For X-rays/MRI/CT — the image with findings circled in red/yellow/green
```

**Two modes:**

| Mode | When | How |
|------|------|-----|
| 📋 Text Report | Blood test, lab report, PDF prescription | OCR extracts text → GPT-4o reads it |
| 🩻 Scan Image | X-ray, MRI, CT scan | GPT-4o Vision looks at the image directly → draws ovals on findings |

---

## 📁 File Structure

```
medical-ai-assistant/
│
├── app.py                ← 🖥️  The screen you see (Streamlit UI)
├── workflow.py           ← 🔀  The brain — decides what to do in what order
├── xray_analyzer.py      ← 🩻  Handles X-ray/MRI/CT scan analysis + image annotation
├── ocr.py                ← 👁️  Reads text from PDF and image files
├── llm.py                ← 🤖  Talks to OpenAI GPT-4o for text analysis
├── rag.py                ← 📚  Stores and searches medical knowledge (Pinecone)
├── hospital.py           ← 🏥  Finds nearby hospitals + Google Maps links
├── config.py             ← ⚙️  All settings and API keys in one place
├── seed_pinecone.py      ← 🌱  One-time script to load medical knowledge into Pinecone
│
├── utils/
│   └── helpers.py        ← 🔧  Small helper functions used across files
│
├── prompts/
│   └── medical_prompts.py ← 📝  The instruction text sent to GPT-4o
│
├── data/
│   └── medical_knowledge/ ← 🗄️  Folder for extra knowledge files (optional)
│
├── requirements.txt      ← 📦  List of Python packages to install
├── .env.example          ← 🔑  Template for your secret API keys
└── README.md             ← 📖  This file
```

---

## 📄 What Each File Does

### `app.py` — The Frontend (What You See)
```
Role: Draws the entire website using Streamlit
Does:
  - Shows the upload button
  - Shows sidebar (patient info, location)
  - Shows progress bar while analysing
  - Displays results: severity box, findings, hospitals
  - For scans: shows the annotated image side-by-side with findings list
  - Has the download button for annotated scan
Talks to: workflow.py (sends file → gets results back)
```

### `workflow.py` — The Orchestrator (The Brain)
```
Role: Controls the entire analysis pipeline using LangGraph
Does:
  - Receives the uploaded file from app.py
  - Decides: is this a scan or a text report?
  - Runs each step in the correct order
  - Passes data from one step to the next
  - Returns the final result to app.py
Talks to: ocr.py, xray_analyzer.py, rag.py, llm.py, hospital.py
```

### `xray_analyzer.py` — The Scan Expert
```
Role: Analyses X-ray, MRI, CT images using AI Vision
Does:
  - Detects if image is a medical scan (by filename + greyscale check)
  - Sends image to GPT-4o Vision with radiologist instructions
  - Gets back findings with approximate locations (cx, cy, width, height)
  - Draws coloured ovals on those locations using Pillow
  - Returns annotated image as PNG bytes
Talks to: OpenAI Vision API, Pillow (image drawing)
Colour coding: 🔴 Emergency  🟡 Moderate  🟢 Mild
```

### `ocr.py` — The Text Reader
```
Role: Extracts text from uploaded files
Does:
  - PDF files → uses pdfplumber to read native text
  - Scanned PDFs → falls back to EasyOCR
  - Image files (JPG/PNG) → uses EasyOCR
  - Returns clean text string
Talks to: pdfplumber, EasyOCR
```

### `llm.py` — The AI Doctor (for text reports)
```
Role: Sends extracted text to GPT-4o and gets medical analysis
Does:
  - Builds a structured prompt with report text + medical knowledge
  - Calls OpenAI API with JSON mode ON
  - Gets back: conditions, severity, doctor type, precautions, guidance
  - Returns structured Python dict
Talks to: OpenAI API, config.py
Note: Skipped for scans (xray_analyzer.py handles that)
```

### `rag.py` — The Medical Knowledge Bank
```
Role: Stores medical knowledge and finds relevant info for each query
Does:
  - Stores 13 medical knowledge chunks in Pinecone (vector database)
  - For each analysis, searches for the most relevant knowledge
  - Returns top 5 matching chunks to help the LLM reason better
  - Knowledge covers: Diabetes, Hypertension, Anaemia, Thyroid,
    Kidney, Liver, Heart, Respiratory, Doctor specialization map
Talks to: Pinecone API, OpenAI Embeddings API
```

### `hospital.py` — The Hospital Finder
```
Role: Finds relevant hospitals and generates Google Maps links
Does:
  - Filters hospitals by doctor specialization
  - If user location provided: sorts by distance (uses Haversine formula)
  - Otherwise: sorts by rating
  - Generates Google Maps directions URL for each hospital
  - Has a built-in database of 11 real hospitals (US/UK/India/AU/CA)
Talks to: Nothing external (self-contained database)
```

### `config.py` — The Settings File
```
Role: One place to read all environment variables and app settings
Does:
  - Reads .env file
  - Defines constants: model names, index names, max file size
  - Defines severity level metadata
  - Has validate_config() to check if required keys are set
Talks to: Everything (all files import from config.py)
```

### `seed_pinecone.py` — The One-Time Setup Script
```
Role: Loads medical knowledge into Pinecone (run once before first use)
Does:
  - Creates the Pinecone index if it doesn't exist
  - Embeds 13 medical knowledge texts using OpenAI embeddings
  - Upserts them into Pinecone
  - Skips if already done (idempotent)
Run with: python seed_pinecone.py
```

---

## 🔗 How Files Connect to Each Other

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│              (User sees this / Streamlit UI)            │
└──────────────────────┬──────────────────────────────────┘
                       │ sends: file_bytes, filename,
                       │        user location, patient info
                       ▼
┌─────────────────────────────────────────────────────────┐
│                     workflow.py                         │
│              (LangGraph pipeline controller)            │
│                                                         │
│  imports and calls:                                     │
│  ┌─────────┐  ┌──────────────┐  ┌─────┐  ┌──────────┐ │
│  │  ocr.py │  │xray_analyzer │  │rag  │  │  llm.py  │ │
│  └─────────┘  └──────────────┘  └─────┘  └──────────┘ │
│                                    ↑            ↑       │
│                               rag.py         llm.py     │
│                          (Pinecone search) (OpenAI GPT) │
│                                                         │
│  ┌──────────┐                                           │
│  │hospital  │                                           │
│  └──────────┘                                           │
└──────────────────────┬──────────────────────────────────┘
                       │ returns: full result dict
                       ▼
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│              (renders results on screen)                │
└─────────────────────────────────────────────────────────┘

config.py ←── imported by ALL files (shared settings)
```

**Data flow between files:**

```
app.py          → workflow.py    : file bytes + metadata
workflow.py     → ocr.py         : file bytes + filename
ocr.py          → workflow.py    : extracted text string
workflow.py     → xray_analyzer  : file bytes + filename
xray_analyzer   → OpenAI Vision  : base64 image
OpenAI Vision   → xray_analyzer  : JSON findings + coordinates
xray_analyzer   → Pillow         : PIL image + findings
Pillow          → xray_analyzer  : annotated PNG bytes
xray_analyzer   → workflow.py    : findings + annotated image
workflow.py     → rag.py         : query text
rag.py          → Pinecone       : embedding vector
Pinecone        → rag.py         : top 5 matching chunks
rag.py          → workflow.py    : knowledge context
workflow.py     → llm.py         : text + context
llm.py          → OpenAI GPT-4o  : prompt
OpenAI GPT-4o   → llm.py         : structured JSON
llm.py          → workflow.py    : conditions, severity, etc.
workflow.py     → hospital.py    : specialization + location
hospital.py     → workflow.py    : hospital list + maps links
workflow.py     → app.py         : complete result dict
app.py          → User Screen    : rendered UI
```

---

## 🚀 Full Workflow — Step by Step

### For a Text Report (Blood Test / Lab Report / PDF)

```
STEP 1 — User uploads file
         app.py receives the bytes and filename

STEP 2 — report_reader node
         Checks: is a file actually there? If not → error

STEP 3 — scan_type_detector node
         Looks at filename and image pixels
         "Is this an X-ray or text report?"
         → Result: is_scan = False (it's a text report)

STEP 4 — ocr_extraction node
         If PDF → pdfplumber reads native text
         If scanned PDF or image → EasyOCR reads pixels
         → Result: raw_text = "Patient: John... Haemoglobin: 9.2 g/dL..."

STEP 5 — medical_parameter_extractor node
         Pass-through (LLM handles this in next step)

STEP 6 — rag_retrieval_pinecone node
         Takes first 1000 chars of raw_text
         Converts to vector using OpenAI embeddings
         Searches Pinecone for top 5 similar knowledge chunks
         → Result: context about anaemia, blood disorders, etc.

STEP 7 — llm_medical_reasoning node
         Builds prompt: report text + RAG context + patient info
         Sends to GPT-4o with JSON mode
         → Result: {conditions, severity, doctor, precautions, guidance}

STEP 8 — severity_classifier node
         Reads severity from LLM result
         Attaches colour + action message
         → Result: severity = "Moderate", action = "See doctor in 24-48hrs"

STEP 9 — doctor_specialization_selector node
         Reads doctor type from LLM result
         → Result: doctor_specialization = "Hematologist"

STEP 10 — hospital_recommender node
          Filters hospital database by "Hematologist"
          If user gave location → sorts by distance
          → Result: top 5 matching hospitals

STEP 11 — navigation_link_generator node
          Builds Google Maps URL for each hospital
          → Result: https://www.google.com/maps/dir/?api=1&destination=...

STEP 12 — result_formatter node
          Logs final state, returns everything to app.py

STEP 13 — app.py renders the results
          Shows: summary, severity box, parameters table,
                 conditions list, precautions, hospitals
```

---

### For a Scan Image (X-ray / MRI / CT)

```
STEP 1-2 — Same as above (upload + validate)

STEP 3 — scan_type_detector node
         Filename contains "xray" or "mri" or "ct"?  → is_scan = True
         OR image is mostly greyscale (R≈G≈B)?       → is_scan = True
         → Workflow takes the SCAN BRANCH

STEP 4 — xray_vision_analysis node  ← (replaces OCR + LLM steps)
         a) Converts image to base64
         b) Sends to GPT-4o Vision with radiologist system prompt
         c) Gets back JSON:
            {
              "findings": [
                {
                  "label": "Consolidation",
                  "description": "Increased opacity suggesting pneumonia",
                  "severity": "Moderate",
                  "region": "lower-right lung",
                  "cx": 0.65,   ← 65% from left
                  "cy": 0.72,   ← 72% from top
                  "w": 0.18,    ← oval width 18% of image
                  "h": 0.14     ← oval height 14% of image
                }
              ]
            }
         d) Draws on the image using Pillow:
            - Semi-transparent oval at (cx, cy) coordinates
            - Coloured border: 🔴 Emergency / 🟡 Moderate / 🟢 Mild
            - Number badge ① ② ③ above oval
            - Finding label next to badge
            - Legend strip at bottom: "1. Consolidation (lower-right lung)"
         e) Saves annotated image as PNG bytes

STEP 5 — rag_retrieval_pinecone node
         Queries: "X-Ray Chest Consolidation Pneumonia"
         Gets relevant medical knowledge

STEP 6 — llm_medical_reasoning node
         SKIPPED for scans (Vision API already did full analysis)

STEP 7-11 — severity, doctor, hospitals (same as text report)

STEP 12 — app.py renders scan results
          LEFT column:  Annotated scan image + download button
          RIGHT column: Findings list with severity badges
          Below: Health guidance, hospitals
```

---

## 🧠 LangGraph Workflow Explained

LangGraph is like a flowchart engine. You define **nodes** (steps) and **edges** (arrows between steps).

```
What is a Node?
  → A Python function that receives the current "state" dict
    and returns an updated "state" dict

What is State?
  → A dictionary that travels through all nodes
    Each node reads from it and writes back to it
    At the end, app.py reads the final state

What is a Conditional Edge?
  → A fork in the road based on a condition
  → After scan_type_detector, we check: is_scan = True or False?
    True  → go to xray_vision_analysis
    False → go to ocr_extraction
```

**The graph visually:**

```
[report_reader]
      ↓
[scan_type_detector]
      ↓
   is_scan?
   /       \
YES         NO
 ↓           ↓
[xray_      [ocr_
 vision]     extraction]
  ↓           ↓
  └─────┬─────┘
        ↓
[rag_retrieval_pinecone]
        ↓
[llm_medical_reasoning]  ← skipped if is_scan
        ↓
[severity_classifier]
        ↓
[doctor_specialization_selector]
        ↓
[hospital_recommender]
        ↓
[navigation_link_generator]
        ↓
[result_formatter]
        ↓
       END
```

**LangSmith** watches every node as it runs — records timing, inputs, outputs, errors — so you can debug on their website.

---

## 🛠️ Technology Explained Simply

| Technology | What it is | Why we use it |
|------------|-----------|---------------|
| **Streamlit** | Python library that turns code into a web app | Build the UI without knowing HTML/CSS/JS |
| **LangGraph** | Workflow engine for AI pipelines | Run steps in order, branch on conditions |
| **OpenAI GPT-4o** | Large Language Model by OpenAI | Read text reports and return structured analysis |
| **GPT-4o Vision** | Same model but can see images | Look at X-rays/MRI directly, no text needed |
| **Pinecone** | Vector database (stores meaning, not just text) | Find relevant medical knowledge for each query |
| **OpenAI Embeddings** | Converts text into a list of numbers (vector) | Make text searchable by meaning in Pinecone |
| **EasyOCR** | Optical Character Recognition library | Read text from scanned images and photos |
| **pdfplumber** | PDF reading library | Extract text from PDF files |
| **Pillow (PIL)** | Python image processing library | Draw ovals, badges, labels on scan images |
| **LangSmith** | Monitoring tool for LangChain/LangGraph | See exactly what happens in each workflow run |
| **uv** | Fast Python package manager | Install packages faster than pip |

---

## ⚙️ Setup & Installation

### Step 1 — Install Python 3.12
Download from: https://python.org/downloads

### Step 2 — Install uv
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3 — Create project environment
```bash
cd medical-ai-assistant

# Initialise uv
uv init

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate.bat       # Windows CMD
.venv\Scripts\Activate.ps1       # Windows PowerShell
```

### Step 4 — Install all packages
```bash
uv pip install -r requirements.txt
```

> ⚠️ EasyOCR downloads ~100 MB of model files on first run. This is normal and only happens once.

---

## 🔑 Environment Variables

```bash
# Copy the template
cp .env.example .env

# Open .env in VS Code
code .env
```

Fill in these values:

```env
# ── REQUIRED ─────────────────────────────────────────────
OPENAI_API_KEY=sk-...        # from platform.openai.com/api-keys
PINECONE_API_KEY=...         # from app.pinecone.io → API Keys

# ── OPTIONAL (for monitoring) ─────────────────────────────
LANGCHAIN_API_KEY=ls__...    # from smith.langchain.com → Settings
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=medical-ai-assistant
```

**Where to get each key:**

```
OpenAI API Key
  → https://platform.openai.com/api-keys
  → Click "+ Create new secret key"
  → Copy and paste

Pinecone API Key
  → https://app.pinecone.io
  → Sign up free → Go to API Keys
  → Copy the default key

LangSmith (optional but useful for debugging)
  → https://smith.langchain.com
  → Sign up free → Settings → API Keys → Create Key
```

---

## 🌱 Seed the Knowledge Base (Run Once)

Before first use, load medical knowledge into Pinecone:

```bash
python seed_pinecone.py
```

You'll see:
```
Seeding Pinecone with medical knowledge…
Seeding 13 medical knowledge vectors into Pinecone…
Seeding complete.
Done! Your Pinecone index is ready.
```

This only needs to run **once**. Running again is safe (it skips if already done).

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Opens automatically at: **http://localhost:8501**

---

## 📱 How to Use the App

### Uploading a Text Report
1. Open the app in browser
2. Optionally fill patient info in sidebar (name, age, location)
3. Click **Browse files** and upload a blood test / lab report (PDF or JPG/PNG)
4. Click **🔬 Analyse Report / Scan**
5. Wait ~10-20 seconds
6. View results in tabs: Parameters / Conditions / Guidance / Raw Text

### Uploading a Scan (X-ray / MRI / CT)
1. **Name your file with a scan keyword** for best detection:
   - `chest_xray.jpg` ✅
   - `brain_mri.png` ✅
   - `ct_abdomen.jpg` ✅
   - `knee_scan.jpg` ✅ (greyscale auto-detection)
2. Upload the file and click Analyse
3. Results show the **annotated image** on the left with coloured ovals
4. Click **⬇️ Download Annotated Scan** to save the marked image

### Understanding the Results

**Severity colours:**
- 🟢 **Mild** — Monitor, routine check-up
- 🟡 **Moderate** — See a doctor within 24-48 hours
- 🔴 **Emergency** — Seek immediate medical attention

**Oval colours on scan images:**
- 🔴 Red oval = Emergency finding
- 🟡 Yellow oval = Moderate finding
- 🟢 Green oval = Mild finding

---

## ❗ Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing OPENAI_API_KEY` | .env file missing or wrong key | Check `.env` file is in project root |
| `Missing PINECONE_API_KEY` | Pinecone key not set | Add key to `.env` |
| `Pinecone index not found` | Knowledge base not seeded | Run `python seed_pinecone.py` |
| `OCR produced no text` | Blank or corrupted file | Check the file opens normally |
| `Scan not detected` | No keyword in filename | Add `xray`/`mri`/`ct` to filename |
| `streamlit: command not found` | Virtual env not activated | Run `source .venv/bin/activate` |
| `EasyOCR very slow` | First-time model download | Wait ~2-3 min, faster next time |
| Vision API error | GPT-4o access needed | Ensure your OpenAI account has GPT-4o access |

---

## 📁 Key Code Locations (Quick Reference)

| I want to change… | File | Look for… |
|-------------------|------|-----------|
| UI colours / layout | `app.py` | `<style>` block at top |
| Add more hospitals | `hospital.py` | `HOSPITAL_DB` list |
| Add medical knowledge | `rag.py` | `MEDICAL_KNOWLEDGE` list |
| Change GPT model | `.env` | `OPENAI_MODEL=` |
| Oval annotation style | `xray_analyzer.py` | `_draw_findings()` function |
| LLM prompt | `llm.py` | `SYSTEM_PROMPT` variable |
| Severity levels | `config.py` | `SEVERITY_LEVELS` dict |
| Workflow order | `workflow.py` | `_build_graph()` function |

---

## ⚠️ Medical Disclaimer

This AI system provides **informational guidance only** and does **not** replace professional medical advice. Please consult a certified doctor before taking any treatment. This is a demonstration tool and is NOT a certified medical device.

---

*Built with ❤️ using Streamlit · LangGraph · GPT-4o · Pinecone · EasyOCR*

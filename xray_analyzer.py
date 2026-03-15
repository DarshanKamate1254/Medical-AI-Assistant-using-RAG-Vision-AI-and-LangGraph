"""
xray_analyzer.py - Vision-based X-ray / MRI / CT scan analysis using GPT-4o Vision.

Pipeline:
  1. Detect if uploaded image is a medical scan (X-ray, MRI, CT, etc.)
  2. Send image to GPT-4o Vision for radiological analysis
  3. Parse structured findings (anomalies with bounding regions)
  4. Draw red oval annotations on the image for each finding
  5. Return annotated image bytes + structured report
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont  # type: ignore

from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Scan type detection keywords
# ─────────────────────────────────────────────────────────────────────────────

SCAN_KEYWORDS = {
    "xray":   ["xray", "x-ray", "x_ray", "chest_xray", "chest-xray", "radiograph"],
    "mri":    ["mri", "brain_mri", "spine_mri", "fmri", "magnetic"],
    "ct":     ["ct", "ctscan", "ct-scan", "ct_scan", "computed_tomography"],
    "scan":   ["scan", "imaging", "radiology", "dicom", "dcm"],
}

# Image-type heuristic: if filename contains any of these → treat as scan
SCAN_FILENAME_HINTS = [kw for kws in SCAN_KEYWORDS.values() for kw in kws]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class XRayFinding:
    """A single radiological finding with an approximate image region."""
    label: str                        # e.g. "Consolidation", "Fracture"
    description: str                  # plain-language explanation
    severity: str                     # Mild | Moderate | Emergency
    region: str                       # anatomical region, e.g. "lower-right lung"
    # Normalised coordinates 0–1 (centre x, centre y, width, height)
    cx: float = 0.5
    cy: float = 0.5
    w:  float = 0.15
    h:  float = 0.15
    confidence: str = "Medium"        # High | Medium | Low


@dataclass
class XRayReport:
    scan_type: str                           # "X-Ray" | "MRI" | "CT Scan" | "Medical Scan"
    body_part: str                           # "Chest" | "Brain" | "Spine" …
    findings: list[XRayFinding] = field(default_factory=list)
    overall_impression: str = ""
    severity: str = "Mild"
    doctor_specialization: str = "Radiologist"
    precautions: list[str] = field(default_factory=list)
    health_guidance: str = ""
    annotated_image_bytes: bytes | None = None   # PNG bytes of annotated image
    raw_llm_response: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Scan-type detection
# ─────────────────────────────────────────────────────────────────────────────

def is_medical_scan(filename: str, file_bytes: bytes) -> bool:
    """
    Return True if the file looks like an X-ray, MRI, or CT scan.
    Checks filename hints first; falls back to a quick greyscale heuristic.
    """
    fname_lower = Path(filename).stem.lower().replace(" ", "_").replace("-", "_")
    for hint in SCAN_FILENAME_HINTS:
        if hint in fname_lower:
            logger.info("Scan detected via filename hint '%s'", hint)
            return True

    # Heuristic: medical scans tend to be mostly greyscale
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # Sample 500 pixels and check colour variance
        import numpy as np
        arr = np.array(img.resize((100, 100)))
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        channel_diff = max(abs(r-g), abs(g-b), abs(r-b))
        if channel_diff < 12:          # very greyscale → likely a scan
            logger.info("Scan detected via greyscale heuristic (channel_diff=%.1f)", channel_diff)
            return True
    except Exception:
        pass

    return False


def detect_scan_type(filename: str) -> str:
    fname = filename.lower()
    if any(k in fname for k in SCAN_KEYWORDS["mri"]):
        return "MRI"
    if any(k in fname for k in SCAN_KEYWORDS["ct"]):
        return "CT Scan"
    if any(k in fname for k in SCAN_KEYWORDS["xray"]):
        return "X-Ray"
    return "Medical Scan"


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o Vision prompt
# ─────────────────────────────────────────────────────────────────────────────

XRAY_SYSTEM_PROMPT = """You are an expert radiologist AI assistant.
Analyse the provided medical scan image (X-ray / MRI / CT) and return ONLY a JSON object.
Do NOT include markdown fences or extra text.

JSON Schema:
{
  "scan_type": "X-Ray|MRI|CT Scan|Medical Scan",
  "body_part": "e.g. Chest, Brain, Spine, Knee",
  "findings": [
    {
      "label": "Short finding name (e.g. Consolidation, Fracture, Mass, Lesion)",
      "description": "Plain-language explanation for a patient",
      "severity": "Mild|Moderate|Emergency",
      "region": "Anatomical region (e.g. lower-right lung, left temporal lobe)",
      "confidence": "High|Medium|Low",
      "cx": 0.55,   // normalised centre-x of anomaly (0.0=left edge, 1.0=right edge)
      "cy": 0.60,   // normalised centre-y of anomaly (0.0=top, 1.0=bottom)
      "w":  0.15,   // normalised width of bounding oval
      "h":  0.12    // normalised height of bounding oval
    }
  ],
  "overall_impression": "Summary paragraph in simple language",
  "severity": "Mild|Moderate|Emergency",
  "doctor_specialization": "e.g. Pulmonologist, Neurologist, Orthopedic Surgeon",
  "precautions": ["precaution 1", "precaution 2"],
  "health_guidance": "Detailed guidance paragraph"
}

Rules:
- If no abnormality is found, return an empty findings array and severity "Mild".
- Provide cx/cy/w/h even for approximate regions — these are used to annotate the image.
- Be concise but medically accurate.
- Always use simple language in description and health_guidance.
"""


def _image_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def _call_vision_api(file_bytes: bytes, scan_type_hint: str) -> dict[str, Any]:
    """Call GPT-4o Vision and return parsed JSON dict."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=OPENAI_API_KEY)
    b64 = _image_to_base64(file_bytes)

    # Detect mime type
    try:
        img = Image.open(io.BytesIO(file_bytes))
        fmt = (img.format or "PNG").lower()
        mime = {"jpeg": "image/jpeg", "jpg": "image/jpeg",
                "png": "image/png", "bmp": "image/bmp"}.get(fmt, "image/png")
    except Exception:
        mime = "image/png"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": XRAY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": f"This is a {scan_type_hint}. Analyse it thoroughly and return the JSON report.",
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Image annotation
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette per severity
SEVERITY_COLOURS = {
    "Emergency": (220, 53,  69,  200),   # red
    "Moderate":  (255, 193,  7,  200),   # amber
    "Mild":      (40,  167, 69,  200),   # green
}
DEFAULT_COLOUR = (255, 100, 100, 200)


def _draw_findings(pil_image: Image.Image, findings: list[XRayFinding]) -> Image.Image:
    """
    Overlay red oval annotations + labels onto *pil_image* for every finding.
    Returns a new RGBA PIL image.
    """
    img = pil_image.convert("RGBA")
    W, H = img.size

    # Overlay layer for semi-transparent fills
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main   = ImageDraw.Draw(img)

    # Try to load a font; fall back to default
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, W // 55))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      max(11, W // 70))
    except Exception:
        font_label = ImageFont.load_default()
        font_small = font_label

    for idx, finding in enumerate(findings, start=1):
        colour_rgba = SEVERITY_COLOURS.get(finding.severity, DEFAULT_COLOUR)
        colour_solid = colour_rgba[:3]

        # Pixel coords of bounding box
        cx_px = int(finding.cx * W)
        cy_px = int(finding.cy * H)
        rw    = int(finding.w  * W / 2)
        rh    = int(finding.h  * H / 2)

        x0, y0 = cx_px - rw, cy_px - rh
        x1, y1 = cx_px + rw, cy_px + rh

        # Semi-transparent fill
        draw_overlay.ellipse([x0, y0, x1, y1], fill=(*colour_solid, 40))
        # Solid border (3px)
        for thickness in range(3):
            draw_overlay.ellipse(
                [x0 - thickness, y0 - thickness, x1 + thickness, y1 + thickness],
                outline=(*colour_solid, 230),
            )

        # Merge overlay so far
        img = Image.alpha_composite(img, overlay)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_main    = ImageDraw.Draw(img)

        # Number badge (circle)
        badge_r = max(14, W // 50)
        bx, by  = x0 + 4, y0 - badge_r * 2 - 4
        bx = max(badge_r + 2, min(W - badge_r - 2, bx))
        by = max(badge_r + 2, min(H - badge_r - 2, by))
        draw_main.ellipse(
            [bx - badge_r, by - badge_r, bx + badge_r, by + badge_r],
            fill=colour_solid,
        )
        num_text = str(idx)
        try:
            bbox = draw_main.textbbox((0, 0), num_text, font=font_label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = badge_r, badge_r
        draw_main.text(
            (bx - tw // 2, by - th // 2), num_text,
            fill="white", font=font_label,
        )

        # Label text below badge
        label_text = f"  {finding.label}"
        lx = bx + badge_r + 4
        ly = by - badge_r
        # Background pill
        try:
            tbbox = draw_main.textbbox((lx, ly), label_text, font=font_label)
            pad = 4
            draw_main.rectangle(
                [tbbox[0]-pad, tbbox[1]-pad, tbbox[2]+pad, tbbox[3]+pad],
                fill=(*colour_solid, 200),
            )
        except Exception:
            pass
        draw_main.text((lx, ly), label_text, fill="white", font=font_label)

    # Final legend strip at the bottom
    legend_h = max(36, H // 18)
    legend_strip = Image.new("RGBA", (W, legend_h), (20, 20, 20, 200))
    ld = ImageDraw.Draw(legend_strip)
    items = []
    for f in findings:
        items.append(f"{findings.index(f)+1}. {f.label} ({f.region})")
    legend_text = "  |  ".join(items) if items else "No anomalies detected"
    try:
        ld.text((8, (legend_h - 14) // 2), legend_text, fill="white", font=font_small)
    except Exception:
        ld.text((8, 4), legend_text, fill="white")

    final = Image.new("RGBA", (W, H + legend_h), (0, 0, 0, 255))
    final.paste(img, (0, 0))
    final.paste(legend_strip, (0, H), legend_strip)
    return final.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_xray(file_bytes: bytes, filename: str) -> XRayReport:
    """
    Full pipeline: vision analysis + image annotation.

    Returns XRayReport with annotated_image_bytes populated.
    """
    scan_type_hint = detect_scan_type(filename)
    logger.info("Analysing %s: %s", scan_type_hint, filename)

    # ── 1. Call GPT-4o Vision ─────────────────────────────────────────────
    try:
        raw = _call_vision_api(file_bytes, scan_type_hint)
    except Exception as exc:
        logger.exception("Vision API call failed")
        return XRayReport(
            scan_type=scan_type_hint,
            body_part="Unknown",
            overall_impression=f"Analysis failed: {exc}",
            health_guidance="Please consult a radiologist.",
        )

    # ── 2. Parse findings ─────────────────────────────────────────────────
    findings: list[XRayFinding] = []
    for f in raw.get("findings", []):
        findings.append(XRayFinding(
            label=f.get("label", "Anomaly"),
            description=f.get("description", ""),
            severity=f.get("severity", "Mild"),
            region=f.get("region", "unspecified"),
            confidence=f.get("confidence", "Medium"),
            cx=float(f.get("cx", 0.5)),
            cy=float(f.get("cy", 0.5)),
            w=float(f.get("w", 0.15)),
            h=float(f.get("h", 0.12)),
        ))

    # ── 3. Annotate image ─────────────────────────────────────────────────
    annotated_bytes: bytes | None = None
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        annotated_pil = _draw_findings(pil_image, findings)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        annotated_bytes = buf.getvalue()
        logger.info("Image annotated with %d findings.", len(findings))
    except Exception as exc:
        logger.warning("Image annotation failed: %s – returning original.", exc)
        annotated_bytes = file_bytes

    return XRayReport(
        scan_type=raw.get("scan_type", scan_type_hint),
        body_part=raw.get("body_part", "Unknown"),
        findings=findings,
        overall_impression=raw.get("overall_impression", ""),
        severity=raw.get("severity", "Mild"),
        doctor_specialization=raw.get("doctor_specialization", "Radiologist"),
        precautions=raw.get("precautions", []),
        health_guidance=raw.get("health_guidance", ""),
        annotated_image_bytes=annotated_bytes,
        raw_llm_response=raw,
    )

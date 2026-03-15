"""
ocr.py - Extract text from medical reports (PDF / image) using EasyOCR + pdfplumber.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Use pdfplumber for native-text PDFs; fall back to EasyOCR for scanned pages."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("pdfplumber is not installed. Run: uv pip install pdfplumber")

    text_pages: list[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) > 20:          # native text found
                text_pages.append(page_text)
                logger.debug("PDF page %d: native text extracted (%d chars)", page_num, len(page_text))
            else:                                      # scanned page → OCR
                logger.debug("PDF page %d: no native text, falling back to OCR", page_num)
                pil_image = page.to_image(resolution=200).original
                ocr_text = _ocr_pil_image(pil_image)
                text_pages.append(ocr_text)

    return "\n\n".join(text_pages)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_pil_image(pil_image) -> str:
    """Run EasyOCR on a PIL image and return extracted text."""
    try:
        import easyocr  # type: ignore
        import numpy as np
    except ImportError:
        raise ImportError("easyocr is not installed. Run: uv pip install easyocr")

    reader = _get_easyocr_reader()
    img_array = np.array(pil_image.convert("RGB"))
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


def _ocr_image_bytes(file_bytes: bytes) -> str:
    """Run EasyOCR on raw image bytes."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        raise ImportError("Pillow is not installed. Run: uv pip install Pillow")

    pil_image = Image.open(io.BytesIO(file_bytes))
    return _ocr_pil_image(pil_image)


# Lazy singleton for EasyOCR reader (model download happens once)
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # type: ignore
        logger.info("Initialising EasyOCR reader (first run downloads ~100 MB model)…")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from an uploaded medical report.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect file type).

    Returns:
        Extracted text string.

    Raises:
        ValueError: If the file type is unsupported.
        RuntimeError: If extraction fails.
    """
    ext = Path(filename).suffix.lower().lstrip(".")

    try:
        if ext == "pdf":
            logger.info("Extracting text from PDF: %s", filename)
            text = _extract_text_from_pdf(file_bytes)
        elif ext in {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}:
            logger.info("Extracting text from image: %s", filename)
            text = _ocr_image_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        if not text.strip():
            raise RuntimeError("OCR produced no text. The file may be blank or corrupted.")

        logger.info("Extraction complete: %d characters extracted.", len(text))
        return text.strip()

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        logger.exception("OCR extraction failed for %s", filename)
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc

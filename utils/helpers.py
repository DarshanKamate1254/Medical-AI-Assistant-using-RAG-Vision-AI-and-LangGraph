"""
utils/helpers.py - Shared utility functions.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── Known city coordinates (extend as needed) ────────────────────────────────
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "toronto": (43.6532, -79.3832),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "tokyo": (35.6762, 139.6503),
    "beijing": (39.9042, 116.4074),
    "dubai": (25.2048, 55.2708),
    "singapore": (1.3521, 103.8198),
    "cape town": (-33.9249, 18.4241),
    "cairo": (30.0444, 31.2357),
}


def geocode_city(city_name: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for a known city name, or None."""
    return CITY_COORDS.get(city_name.strip().lower())


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_parameter_table(parameters: list[dict]) -> list[dict]:
    """
    Clean and normalise extracted parameter list for display.

    Input items expected: {parameter, value, normal_range, status}
    """
    cleaned = []
    for p in parameters:
        cleaned.append({
            "Parameter": str(p.get("parameter", "–")),
            "Value": str(p.get("value", "–")),
            "Normal Range": str(p.get("normal_range", "–")),
            "Status": str(p.get("status", "–")),
        })
    return cleaned


def severity_badge(severity: str) -> str:
    """Return an emoji badge for a severity string."""
    badges = {
        "Mild": "🟢 Mild",
        "Moderate": "🟡 Moderate",
        "Emergency": "🔴 Emergency",
    }
    return badges.get(severity, f"⚪ {severity}")


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate long strings for display."""
    return text if len(text) <= max_len else text[:max_len] + "…"

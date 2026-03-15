"""
hospital.py - Hospital recommendation and Google Maps navigation link generation.

Note: This module uses a curated sample dataset to demonstrate functionality.
For production, integrate with Google Places API or a hospitals directory API.
"""
from __future__ import annotations

import math
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Hospital:
    name: str
    address: str
    city: str
    country: str
    latitude: float
    longitude: float
    specializations: list[str]
    phone: str = ""
    website: str = ""
    rating: float = 0.0
    doctor_name: str = ""
    doctor_qualification: str = ""
    distance_km: float = field(default=0.0, compare=False)

    @property
    def maps_link(self) -> str:
        encoded = urllib.parse.quote_plus(f"{self.name}, {self.address}, {self.city}")
        return f"https://www.google.com/maps/dir/?api=1&destination={encoded}"

    @property
    def full_address(self) -> str:
        return f"{self.address}, {self.city}, {self.country}"


# ─────────────────────────────────────────────────────────────────────────────
# Sample hospital database (extend with real API in production)
# ─────────────────────────────────────────────────────────────────────────────

HOSPITAL_DB: list[Hospital] = [
    # ── USA ──────────────────────────────────────────────────────────────
    Hospital("Mayo Clinic", "200 First St SW", "Rochester, MN", "USA", 44.0225, -92.4669,
             ["Cardiology", "Endocrinology", "Nephrology", "Hematology", "Neurology", "General Medicine"],
             "+1-507-284-2511", "https://www.mayoclinic.org", 4.9,
             "Dr. John Smith MD", "MD, PhD – Internal Medicine, Cardiology"),
    Hospital("Johns Hopkins Hospital", "1800 Orleans St", "Baltimore, MD", "USA", 39.2963, -76.5927,
             ["Oncology", "Cardiology", "Neurology", "Pulmonology", "General Surgery"],
             "+1-410-955-5000", "https://www.hopkinsmedicine.org", 4.8,
             "Dr. Emily Chen MD", "MD, FRCS – Cardiology & Vascular Surgery"),
    Hospital("Cleveland Clinic", "9500 Euclid Ave", "Cleveland, OH", "USA", 41.5020, -81.6209,
             ["Cardiology", "Gastroenterology", "Endocrinology", "Orthopedics"],
             "+1-800-223-2273", "https://www.clevelandclinic.org", 4.8,
             "Dr. Robert Lee MD", "MD, MS – Cardiology"),
    Hospital("Massachusetts General Hospital", "55 Fruit St", "Boston, MA", "USA", 42.3631, -71.0686,
             ["Neurology", "Oncology", "Pulmonology", "Hematology", "Radiology"],
             "+1-617-726-2000", "https://www.massgeneral.org", 4.7,
             "Dr. Sarah Williams MD", "MD, PhD – Neurology & Radiology"),

    # ── UK ───────────────────────────────────────────────────────────────
    Hospital("St Thomas' Hospital", "Westminster Bridge Rd", "London", "UK", 51.4985, -0.1190,
             ["Cardiology", "Nephrology", "General Medicine", "Hematology"],
             "+44-20-7188-7188", "https://www.guysandstthomas.nhs.uk", 4.6,
             "Dr. James Thompson MBBS", "MBBS, MRCP – Cardiology"),
    Hospital("Royal Free Hospital", "Pond St", "London", "UK", 51.5533, -0.1659,
             ["Hepatology", "Gastroenterology", "Nephrology", "General Surgery"],
             "+44-20-7794-0500", "https://www.royalfree.nhs.uk", 4.5,
             "Dr. Priya Patel MBBS", "MBBS, MD – Hepatology & Gastroenterology"),

    # ── India ─────────────────────────────────────────────────────────────
    Hospital("AIIMS New Delhi", "Ansari Nagar", "New Delhi", "India", 28.5672, 77.2100,
             ["Cardiology", "Endocrinology", "Neurology", "Oncology", "General Medicine"],
             "+91-11-26588500", "https://www.aiims.edu", 4.7,
             "Dr. Ramesh Kumar MD", "MD, DM – Endocrinology & Internal Medicine"),
    Hospital("Apollo Hospital", "21 Greams Lane", "Chennai", "India", 13.0604, 80.2496,
             ["Cardiology", "Orthopedics", "Nephrology", "Gastroenterology"],
             "+91-44-28293333", "https://www.apollohospitals.com", 4.6,
             "Dr. Anitha Raj MD", "MD, DNB – Nephrology"),
    Hospital("Fortis Memorial Research Institute", "Sector 44", "Gurugram", "India", 28.4595, 77.0266,
             ["Cardiology", "Oncology", "Neurology", "Pulmonology"],
             "+91-124-4921021", "https://www.fortishealthcare.com", 4.5,
             "Dr. Vikram Singh MD", "MD, DM – Cardiology & Interventional Cardiology"),

    # ── Australia ─────────────────────────────────────────────────────────
    Hospital("Royal Melbourne Hospital", "300 Grattan St", "Melbourne, VIC", "Australia", -37.7993, 144.9554,
             ["Cardiology", "Neurology", "Hematology", "General Medicine"],
             "+61-3-9342-7000", "https://www.thermh.org.au", 4.6,
             "Dr. Laura White MBBS", "MBBS, FRACP – Cardiology"),
    Hospital("Royal Prince Alfred Hospital", "Missenden Rd", "Camperdown, NSW", "Australia", -33.8895, 151.1863,
             ["Gastroenterology", "Hepatology", "Nephrology", "Pulmonology"],
             "+61-2-9515-6111", "https://www.slhd.nsw.gov.au", 4.5,
             "Dr. Michael Brown MBBS", "MBBS, FRACP – Gastroenterology"),

    # ── Canada ────────────────────────────────────────────────────────────
    Hospital("Toronto General Hospital", "200 Elizabeth St", "Toronto, ON", "Canada", 43.6596, -79.3878,
             ["Cardiology", "Transplant Medicine", "Nephrology", "General Surgery"],
             "+1-416-340-4800", "https://www.uhn.ca", 4.7,
             "Dr. Angela Moore MD", "MD, FRCPC – Cardiology & Transplant Medicine"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommend_hospitals(
    specialization: str,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    top_n: int = 5,
) -> list[Hospital]:
    """
    Return top-N hospitals that match *specialization*, sorted by distance
    (if coordinates provided) or by rating.

    Args:
        specialization: Doctor specialization string (e.g. "Cardiologist").
        user_lat:       User latitude for distance sorting.
        user_lon:       User longitude for distance sorting.
        top_n:          Number of results to return.

    Returns:
        List of Hospital dataclass instances.
    """
    # Normalise: "Cardiologist" → "Cardiology"
    spec_key = (
        specialization.replace("ist", "y")
        .replace("ologist", "ology")
        .replace("Surgeon", "Surgery")
        .strip()
    )
    if "General" in specialization:
        spec_key = "General Medicine"

    logger.info("Searching hospitals for specialization: '%s' (key='%s')", specialization, spec_key)

    matched: list[Hospital] = []
    for hospital in HOSPITAL_DB:
        if any(spec_key.lower() in s.lower() or s.lower() in spec_key.lower()
               for s in hospital.specializations):
            h = Hospital(**hospital.__dict__)  # shallow copy
            if user_lat is not None and user_lon is not None:
                h.distance_km = round(_haversine(user_lat, user_lon, h.latitude, h.longitude), 1)
            matched.append(h)

    if not matched:
        # Return all hospitals sorted by rating as fallback
        matched = list(HOSPITAL_DB)
        for h in matched:
            if user_lat is not None and user_lon is not None:
                h.distance_km = round(_haversine(user_lat, user_lon, h.latitude, h.longitude), 1)

    if user_lat is not None and user_lon is not None:
        matched.sort(key=lambda h: h.distance_km)
    else:
        matched.sort(key=lambda h: h.rating, reverse=True)

    logger.info("Found %d matching hospitals; returning top %d.", len(matched), top_n)
    return matched[:top_n]


def generate_maps_link(hospital: Hospital) -> str:
    """Return a Google Maps directions URL for the given hospital."""
    return hospital.maps_link

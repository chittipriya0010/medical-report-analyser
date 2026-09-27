"""
Application Configuration and Constants
Handles environment variable loading, API settings, and medical reference ranges.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment files: check both .env and .env.local
env_file = BASE_DIR / ".env"
env_local_file = BASE_DIR / ".env.local"

if env_file.exists():
    load_dotenv(dotenv_path=env_file)
if env_local_file.exists():
    load_dotenv(dotenv_path=env_local_file, override=True)


def get_gemini_api_key() -> str:
    """
    Retrieve Gemini API key from environment (.env, .env.local) or Streamlit Cloud secrets.
    Supports both GEMINI_API_KEY and GOOGLE_API_KEY naming conventions.
    """
    key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if "GEMINI_API_KEY" in st.secrets:
                    key = str(st.secrets["GEMINI_API_KEY"])
                elif "GOOGLE_API_KEY" in st.secrets:
                    key = str(st.secrets["GOOGLE_API_KEY"])
                elif "general" in st.secrets:
                    key = str(st.secrets["general"].get("GEMINI_API_KEY", st.secrets["general"].get("GOOGLE_API_KEY", "")))
        except Exception:
            pass
    return key.strip()


# Supported Modern Gemini Models (Gemini 3.x)
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

DEFAULT_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.5-flash-lite")

# Standard Clinical Reference Ranges
CLINICAL_REFERENCE_RANGES: Dict[str, Any] = {
    # Complete Blood Count (CBC)
    "hemoglobin": {
        "unit": "g/dL",
        "male": (13.0, 17.5),
        "female": (12.0, 15.5),
        "critical_low": 7.0,
        "critical_high": 20.0
    },
    "wbc": {
        "unit": "x10^3/uL",
        "normal": (4.5, 11.0),
        "critical_low": 2.0,
        "critical_high": 30.0
    },
    "platelets": {
        "unit": "x10^3/uL",
        "normal": (150, 450),
        "critical_low": 50,
        "critical_high": 1000
    },
    "rbc": {
        "unit": "x10^6/uL",
        "male": (4.5, 5.9),
        "female": (4.1, 5.1)
    },
    
    # Diabetes & Glucose Metabolism
    "glucose_fasting": {
        "unit": "mg/dL",
        "normal": (70, 99),
        "prediabetes": (100, 125),
        "diabetes": 126,
        "critical_low": 50,
        "critical_high": 400
    },
    "glucose_random": {
        "unit": "mg/dL",
        "normal": (70, 140),
        "critical_high": 300
    },
    "hba1c": {
        "unit": "%",
        "normal": (4.0, 5.6),
        "prediabetes": (5.7, 6.4),
        "diabetes": 6.5,
        "critical_high": 10.0
    },
    
    # Lipid Panel (Cardiovascular Health)
    "cholesterol_total": {
        "unit": "mg/dL",
        "desirable": 200,
        "borderline": (200, 239),
        "high": 240
    },
    "ldl_cholesterol": {
        "unit": "mg/dL",
        "optimal": 100,
        "near_optimal": (100, 129),
        "borderline": (130, 159),
        "high": 160
    },
    "hdl_cholesterol": {
        "unit": "mg/dL",
        "male_optimal": 40,
        "female_optimal": 50
    },
    "triglycerides": {
        "unit": "mg/dL",
        "normal": 150,
        "borderline": (150, 199),
        "high": 200,
        "very_high": 500
    },
    
    # Renal / Kidney Function
    "creatinine": {
        "unit": "mg/dL",
        "male": (0.74, 1.35),
        "female": (0.59, 1.04),
        "critical_high": 4.0
    },
    "bun": {
        "unit": "mg/dL",
        "normal": (7, 20),
        "critical_high": 60
    },
    "uric_acid": {
        "unit": "mg/dL",
        "male": (3.4, 7.0),
        "female": (2.4, 6.0)
    },
    
    # Liver Function
    "alt_sgpt": {
        "unit": "U/L",
        "normal": (7, 56)
    },
    "ast_sgot": {
        "unit": "U/L",
        "normal": (10, 40)
    },
    "bilirubin_total": {
        "unit": "mg/dL",
        "normal": (0.2, 1.2)
    },
    
    # Thyroid Profile
    "tsh": {
        "unit": "uIU/mL",
        "normal": (0.4, 4.5),
        "critical_low": 0.1,
        "critical_high": 10.0
    },
    
    # Micronutrients & Vitamins
    "vitamin_d": {
        "unit": "ng/mL",
        "deficiency": 20,
        "insufficiency": (20, 30),
        "sufficiency": (30, 100)
    },
    "vitamin_b12": {
        "unit": "pg/mL",
        "normal": (200, 900),
        "deficiency": 200
    }
}

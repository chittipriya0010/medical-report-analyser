"""
Utility functions for text processing, JSON cleaning, parsing, and logging.
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    return logging.getLogger(name)


logger = get_logger(__name__)


def clean_json_response(response_text: str) -> str:
    """
    Cleans raw LLM response to extract valid JSON substring.
    Handles ```json blocks, trailing commas, unescaped characters.
    """
    if not response_text:
        return "{}"

    text = response_text.strip()

    # Extract content inside markdown ```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()

    # Find curly braces if not cleanly wrapped
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group()

    # Remove invalid trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text.strip()


def extract_numeric_value(val_str: Any) -> Optional[float]:
    """
    Safely extract numeric float from strings like '14.2 g/dL', '<0.5', '>100'.
    """
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
    
    s = str(val_str).strip()
    # Find decimal or integer numbers
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def safe_str(val: Any, default: str = "Not specified") -> str:
    """Convert value safely to string."""
    if val is None or str(val).strip().lower() in ["", "none", "null", "not specified"]:
        return default
    return str(val).strip()


def parse_fallback_text(text: str) -> Dict[str, Any]:
    """
    Emergency rule-based text parser if AI service is completely unavailable.
    """
    lines = text.split("\n")
    tests: List[Dict[str, Any]] = []
    patient_name = "Not specified"
    patient_age = "Not specified"
    patient_gender = "Not specified"

    for line in lines[:25]:
        lower = line.lower()
        if any(title in lower for title in ["mr.", "mrs.", "ms.", "patient:", "name:"]):
            m = re.search(r"(?:name[:\s]+|(?:mr\.|mrs\.|ms\.)\s*)([a-zA-Z\s]{3,30})", line, re.I)
            if m:
                patient_name = m.group(1).strip()
        if "year" in lower or "age" in lower:
            m = re.search(r"(\d{1,3})\s*(?:years?|yrs?|y/o)", line, re.I)
            if m:
                patient_age = f"{m.group(1)} years"
        if "male" in lower or "female" in lower:
            m = re.search(r"\b(male|female)\b", line, re.I)
            if m:
                patient_gender = m.group(1).capitalize()

    # Pattern: Test Name: value unit (ref range)
    test_pattern = re.compile(
        r"([A-Za-z0-9\s\-_/()]+?)\s*[:=\t]\s*([0-9.]+)\s*([A-Za-z/%^0-9]+)?\s*(?:\(?([0-9.\s\-–to]+)\)?)?"
    )

    for line in lines:
        match = test_pattern.search(line)
        if match:
            tname, tval, tunit, trange = match.groups()
            tname = tname.strip()
            if len(tname) > 2 and not tname.lower().startswith("page"):
                tests.append({
                    "test_name": tname,
                    "value": tval or "",
                    "unit": (tunit or "").strip(),
                    "reference_range": (trange or "Not specified").strip(),
                    "status": "unknown"
                })

    return {
        "patient_info": {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "report_date": "Not specified",
            "lab_number": "Not specified"
        },
        "test_categories": [{
            "category": "Extracted Lab Tests",
            "tests": tests[:20]
        }] if tests else [],
        "abnormal_findings": [],
        "critical_values": []
    }

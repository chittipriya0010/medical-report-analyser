"""
Gemini Multimodal Client
Handles direct visual and textual medical document analysis using modern Google GenAI SDK.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Check for modern google.genai SDK
try:
    from google import genai
    from google.genai import types
    GENAI_V2_AVAILABLE = True
except ImportError:
    GENAI_V2_AVAILABLE = False

# Fallback to google.generativeai if needed
try:
    import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    legacy_genai = None
    LEGACY_GENAI_AVAILABLE = False

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL
from src.models.schemas import MedicalReportData, DiagnosisInsights
from src.core.extractor import ExtractedDocument
from src.utils.helpers import clean_json_response, parse_fallback_text, get_logger

logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """
You are an expert Clinical Diagnostic Assistant and Medical Lab Technologist.
Analyze the provided medical report document (image/pages or text) with high precision.

Extract all test results, patient demographics, clinical flags, reference intervals, and potential health insights.

Return ONLY a valid JSON object matching exactly this structure:
{
    "patient_info": {
        "name": "Patient full name or 'Not specified'",
        "age": "Age in years (e.g., '45 years') or 'Not specified'",
        "gender": "Male / Female / Other or 'Not specified'",
        "report_date": "Date of collection or reporting (e.g. '2026-03-15') or 'Not specified'",
        "lab_number": "Lab ID / Barcode / Sample ID or 'Not specified'",
        "referring_doctor": "Doctor name or 'Not specified'"
    },
    "test_categories": [
        {
            "category": "Category name (e.g. 'Complete Blood Count', 'Lipid Panel', 'Liver Function', 'Thyroid Profile', 'Renal Function')",
            "tests": [
                {
                    "test_name": "Exact standard name of test (e.g. 'Hemoglobin', 'HbA1c', 'Total Cholesterol')",
                    "value": "Measured numerical or qualitative value (e.g. '14.2', 'Negative')",
                    "unit": "Unit of measurement (e.g. 'g/dL', 'mg/dL', 'uIU/mL') or empty",
                    "reference_range": "Normal range provided by lab (e.g. '13.0 - 17.0') or 'Not specified'",
                    "status": "normal / high / low / borderline / critical / unspecified",
                    "status_source": "report / unspecified"
                }
            ]
        }
    ],
    "abnormal_findings": [
        "Concise list of all tests outside the reference interval with their abnormal values"
    ],
    "critical_values": [
        "Any life-threatening or severely critical values requiring immediate medical attention"
    ],
    "diagnosis": {
        "risk_assessment": {
            "overall_risk": "low / moderate / high",
            "cardiovascular_risk": "low / moderate / high",
            "diabetes_risk": "low / moderate / high",
            "metabolic_risk": "low / moderate / high",
            "risk_factors": ["Key risk factors identified from the findings"]
        },
        "potential_conditions": [
            {
                "condition": "Medical condition name to discuss with doctor (e.g. 'Iron Deficiency Anemia', 'Prediabetes')",
                "probability": "low / moderate / high",
                "supporting_evidence": ["Specific test results supporting this observation"],
                "description": "Short clinical rationale"
            }
        ],
        "recommendations": [
            {
                "category": "lifestyle / dietary / medical / follow-up",
                "recommendation": "Clear, actionable recommendation",
                "priority": "low / medium / high",
                "rationale": "Why this is recommended"
            }
        ],
        "follow_up_tests": [
            "Recommended confirmatory or monitoring lab tests"
        ],
        "red_flags": [
            "Critical warning signs requiring immediate consultation"
        ],
        "positive_findings": [
            "Normal, healthy test results that are reassuring"
        ],
        "summary": "Clear, compassionate 2-3 paragraph summary of the overall lab results, explaining what looks healthy and what needs attention."
    }
}

Guidelines:
1. Preserve numbers, decimals, and units exactly as printed on the lab report.
2. If values are flagged with H (High), L (Low), *, or bold, assign the corresponding status ('high', 'low', 'critical').
3. If a section is unclear or absent, set fields to 'Not specified' or empty list.
4. Output valid, parseable JSON with NO markdown commentary.
"""


class GeminiReportAnalyzer:
    """Orchestrates Gemini AI inference for medical document extraction and diagnostic insights."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.client_v2 = None

        if not self.api_key:
            raise ValueError("Gemini API Key is missing. Please provide a valid API key.")

        if GENAI_V2_AVAILABLE:
            try:
                self.client_v2 = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize google.genai Client: {e}")

        if LEGACY_GENAI_AVAILABLE:
            try:
                legacy_genai.configure(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize legacy google.generativeai: {e}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test API connectivity and key validity."""
        try:
            if self.client_v2:
                resp = self.client_v2.models.generate_content(
                    model=self.model_name,
                    contents="Ping"
                )
                if resp and resp.text:
                    return True, f"Connected to {self.model_name} successfully."

            if LEGACY_GENAI_AVAILABLE:
                model = legacy_genai.GenerativeModel(self.model_name)
                resp = model.generate_content("Ping")
                if resp and resp.text:
                    return True, f"Connected to {self.model_name} successfully."

            return False, "Received empty response from Gemini API."
        except Exception as e:
            return False, str(e)

    def analyze_document(self, doc: ExtractedDocument) -> MedicalReportData:
        """
        Analyze medical document.
        Uses Multimodal Vision if page images exist, or text mode if only text is present.
        """
        # Candidate modern models in priority order
        candidate_models = [self.model_name]
        for m in AVAILABLE_MODELS:
            if m not in candidate_models:
                candidate_models.append(m)

        last_error = None

        for model_to_try in candidate_models:
            # Skip any deprecated 2.0 or 1.5 or 2.5 legacy names that trigger 404
            if any(legacy in model_to_try for legacy in ["gemini-2.0", "gemini-1.5", "gemini-2.5"]):
                continue

            try:
                logger.info(f"Attempting analysis with model: {model_to_try}")

                # Prepare multimodal content parts
                content_parts: List[Any] = [EXTRACTION_SYSTEM_PROMPT]

                if doc.page_images:
                    # Multimodal vision mode: pass page images directly
                    for idx, img in enumerate(doc.page_images):
                        content_parts.append(f"Document Page {idx + 1}:")
                        content_parts.append(img)

                    if doc.raw_text:
                        content_parts.append(f"Optional OCR/Digital Text:\n{doc.raw_text[:2000]}")

                    extraction_mode = "multimodal_vision"
                else:
                    if not doc.raw_text or len(doc.raw_text.strip()) < 10:
                        raise ValueError("No visual pages or text found in document.")
                    content_parts.append(f"Medical Report Text:\n{doc.raw_text}")
                    extraction_mode = "digital_text"

                # Generate content using V2 Client first
                raw_response_text = None
                if self.client_v2:
                    try:
                        resp = self.client_v2.models.generate_content(
                            model=model_to_try,
                            contents=content_parts
                        )
                        raw_response_text = resp.text
                    except Exception as err_v2:
                        logger.warning(f"V2 Client call failed on {model_to_try}: {err_v2}")
                        last_error = err_v2

                # If V2 didn't produce text, fallback to legacy genai
                if not raw_response_text and LEGACY_GENAI_AVAILABLE:
                    try:
                        legacy_model = legacy_genai.GenerativeModel(model_to_try)
                        resp = legacy_model.generate_content(content_parts)
                        raw_response_text = resp.text
                    except Exception as err_legacy:
                        logger.warning(f"Legacy Client call failed on {model_to_try}: {err_legacy}")
                        last_error = err_legacy

                if not raw_response_text:
                    continue

                cleaned_json = clean_json_response(raw_response_text)
                parsed_dict = json.loads(cleaned_json)

                # Set metadata
                parsed_dict["extraction_mode"] = extraction_mode
                parsed_dict["raw_text"] = doc.raw_text or ""
                parsed_dict["page_count"] = doc.page_count

                report_data = MedicalReportData.model_validate(parsed_dict)
                logger.info(f"Successfully parsed report with {len(report_data.test_categories)} categories using {model_to_try}.")
                return report_data

            except Exception as e:
                logger.warning(f"Model {model_to_try} failed: {e}")
                last_error = e
                continue

        # If all AI models failed, use emergency fallback if raw_text exists
        logger.error(f"All Gemini models failed. Last error: {last_error}")
        if doc.raw_text and len(doc.raw_text.strip()) > 30:
            logger.info("Using emergency rule-based fallback parser.")
            fallback_dict = parse_fallback_text(doc.raw_text)
            fallback_dict["extraction_mode"] = "rule_based_fallback"
            fallback_dict["page_count"] = doc.page_count
            fallback_dict["raw_text"] = doc.raw_text
            return MedicalReportData.model_validate(fallback_dict)

        raise RuntimeError(f"Medical analysis failed: {last_error}")

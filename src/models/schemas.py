"""
Data structures and schemas for medical report analysis.
Provides structured types with fallback defaults.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    name: str = Field(default="Not specified")
    age: str = Field(default="Not specified")
    gender: str = Field(default="Not specified")
    report_date: str = Field(default="Not specified")
    lab_number: str = Field(default="Not specified")
    referring_doctor: str = Field(default="Not specified")


class TestItem(BaseModel):
    test_name: str
    value: str
    unit: str = ""
    reference_range: str = "Not specified"
    status: str = "unknown"  # normal, high, low, borderline, critical, unknown
    status_source: str = "report"  # 'report' (mentioned in report) or 'ai_assumption' (derived from reference interval)
    is_ai_assumption: bool = False
    clinical_significance: Optional[str] = None


class TestCategory(BaseModel):
    category: str
    tests: List[TestItem] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    overall_risk: str = "moderate"  # low, moderate, high
    cardiovascular_risk: str = "moderate"
    diabetes_risk: str = "moderate"
    metabolic_risk: str = "moderate"
    risk_factors: List[str] = Field(default_factory=list)


class PotentialCondition(BaseModel):
    condition: str
    probability: str = "moderate"  # low, moderate, high
    supporting_evidence: List[str] = Field(default_factory=list)
    description: str = ""


class Recommendation(BaseModel):
    category: str = "general"  # lifestyle, dietary, medical, follow-up
    recommendation: str
    priority: str = "medium"  # low, medium, high
    rationale: str = ""


class DiagnosisInsights(BaseModel):
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    potential_conditions: List[PotentialCondition] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    follow_up_tests: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    positive_findings: List[str] = Field(default_factory=list)
    summary: str = "No automated summary available."


class MedicalReportData(BaseModel):
    patient_info: PatientInfo = Field(default_factory=PatientInfo)
    test_categories: List[TestCategory] = Field(default_factory=list)
    abnormal_findings: List[str] = Field(default_factory=list)
    critical_values: List[str] = Field(default_factory=list)
    diagnosis: Optional[DiagnosisInsights] = None
    extraction_mode: str = "text"  # 'multimodal_pdf', 'multimodal_image', 'digital_text'
    raw_text: Optional[str] = ""
    page_count: int = 1

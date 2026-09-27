"""
Clinical Analyzer
Performs post-extraction medical validation, rule-based range checking, and report generation.
Distinguishes between report-specified statuses and AI-calculated assumptions from reference intervals.
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from src.config import CLINICAL_REFERENCE_RANGES
from src.models.schemas import MedicalReportData, TestItem, TestCategory
from src.utils.helpers import extract_numeric_value, get_logger

logger = get_logger(__name__)


def parse_range_bounds(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse lower and upper numerical bounds from arbitrary reference range strings.
    Examples:
        '13.0 - 17.5' -> (13.0, 17.5)
        '< 200' -> (None, 200.0)
        '> 40' -> (40.0, None)
        '70.0 - 100.0 mg/dL' -> (70.0, 100.0)
        'Up to 150' -> (None, 150.0)
        '4,000 - 11,000' -> (4000.0, 11000.0)
    """
    if not range_str or range_str.lower().strip() in ["not specified", "unknown", "n/a", "-", "nil", ""]:
        return None, None

    # Remove commas
    clean = range_str.replace(",", "").strip()

    # Pattern 1: "< 200", "<= 150", "less than 200", "up to 150"
    m_less = re.search(r'(?:<|<=|less than|up to|below)\s*([\d.]+)', clean, re.I)
    if m_less:
        try:
            return None, float(m_less.group(1))
        except ValueError:
            pass

    # Pattern 2: "> 40", ">= 50", "greater than 40", "more than 50"
    m_greater = re.search(r'(?:>|>=|greater than|more than|above)\s*([\d.]+)', clean, re.I)
    if m_greater:
        try:
            return float(m_greater.group(1)), None
        except ValueError:
            pass

    # Pattern 3: Range "X - Y", "X to Y", "X – Y"
    m_range = re.search(r'([\d.]+)\s*(?:-|–|—|to)\s*([\d.]+)', clean)
    if m_range:
        try:
            low_val = float(m_range.group(1))
            high_val = float(m_range.group(2))
            return min(low_val, high_val), max(low_val, high_val)
        except ValueError:
            pass

    return None, None


class ClinicalAnalyzer:
    """Clinical logic processor for medical lab metrics and clinical validation."""

    def __init__(self, reference_ranges: Dict[str, Any] = CLINICAL_REFERENCE_RANGES):
        self.reference_ranges = reference_ranges

    def enrich_report_data(self, report_data: MedicalReportData) -> MedicalReportData:
        """
        Validates extracted values against reported reference intervals and standard ranges.
        Distinguishes between statuses explicitly stated in the report vs AI-calculated assumptions.
        """
        for cat in report_data.test_categories:
            for test in cat.tests:
                self._enrich_single_test(test, report_data.patient_info.gender)

        # Recalculate abnormal findings
        abnormals = []
        for cat in report_data.test_categories:
            for test in cat.tests:
                if test.status.lower() in ["high", "low", "abnormal", "critical"]:
                    is_ai = getattr(test, "is_ai_assumption", False) or getattr(test, "status_source", "") == "ai_assumption"
                    source_tag = "[Report Mention]" if not is_ai else "[AI Calculated]"
                    abnormals.append(f"{test.test_name}: {test.value} {test.unit} (Ref: {test.reference_range}) {source_tag}")
        report_data.abnormal_findings = abnormals

        return report_data

    def _enrich_single_test(self, test: TestItem, gender: str = ""):
        """
        Determine if status is already provided in report or must be calculated by AI.
        If calculated by AI, explicitly mark it as an AI assumption derived from the reference range.
        """
        orig_status = (test.status or "").lower().strip()
        val = extract_numeric_value(test.value)

        # Check if the report explicitly stamped a status
        has_explicit_report_status = (
            test.status_source == "report" and
            orig_status not in ["", "unknown", "not_specified", "unspecified", "none"]
        )

        if has_explicit_report_status:
            test.status_source = "report"
            test.is_ai_assumption = False
            if not test.clinical_significance:
                test.clinical_significance = f"Status explicitly stated as '{test.status.upper()}' in laboratory report"
            return

        # If not specified in the report, evaluate using the report's reference range
        min_b, max_b = parse_range_bounds(test.reference_range)

        if val is not None and (min_b is not None or max_b is not None):
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True

            if min_b is not None and val < min_b:
                test.status = "low"
                test.clinical_significance = f"AI Assumption: Observed value ({val}) is below reference lower limit ({min_b})"
            elif max_b is not None and val > max_b:
                test.status = "high"
                test.clinical_significance = f"AI Assumption: Observed value ({val}) exceeds reference upper limit ({max_b})"
            else:
                test.status = "normal"
                test.clinical_significance = f"AI Assumption: Observed value ({val}) is within reference range ({test.reference_range})"
            return

        # If report reference range is absent or unparseable, compare against standard clinical guidelines
        self._apply_standard_clinical_knowledge(test, gender, val)

    def _apply_standard_clinical_knowledge(self, test: TestItem, gender: str, val: Optional[float]):
        """Fallback check against standard medical guidelines if report had no reference range."""
        if val is None:
            return

        t_name = test.test_name.lower().strip()
        is_female = "female" in gender.lower()

        # HbA1c
        if "hba1c" in t_name or "glycated" in t_name or "glycosylated" in t_name:
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            if val >= 6.5:
                test.status = "high"
                test.clinical_significance = "AI Assumption: Diabetic range (>= 6.5%)"
            elif val >= 5.7:
                test.status = "borderline"
                test.clinical_significance = "AI Assumption: Prediabetes range (5.7 - 6.4%)"
            elif val < 4.0:
                test.status = "low"
                test.clinical_significance = "AI Assumption: Below normal range (< 4.0%)"
            else:
                test.status = "normal"
                test.clinical_significance = "AI Assumption: Within normal glycemic range (4.0 - 5.6%)"

        # Fasting Glucose
        elif "glucose" in t_name and ("fast" in t_name or "fbs" in t_name):
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            if val >= 126:
                test.status = "high"
                test.clinical_significance = "AI Assumption: Elevated fasting blood sugar (>= 126 mg/dL)"
            elif val >= 100:
                test.status = "borderline"
                test.clinical_significance = "AI Assumption: Impaired fasting glucose (100 - 125 mg/dL)"
            elif val < 70:
                test.status = "low"
                test.clinical_significance = "AI Assumption: Hypoglycemia risk (< 70 mg/dL)"
            else:
                test.status = "normal"
                test.clinical_significance = "AI Assumption: Normal fasting glucose (70 - 99 mg/dL)"

        # Total Cholesterol
        elif "cholesterol" in t_name and "total" in t_name:
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            if val >= 240:
                test.status = "high"
                test.clinical_significance = "AI Assumption: Elevated cardiovascular risk (>= 240 mg/dL)"
            elif val >= 200:
                test.status = "borderline"
                test.clinical_significance = "AI Assumption: Borderline elevated (200 - 239 mg/dL)"
            else:
                test.status = "normal"
                test.clinical_significance = "AI Assumption: Desirable level (< 200 mg/dL)"

        # LDL Cholesterol
        elif "ldl" in t_name:
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            if val >= 160:
                test.status = "high"
                test.clinical_significance = "AI Assumption: High atherogenic risk (>= 160 mg/dL)"
            elif val >= 130:
                test.status = "borderline"
                test.clinical_significance = "AI Assumption: Borderline high (130 - 159 mg/dL)"
            else:
                test.status = "normal"
                test.clinical_significance = "AI Assumption: Optimal level (< 100-129 mg/dL)"

        # Triglycerides
        elif "triglyceride" in t_name:
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            if val >= 500:
                test.status = "critical"
                test.clinical_significance = "AI Assumption: Severe hypertriglyceridemia (>= 500 mg/dL)"
            elif val >= 200:
                test.status = "high"
                test.clinical_significance = "AI Assumption: High (200 - 499 mg/dL)"
            elif val >= 150:
                test.status = "borderline"
                test.clinical_significance = "AI Assumption: Borderline high (150 - 199 mg/dL)"
            else:
                test.status = "normal"
                test.clinical_significance = "AI Assumption: Normal (< 150 mg/dL)"

        # Hemoglobin
        elif "hemoglobin" in t_name or "hb" == t_name or "hgb" in t_name:
            test.status_source = "ai_assumption"
            test.is_ai_assumption = True
            low_thresh = 12.0 if is_female else 13.0
            high_thresh = 15.5 if is_female else 17.5
            if val < 7.0:
                test.status = "critical"
                test.clinical_significance = "AI Assumption: Critical Anemia (< 7.0 g/dL)"
            elif val < low_thresh:
                test.status = "low"
                test.clinical_significance = f"AI Assumption: Anemia indicator (< {low_thresh} g/dL)"
            elif val > high_thresh:
                test.status = "high"
                test.clinical_significance = f"AI Assumption: Polycythemia indicator (> {high_thresh} g/dL)"
            else:
                test.status = "normal"
                test.clinical_significance = f"AI Assumption: Normal ({low_thresh} - {high_thresh} g/dL)"

    def get_summary_statistics(self, report_data: MedicalReportData) -> Dict[str, int]:
        """Calculate counts of normal, abnormal, borderline, and total tests."""
        total = 0
        normal = 0
        abnormal = 0
        borderline = 0
        critical = 0

        for cat in report_data.test_categories:
            for test in cat.tests:
                total += 1
                st = test.status.lower()
                if st == "normal":
                    normal += 1
                elif st in ["high", "low", "abnormal"]:
                    abnormal += 1
                elif st == "borderline":
                    borderline += 1
                elif st == "critical":
                    critical += 1

        return {
            "total": total,
            "normal": normal,
            "abnormal": abnormal,
            "borderline": borderline,
            "critical": critical,
            "categories": len(report_data.test_categories)
        }

    def generate_markdown_report(self, report_data: MedicalReportData) -> str:
        """Create a beautifully formatted professional clinical report export."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p = report_data.patient_info

        lines = [
            "# CLINICAL LABORATORY DIAGNOSTIC EVALUATION REPORT",
            f"**Generated:** {timestamp}  ",
            f"**Document Ingestion:** {report_data.extraction_mode.replace('_', ' ').title()}  ",
            f"**Verified Pages:** {report_data.page_count}  ",
            "",
            "---",
            "",
            "## CRITICAL CLINICAL NOTICE & EXPERT CONSULTATION MANDATE",
            "> ⚠️ **IMPORTANT ADVISORY: AI-GENERATED CLINICAL ANALYSIS**",
            "> This evaluation is generated by an Artificial Intelligence system for diagnostic screening and decision support.",
            "> **MANDATORY REQUIREMENT:** Yeh analysis AI dwara ki gayi hai. Kisi bhi medication, treatment ya medical decision se pehle clinical confirmation ke liye yeh report kisi qualified medical doctor, pathologist ya specialist ko zaroor dikhaayein.",
            "> ",
            "> **Biomarker Status Attribution Legend:**",
            "> - **[Mentioned in Report]:** Explicitly stated on the original laboratory document.",
            "> - **[AI Assumption]:** Derived mathematically by comparing observed patient levels against reported reference intervals.",
            "",
            "---",
            "",
            "## PATIENT DEMOGRAPHICS",
            f"- **Patient Name:** {p.name}",
            f"- **Age:** {p.age}",
            f"- **Gender:** {p.gender}",
            f"- **Collection Date:** {p.report_date}",
            f"- **Accession / Lab ID:** {p.lab_number}",
            f"- **Ordering Physician:** {p.referring_doctor}",
            "",
            "---",
            "",
            "## LABORATORY BIOMARKER PANELS",
            ""
        ]

        status_labels = {
            "normal": "Normal",
            "high": "High",
            "low": "Low",
            "borderline": "Borderline",
            "critical": "Critical",
            "unknown": "Not Specified"
        }

        for cat in report_data.test_categories:
            lines.append(f"### {cat.category}")
            lines.append("| Biomarker | Measured Value | Reference Interval | Status | Attribution | Clinical Correlate |")
            lines.append("|:---|:---|:---|:---|:---|:---|")
            for t in cat.tests:
                status_text = status_labels.get(t.status.lower(), t.status.title())
                is_ai = getattr(t, "is_ai_assumption", False) or getattr(t, "status_source", "") == "ai_assumption"
                source_text = "AI Assumption" if is_ai else "Report Mention"
                signif = t.clinical_significance or "Within expected parameters"
                val_disp = f"{t.value} {t.unit}".strip()
                lines.append(f"| **{t.test_name}** | {val_disp} | {t.reference_range} | **{status_text}** | *{source_text}* | {signif} |")
            lines.append("")

        # Diagnosis section
        diag = report_data.diagnosis
        if diag:
            lines.append("---")
            lines.append("## CLINICAL RISK STRATIFICATION")
            lines.append(f"- **Overall Health Risk:** **{diag.risk_assessment.overall_risk.upper()}**")
            lines.append(f"- **Cardiovascular Risk Index:** {diag.risk_assessment.cardiovascular_risk.title()}")
            lines.append(f"- **Metabolic / Glycemic Risk Index:** {diag.risk_assessment.diabetes_risk.title()}")
            if diag.risk_assessment.risk_factors:
                lines.append("- **Identified Clinical Correlates:**")
                for rf in diag.risk_assessment.risk_factors:
                    lines.append(f"  - {rf}")
            lines.append("")

            if diag.red_flags:
                lines.append("## CRITICAL ALERTS REQUIRING IMMEDIATE ATTENTION")
                for rf in diag.red_flags:
                    lines.append(f"- **{rf}**")
                lines.append("")

            if diag.potential_conditions:
                lines.append("## DIFFERENTIAL CONSIDERATIONS FOR CLINICAL REVIEW")
                for pc in diag.potential_conditions:
                    lines.append(f"### • {pc.condition} (Probability: {pc.probability.title()})")
                    if pc.description:
                        lines.append(f"*{pc.description}*")
                    if pc.supporting_evidence:
                        lines.append(f"**Laboratory Evidence:** {', '.join(pc.supporting_evidence)}")
                    lines.append("")

            if diag.recommendations:
                lines.append("## EVIDENCE-BASED CLINICAL RECOMMENDATIONS")
                for rec in diag.recommendations:
                    lines.append(f"- **[{rec.priority.upper()}] {rec.category.title()}:** {rec.recommendation}")
                    if rec.rationale:
                        lines.append(f"  *Rationale: {rec.rationale}*")
                lines.append("")

            if diag.summary:
                lines.append("## CLINICAL NARRATIVE EXECUTIVE SUMMARY")
                lines.append(f"> {diag.summary}")
                lines.append("")

        return "\n".join(lines)

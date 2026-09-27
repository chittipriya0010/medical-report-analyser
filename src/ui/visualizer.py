"""
Clinical Data Visualizations and Tabular Views.
Provides clean, professional pathology tables, dark-theme Plotly charts, and export facilities.
Distinguishes between report-stamped statuses and AI-calculated assumptions.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.schemas import TestCategory, MedicalReportData, RiskAssessment
from src.utils.exporter import (
    generate_pdf_report,
    generate_docx_report,
    generate_structured_json,
    AI_EXPERT_DISCLAIMER,
    STATUS_LEGEND_NOTICE,
)


def render_test_category_table(category: TestCategory):
    """
    Render pathology panel table with clean clinical styling.
    Clearly marks whether status was 'Mentioned in Report' or 'AI Assumption (from Range)'.
    """
    if not category.tests:
        st.info("No verified biomarkers extracted for this panel.")
        return

    rows = []
    for t in category.tests:
        status_raw = t.status.lower()
        if status_raw == "normal":
            status_word = "Normal"
        elif status_raw in ["high", "abnormal"]:
            status_word = "High"
        elif status_raw == "low":
            status_word = "Low"
        elif status_raw == "borderline":
            status_word = "Borderline"
        elif status_raw == "critical":
            status_word = "Critical"
        else:
            status_word = t.status.title()

        # Distinguish Source Attribution safely
        is_ai = getattr(t, "is_ai_assumption", False) or getattr(t, "status_source", "") == "ai_assumption"
        if is_ai:
            status_display = f"{status_word} [AI Assumption]"
            source_attribution = "AI Assumption (Calculated from Range)"
        else:
            status_display = f"{status_word} [Mentioned in Report]"
            source_attribution = "Mentioned in Report"

        val_display = f"{t.value} {t.unit}".strip()

        rows.append({
            "Biomarker": t.test_name,
            "Observed Level": val_display,
            "Reference Interval": t.reference_range,
            "Evaluation Status": status_display,
            "Attribution Source": source_attribution,
            "Clinical Correlate": t.clinical_significance or "Within normal bounds"
        })

    df = pd.DataFrame(rows)

    def highlight_status(val):
        s = str(val).lower()
        if "normal" in s:
            return "background-color: rgba(34, 197, 94, 0.15); color: #4ade80; font-weight: 600;"
        elif "critical" in s or "high" in s:
            return "background-color: rgba(239, 68, 68, 0.18); color: #f87171; font-weight: 600;"
        elif "low" in s:
            return "background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; font-weight: 600;"
        elif "borderline" in s:
            return "background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; font-weight: 600;"
        return ""

    def highlight_source(val):
        if "AI Assumption" in str(val):
            return "color: #c084fc; font-style: italic; font-weight: 500;"
        elif "Mentioned in Report" in str(val):
            return "color: #93c5fd; font-weight: 500;"
        return ""

    styled_df = df.style.map(highlight_status, subset=["Evaluation Status"]).map(highlight_source, subset=["Attribution Source"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_test_distribution_donut(stats: Dict[str, int]) -> go.Figure:
    """Create a high-contrast dark donut chart showing biomarker distribution."""
    labels = ["Normal Range", "Elevated / Out of Range", "Borderline", "Critical"]
    values = [
        stats.get("normal", 0),
        stats.get("abnormal", 0),
        stats.get("borderline", 0),
        stats.get("critical", 0)
    ]
    colors = ["#22c55e", "#ef4444", "#f59e0b", "#991b1b"]

    active_labels = [l for l, v in zip(labels, values) if v > 0]
    active_values = [v for v in values if v > 0]
    active_colors = [c for c, v in zip(colors, values) if v > 0]

    if not active_values:
        active_labels = ["No Data"]
        active_values = [1]
        active_colors = ["#64748b"]

    fig = go.Figure(data=[go.Pie(
        labels=active_labels,
        values=active_values,
        hole=0.68,
        marker=dict(colors=active_colors, line=dict(color="#111827", width=2)),
        textinfo="label+value",
        textfont=dict(color="#f8fafc", family="Inter", size=11),
        hoverinfo="label+percent"
    )])

    fig.update_layout(
        title=dict(
            text="Biomarker Distribution",
            font=dict(size=14, color="#f8fafc", family="Inter")
        ),
        margin=dict(t=45, b=20, l=20, r=20),
        height=260,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#f8fafc")
    )
    return fig


def render_risk_gauge_chart(risk_assessment: RiskAssessment) -> go.Figure:
    """Create high-contrast horizontal clinical risk index chart with white legible text."""
    categories = ["Overall Health Index", "Cardiovascular Risk", "Metabolic / Glycemic Risk"]

    risk_map = {"low": 25, "moderate": 60, "high": 90}
    scores = [
        risk_map.get(risk_assessment.overall_risk.lower(), 50),
        risk_map.get(risk_assessment.cardiovascular_risk.lower(), 50),
        risk_map.get(risk_assessment.diabetes_risk.lower(), 50)
    ]

    colors = []
    for s in scores:
        if s <= 35:
            colors.append("#22c55e")
        elif s <= 70:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=[f"{s}%" for s in scores],
        textfont=dict(color="#ffffff", family="Inter", size=12, weight=700),
        textposition="inside",
        insidetextanchor="middle"
    ))

    fig.update_layout(
        title=dict(
            text="Risk Stratification Index",
            font=dict(size=14, color="#f8fafc", family="Inter")
        ),
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(family="Inter", size=11, color="#94a3b8")
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Inter", size=12, color="#f8fafc")  # Crisp legible white!
        ),
        margin=dict(t=45, b=20, l=20, r=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc")
    )
    return fig


def render_export_buttons(report_data: MedicalReportData, markdown_text: str):
    """
    Render clean, professional report export buttons including PDF, DOCX, JSON, and Markdown.
    Includes prominent AI analysis disclaimer and mandatory expert consultation mandate.
    """
    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="color: #ef4444; font-weight: 700; font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.05em;">
                Critical Clinical Notice: AI Screening & Mandatory Expert Review
            </span>
        </div>
        <p style="color: #f87171; font-size: 0.85rem; line-height: 1.5; margin: 0 0 6px 0; font-weight: 500;">
            Yeh analysis Artificial Intelligence (AI) dwara generate ki gayi automated screening hai. Yeh kisi bhi medical ya pathology diagnosis ke liye definitive nahi hai.
        </p>
        <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5; margin: 0;">
            <strong>Mandatory Doctor Consultation:</strong> Kisi bhi treatment, medication ya clinical decision se pehle confirmation ke liye yeh report kisi certified medical doctor, pathologist ya healthcare specialist ko zaroor dikhaayein.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Document Export & Download Facilities")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.85rem; margin-bottom: 1rem;'>"
        "Download official clinical evaluation reports in PDF, Microsoft Word (.docx), or machine-readable Structured Health Data (.json)."
        "</p>",
        unsafe_allow_html=True
    )

    # Cache binary report generations to ensure fast UI updates
    report_cache_id = f"report_{id(report_data)}"
    if st.session_state.get("cached_export_id") != report_cache_id:
        with st.spinner("Compiling official PDF and DOCX clinical documents..."):
            try:
                st.session_state["cached_pdf"] = generate_pdf_report(report_data)
            except Exception as e:
                st.session_state["cached_pdf"] = None
                st.error(f"PDF compilation error: {e}")
            try:
                st.session_state["cached_docx"] = generate_docx_report(report_data)
            except Exception as e:
                st.session_state["cached_docx"] = None
                st.error(f"Word document compilation error: {e}")
            try:
                st.session_state["cached_json"] = generate_structured_json(report_data)
            except Exception as e:
                st.session_state["cached_json"] = json.dumps(report_data.model_dump(), indent=2, default=str)
            st.session_state["cached_export_id"] = report_cache_id

    pdf_bytes = st.session_state.get("cached_pdf")
    docx_bytes = st.session_state.get("cached_docx")
    json_str = st.session_state.get("cached_json")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_raw = getattr(report_data.patient_info, 'name', 'Patient') or 'Patient'
    patient_slug = "".join([c if c.isalnum() else "_" for c in patient_raw]).strip("_") or "Patient"

    col1, col2 = st.columns(2)
    with col1:
        if pdf_bytes:
            st.download_button(
                label="Download Clinical PDF Report (.pdf)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{patient_slug}_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Official pathology report in high-contrast PDF format with clinical tables, risk assessment, and AI disclaimer."
            )
        else:
            st.button("PDF Generation Error", disabled=True, use_container_width=True)

    with col2:
        if docx_bytes:
            st.download_button(
                label="Download Word Document (.docx)",
                data=docx_bytes,
                file_name=f"Clinical_Report_{patient_slug}_{timestamp_str}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                help="Official Microsoft Word document format with tables, risk metrics, and AI disclaimer for hospital records."
            )
        else:
            st.button("DOCX Generation Error", disabled=True, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.download_button(
            label="Download Structured Health Data (.json)",
            data=json_str,
            file_name=f"Structured_Health_Data_{patient_slug}_{timestamp_str}.json",
            mime="application/json",
            use_container_width=True,
            help="Enriched JSON schema with biomarker telemetry, calculated ranges, and AI attribution metadata."
        )

    with col4:
        st.download_button(
            label="Download Diagnostic Summary (.md)",
            data=markdown_text,
            file_name=f"Diagnostic_Summary_{patient_slug}_{timestamp_str}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Clean markdown diagnostic summary notes formatted for EHR integration."
        )


def render_structured_health_data_section(report_data: MedicalReportData):
    """
    Render comprehensive structured health data telemetry with interactive JSON inspector
    and status attribution breakdown.
    """
    st.markdown("#### Structured Health Data Telemetry")
    
    # Calculate telemetry metrics
    total_tests = 0
    ai_assumptions_count = 0
    report_mentions_count = 0
    categories_count = len(report_data.test_categories)

    for cat in report_data.test_categories:
        for t in cat.tests:
            total_tests += 1
            is_ai = getattr(t, "is_ai_assumption", False) or getattr(t, "status_source", "") == "ai_assumption"
            if is_ai:
                ai_assumptions_count += 1
            else:
                report_mentions_count += 1

    # Render telemetry summary badges/metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Total Biomarkers</div>
            <div style="color: #ffffff; font-size: 1.5rem; font-weight: 700; margin: 4px 0;">{total_tests}</div>
            <div style="color: #64748b; font-size: 0.75rem;">Across {categories_count} panels</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #93c5fd; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Report Mentions</div>
            <div style="color: #93c5fd; font-size: 1.5rem; font-weight: 700; margin: 4px 0;">{report_mentions_count}</div>
            <div style="color: #64748b; font-size: 0.75rem;">Stated on original lab slip</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #c084fc; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">AI Assumptions</div>
            <div style="color: #c084fc; font-size: 1.5rem; font-weight: 700; margin: 4px 0;">{ai_assumptions_count}</div>
            <div style="color: #64748b; font-size: 0.75rem;">Calculated from intervals</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        mode_disp = report_data.extraction_mode.replace("_", " ").title()
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Ingestion Pipeline</div>
            <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600; margin: 6px 0;">{mode_disp}</div>
            <div style="color: #64748b; font-size: 0.75rem;">{report_data.page_count} verified page(s)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    with st.expander("Inspect Raw Structured Health Data (JSON)", expanded=False):
        raw_json_str = st.session_state.get("cached_json") or generate_structured_json(report_data)
        st.json(json.loads(raw_json_str))


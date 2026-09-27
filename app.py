"""
Clinical Diagnostic Intelligence Portal
Professional Streamlit web application with modern enterprise styling,
multimodal visual document ingestion, and comprehensive pathology analysis.
"""

import io
import streamlit as st
from PIL import Image

from src.core.extractor import DocumentExtractor, ExtractedDocument
from src.core.gemini_client import GeminiReportAnalyzer
from src.core.analyzer import ClinicalAnalyzer
from src.ui.components import (
    apply_shadcn_theme,
    render_header,
    render_compliance_notice,
    render_sidebar,
    render_metric_card,
    render_patient_info_card,
    render_risk_banner,
)
from src.ui.visualizer import (
    render_test_category_table,
    render_risk_gauge_chart,
    render_test_distribution_donut,
    render_export_buttons,
    render_structured_health_data_section,
)
from src.utils.sample_data import SAMPLE_REPORT_TEXT
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Clinical Diagnostic Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Minimalist Enterprise Theme
apply_shadcn_theme()


def main():
    # Render Top Navigation Header
    render_header()

    # Render Regulatory & Screening Notice
    render_compliance_notice()

    # Render Configuration Sidebar
    api_key, selected_model, detailed_analysis, auto_enrich = render_sidebar()

    # Document Ingestion Engine & Clinical Logic
    extractor = DocumentExtractor(max_pages_to_render=5, render_dpi=200)
    clinical_analyzer = ClinicalAnalyzer()

    # File Ingestion Section
    col_upload, col_preview = st.columns([3, 2])

    with col_upload:
        st.markdown("### Document Intake & Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Laboratory Diagnostic Report",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            help="Supported formats: Multi-page PDF (digital/scanned), high-resolution pathology captures."
        )

        sample_btn = st.button("Load Reference Clinical Sample (CBC + Metabolic)", type="secondary")

    extracted_doc = None

    if sample_btn:
        st.session_state.is_sample = True
        extracted_doc = ExtractedDocument(
            file_name="reference_metabolic_cbc_sample.txt",
            file_type="text",
            raw_text=SAMPLE_REPORT_TEXT,
            page_images=[],
            page_count=1,
            is_scanned=False,
            status_message="Verified reference sample loaded successfully."
        )
        st.session_state.extracted_doc = extracted_doc

    elif uploaded_file is not None:
        st.session_state.is_sample = False
        try:
            with st.spinner("Processing document intake..."):
                extracted_doc = extractor.process_file(uploaded_file)
                st.session_state.extracted_doc = extracted_doc
        except Exception as e:
            st.error(f"Ingestion Error: {e}")
            logger.error(f"Document extraction error: {e}")

    # Preview & Metadata Column
    with col_preview:
        if "extracted_doc" in st.session_state and st.session_state.extracted_doc:
            doc = st.session_state.extracted_doc
            st.markdown("### Ingestion Meta")
            st.info(f"**Document:** {doc.file_name} — {doc.page_count} page{'s' if doc.page_count > 1 else ''}")
            st.caption(f"Pipeline: {doc.status_message}")
            if doc.page_images:
                with st.expander("Document View (Page 1)", expanded=False):
                    st.image(doc.page_images[0], use_container_width=True)

    # Action Execution
    if "extracted_doc" in st.session_state and st.session_state.extracted_doc:
        doc = st.session_state.extracted_doc
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("Run Diagnostic Analysis", type="primary", use_container_width=True)

        if analyze_clicked:
            if not api_key:
                st.error("API Key required. Please provide a key in the Configuration sidebar or via .env.local.")
                return

            with st.spinner(f"Executing diagnostic inference with {selected_model}..."):
                try:
                    analyzer = GeminiReportAnalyzer(api_key=api_key, model_name=selected_model)
                    report_data = analyzer.analyze_document(doc)

                    if auto_enrich:
                        report_data = clinical_analyzer.enrich_report_data(report_data)

                    stats = clinical_analyzer.get_summary_statistics(report_data)
                    markdown_report = clinical_analyzer.generate_markdown_report(report_data)

                    st.session_state.report_data = report_data
                    st.session_state.stats = stats
                    st.session_state.markdown_report = markdown_report
                    st.success("Analysis finalized successfully.")

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
                    logger.error(f"Analysis failed: {e}")

    # Analysis Results Tabs
    if "report_data" in st.session_state and st.session_state.report_data:
        report_data = st.session_state.report_data
        stats = st.session_state.get("stats", {})
        markdown_report = st.session_state.get("markdown_report", "")

        # Defensive attribute migration for hot-reloaded session state objects
        for cat in getattr(report_data, 'test_categories', []):
            for test in getattr(cat, 'tests', []):
                if not hasattr(test, 'is_ai_assumption'):
                    try:
                        test.is_ai_assumption = getattr(test, 'status_source', '') == 'ai_assumption'
                    except Exception:
                        pass
                if not hasattr(test, 'status_source'):
                    try:
                        test.status_source = 'report'
                    except Exception:
                        pass

        st.markdown("---")
        st.markdown("### Diagnostic Evaluation & Clinical Panels")

        tab_overview, tab_panels, tab_insights, tab_report = st.tabs([
            "Clinical Overview",
            "Pathology Panels",
            "Differential Findings & Insights",
            "Diagnostic Summary & Export"
        ])

        # TAB 1: CLINICAL OVERVIEW
        with tab_overview:
            # Patient Info Card
            render_patient_info_card(report_data.patient_info)

            # High-level Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Total Biomarkers", stats.get("total", 0), "Extracted parameters")
            with m2:
                render_metric_card("Normal Range", stats.get("normal", 0), "Within reference bounds", border_indicator="#16a34a")
            with m3:
                render_metric_card("Atypical / Elevated", stats.get("abnormal", 0), "Out of reference bounds", border_indicator="#d97706")
            with m4:
                render_metric_card("Critical Flags", stats.get("critical", 0), "Actionable pathology flags", border_indicator="#dc2626")

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts Row
            chart1, chart2 = st.columns(2)
            with chart1:
                st.plotly_chart(render_test_distribution_donut(stats), use_container_width=True)
            with chart2:
                if report_data.diagnosis:
                    st.plotly_chart(render_risk_gauge_chart(report_data.diagnosis.risk_assessment), use_container_width=True)

            # Overall Clinical Risk Banner
            if report_data.diagnosis:
                render_risk_banner(
                    report_data.diagnosis.risk_assessment.overall_risk,
                    report_data.diagnosis.risk_assessment.risk_factors
                )

        # TAB 2: PATHOLOGY PANELS
        with tab_panels:
            st.markdown("#### Laboratory Panel Breakdown")
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.85rem; padding: 0.5rem 0.75rem; background: rgba(255,255,255,0.03); border-radius: 0.375rem; border: 1px solid rgba(255,255,255,0.06);">
                <strong>Status Attribution Legend:</strong> 
                <span style="color: #93c5fd; font-weight: 600; margin-left: 0.5rem;">[Mentioned in Report]</span> = Explicitly printed on the lab report document. 
                <span style="color: #c084fc; font-weight: 600; margin-left: 0.5rem;">[AI Assumption]</span> = Derived by comparing patient's level against the reported reference interval.
            </div>
            """, unsafe_allow_html=True)
            if not report_data.test_categories:
                st.warning("No structured biomarker categories identified.")
            else:
                for idx, cat in enumerate(report_data.test_categories):
                    with st.expander(f"{cat.category} ({len(cat.tests)} biomarkers)", expanded=(idx == 0)):
                        render_test_category_table(cat)

            # Highlighted abnormal findings box
            if report_data.abnormal_findings:
                st.markdown("#### Highlighted Anomalous Findings")
                for item in report_data.abnormal_findings:
                    st.warning(f"• {item}")

        # TAB 3: DIFFERENTIAL FINDINGS & INSIGHTS
        with tab_insights:
            diag = report_data.diagnosis
            if diag:
                if diag.red_flags:
                    st.markdown("""
                    <div class="clinical-alert alert-critical">
                        <strong>CRITICAL CLINICAL OBSERVATIONS REQUIRING IMMEDIATE ATTENTION</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    for rf in diag.red_flags:
                        st.error(f"• {rf}")

                # Clinical Narrative Summary
                if diag.summary:
                    st.markdown("#### Clinical Narrative Summary")
                    st.info(diag.summary)

                # Potential Conditions
                if diag.potential_conditions:
                    st.markdown("#### Differential Considerations for Clinical Review")
                    for pc in diag.potential_conditions:
                        with st.expander(f"{pc.condition} (Probability: {pc.probability.title()})"):
                            if pc.description:
                                st.write(f"**Clinical Context:** {pc.description}")
                            if pc.supporting_evidence:
                                st.write("**Supporting Laboratory Evidence:**")
                                for ev in pc.supporting_evidence:
                                    st.write(f"• {ev}")

                # Recommendations
                if diag.recommendations:
                    st.markdown("#### Evidence-Based Clinical Recommendations")
                    rec_groups = {}
                    for rec in diag.recommendations:
                        cat_name = rec.category.title()
                        if cat_name not in rec_groups:
                            rec_groups[cat_name] = []
                        rec_groups[cat_name].append(rec)

                    for cat_name, rec_list in rec_groups.items():
                        st.markdown(f"**{cat_name} Interventions:**")
                        for r in rec_list:
                            st.markdown(f"- **{r.recommendation}** *(Priority: {r.priority.title()})*")
                            if r.rationale:
                                st.caption(f"Rationale: {r.rationale}")

                # Follow-up Testing
                if diag.follow_up_tests:
                    st.markdown("#### Suggested Confirmatory / Follow-Up Diagnostics")
                    for ft in diag.follow_up_tests:
                        st.info(f"• {ft}")
            else:
                st.info("Clinical narrative is disabled. Check 'Generate Clinical Narrative' in the sidebar to view differential insights.")

        # TAB 4: DIAGNOSTIC SUMMARY & STRUCTURED HEALTH DATA EXPORT
        with tab_report:
            render_export_buttons(report_data, markdown_report)
            st.markdown("---")
            render_structured_health_data_section(report_data)
            st.markdown("---")
            st.markdown("#### Clinical Diagnostic Summary")
            st.markdown(markdown_report)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 1.5rem 0;'>
        <p><strong>Clinical Diagnostic Intelligence Portal v2.1</strong> | Powered by Google Gemini Multimodal Vision</p>
        <p>Notice: Intended for clinical laboratory screening and research decision-support only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

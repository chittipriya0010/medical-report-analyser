"""
Enterprise Clinical UI Components with High-Contrast Dark-Mode Design System.
Clean typography, zero emoji clutter, seamless dark-mode cards, and hidden scrollbars.
"""

from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, get_gemini_api_key

# Attempt to import streamlit-shadcn-ui
try:
    import streamlit_shadcn_ui as ui
    SHADCN_PACKAGE_AVAILABLE = True
except ImportError:
    ui = None
    SHADCN_PACKAGE_AVAILABLE = False


def apply_shadcn_theme():
    """Inject modern, minimalist high-contrast dark theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hide header link chain icons */
    a.anchorjs-link, [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    
    /* Hide all scrollbars smoothly */
    ::-webkit-scrollbar {
        width: 0px !important;
        height: 0px !important;
        display: none !important;
    }
    * {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }
    [data-testid="stSidebar"] {
        scrollbar-width: none !important;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar {
        display: none !important;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #f8fafc !important;
        margin: 0;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        margin-top: 0.25rem;
    }
    
    /* High-Contrast Clinical Card */
    .clinical-card {
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.75rem !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
        color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .card-heading {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.5rem;
    }
    
    /* High-Contrast Metric Container */
    .metric-container {
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.75rem !important;
        padding: 1.15rem 1.25rem !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0.35rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-caption {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    .metric-val {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
    }
    .metric-desc {
        font-size: 0.8rem !important;
        color: #64748b !important;
    }
    
    /* Clinical Badges */
    .c-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 0.375rem;
        padding: 0.2rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.2;
        border: 1px solid transparent;
        letter-spacing: 0.02em;
    }
    .badge-normal {
        background-color: rgba(34, 197, 94, 0.15) !important;
        color: #4ade80 !important;
        border-color: rgba(34, 197, 94, 0.3) !important;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15) !important;
        color: #f87171 !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
    }
    .badge-low {
        background-color: rgba(59, 130, 246, 0.15) !important;
        color: #60a5fa !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }
    .badge-borderline {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #fbbf24 !important;
        border-color: rgba(245, 158, 11, 0.3) !important;
    }
    .badge-critical {
        background-color: rgba(220, 38, 38, 0.25) !important;
        color: #fca5a5 !important;
        border-color: #ef4444 !important;
        font-weight: 700 !important;
    }
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.15) !important;
        color: #cbd5e1 !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
    }
    
    /* Attribution Pills */
    .source-pill-report {
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
        background-color: rgba(59, 130, 246, 0.12);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
        font-weight: 500;
    }
    .source-pill-ai {
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
        background-color: rgba(168, 85, 247, 0.12);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.3);
        font-weight: 500;
    }

    /* Clinical Alerts */
    .clinical-alert {
        padding: 1rem 1.25rem !important;
        border-radius: 0.5rem !important;
        border-left: 4px solid !important;
        margin: 1rem 0 !important;
        font-size: 0.9rem !important;
    }
    .alert-critical {
        background-color: rgba(220, 38, 38, 0.12) !important;
        border-color: #ef4444 !important;
        color: #fca5a5 !important;
    }
    .alert-warning {
        background-color: rgba(217, 119, 6, 0.12) !important;
        border-color: #f59e0b !important;
        color: #fcd34d !important;
    }
    .alert-info {
        background-color: rgba(37, 99, 235, 0.1) !important;
        border-color: #3b82f6 !important;
        color: #93c5fd !important;
    }
    .alert-success {
        background-color: rgba(22, 163, 74, 0.1) !important;
        border-color: #22c55e !important;
        color: #86efac !important;
    }
    
    /* Header Disclaimer */
    .compliance-banner {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 0.5rem !important;
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
        margin-bottom: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render top application navigation bar."""
    st.markdown("""
    <div class="top-nav">
        <div>
            <h1 class="brand-title">Clinical Diagnostic Intelligence</h1>
            <div class="brand-sub">Laboratory Analysis, Reference Validation & Diagnostic Decision Support</div>
        </div>
        <div style="text-align: right;">
            <span class="c-badge badge-normal">● Multimodal Vision Active</span>
            <span class="c-badge badge-neutral" style="margin-left: 0.5rem;">Release v2.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_compliance_notice():
    """Render standard regulatory and clinical disclaimer with attribution transparency."""
    st.markdown("""
    <div class="compliance-banner">
        <strong>Clinical Notice:</strong> This platform is a diagnostic decision-support screening tool. 
        <br>
        <strong>Status Legend:</strong> 
        <span class="source-pill-report" style="margin: 0 0.3rem;">Report Mention</span> indicates status was explicitly stated on the lab document. 
        <span class="source-pill-ai" style="margin: 0 0.3rem;">AI Assumption</span> indicates status was calculated by AI comparing observed patient levels against reported reference intervals.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> Tuple[str, str, bool, bool]:
    """
    Render configuration sidebar without exposing API Key.
    Only shows the selected engine and parameters.
    """
    api_key = get_gemini_api_key()

    with st.sidebar:
        st.markdown("### Configuration")

        if not api_key:
            st.error("API Key missing. Please set GEMINI_API_KEY in .env.local")
            api_key_input = st.text_input("Enter Gemini API Key", type="password")
            if api_key_input:
                api_key = api_key_input.strip()
        else:
            st.caption("Authentication: Active via environment (.env.local)")

        selected_model = st.selectbox(
            "Inference Engine",
            options=AVAILABLE_MODELS,
            index=0,
            help="Select verified Google Gemini multimodal model."
        )

        st.markdown("---")
        st.markdown("### Protocol Parameters")
        detailed_analysis = st.checkbox("Generate Clinical Narrative", value=True)
        auto_enrich = st.checkbox("Evaluate Missing Status from Range", value=True, help="Derives status by comparing patient level against reference range.")

        st.markdown("---")
        st.markdown("### System Ingestion Formats")
        st.markdown("""
        - **PDF Documents** (Digital & Scanned multi-page)
        - **High-Resolution Images** (PNG, JPG, TIFF, BMP)
        - **Pathology Panels:** Hematology, Biochemistry, Lipid Screen, Endocrinology, Renal & Hepatic Function
        """)

        st.caption("Confidentiality: Files processed in volatile memory. No permanent storage.")

    return api_key, selected_model, detailed_analysis, auto_enrich


def render_metric_card(title: str, value: Any, description: str = "", border_indicator: Optional[str] = None):
    """Render clean high-contrast SaaS metric container."""
    indicator_style = f"border-top: 3px solid {border_indicator} !important;" if border_indicator else ""
    st.markdown(f"""
    <div class="metric-container" style="{indicator_style}">
        <div class="metric-caption">{title}</div>
        <div class="metric-val">{value}</div>
        <div class="metric-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, status_type: str = "normal") -> str:
    """Return clean HTML string for status badge."""
    css_class = f"badge-{status_type.lower()}"
    return f'<span class="c-badge {css_class}">{text}</span>'


def render_patient_info_card(patient_info):
    """Render patient demographics card with clean typography in dark theme."""
    st.markdown("""
    <div class="clinical-card">
        <div class="card-heading">Patient Demographics & Accession Meta</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Patient Name:** `{patient_info.name}`")
        st.markdown(f"**Age:** `{patient_info.age}`")
    with col2:
        st.markdown(f"**Gender:** `{patient_info.gender}`")
        st.markdown(f"**Specimen Date:** `{patient_info.report_date}`")
    with col3:
        st.markdown(f"**Accession / Lab ID:** `{patient_info.lab_number}`")
        st.markdown(f"**Ordering Physician:** `{patient_info.referring_doctor}`")

    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_banner(overall_risk: str, factors: List[str]):
    """Render clinical risk summary banner."""
    risk_classes = {
        "low": ("alert-success", "OVERALL RISK INDEX: LOW", "Biomarkers indicate general physiological homeostasis."),
        "moderate": ("alert-warning", "OVERALL RISK INDEX: MODERATE", "Multiple parameters outside optimal reference bounds."),
        "high": ("alert-critical", "OVERALL RISK INDEX: ELEVATED / HIGH", "Significant pathology flags detected. Clinical review advised.")
    }
    css_class, title, desc = risk_classes.get(overall_risk.lower(), ("alert-warning", "RISK INDEX: MODERATE", "Clinical correlation required."))

    factors_html = "".join([f"<li>{f}</li>" for f in factors]) if factors else "<li>No secondary risk criteria identified.</li>"

    st.markdown(f"""
    <div class="clinical-alert {css_class}">
        <div style="font-weight: 700; font-size: 0.95rem; letter-spacing: 0.03em;">{title}</div>
        <div style="margin: 0.25rem 0 0.5rem 0; font-size: 0.85rem;">{desc}</div>
        <div style="font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;">Primary Clinical Correlates:</div>
        <ul style="margin: 0.25rem 0 0 0; padding-left: 1.25rem; font-size: 0.85rem;">
            {factors_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

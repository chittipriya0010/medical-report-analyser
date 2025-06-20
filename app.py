import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
from pathlib import Path
import io

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Third-party imports
import google.generativeai as genai
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PDF processing not available. Install PyPDF2 and pdfplumber for PDF support.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReportScanner:
    """AI-powered medical report analysis system using Gemini AI"""
    
    def __init__(self, api_key: str):
        """Initialize the scanner with Gemini API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.analysis_history = []
        
        # Medical reference ranges for common tests
        self.reference_ranges = {
            'hemoglobin': {'male': (13.0, 17.0), 'female': (12.0, 15.5)},
            'glucose_fasting': (70, 100),
            'hba1c': (4.0, 5.6),
            'cholesterol_total': (0, 200),
            'ldl_cholesterol': (0, 100),
            'hdl_cholesterol': {'male': 40, 'female': 50},
            'triglycerides': (0, 150),
            'creatinine': {'male': (0.7, 1.3), 'female': (0.6, 1.1)},
            'tsh': (0.55, 4.78),
            'vitamin_d': (75, 250),
            'vitamin_b12': (211, 911)
        }
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file using multiple methods"""
        if not PDF_AVAILABLE:
            return "PDF processing libraries not available. Please install PyPDF2 and pdfplumber."
        
        extracted_text = ""
        
        try:
            # Method 1: Try pdfplumber first (better for complex layouts)
            pdf_file.seek(0)  # Reset file pointer
            with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- Page {page_num + 1} ---\n"
                            extracted_text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            # If pdfplumber got good results, return it
            if len(extracted_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(extracted_text)} characters using pdfplumber")
                return extracted_text.strip()
            
            # Method 2: Fallback to PyPDF2
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- Page {page_num + 1} ---\n"
                        extracted_text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} with PyPDF2: {e}")
                    continue
            
            if extracted_text.strip():
                logger.info(f"Successfully extracted {len(extracted_text)} characters using PyPDF2")
                return extracted_text.strip()
            else:
                return "No text could be extracted from the PDF. The PDF might contain only images or be password-protected."
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return f"Error processing PDF: {str(e)}"
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from medical report image using Google Vision API via Gemini"""
        try:
            image = Image.open(image_path)
            prompt = """
            Extract all text from this medical report image. 
            Maintain the structure and formatting as much as possible.
            Include all test names, values, reference ranges, and any notes.
            Focus on numerical values and their associated test names.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process uploaded file (PDF or Image) and extract text"""
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            st.info("📄 Processing PDF file...")
            extracted_text = self.extract_text_from_pdf(uploaded_file)
            return extracted_text
            
        elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            st.info("🖼️ Processing image file...")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            try:
                extracted_text = self.extract_text_from_image(temp_path)
                return extracted_text
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        else:
            return f"Unsupported file format: {file_extension}. Please upload PDF, PNG, JPG, or JPEG files."
    
    def clean_json_response(self, response_text: str) -> str:
        """Clean and extract JSON from AI response"""
        # Remove markdown code blocks
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.rfind('```')
            if end > start:
                response_text = response_text[start:end]
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.rfind('```')
            if end > start:
                response_text = response_text[start:end]
        
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        # Try to find JSON-like content if not already clean
        if not response_text.startswith('{'):
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
        
        return response_text
    
    def parse_medical_data(self, text: str) -> Dict:
        """Parse extracted text to identify medical parameters"""
        prompt = f"""
        Analyze this medical report text and extract structured information. This appears to be a comprehensive lab report.
        
        Text to analyze:
        {text}
        
        Please provide a JSON response with exactly this structure:
        {{
            "patient_info": {{
                "name": "patient name if available, otherwise 'Not specified'",
                "age": "age if available, otherwise 'Not specified'",
                "gender": "gender if available, otherwise 'Not specified'",
                "report_date": "date if available, otherwise 'Not specified'",
                "lab_number": "lab number if available, otherwise 'Not specified'"
            }},
            "test_categories": [
                {{
                    "category": "test category name (e.g., 'Complete Blood Count', 'Liver Function', 'Lipid Profile')",
                    "tests": [
                        {{
                            "test_name": "name of test",
                            "value": "measured value",
                            "unit": "unit of measurement",
                            "reference_range": "normal range",
                            "status": "normal/abnormal/high/low/borderline"
                        }}
                    ]
                }}
            ],
            "abnormal_findings": [
                "list of abnormal test results with brief description"
            ],
            "critical_values": [
                "list of any critical or extremely abnormal values"
            ]
        }}
        
        Important guidelines:
        - Look for common lab categories like CBC, Liver Panel, Kidney Panel, Lipid Screen, Thyroid Profile, HbA1c, Vitamins
        - Pay attention to numerical values and their units
        - Compare values to reference ranges when provided
        - If no test results are found, include an empty array for test_categories
        - Always include all required fields even if empty or "Not specified"
        - Only return valid JSON, no additional text or explanations
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                json_text = self.clean_json_response(response.text)
                
                # Log the raw response for debugging
                logger.info(f"Attempt {attempt + 1} - Parsing medical data...")
                
                # Parse JSON
                parsed_data = json.loads(json_text)
                
                # Validate structure
                if self.validate_parsed_data(parsed_data):
                    logger.info("Successfully parsed and validated medical data")
                    # Enhance with status analysis
                    parsed_data = self.enhance_test_status(parsed_data)
                    return parsed_data
                else:
                    logger.warning(f"Attempt {attempt + 1} - Invalid data structure, retrying...")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Attempt {attempt + 1} - JSON decode error: {e}")
                if attempt == max_retries - 1:
                    return self.create_fallback_structure(text)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error parsing medical data: {e}")
                if attempt == max_retries - 1:
                    return self.create_fallback_structure(text)
        
        return self.create_fallback_structure(text)
    
    def enhance_test_status(self, parsed_data: Dict) -> Dict:
        """Enhance test results with improved status analysis"""
        for category in parsed_data.get('test_categories', []):
            for test in category.get('tests', []):
                test_name = test.get('test_name', '').lower()
                value_str = test.get('value', '')
                
                # Try to extract numeric value
                try:
                    if isinstance(value_str, (int, float)):
                        value = float(value_str)
                    else:
                        # Extract first number from string
                        numbers = re.findall(r'[\d.]+', str(value_str))
                        if numbers:
                            value = float(numbers[0])
                        else:
                            continue
                    
                    # Enhanced status determination based on known ranges
                    if 'hba1c' in test_name or 'glycosylated hemoglobin' in test_name:
                        if value >= 6.5:
                            test['status'] = 'high'
                            test['clinical_significance'] = 'Suggests diabetes'
                        elif value >= 5.7:
                            test['status'] = 'borderline'
                            test['clinical_significance'] = 'Prediabetic range'
                        else:
                            test['status'] = 'normal'
                    
                    elif 'glucose' in test_name and 'fasting' in test_name:
                        if value >= 126:
                            test['status'] = 'high'
                            test['clinical_significance'] = 'Diabetic range'
                        elif value >= 100:
                            test['status'] = 'borderline'
                            test['clinical_significance'] = 'Impaired fasting glucose'
                        else:
                            test['status'] = 'normal'
                    
                    elif 'cholesterol' in test_name and 'total' in test_name:
                        if value >= 240:
                            test['status'] = 'high'
                            test['clinical_significance'] = 'High cardiovascular risk'
                        elif value >= 200:
                            test['status'] = 'borderline'
                            test['clinical_significance'] = 'Borderline high'
                        else:
                            test['status'] = 'normal'
                    
                except (ValueError, TypeError):
                    continue
        
        return parsed_data
    
    def validate_parsed_data(self, data: Dict) -> bool:
        """Validate the structure of parsed medical data"""
        required_keys = ['patient_info', 'test_categories', 'abnormal_findings']
        
        # Check if all required keys exist
        if not all(key in data for key in required_keys):
            return False
        
        # Check patient_info structure
        if not isinstance(data['patient_info'], dict):
            return False
        
        # Check test_categories structure
        if not isinstance(data['test_categories'], list):
            return False
        
        # Check abnormal_findings structure
        if not isinstance(data['abnormal_findings'], list):
            return False
        
        return True
    
    def create_fallback_structure(self, original_text: str) -> Dict:
        """Create a basic fallback structure when parsing fails"""
        logger.info("Creating fallback structure for medical data")
        
        # Try to extract some basic information using simple regex
        lines = original_text.split('\n')
        potential_tests = []
        patient_name = "Not specified"
        patient_age = "Not specified"
        patient_gender = "Not specified"
        
        # Try to extract patient info
        for line in lines[:20]:  # Check first 20 lines
            if 'Mr.' in line or 'Mrs.' in line or 'Ms.' in line:
                name_match = re.search(r'(Mr\.|Mrs\.|Ms\.)\s+([A-Za-z\s]+)', line)
                if name_match:
                    patient_name = name_match.group(0).strip()
            
            if 'Years' in line or 'years' in line:
                age_match = re.search(r'(\d+)\s+[Yy]ears', line)
                if age_match:
                    patient_age = age_match.group(1) + " years"
            
            if 'Male' in line or 'Female' in line:
                gender_match = re.search(r'(Male|Female)', line)
                if gender_match:
                    patient_gender = gender_match.group(1)
        
        # Extract test results
        for line in lines:
            # Look for patterns like "TestName: value unit (range)"
            test_pattern = r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]*)\s*\(?([0-9.-]+\s*[-–]\s*[0-9.-]+)?\)?'
            match = re.search(test_pattern, line)
            if match:
                test_name, value, unit, ref_range = match.groups()
                potential_tests.append({
                    "test_name": test_name.strip(),
                    "value": value,
                    "unit": unit or "",
                    "reference_range": ref_range or "Not specified",
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
                "category": "General Tests",
                "tests": potential_tests[:15]  # Limit to first 15 found tests
            }] if potential_tests else [],
            "abnormal_findings": ["Unable to automatically detect abnormal findings - manual review recommended"],
            "critical_values": []
        }
    
    def analyze_diagnosis(self, parsed_data: Dict) -> Dict:
        """Generate AI-powered diagnosis insights"""
        prompt = f"""
        As a medical AI assistant, analyze these comprehensive lab results and provide detailed insights:
        
        {json.dumps(parsed_data, indent=2)}
        
        Provide analysis in the following JSON format:
        {{
            "risk_assessment": {{
                "overall_risk": "low/moderate/high",
                "cardiovascular_risk": "low/moderate/high",
                "diabetes_risk": "low/moderate/high",
                "risk_factors": ["list identified risk factors"]
            }},
            "potential_conditions": [
                {{
                    "condition": "condition name",
                    "probability": "low/moderate/high",
                    "supporting_evidence": ["specific test results that support this"],
                    "description": "brief clinical explanation"
                }}
            ],
            "recommendations": [
                {{
                    "category": "lifestyle/dietary/medical/follow-up",
                    "recommendation": "specific actionable recommendation",
                    "priority": "low/medium/high",
                    "rationale": "why this recommendation is important"
                }}
            ],
            "follow_up_tests": [
                "suggested additional tests with rationale"
            ],
            "red_flags": [
                "critical findings requiring immediate medical attention"
            ],
            "positive_findings": [
                "normal or good results worth highlighting"
            ],
            "summary": "A comprehensive overall assessment summary including key findings and next steps"
        }}
        
        Consider:
        - HbA1c levels for diabetes assessment
        - Lipid profile for cardiovascular risk
        - Liver and kidney function markers
        - Vitamin levels and deficiencies
        - Blood count abnormalities
        - Thyroid function
        
        Important: Only return valid JSON, include disclaimer about professional medical consultation
        """
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                json_text = self.clean_json_response(response.text)
                
                logger.info(f"Diagnosis attempt {attempt + 1} - Analyzing comprehensive report...")
                
                diagnosis = json.loads(json_text)
                
                # Validate diagnosis structure
                if self.validate_diagnosis_data(diagnosis):
                    return diagnosis
                else:
                    logger.warning(f"Diagnosis attempt {attempt + 1} - Invalid structure, retrying...")
                    
            except Exception as e:
                logger.error(f"Diagnosis attempt {attempt + 1} - Error: {e}")
                if attempt == max_retries - 1:
                    return self.create_fallback_diagnosis()
        
        return self.create_fallback_diagnosis()
    
    def validate_diagnosis_data(self, data: Dict) -> bool:
        """Validate diagnosis data structure"""
        required_keys = ['risk_assessment', 'potential_conditions', 'recommendations', 
                        'follow_up_tests', 'red_flags', 'summary']
        return all(key in data for key in required_keys)
    
    def create_fallback_diagnosis(self) -> Dict:
        """Create fallback diagnosis when AI analysis fails"""
        return {
            "risk_assessment": {
                "overall_risk": "moderate",
                "cardiovascular_risk": "moderate",
                "diabetes_risk": "moderate",
                "risk_factors": ["Unable to complete automated risk assessment"]
            },
            "potential_conditions": [],
            "recommendations": [{
                "category": "medical",
                "recommendation": "Consult with healthcare provider for proper interpretation",
                "priority": "high",
                "rationale": "Professional medical interpretation required"
            }],
            "follow_up_tests": [],
            "red_flags": [],
            "positive_findings": [],
            "summary": "Automated analysis could not be completed. Please consult with a qualified healthcare professional for proper interpretation of these comprehensive lab results."
        }
    
    def generate_comprehensive_report(self, parsed_data: Dict, diagnosis: Dict) -> str:
        """Generate comprehensive medical report with enhanced formatting"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 🏥 COMPREHENSIVE MEDICAL REPORT ANALYSIS
**Generated on:** {timestamp}  
**Analysis System:** AI Medical Report Scanner v2.0  
**Report Type:** Comprehensive Lab Panel Analysis

---

## ⚠️ MEDICAL DISCLAIMER
This analysis is generated by AI for educational and informational purposes only. 
**ALWAYS consult with qualified healthcare professionals for medical decisions and treatment.**

---

## 👤 PATIENT INFORMATION
"""
        
        if 'patient_info' in parsed_data:
            patient = parsed_data['patient_info']
            for key, value in patient.items():
                if value and value != "Not specified":
                    icon = {"name": "👤", "age": "📅", "gender": "⚧", "report_date": "📋", "lab_number": "🆔"}.get(key, "📌")
                    report += f"{icon} **{key.replace('_', ' ').title()}:** {value}\n"
        
        # Risk Assessment Summary
        if diagnosis and 'risk_assessment' in diagnosis:
            risk = diagnosis['risk_assessment']
            report += f"\n## 🎯 RISK ASSESSMENT SUMMARY\n"
            
            overall_risk = risk.get('overall_risk', 'moderate')
            risk_colors = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
            report += f"**Overall Health Risk:** {risk_colors.get(overall_risk, '🟡')} **{overall_risk.upper()}**\n\n"
            
            # Specific risk categories
            if 'cardiovascular_risk' in risk:
                cv_risk = risk['cardiovascular_risk']
                report += f"**Cardiovascular Risk:** {risk_colors.get(cv_risk, '🟡')} {cv_risk.title()}\n"
            
            if 'diabetes_risk' in risk:
                dm_risk = risk['diabetes_risk']
                report += f"**Diabetes Risk:** {risk_colors.get(dm_risk, '🟡')} {dm_risk.title()}\n"
        
        # Test Results by Category
        report += "\n## 📊 DETAILED TEST RESULTS\n"
        
        if 'test_categories' in parsed_data and parsed_data['test_categories']:
            for category in parsed_data['test_categories']:
                if category.get('tests'):
                    report += f"\n### 🔬 {category.get('category', 'Unknown Category')}\n"
                    
                    # Create a table-like format
                    report += "| Test | Value | Reference Range | Status |\n"
                    report += "|------|-------|-----------------|--------|\n"
                    
                    for test in category['tests']:
                        test_name = test.get('test_name', 'Unknown Test')
                        value = f"{test.get('value', 'N/A')} {test.get('unit', '')}"
                        ref_range = test.get('reference_range', 'N/A')
                        status = test.get('status', 'unknown')
                        
                        status_icons = {
                            'normal': '🟢 Normal',
                            'abnormal': '🔴 Abnormal', 
                            'high': '🔺 High',
                            'low': '🔻 Low',
                            'borderline': '🟡 Borderline',
                            'unknown': '⚪ Unknown'
                        }
                        status_display = status_icons.get(status, '⚪ Unknown')
                        
                        report += f"| {test_name} | {value} | {ref_range} | {status_display} |\n"
                    
                    report += "\n"
        else:
            report += "No structured test results could be extracted from the report.\n"
        
        # Critical and Abnormal Findings
        if diagnosis and diagnosis.get('red_flags'):
            report += "\n## 🚨 CRITICAL FINDINGS - IMMEDIATE ATTENTION REQUIRED\n"
            for flag in diagnosis['red_flags']:
                report += f"- ⚠️ **{flag}**\n"
            report += "\n**🚑 ACTION REQUIRED:** Contact your healthcare provider immediately.\n"
        
        if 'abnormal_findings' in parsed_data and parsed_data['abnormal_findings']:
            report += "\n## ⚠️ ABNORMAL FINDINGS\n"
            for finding in parsed_data['abnormal_findings']:
                report += f"- 🔴 {finding}\n"
        
        # Positive Findings
        if diagnosis and diagnosis.get('positive_findings'):
            report += "\n## ✅ POSITIVE FINDINGS\n"
            for finding in diagnosis['positive_findings']:
                report += f"- 🟢 {finding}\n"
        
        # AI Analysis and Insights
        if diagnosis:
            if diagnosis.get('potential_conditions'):
                report += "\n## 🔍 POTENTIAL CONDITIONS TO DISCUSS WITH YOUR DOCTOR\n"
                for condition in diagnosis['potential_conditions']:
                    prob_icons = {"low": "🟢 Low", "moderate": "🟡 Moderate", "high": "🔴 High"}
                    prob_display = prob_icons.get(condition.get('probability', 'moderate'), '🟡 Moderate')
                    
                    report += f"\n**{condition.get('condition', 'Unknown')}** - Probability: {prob_display}\n"
                    report += f"- **Description:** {condition.get('description', 'No description available')}\n"
                    if condition.get('supporting_evidence'):
                        report += f"- **Supporting Evidence:** {', '.join(condition['supporting_evidence'])}\n"
            
            # Comprehensive Recommendations
            if diagnosis.get('recommendations'):
                report += "\n## 💡 PERSONALIZED RECOMMENDATIONS\n"
                categories = {}
                for rec in diagnosis['recommendations']:
                    cat = rec.get('category', 'general').title()
                    if cat not in categories:
                        categories[cat] = []
                    
                    priority_icons = {"low": "🔵", "medium": "🟡", "high": "🔴"}
                    priority = rec.get('priority', 'medium')
                    priority_display = priority_icons.get(priority, '🟡')
                    
                    rec_text = f"{priority_display} **{rec.get('recommendation', '')}**"
                    if rec.get('rationale'):
                        rec_text += f"\n  - *Rationale:* {rec['rationale']}"
                    
                    categories[cat].append(rec_text)
                
                category_icons = {
                    'Lifestyle': '🏃‍♂️',
                    'Dietary': '🥗',
                    'Medical': '⚕️',
                    'Follow-Up': '📅',
                    'General': '📌'
                }
                
                for cat, recs in categories.items():
                    icon = category_icons.get(cat, '📌')
                    report += f"\n### {icon} {cat} Recommendations\n"
                    for rec in recs:
                        report += f"- {rec}\n"
            
            # Follow-up Tests
            if diagnosis.get('follow_up_tests'):
                report += "\n## 🧪 SUGGESTED FOLLOW-UP TESTS\n"
                for test in diagnosis['follow_up_tests']:
                    report += f"- 📋 {test}\n"
            
            # Executive Summary
            if diagnosis.get('summary'):
                report += f"\n## 📋 EXECUTIVE SUMMARY\n"
                report += f"{diagnosis['summary']}\n"
        
        # Footer
        report += "\n---\n"
        report += "## 📞 NEXT STEPS\n"
        report += "1. **Schedule an appointment** with your healthcare provider to discuss these results\n"
        report += "2. **Bring this report** to your medical consultation\n"
        report += "3. **Ask questions** about any findings you don't understand\n"
        report += "4. **Follow recommended** lifestyle modifications and follow-up tests\n\n"
        
        report += "**🔒 Privacy Note:** Keep this report confidential and share only with authorized healthcare providers.\n"
        report += "**⚕️ Remember:** This AI analysis supports but does not replace professional medical advice.\n"
        
        return report
    
    def create_test_trend_chart(self, analyses: List[Dict]) -> go.Figure:
        """Create trend charts for key health markers"""
        if len(analyses) < 2:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('HbA1c Trend', 'Cholesterol Trend', 'Blood Pressure Trend', 'Key Markers'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract dates and key values
        dates = [a['timestamp'] for a in analyses]
        
        # Add trend lines for key markers
        # This is a simplified example - you'd extract actual values from parsed_data
        
        fig.update_layout(
            title_text="Health Markers Trend Analysis",
            showlegend=True,
            height=600
        )
        
        return fig

# Enhanced Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Medical Report Scanner",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #00050f;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { border-left-color: #d62728 !important; }
    .risk-moderate { border-left-color: #ff7f0e !important; }
    .risk-low { border-left-color: #2ca02c !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 AI Medical Report Scanner")
    st.markdown("**Advanced AI-powered analysis of comprehensive medical lab reports**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Load API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found in environment variables")
            st.info("Please add GEMINI_API_KEY to your .env file")
            return
        
        st.success("✅ Your AI Medical Scanner is ready")
        
        # Analysis options
        st.header("📊 Analysis Options")
        include_trends = st.checkbox("📈 Include trend analysis", value=True)
        detailed_analysis = st.checkbox("🔍 Detailed AI analysis", value=True)
        export_options = st.checkbox("💾 Enable export options", value=True)
        
        st.header("ℹ️ About")
        st.info("""
        This AI Medical Report Scanner can:
        - 📄 Process PDF and image files
        - 🔍 Extract test results automatically  
        - 📊 Analyze comprehensive lab panels
        - ⚕️ Provide AI-powered insights
        - 📈 Track health trends over time
        """)
        
        st.warning("⚠️ For educational purposes only. Always consult healthcare professionals.")
    
    # Initialize scanner
    try:
        scanner = MedicalReportScanner(api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize scanner: {e}")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Medical Report")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload PDF or image files of medical reports"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📏 File size: {uploaded_file.size / 1024:.1f} KB")
            
            # Process file button
            if st.button("🚀 Analyze Report", type="primary"):
                with st.spinner("🔄 Processing medical report..."):
                    try:
                        # Extract text from file
                        extracted_text = scanner.process_uploaded_file(uploaded_file)
                        
                        if not extracted_text or len(extracted_text.strip()) < 50:
                            st.error("❌ Could not extract sufficient text from the file")
                            st.info("Please ensure the file contains readable medical report data")
                            return
                        
                        # Store extracted text in session state
                        st.session_state.extracted_text = extracted_text
                        
                        # Parse medical data
                        st.info("🧠 Analyzing medical data with AI...")
                        parsed_data = scanner.parse_medical_data(extracted_text)
                        st.session_state.parsed_data = parsed_data
                        
                        # Generate diagnosis if detailed analysis is enabled
                        if detailed_analysis:
                            st.info("⚕️ Generating comprehensive medical insights...")
                            diagnosis = scanner.analyze_diagnosis(parsed_data)
                            st.session_state.diagnosis = diagnosis
                        else:
                            st.session_state.diagnosis = None
                        
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        logger.error(f"Analysis error: {e}")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if 'parsed_data' in st.session_state:
            data = st.session_state.parsed_data
            
            # Count metrics
            total_tests = sum(len(cat.get('tests', [])) for cat in data.get('test_categories', []))
            abnormal_count = len(data.get('abnormal_findings', []))
            categories_count = len(data.get('test_categories', []))
            
            # Display metrics
            st.metric("🧪 Total Tests", total_tests)
            st.metric("⚠️ Abnormal Findings", abnormal_count)
            st.metric("📋 Test Categories", categories_count)
            
            # Risk assessment if available
            if 'diagnosis' in st.session_state and st.session_state.diagnosis:
                risk_data = st.session_state.diagnosis.get('risk_assessment', {})
                overall_risk = risk_data.get('overall_risk', 'moderate')
                
                risk_colors = {
                    'low': '🟢',
                    'moderate': '🟡', 
                    'high': '🔴'
                }
                
                st.metric(
                    "🎯 Overall Risk",
                    f"{risk_colors.get(overall_risk, '🟡')} {overall_risk.title()}"
                )
        else:
            st.info("Upload and analyze a report to see stats")
    
    # Results display
    if 'parsed_data' in st.session_state:
        st.header("📋 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Test Results", "⚕️ AI Insights", "📄 Full Report"])
        
        with tab1:
            st.subheader("👤 Patient Information")
            
            patient_info = st.session_state.parsed_data.get('patient_info', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Name:** {patient_info.get('name', 'Not specified')}")
                st.info(f"**Age:** {patient_info.get('age', 'Not specified')}")
            with col2:
                st.info(f"**Gender:** {patient_info.get('gender', 'Not specified')}")
                st.info(f"**Report Date:** {patient_info.get('report_date', 'Not specified')}")
            with col3:
                st.info(f"**Lab Number:** {patient_info.get('lab_number', 'Not specified')}")
            
            # Summary cards
            st.subheader("📈 Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            data = st.session_state.parsed_data
            total_tests = sum(len(cat.get('tests', [])) for cat in data.get('test_categories', []))
            abnormal_count = len(data.get('abnormal_findings', []))
            normal_count = total_tests - abnormal_count
            categories_count = len(data.get('test_categories', []))
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🧪 {total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✅ {normal_count}</h3>
                    <p>Normal Results</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card risk-moderate">
                    <h3>⚠️ {abnormal_count}</h3>
                    <p>Abnormal Findings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📋 {categories_count}</h3>
                    <p>Test Categories</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("🔬 Detailed Test Results")
            
            test_categories = st.session_state.parsed_data.get('test_categories', [])
            
            if test_categories:
                for i, category in enumerate(test_categories):
                    with st.expander(f"📊 {category.get('category', f'Category {i+1}')}", expanded=True):
                        tests = category.get('tests', [])
                        
                        if tests:
                            # Create DataFrame for better display
                            df_data = []
                            for test in tests:
                                df_data.append({
                                    'Test Name': test.get('test_name', 'N/A'),
                                    'Value': f"{test.get('value', 'N/A')} {test.get('unit', '')}".strip(),
                                    'Reference Range': test.get('reference_range', 'N/A'),
                                    'Status': test.get('status', 'unknown').title()
                                })
                            
                            df = pd.DataFrame(df_data)
                            
                            # Color code the status
                            def highlight_status(val):
                                if val.lower() == 'normal':
                                    return 'background-color: #d4edda'
                                elif val.lower() in ['high', 'abnormal']:
                                    return 'background-color: #f8d7da'
                                elif val.lower() in ['low', 'borderline']:
                                    return 'background-color: #fff3cd'
                                return ''
                            
                            styled_df = df.style.applymap(highlight_status, subset=['Status'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.info("No test results found in this category")
            else:
                st.warning("No test categories found in the analysis")
            
            # Abnormal findings
            abnormal_findings = st.session_state.parsed_data.get('abnormal_findings', [])
            if abnormal_findings:
                st.subheader("⚠️ Abnormal Findings")
                for finding in abnormal_findings:
                    st.warning(f"🔴 {finding}")
        
        with tab3:
            if 'diagnosis' in st.session_state and st.session_state.diagnosis:
                diagnosis = st.session_state.diagnosis
                
                st.subheader("🎯 Risk Assessment")
                
                risk_assessment = diagnosis.get('risk_assessment', {})
                
                col1, col2, col3 = st.columns(3)
                
                risks = [
                    ('Overall Risk', risk_assessment.get('overall_risk', 'moderate')),
                    ('Cardiovascular Risk', risk_assessment.get('cardiovascular_risk', 'moderate')),
                    ('Diabetes Risk', risk_assessment.get('diabetes_risk', 'moderate'))
                ]
                
                for i, (risk_type, risk_level) in enumerate(risks):
                    with [col1, col2, col3][i]:
                        risk_class = f"risk-{risk_level}"
                        st.markdown(f"""
                        <div class="metric-card {risk_class}">
                            <h4>{risk_type}</h4>
                            <h3>{risk_level.title()}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Risk factors
                risk_factors = risk_assessment.get('risk_factors', [])
                if risk_factors:
                    st.subheader("🚨 Identified Risk Factors")
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                
                # Potential conditions
                conditions = diagnosis.get('potential_conditions', [])
                if conditions:
                    st.subheader("🔍 Potential Conditions to Discuss")
                    
                    for condition in conditions:
                        probability = condition.get('probability', 'moderate')
                        prob_colors = {'low': '🟢', 'moderate': '🟡', 'high': '🔴'}
                        
                        with st.expander(f"{prob_colors.get(probability, '🟡')} {condition.get('condition', 'Unknown')} - {probability.title()} Probability"):
                            st.write(f"**Description:** {condition.get('description', 'No description available')}")
                            
                            evidence = condition.get('supporting_evidence', [])
                            if evidence:
                                st.write("**Supporting Evidence:**")
                                for item in evidence:
                                    st.write(f"• {item}")
                
                # Recommendations
                recommendations = diagnosis.get('recommendations', [])
                if recommendations:
                    st.subheader("💡 Personalized Recommendations")
                    
                    # Group by category
                    rec_categories = {}
                    for rec in recommendations:
                        category = rec.get('category', 'general').title()
                        if category not in rec_categories:
                            rec_categories[category] = []
                        rec_categories[category].append(rec)
                    
                    for category, recs in rec_categories.items():
                        st.write(f"**{category} Recommendations:**")
                        
                        for rec in recs:
                            priority = rec.get('priority', 'medium')
                            priority_icons = {'low': '🔵', 'medium': '🟡', 'high': '🔴'}
                            
                            st.write(f"{priority_icons.get(priority, '🟡')} {rec.get('recommendation', '')}")
                            if rec.get('rationale'):
                                st.caption(f"*Rationale: {rec['rationale']}*")
                        st.write("")
                
                # Follow-up tests
                follow_up = diagnosis.get('follow_up_tests', [])
                if follow_up:
                    st.subheader("🧪 Suggested Follow-up Tests")
                    for test in follow_up:
                        st.info(f"📋 {test}")
                
                # Red flags
                red_flags = diagnosis.get('red_flags', [])
                if red_flags:
                    st.subheader("🚨 Critical Findings")
                    st.error("⚠️ **IMMEDIATE MEDICAL ATTENTION REQUIRED**")
                    for flag in red_flags:
                        st.error(f"🚑 {flag}")
                
                # Summary
                summary = diagnosis.get('summary', '')
                if summary:
                    st.subheader("📋 AI Analysis Summary")
                    st.info(summary)
                    
            else:
                st.info("Enable 'Detailed AI analysis' in the sidebar to see comprehensive insights")
        
        with tab4:
            st.subheader("📄 Complete Medical Report")
            
            if 'diagnosis' in st.session_state:
                # Generate comprehensive report
                full_report = scanner.generate_comprehensive_report(
                    st.session_state.parsed_data,
                    st.session_state.diagnosis
                )
                
                st.markdown(full_report)
                
                # Export options
                if export_options:
                    st.subheader("💾 Export Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📄 Download Report (Markdown)"):
                            st.download_button(
                                label="📥 Download MD",
                                data=full_report,
                                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                    
                    with col2:
                        if st.button("📊 Download Data (JSON)"):
                            json_data = {
                                'parsed_data': st.session_state.parsed_data,
                                'diagnosis': st.session_state.diagnosis,
                                'timestamp': datetime.now().isoformat()
                            }
                            st.download_button(
                                label="📥 Download JSON",
                                data=json.dumps(json_data, indent=2),
                                file_name=f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with col3:
                        if st.button("🔗 Share Report"):
                            # Generate shareable link (simplified)
                            st.info("🔗 Report URL generated (feature under development)")
            else:
                st.info("Complete analysis to generate full report")
    
    # Additional features
    st.header("🔧 Additional Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗂️ View Analysis History"):
            if hasattr(scanner, 'analysis_history') and scanner.analysis_history:
                st.subheader("📈 Previous Analyses")
                for i, analysis in enumerate(scanner.analysis_history[-5:]):  # Show last 5
                    with st.expander(f"Analysis {i+1} - {analysis.get('timestamp', 'Unknown')}"):
                        st.json(analysis)
            else:
                st.info("No previous analyses found")
    
    with col2:
        if st.button("❓ Help & Documentation"):
            st.subheader("📚 How to Use")
            st.markdown("""
            ### 🚀 Getting Started
            1. **Enter API Key**: Get your Google AI API key from the provided link
            2. **Upload File**: Choose a PDF or image file of your medical report
            3. **Analyze**: Click the analyze button to process your report
            4. **Review Results**: Check the different tabs for comprehensive analysis
            
            ### 📋 Supported File Types
            - **PDF**: Medical lab reports in PDF format
            - **Images**: PNG, JPG, JPEG, TIFF, BMP files
            
            ### ⚠️ Important Notes
            - This tool is for educational purposes only
            - Always consult healthcare professionals for medical decisions
            - Keep your medical data confidential
            - Results are AI-generated and may not be 100% accurate
            
            ### 🔧 Troubleshooting
            - **No text extracted**: Ensure the file contains readable text
            - **Analysis failed**: Check your internet connection and API key
            - **Poor results**: Try uploading a clearer image or better quality PDF
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 AI Medical Report Scanner v2.0 | Built with Streamlit & Google AI</p>
        <p>⚠️ For educational purposes only. Always consult qualified healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

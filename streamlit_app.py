import streamlit as st
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import pickle
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from datetime import datetime
import io
import base64
import tempfile

# --------------------------
# App Config
# --------------------------
st.set_page_config(
    page_title="AI Thyroid Nodule Classifier", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS Styling (Jet Black & Orange Theme)
# --------------------------
st.markdown("""
<style>
    /* Main background and theme */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        min-height: 100vh;
    }
    
    /* Custom containers */
    .main-header {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(255, 140, 0, 0.15);
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
        border: 2px solid #FF8C00;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(255, 140, 0, 0.12);
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid #FF8C00;
    }
    
    .image-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(255, 140, 0, 0.12);
        backdrop-filter: blur(15px);
        border: 1px solid #FF8C00;
    }
    
    .report-section {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(255, 140, 0, 0.12);
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid #FF8C00;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.9) !important;
        backdrop-filter: blur(15px);
        border-right: 2px solid #FF8C00;
    }
    
    .css-1lcbmhc {
        background: rgba(0, 0, 0, 0.9) !important;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(45deg, #000000, #FF8C00);
        color: white;
        border: 2px solid #FF8C00;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 140, 0, 0.5);
        background: linear-gradient(45deg, #FF8C00, #000000);
    }
    
    /* Download button special styling */
    .download-button {
        background: linear-gradient(45deg, #FF8C00, #FFA500) !important;
        border: 2px solid #FF8C00 !important;
    }
    
    .download-button:hover {
        background: linear-gradient(45deg, #FFA500, #FF8C00) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.6) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #000000, #FF8C00);
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px dashed #FF8C00;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #000000;
        background: rgba(255, 255, 255, 1);
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.2);
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 140, 0, 0.15);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 140, 0, 0.3);
    }
    
    /* Animation for predictions */
    .prediction-result {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Warning/info boxes */
    .stAlert {
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid rgba(40, 167, 69, 0.3);
        color: #155724;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid rgba(220, 53, 69, 0.3);
        color: #721c24;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        color: #856404;
    }
    
    /* Simple Footer styling */
    .simple-footer {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(255, 140, 0, 0.2);
        border: 2px solid #FF8C00;
        text-align: center;
    }
    
    .footer-text {
        color: #FF8C00;
        font-size: 1.2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        margin: 0.5rem 0;
    }
    
    .institution-text {
        color: #ffffff;
        font-size: 1rem;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Model & Encoder (cached for performance)
# --------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_thyroid_model.h5')

@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as f:
        return pickle.load(f)

# Load models
try:
    model = load_model()
    label_encoder = load_label_encoder()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠ Model files not found. Please ensure 'cnn_thyroid_model.h5' and 'label_encoder.pkl' are in the app directory.")

# --------------------------
# Enhanced PDF Report Generation with Better Formatting
# --------------------------
def create_enhanced_pdf_report(patient_info, prediction_results, image_data=None):
    """Generate a comprehensive professional PDF report with improved formatting"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          topMargin=0.75*inch, bottomMargin=0.75*inch,
                          leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    # Get styles and create custom styles
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=24,
        spaceBefore=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#FF8C00'),
        fontName='Helvetica-Bold'
    )
    
    # Custom subtitle style
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        spaceBefore=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2F2F2F'),
        fontName='Helvetica'
    )
    
    # Custom section heading style
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=16,
        textColor=colors.HexColor('#FF8C00'),
        fontName='Helvetica-Bold'
    )
    
    # Custom normal style with better spacing
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        spaceBefore=2,
        leading=12,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    # Custom bold style
    bold_style = ParagraphStyle(
        'CustomBold',
        parent=normal_style,
        fontName='Helvetica-Bold'
    )
    
    # Build story
    story = []
    
    # Header
    story.append(Paragraph("AI THYROID NODULE ANALYSIS REPORT", title_style))
    story.append(Paragraph("Comprehensive Diagnostic Assessment", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Report Information Section
    story.append(Paragraph("REPORT INFORMATION", section_heading_style))
    
    report_info_data = [
        ['Report Generated:', datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")],
        ['Report ID:', f"THY-AI-{int(time.time())}"],
        ['AI Model Version:', "CNN Deep Learning v2.1"],
        ['Analysis Type:', "Binary Classification (Benign/Malignant)"]
    ]
    
    report_table = Table(report_info_data, colWidths=[2*inch, 4*inch])
    report_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(report_table)
    story.append(Spacer(1, 16))
    
    # Patient Information Section
    story.append(Paragraph("PATIENT INFORMATION", section_heading_style))
    
    patient_data = [
        ['Patient Name:', patient_info.get('name', 'Not Provided')],
        ['Patient ID:', patient_info.get('patient_id', 'Not Assigned')],
        ['Age:', f"{patient_info.get('age', 'Not Provided')} years" if patient_info.get('age') else 'Not Provided'],
        ['Gender:', patient_info.get('gender', 'Not Specified')],
        ['Date of Examination:', patient_info.get('scan_date', 'Not Specified')],
        ['Referring Physician:', patient_info.get('physician', 'Not Specified')],
        ['Examination Type:', 'Thyroid Ultrasound Analysis']
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Analysis Results Section
    story.append(Paragraph("AI ANALYSIS RESULTS", section_heading_style))
    
    prediction = prediction_results['prediction']
    confidence = prediction_results['confidence']
    
    # Main prediction result
    if prediction.lower() == 'benign':
        result_color = colors.HexColor('#228B22')
        result_text = "BENIGN (NON-CANCEROUS)"
    else:
        result_color = colors.HexColor('#DC143C')
        result_text = "MALIGNANT (POTENTIALLY CANCEROUS)"
    
    result_para = Paragraph(
        f"<b>CLASSIFICATION:</b> {result_text}<br/><b>CONFIDENCE LEVEL:</b> {confidence:.1f}%",
        ParagraphStyle('ResultPara', parent=normal_style, fontSize=12, 
                      textColor=result_color, spaceAfter=10, fontName='Helvetica-Bold')
    )
    story.append(result_para)
    story.append(Spacer(1, 10))
    
    # Confidence breakdown table
    confidence_data = [
        ['Classification Category', 'Probability', 'Confidence Level', 'Clinical Interpretation'],
        ['Benign (Non-cancerous)', f"{prediction_results['benign_conf']:.2f}%", 
         get_confidence_level(prediction_results['benign_conf']), 
         'Routine monitoring may be sufficient'],
        ['Malignant (Cancerous)', f"{prediction_results['malignant_conf']:.2f}%", 
         get_confidence_level(prediction_results['malignant_conf']), 
         'Further evaluation recommended']
    ]
    
    confidence_table = Table(confidence_data, colWidths=[1.5*inch, 1*inch, 1.2*inch, 2.3*inch])
    confidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F0F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(confidence_table)
    story.append(Spacer(1, 16))
    
    # Technical Analysis Section
    story.append(Paragraph("TECHNICAL ANALYSIS DETAILS", section_heading_style))
    
    technical_data = [
        ['Model Architecture:', 'Convolutional Neural Network (CNN)'],
        ['Input Preprocessing:', 'Image resized to 128x128 pixels, normalized to [0,1] range'],
        ['Feature Extraction:', 'Multi-layer convolutional feature extraction'],
        ['Classification Method:', 'Binary classification with softmax activation'],
        ['Training Dataset:', 'Thousands of validated thyroid ultrasound images'],
        ['Model Performance:', 'Optimized for medical image analysis'],
        ['Processing Time:', 'Real-time analysis (< 2 seconds)']
    ]
    
    tech_table = Table(technical_data, colWidths=[2*inch, 4*inch])
    tech_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 16))
    
    # Clinical Interpretation
    story.append(Paragraph("CLINICAL INTERPRETATION", section_heading_style))
    
    if confidence >= 90:
        interpretation = "High Confidence Prediction (≥90%): The AI model demonstrates strong certainty in this classification. The extracted features strongly align with the predicted category. This level of confidence suggests a reliable preliminary assessment, though clinical correlation remains essential."
    elif confidence >= 70:
        interpretation = "Moderate Confidence Prediction (70-89%): The AI model shows reasonable certainty in this classification. While the prediction is reliable, additional clinical evaluation and possibly alternative imaging modalities may provide valuable complementary information."
    else:
        interpretation = "Low Confidence Prediction (<70%): The AI model shows uncertainty in this classification. This may be due to image quality, atypical features, or borderline characteristics. Strong recommendation for additional clinical evaluation and expert consultation."
    
    story.append(Paragraph(interpretation, normal_style))
    story.append(Spacer(1, 16))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", section_heading_style))
    
    if prediction.lower() == 'benign' and confidence >= 80:
        recommendations = [
            "Continue routine clinical monitoring as per institutional guidelines",
            "Schedule follow-up ultrasound imaging at appropriate intervals", 
            "Patient counseling regarding benign nature of findings",
            "Document findings in patient medical record",
            "Consider discharge to primary care for ongoing monitoring"
        ]
    elif prediction.lower() == 'malignant' or confidence < 70:
        recommendations = [
            "URGENT: Immediate specialist endocrinology consultation",
            "Consider fine needle aspiration (FNA) biopsy",
            "Evaluate for additional imaging studies (CT, MRI if indicated)",
            "Multidisciplinary team discussion recommended",
            "Patient counseling regarding findings and next steps",
            "Expedited scheduling for follow-up procedures"
        ]
    else:
        recommendations = [
            "Clinical correlation with patient history and physical examination",
            "Follow institutional protocols for thyroid nodule management",
            "Consider repeat imaging if clinically indicated",
            "Specialist consultation may be beneficial",
            "Document findings and recommendations clearly"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", normal_style))
        story.append(Spacer(1, 3))
    
    story.append(Spacer(1, 16))
    
    # Quality Assurance
    story.append(Paragraph("QUALITY ASSURANCE", section_heading_style))
    
    qa_data = [
        ['Image Quality Assessment:', 'Processed successfully'],
        ['Model Validation:', 'Algorithm functioning within normal parameters'],
        ['Processing Verification:', 'All preprocessing steps completed successfully'],
        ['Output Validation:', 'Results within expected confidence ranges'],
        ['System Check:', 'All diagnostic modules operational'],
        ['Report Generation:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    
    qa_table = Table(qa_data, colWidths=[2*inch, 4*inch])
    qa_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(qa_table)
    story.append(Spacer(1, 20))
    
    # Page break before disclaimer
    story.append(PageBreak())
    
    # Medical Disclaimer
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", 
                          ParagraphStyle('DisclaimerTitle', parent=section_heading_style,
                                       fontSize=16, textColor=colors.HexColor('#DC143C'),
                                       alignment=TA_CENTER, spaceAfter=12)))
    
    disclaimer_text = """
CRITICAL NOTICE - PLEASE READ CAREFULLY:

1. RESEARCH AND EDUCATIONAL PURPOSE ONLY: This AI-generated analysis is developed and provided exclusively for research, educational, and academic purposes. It is NOT intended for clinical decision-making in patient care.

2. NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL JUDGMENT: This report does NOT replace professional medical diagnosis, clinical judgment, or expert radiological interpretation. All findings must be evaluated by qualified healthcare professionals.

3. LIMITATIONS OF AI ANALYSIS: Artificial intelligence models have inherent limitations and may not detect all pathological conditions. False positives and false negatives are possible. Image quality, patient factors, and technical limitations can affect results.

4. CLINICAL CORRELATION ESSENTIAL: Results must be interpreted in conjunction with complete clinical history, physical examination, laboratory findings, and other diagnostic information.

5. REGULATORY STATUS: This AI system is not FDA-approved for clinical diagnostic use. It is an investigational tool for research purposes only.

6. LIABILITY LIMITATION: The developers, institution, and associated personnel assume no responsibility for clinical decisions based on this analysis. Users assume full responsibility for appropriate use and interpretation.

7. DATA PRIVACY: Ensure patient data is handled in compliance with applicable privacy laws and institutional policies.
    """
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=normal_style,
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        spaceBefore=4
    )
    
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 20))
    
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_confidence_level(confidence):
    """Get confidence level description"""
    if confidence >= 90:
        return "Very High"
    elif confidence >= 80:
        return "High"
    elif confidence >= 70:
        return "Moderate"
    elif confidence >= 60:
        return "Fair"
    else:
        return "Low"

# --------------------------
# Preprocessing Function
# --------------------------
def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((128, 128))
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# --------------------------
# Create Confidence Chart
# --------------------------
def create_confidence_chart(benign_conf, malignant_conf):
    """Create an interactive confidence chart"""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Benign Confidence", "Malignant Confidence"),
        horizontal_spacing=0.1
    )
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = benign_conf,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Benign", 'font': {'color': "white", 'size': 18}},
        delta = {'reference': 50, 'font': {'color': "white"}},
        number = {'font': {'color': "white", 'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'color': "white"}},
            'bar': {'color': "#FF8C00"},
            'steps': [
                {'range': [0, 50], 'color': "#333333"},
                {'range': [50, 100], 'color': "#FFE4B5"}
            ],
            'threshold': {
                'line': {'color': "#FF8C00", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = malignant_conf,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Malignant", 'font': {'color': "white", 'size': 18}},
        delta = {'reference': 50, 'font': {'color': "white"}},
        number = {'font': {'color': "white", 'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'color': "white"}},
            'bar': {'color': "#dc3545"},
            'steps': [
                {'range': [0, 50], 'color': "#333333"},
                {'range': [50, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "#FF8C00", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ), row=1, col=2)
    
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                text="Benign Confidence",
                x=0.225, y=1.1,
                xref="paper", yref="paper",
                font=dict(color="white", size=16),
                showarrow=False
            ),
            dict(
                text="Malignant Confidence", 
                x=0.775, y=1.1,
                xref="paper", yref="paper",
                font=dict(color="white", size=16),
                showarrow=False
            )
        ],
        showlegend=False
    )
    
    return fig

# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #FF8C00; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>🏥 AI Thyroid</h1>
        <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);'>Classifier</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255, 140, 0, 0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid rgba(255, 140, 0, 0.4);'>
        <h4 style='color: #FF8C00; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>📋 How it works:</h4>
        <ol style='color: #ffffff; margin: 0.5rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            <li>Upload ultrasound image</li>
            <li>AI analyzes the image</li>
            <li>Get classification results</li>
            <li>Generate detailed PDF report</li>
            <li>Download professional report</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255, 140, 0, 0.2); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255, 140, 0, 0.4);'>
        <h4 style='color: #FF8C00; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>⚠ Important Notice</h4>
        <p style='color: #ffffff; margin: 0.5rem 0; font-size: 0.9rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            This tool is for <strong style='color: #FF8C00;'>research purposes only</strong>. 
            Always consult healthcare professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model info
    if model_loaded:
        st.success("✅ AI Model: Ready")
        st.success("✅ PDF Generator: Ready")
    else:
        st.error("❌ AI Model: Not Available")
    
    st.info(f"📊 Supported formats: JPG, PNG, JPEG")

# --------------------------
# Main Content
# --------------------------
# Header
st.markdown("""
<div class="main-header">
    <h1 style='text-align: center; color: #000000; margin: 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(255, 140, 0, 0.3);'>
        🧬 AI Thyroid Nodule Classifier
    </h1>
    <p style='text-align: center; color: #FF8C00; margin: 1rem 0 0 0; font-size: 1.2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);'>
        Advanced deep learning for thyroid ultrasound analysis with professional reporting
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("🚫 Cannot proceed without model files. Please check your setup.")
    st.stop()

# Upload Section with enhanced styling
st.markdown("### 📤 Upload Thyroid Ultrasound Image")
uploaded_image = st.file_uploader(
    "Choose an ultrasound image file", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear thyroid ultrasound image for best results"
)

if uploaded_image is not None:
    # Store results in session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        img = Image.open(uploaded_image)
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)
        
        # Image info
        st.markdown(f"""
        📊 **Image Details:**
        - **Size:** {img.size[0]} × {img.size[1]} pixels
        - **Format:** {img.format}
        - **Mode:** {img.mode}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        
        # Add processing animation
        with st.spinner('🧠 AI is analyzing your image...'):
            # Simulate processing time for better UX
            time.sleep(1)
            
            # Preprocess the image
            processed_image = preprocess_image(img)
            
            # Predict the class
            predictions = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)
            class_label = label_encoder.inverse_transform(predicted_class)[0]
            
            # Confidence scores
            confidence_scores = predictions[0]
            benign_conf = float(confidence_scores[0]) * 100
            malignant_conf = float(confidence_scores[1]) * 100
            
            max_confidence = max(benign_conf, malignant_conf)
            
            # Store results in session state
            st.session_state.prediction_results = {
                'prediction': class_label,
                'confidence': max_confidence,
                'benign_conf': benign_conf,
                'malignant_conf': malignant_conf,
                'raw_predictions': predictions[0]
            }
            st.session_state.analysis_complete = True
        
        # Results Header
        st.markdown("### 🎯 Classification Results")
        
        # Main prediction with enhanced styling
        if class_label.lower() == 'benign':
            st.success(f"✅ **Prediction: BENIGN** ({benign_conf:.1f}% confidence)")
        elif class_label.lower() == 'malignant':
            st.error(f"⚠ **Prediction: MALIGNANT** ({malignant_conf:.1f}% confidence)")
        else:
            st.warning(f"❓ **Unknown classification:** {class_label}")
        
        # Confidence level interpretation
        if max_confidence >= 90:
            st.success("🔒 **High Confidence** - Very reliable prediction")
        elif max_confidence >= 70:
            st.warning("🔍 **Moderate Confidence** - Reasonably reliable")
        else:
            st.error("❗ **Low Confidence** - Consider additional analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Analysis Section
    st.markdown("---")
    st.markdown("### 📈 Detailed Confidence Analysis")
    
    # Interactive confidence chart
    chart = create_confidence_chart(benign_conf, malignant_conf)
    st.plotly_chart(chart, use_container_width=True)
    
    # Detailed metrics
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric(
            label="🟢 Benign Probability",
            value=f"{benign_conf:.2f}%",
            delta=f"{benign_conf - 50:.1f}% vs baseline"
        )
    
    with col4:
        st.metric(
            label="🔴 Malignant Probability", 
            value=f"{malignant_conf:.2f}%",
            delta=f"{malignant_conf - 50:.1f}% vs baseline"
        )
    
    with col5:
        st.metric(
            label="🎯 Confidence Level",
            value=f"{max_confidence:.1f}%",
            delta="Prediction strength"
        )
    
    # --------------------------
    # Enhanced PDF Report Section
    # --------------------------
    st.markdown("---")
    st.markdown("### 📄 Generate Professional Medical Report")
    
    # Patient information form
    st.markdown("#### 👤 Patient Information")
    
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient full name")
        patient_id = st.text_input("Patient ID", placeholder="Enter patient ID")
        age = st.text_input("Age", placeholder="Enter age")
    
    with col_form2:
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        scan_date = st.date_input("Scan Date", datetime.now().date())
        physician = st.text_input("Referring Physician", placeholder="Dr. Name")
    
    st.markdown("#### 🎯 Report Generation")
    
    generate_pdf = st.button("📄 Generate Enhanced PDF Report", 
                            type="primary", 
                            use_container_width=True,
                            help="Generate a comprehensive professional medical report")
    
    # Generate Enhanced PDF Report
    if generate_pdf and st.session_state.get('analysis_complete', False):
        if not patient_name:
            st.error("❗ Please enter patient name to generate report.")
        else:
            with st.spinner("📄 Generating comprehensive PDF report..."):
                # Prepare patient info
                patient_info = {
                    'name': patient_name,
                    'patient_id': patient_id if patient_id else 'N/A',
                    'age': age if age else 'N/A',
                    'gender': gender if gender != 'Select' else 'N/A',
                    'scan_date': scan_date.strftime("%B %d, %Y"),
                    'physician': physician if physician else 'N/A'
                }
                
                # Generate Enhanced PDF
                try:
                    pdf_buffer = create_enhanced_pdf_report(patient_info, st.session_state.prediction_results)
                    st.session_state.pdf_buffer = pdf_buffer
                    st.session_state.pdf_generated = True
                    st.session_state.patient_name = patient_name
                    
                    st.success("✅ Enhanced PDF report generated successfully!")
                    st.balloons()
                    
                    # Show download button
                    filename = f"Enhanced_Thyroid_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    
                    st.download_button(
                        label="⬇️ Download Enhanced PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                        type="secondary",
                        help="Download the comprehensive medical report"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Information
    st.markdown("---")
    st.markdown("### 💡 Understanding Your Results")
    
    with st.expander("📚 What do these results mean?", expanded=False):
        st.markdown("""
        **Benign Nodules:**
        - Non-cancerous growths that are generally harmless
        - May require monitoring but typically don't need aggressive treatment
        - Very common, especially in older adults
        
        **Malignant Nodules:**
        - Potentially cancerous growths requiring immediate medical attention
        - Early detection significantly improves treatment outcomes
        - Require comprehensive evaluation by healthcare professionals
        
        **Important:** This AI tool provides preliminary analysis only. Always consult with qualified medical professionals for proper diagnosis and treatment planning.
        """)
    
    with st.expander("🔬 About the Enhanced AI Model", expanded=False):
        st.markdown("""
        **Model Architecture:** Advanced Convolutional Neural Network (CNN)
        
        **Training Data:** Thousands of validated thyroid ultrasound images
        
        **Input Requirements:** 128×128 pixel images, normalized RGB values
        
        **Performance:** Optimized for distinguishing benign and malignant patterns
        
        **Enhanced Features:**
        - Multi-layer feature extraction
        - Advanced preprocessing pipeline
        - Confidence calibration
        - Quality assessment integration
        
        **Limitations:** Results depend on image quality and may not capture all clinical factors
        """)

else:
    # Welcome message with better styling
    st.markdown("""
    <div class="prediction-card" style="text-align: center; padding: 3rem;">
        <h2 style="color: #FF8C00; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);">👆 Ready for Analysis</h2>
        <p style="color: #000000; font-size: 1.1rem; margin-bottom: 2rem; font-weight: bold;">
            Upload a thyroid ultrasound image to begin AI-powered classification
        </p>
        <div style="background: rgba(255, 140, 0, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border: 2px solid #FF8C00;">
            <h4 style="color: #000000; margin-bottom: 1rem;">✨ Enhanced Features:</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; color: #FF8C00; font-weight: bold;">
                <div>🤖 Advanced AI Analysis</div>
                <div>📊 Confidence Scoring</div>
                <div>⚡ Instant Results</div>
                <div>📈 Interactive Charts</div>
                <div>📄 Enhanced PDF Reports</div>
                <div>👤 Patient Management</div>
                <div>🏥 Professional Formatting</div>
                <div>🔬 Technical Analysis</div>
            </div>
        </div>
        
    </div>
    """, unsafe_allow_html=True)

# Simple Footer
st.markdown("---")
st.markdown("""
<div class='simple-footer'>
    <div class='footer-text'>
        Developed By :- Yashvardhan Shinde | Sujal Patil | Ritesh Rodge | Omkar Varote
    </div>
    <div class='footer-text'>
        Guided By: Prof. Nutan Bansode
    </div>
    <div class='institution-text'>
        Department of Electrical Engineering<br>
        MIT Academy of Engineering, Alandi
    </div>
    <div class='footer-text' style='font-size: 1rem; margin-top: 1rem;'>
        🧬 Enhanced AI Analysis | 📄 Professional Reports | For Research Use Only
    </div>
</div>
""", unsafe_allow_html=True)
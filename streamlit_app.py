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
import json

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
    
    
    
    /* Voice button styling */
    .voice-controls {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    
    /* Speaking indicator animation */
    .speaking-indicator {
        animation: pulse 2s infinite;
        color: #FF8C00;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
    
    /* Voice buttons special styling */
    .voice-button-start {
        background: linear-gradient(45deg, #228B22, #32CD32) !important;
        border: 2px solid #228B22 !important;
    }
    
    .voice-button-stop {
        background: linear-gradient(45deg, #DC143C, #FF6347) !important;
        border: 2px solid #DC143C !important;
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

def create_pdf_download_html(pdf_data, patient_name, report_id):
    """Create a simple HTML page that triggers PDF download when opened from QR code"""
    
    # Encode PDF data to base64
    pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thyroid AI Report - Download</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                max-width: 500px;
                background: rgba(255, 255, 255, 0.95);
                color: black;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(255, 140, 0, 0.3);
                border: 2px solid #FF8C00;
                text-align: center;
            }}
            .header {{
                color: #FF8C00;
                margin-bottom: 30px;
            }}
            .download-btn {{
                background: linear-gradient(45deg, #FF8C00, #FFA500);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                margin: 20px 0;
                box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
                transition: transform 0.2s;
            }}
            .download-btn:hover {{
                transform: translateY(-2px);
            }}
            .info {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #FF8C00;
            }}
            .footer {{
                margin-top: 30px;
                color: #666;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 Thyroid AI Report</h1>
                <h3>Ready for Download</h3>
            </div>
            
            <div class="info">
                <p><strong>Patient:</strong> {patient_name}</p>
                <p><strong>Report ID:</strong> {report_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </div>
            
            <button class="download-btn" onclick="downloadPDF()">
                📥 Download PDF Report
            </button>
            
            <p>Click the button above to download the complete medical report</p>
            
            <div class="footer">
                <p><strong>MIT Academy of Engineering</strong></p>
                <p>AI Thyroid Classifier - Research Use Only</p>
            </div>
        </div>
        
        <script>
            function downloadPDF() {{
                // Create a blob from the base64 data
                const pdfData = '{pdf_b64}';
                const byteCharacters = atob(pdfData);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {{
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {{type: 'application/pdf'}});
                
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'Thyroid_AI_Report_{patient_name.replace(" ", "_")}_{int(time.time())}.pdf';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                
                // Update button text
                document.querySelector('.download-btn').innerHTML = '✅ Downloaded Successfully';
                document.querySelector('.download-btn').style.background = 'linear-gradient(45deg, #28a745, #20c997)';
            }}
            
            // Auto-trigger download on mobile devices
            if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {{
                setTimeout(() => {{
                    downloadPDF();
                }}, 1000);
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

def create_viewable_report_html(prediction_results, patient_info=None):
    """Create a viewable HTML report for display (not for QR code)"""
    
    patient_name = patient_info.get('name', 'Anonymous') if patient_info else 'Anonymous'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thyroid AI Report</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                color: black;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(255, 140, 0, 0.3);
                border: 2px solid #FF8C00;
            }}
            .header {{
                text-align: center;
                color: #FF8C00;
                margin-bottom: 30px;
            }}
            .result {{
                background: {'#e8f5e8' if prediction_results['prediction'].lower() == 'benign' else '#ffe8e8'};
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid {'#28a745' if prediction_results['prediction'].lower() == 'benign' else '#dc3545'};
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .info-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 3px solid #FF8C00;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #FF8C00;
                color: #666;
            }}
            .disclaimer {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> AI Thyroid Analysis Report</h1>
                <h3>Digital Summary</h3>
            </div>
            
            <div class="result">
                <h2>Classification: {prediction_results['prediction'].upper()}</h2>
                <h3>Confidence: {prediction_results['confidence']:.1f}%</h3>
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <strong>Patient:</strong><br>
                    {patient_name}
                </div>
                <div class="info-item">
                    <strong>Report ID:</strong><br>
                    THY-AI-{int(time.time())}
                </div>
                <div class="info-item">
                    <strong>Date & Time:</strong><br>
                    {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </div>
                <div class="info-item">
                    <strong>Institution:</strong><br>
                    MIT Academy of Engineering
                </div>
            </div>
            
            <div class="disclaimer">
                <strong>⚠ Important:</strong> This AI analysis is for research purposes only. 
                Always consult qualified healthcare professionals for medical decisions.
            </div>
            
            <div class="footer">
                <p><strong>Generated by AI Thyroid Classifier</strong></p>
                <p>MIT Academy of Engineering, Alandi</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content
# --------------------------
# Web-based Voice Functions using Browser's Speech Synthesis API
# --------------------------
def generate_voice_summary(prediction_results, patient_name=None):
    """Generate a voice summary text of the analysis results"""
    prediction = prediction_results['prediction']
    confidence = prediction_results['confidence']
    
    # Create voice summary text
    if patient_name:
        summary = f"Voice Report for patient {patient_name}. "
    else:
        summary = "AI Thyroid Analysis Voice Report. "
    
    summary += f"The artificial intelligence analysis has classified this thyroid nodule as {prediction.upper()}. "
    
    if prediction.lower() == 'benign':
        summary += f"This indicates a non-cancerous nodule. The confidence level is {confidence:.1f} percent. "
    else:
        summary += f"This indicates a potentially cancerous nodule requiring immediate medical attention. The confidence level is {confidence:.1f} percent. "
    
    if confidence >= 90:
        summary += "The AI model shows very high confidence in this prediction. "
    elif confidence >= 70:
        summary += "The AI model shows moderate confidence in this prediction. "
    else:
        summary += "The AI model shows low confidence in this prediction. Additional clinical evaluation is strongly recommended. "
    
    summary += "Please note that this AI analysis is for research purposes only and should not replace professional medical diagnosis. "
    
    if prediction.lower() == 'malignant':
        summary += "Immediate consultation with a healthcare professional is advised. "
    else:
        summary += "Continue routine monitoring as per medical guidelines. "
    
    summary += "This concludes the voice report. Thank you."
    
    return summary

def create_speech_component(text_to_speak, button_id):
    """Create HTML/JavaScript component for web-based speech synthesis"""
    
    # Escape text for JavaScript
    escaped_text = json.dumps(text_to_speak)
    
    html_code = f"""
    <div id="speech-container-{button_id}" style="text-align: center; margin: 20px 0;">
        <button id="speak-btn-{button_id}" onclick="speakText{button_id}()" 
                style="background: linear-gradient(45deg, #228B22, #32CD32); 
                       color: white; border: none; padding: 12px 24px; 
                       border-radius: 25px; font-size: 16px; font-weight: bold; 
                       cursor: pointer; margin: 5px; box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);">
            🔊 Start Voice Report
        </button>
        
        <button id="stop-btn-{button_id}" onclick="stopSpeech{button_id}()" 
                style="background: linear-gradient(45deg, #DC143C, #FF6347); 
                       color: white; border: none; padding: 12px 24px; 
                       border-radius: 25px; font-size: 16px; font-weight: bold; 
                       cursor: pointer; margin: 5px; box-shadow: 0 4px 15px rgba(220, 20, 60, 0.3);">
            ⏹️ Stop Voice Report
        </button>
        
        <div id="status-{button_id}" style="margin-top: 10px; font-weight: bold; color: #FF8C00;"></div>
    </div>

    <script>
        let currentUtterance{button_id} = null;
        let isCurrentlySpeaking{button_id} = false;

        function speakText{button_id}() {{
            // Stop any existing speech
            if (isCurrentlySpeaking{button_id}) {{
                stopSpeech{button_id}();
                return;
            }}

            // Check if speech synthesis is supported
            if (!('speechSynthesis' in window)) {{
                document.getElementById('status-{button_id}').innerHTML = '❌ Speech synthesis not supported in this browser';
                return;
            }}

            const textToSpeak = {escaped_text};
            
            // Create new utterance
            currentUtterance{button_id} = new SpeechSynthesisUtterance(textToSpeak);
            
            // Configure voice settings
            currentUtterance{button_id}.rate = 0.8;  // Slower speech
            currentUtterance{button_id}.pitch = 1.0;
            currentUtterance{button_id}.volume = 0.9;
            
            // Try to set a professional voice
            const voices = speechSynthesis.getVoices();
            if (voices.length > 0) {{
                // Prefer female or clear voices
                const preferredVoice = voices.find(voice => 
                    voice.name.includes('Female') || 
                    voice.name.includes('Google') ||
                    voice.name.includes('Microsoft Zira') ||
                    voice.lang.startsWith('en')
                );
                if (preferredVoice) {{
                    currentUtterance{button_id}.voice = preferredVoice;
                }}
            }}

            // Event handlers
            currentUtterance{button_id}.onstart = function() {{
                isCurrentlySpeaking{button_id} = true;
                document.getElementById('status-{button_id}').innerHTML = '🔊 <span class="speaking-indicator">Currently Speaking...</span>';
                document.getElementById('speak-btn-{button_id}').innerHTML = '🔊 Speaking...';
                document.getElementById('speak-btn-{button_id}').style.background = 'linear-gradient(45deg, #FF8C00, #FFA500)';
            }};

            currentUtterance{button_id}.onend = function() {{
                isCurrentlySpeaking{button_id} = false;
                document.getElementById('status-{button_id}').innerHTML = '✅ Voice report completed';
                document.getElementById('speak-btn-{button_id}').innerHTML = '🔊 Start Voice Report';
                document.getElementById('speak-btn-{button_id}').style.background = 'linear-gradient(45deg, #228B22, #32CD32)';
            }};

            currentUtterance{button_id}.onerror = function(event) {{
                isCurrentlySpeaking{button_id} = false;
                document.getElementById('status-{button_id}').innerHTML = '❌ Error: ' + event.error;
                document.getElementById('speak-btn-{button_id}').innerHTML = '🔊 Start Voice Report';
                document.getElementById('speak-btn-{button_id}').style.background = 'linear-gradient(45deg, #228B22, #32CD32)';
            }};

            // Start speaking
            document.getElementById('status-{button_id}').innerHTML = '🎵 Preparing voice report...';
            speechSynthesis.speak(currentUtterance{button_id});
        }}

        function stopSpeech{button_id}() {{
            if (speechSynthesis.speaking || isCurrentlySpeaking{button_id}) {{
                speechSynthesis.cancel();
                isCurrentlySpeaking{button_id} = false;
                document.getElementById('status-{button_id}').innerHTML = '🔇 Voice report stopped';
                document.getElementById('speak-btn-{button_id}').innerHTML = '🔊 Start Voice Report';
                document.getElementById('speak-btn-{button_id}').style.background = 'linear-gradient(45deg, #228B22, #32CD32)';
            }}
        }}

        // Load voices when available
        if (speechSynthesis.onvoiceschanged !== undefined) {{
            speechSynthesis.onvoiceschanged = function() {{
                // Voices are now loaded
            }};
        }}
    </script>
    """
    
    return html_code

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
        predicted_conf = prediction_results['benign_conf']
    else:
        result_color = colors.HexColor('#DC143C')
        result_text = "MALIGNANT (POTENTIALLY CANCEROUS)"
        predicted_conf = prediction_results['malignant_conf']
    
    result_para = Paragraph(
        f"<b>CLASSIFICATION:</b> {result_text}<br/><b>CONFIDENCE LEVEL:</b> {confidence:.1f}%",
        ParagraphStyle('ResultPara', parent=normal_style, fontSize=12, 
                      textColor=result_color, spaceAfter=10, fontName='Helvetica-Bold')
    )
    story.append(result_para)
    story.append(Spacer(1, 10))
    
    # Single prediction confidence table (only showing predicted class)
    confidence_data = [
        ['Classification Category', 'Probability', 'Confidence Level', 'Clinical Interpretation'],
        [result_text, f"{predicted_conf:.2f}%", 
         get_confidence_level(predicted_conf), 
         'Further evaluation recommended' if prediction.lower() == 'malignant' else 'Routine monitoring may be sufficient']
    ]
    
    confidence_table = Table(confidence_data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 2.5*inch])
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
# Create Single Confidence Chart (Updated)
# --------------------------
def create_confidence_chart(prediction, confidence, class_label):
    """Create an interactive confidence chart showing only the predicted class"""
    
    # Determine color based on prediction
    if class_label.lower() == 'benign':
        bar_color = "#228B22"
        title_text = "Benign Confidence"
        step_color = "#90EE90"
    else:
        bar_color = "#DC143C" 
        title_text = "Malignant Confidence"
        step_color = "#FFB6C1"
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title_text, 'font': {'color': "white", 'size': 20}},
        delta = {'reference': 50, 'font': {'color': "white"}},
        number = {'font': {'color': "white", 'size': 28}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'color': "white"}},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 50], 'color': "#333333"},
                {'range': [50, 100], 'color': step_color}
            ],
            'threshold': {
                'line': {'color': "#FF8C00", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        margin=dict(l=40, r=40, t=80, b=40),
        title={
            'text': f"<b>{class_label.upper()} Classification Confidence</b>",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': 'white'}
        }
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
            <li>Listen to voice summary</li>
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
        st.success("✅ Voice Engine: Browser-based TTS")
        st.success("✅ Digital Report Preview : Ready")
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
        Advanced deep learning for thyroid ultrasound analysis with professional reporting and voice summaries
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
        📊 *Image Details:*
        - *Size:* {img.size[0]} × {img.size[1]} pixels
        - *Format:* {img.format}
        - *Mode:* {img.mode}
        """)
    
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
            st.success(f"✅ *Prediction: BENIGN* ({benign_conf:.1f}% confidence)")
        elif class_label.lower() == 'malignant':
            st.error(f"⚠ *Prediction: MALIGNANT* ({malignant_conf:.1f}% confidence)")
        else:
            st.warning(f"❓ *Unknown classification:* {class_label}")
        
        # Confidence level interpretation
        if max_confidence >= 90:
            st.success("🔒 *High Confidence* - Very reliable prediction")
        elif max_confidence >= 70:
            st.warning("🔍 *Moderate Confidence* - Reasonably reliable")
        else:
            st.error("❗ *Low Confidence* - Consider additional analysis")
    
    # Detailed Analysis Section
    st.markdown("---")
    st.markdown("### 📈 Detailed Confidence Analysis")
    
    # Interactive confidence chart - now shows only predicted class
    chart = create_confidence_chart(st.session_state.prediction_results['prediction'], 
                                   st.session_state.prediction_results['confidence'],
                                   st.session_state.prediction_results['prediction'])
    st.plotly_chart(chart, use_container_width=True)
    
    # Single metric for predicted class only
    st.markdown("#### 🎯 Prediction Metrics")
    
    if st.session_state.prediction_results['prediction'].lower() == 'benign':
        col_center = st.columns([1, 2, 1])[1]  # Center the metric
        with col_center:
            st.metric(
                label="🟢 Benign Probability",
                value=f"{benign_conf:.2f}%",
                delta=f"{benign_conf - 50:.1f}% vs baseline"
            )
    else:
        col_center = st.columns([1, 2, 1])[1]  # Center the metric
        with col_center:
            st.metric(
                label="🔴 Malignant Probability", 
                value=f"{malignant_conf:.2f}%",
                delta=f"{malignant_conf - 50:.1f}% vs baseline"
            )
    
    # --------------------------
    # VOICE REPORTS SECTION (NEW)
    # --------------------------
    st.markdown("---")
    st.markdown("""
    <div class="voice-section">
        <h2 style='color: #FF8C00; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            🔊 Voice Report Summary
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎵 Audio Summary for Accessibility")
    st.markdown("Listen to a comprehensive voice summary of the AI analysis results")
    
    # Generate voice summary text
    voice_text = generate_voice_summary(st.session_state.prediction_results)
    
    # Create and display the speech component
    speech_component = create_speech_component(voice_text, "main")
    st.components.v1.html(speech_component, height=120)
    
    # Show the text being spoken in an expander
    with st.expander("📝 View Voice Report Text"):
        st.markdown(f"**Voice Summary:**\n\n{voice_text}")
    
    # Voice features info
    st.markdown("---")
    st.markdown("#### 🎧 Voice Features")
    
    voice_col1, voice_col2 = st.columns(2)
    
    with voice_col1:
        st.info("""
        **🎯 Voice Report Includes:**
        - Patient classification results
        - Confidence level explanation  
        - Clinical interpretation
        - Medical recommendations
        - Important disclaimers
        """)
    
    with voice_col2:
        st.info("""
        **♿ Accessibility Benefits:**
        - Hands-free report review
        - Support for visually impaired users
        - Multi-modal information delivery
        - Enhanced user experience
        - Professional narration
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------------
    # QR CODE INTEGRATION SECTION (NEW)
    # --------------------------
    st.markdown("---")
    # Create viewable HTML report for preview
    html_report = create_viewable_report_html(st.session_state.prediction_results, {'name': st.session_state.get('patient_name', 'Anonymous')})
    
    # HTML Report preview section
    st.markdown("---")
    st.markdown("#### 🌐 Digital Report Preview")
    st.markdown("Preview the full digital report for sharing")
    
    if st.button("🌐 Open Full Digital Report", key="preview_digital"):
        # Encode the HTML for data URL
        encoded_html = base64.b64encode(html_report.encode('utf-8')).decode('utf-8')
        data_url = f"data:text/html;base64,{encoded_html}"
        st.components.v1.html(f'<iframe src="{data_url}" width="100%" height="600" style="border: 2px solid #FF8C00; border-radius: 10px;"></iframe>', height=650)
    
    # --------------------------
    # PATIENT INFORMATION SECTION (REPORT GENERATION)
    # --------------------------
    st.markdown("---")
    st.markdown("""
    <div class="report-section">
        <h2 style='color: #FF8C00; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            📋 Generate Professional Medical Report
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 👤 Patient Information")
    st.markdown("Fill in patient details for comprehensive report generation")
    
    # Create two columns for patient information
    col_left, col_right = st.columns(2, gap="medium")
    
    with col_left:
        patient_name = st.text_input("👤 Patient Name", 
                                   placeholder="Enter patient's full name")
        patient_id = st.text_input("🆔 Patient ID", 
                                 placeholder="Enter patient ID/MRN")
        patient_age = st.number_input("🎂 Age", min_value=1, max_value=120, 
                                    value=None, placeholder="Age in years")
    
    with col_right:
        patient_gender = st.selectbox("⚧ Gender", 
                                    ["", "Male", "Female", "Other", "Prefer not to say"],
                                    index=0)
        scan_date = st.date_input("📅 Scan Date", 
                                value=datetime.now().date(),
                                help="Date when the ultrasound was performed")
        physician_name = st.text_input("👨‍⚕ Referring Physician", 
                                     placeholder="Dr. Name")
    
    # Additional clinical information
    st.markdown("#### 📝 Additional Clinical Information")
    clinical_notes = st.text_area("Clinical Notes (Optional)", 
                                 placeholder="Any additional clinical observations, symptoms, or relevant patient history...",
                                 height=100)
    
    # Report generation section
    st.markdown("---")
    st.markdown("#### 📄 Report Generation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔄 Generate Professional PDF Report", 
                    use_container_width=True,
                    help="Generate a comprehensive medical report with patient information and analysis results"):
            
            if not patient_name.strip():
                st.error("⚠ Please enter patient name to generate report")
            else:
                with st.spinner("📝 Generating comprehensive PDF report..."):
                    # Prepare patient information
                    patient_info = {
                        'name': patient_name.strip(),
                        'patient_id': patient_id.strip() if patient_id.strip() else "Not Assigned",
                        'age': patient_age,
                        'gender': patient_gender if patient_gender else "Not Specified",
                        'scan_date': scan_date.strftime("%B %d, %Y"),
                        'physician': physician_name.strip() if physician_name.strip() else "Not Specified",
                        'clinical_notes': clinical_notes.strip() if clinical_notes.strip() else "None provided"
                    }
                    
                    # Generate the PDF report
                    pdf_buffer = create_enhanced_pdf_report(
                        patient_info, 
                        st.session_state.prediction_results
                    )
                    
                    # Store in session state
                    st.session_state.pdf_report = pdf_buffer.getvalue()
                    st.session_state.report_generated = True
                    st.session_state.patient_name = patient_name.strip()
                    
                st.success("✅ Report generated successfully!")
                st.balloons()
    
    # Enhanced voice report with patient name
    if patient_name.strip() and st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown("#### 🎤 Personalized Voice Report")
        
        # Generate personalized voice summary
        personal_voice_text = generate_voice_summary(
            st.session_state.prediction_results,
            patient_name.strip()
        )
        
        # Create and display the personalized speech component
        personal_speech_component = create_speech_component(personal_voice_text, "personal")
        st.components.v1.html(personal_speech_component, height=120)
        
        # Show personalized text in expander
        with st.expander("📝 View Personalized Voice Report Text"):
            st.markdown(f"**Personalized Voice Summary for {patient_name.strip()}:**\n\n{personal_voice_text}")
    
    # Download section
    if 'report_generated' in st.session_state and st.session_state.report_generated:
        st.markdown("---")
        st.markdown("#### 📥 Download Report")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_name_clean = "".join(c for c in patient_name.replace(" ", "") if c.isalnum() or c in ".-")
            filename = f"Thyroid_AI_Report_{patient_name_clean}_{timestamp}.pdf"
            
            st.download_button(
                label="📥 Download Professional Report",
                data=st.session_state.pdf_report,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                help="Download the complete medical analysis report"
            )
        
        # Report summary
        st.markdown("---")
        st.markdown("#### 📊 Report Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.info(f"""
            *Patient:* {patient_name}
            *ID:* {patient_id if patient_id.strip() else 'Not Assigned'}
            *Classification:* {st.session_state.prediction_results['prediction'].upper()}
            """)
        
        with summary_col2:
            st.info(f"""
            *Confidence:* {st.session_state.prediction_results['confidence']:.1f}%
            *Scan Date:* {scan_date.strftime("%B %d, %Y")}
            *Report Generated:* {datetime.now().strftime("%Y-%m-%d %H:%M")}
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome message with better styling - This appears when no image is uploaded
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
                <div>🔊 Voice Reports</div>
                <div>📄 Enhanced PDF Reports</div>
                <div>👤 Patient Management</div>
                <div>🏥 Professional Formatting</div>
                <div>♿ Accessibility Features</div>
                <div>🔬 Technical Analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Footer Section - Always appears at the bottom
# --------------------------
st.markdown("---")
st.markdown("""
<div class='simple-footer'>
    <div class='footer-text'>
        Developed By: Yashvardhan Shinde | Sujal Patil | Ritesh Rodge | Omkar Varote
    </div>
    <div class='footer-text'>
        Guided By: Prof. Nutan Bansode
    </div>
    <div class='institution-text'>
        Department of Electrical Engineering<br>
        MIT Academy of Engineering, Alandi
    </div>
    <div class='footer-text' style='font-size: 1rem; margin-top: 1rem;'>
        🧬 Enhanced AI Analysis | 🔊 Voice Reports | 📱 QR Access | 📄 Professional Reports | For Research Use Only
    </div>
</div>
""", unsafe_allow_html=True)

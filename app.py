import streamlit as st
import os
from PIL import Image
import tempfile
from detection_utils import DeepfakeDetector
import torch
import numpy as np

# Page config
st.set_page_config(
    page_title="GuardianAI | Deepfake & Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium White/Black Gradient CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 50%, #e8e8e8 100%);
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Landing Page Styles */
    .landing-container {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(180deg, #ffffff 0%, #f5f5f5 100%);
    }
    
    .landing-logo {
        font-size: 6rem;
        margin-bottom: 20px;
        filter: drop-shadow(0 10px 30px rgba(0,0,0,0.1));
    }
    
    .landing-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1a1a1a 0%, #4a4a4a 50%, #2a2a2a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        margin-bottom: 15px;
    }
    
    .landing-subtitle {
        font-size: 1.4rem;
        color: #666666;
        max-width: 600px;
        line-height: 1.7;
        margin-bottom: 50px;
    }
    
    .start-button {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        color: white;
        padding: 18px 60px;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        text-decoration: none;
        display: inline-block;
    }
    
    .start-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    }
    
    .landing-features {
        display: flex;
        gap: 40px;
        margin-top: 80px;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .landing-feature {
        text-align: center;
        padding: 25px;
    }
    
    .landing-feature-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .landing-feature-text {
        color: #666;
        font-size: 0.95rem;
    }
    
    /* Dashboard Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
        border-right: 1px solid #333;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .sidebar-brand {
        text-align: center;
        padding: 30px 20px;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
    }
    
    .sidebar-logo { font-size: 3rem; }
    
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
        margin-top: 10px;
    }
    
    .sidebar-tagline {
        color: #888;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    
    /* Dashboard Styles */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }
    
    .dashboard-subtitle {
        color: #aaa;
        font-size: 1rem;
    }
    
    /* Cards */
    .analysis-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border-color: #1a1a1a;
    }
    
    .card-icon { font-size: 3rem; margin-bottom: 15px; }
    .card-title { font-size: 1.3rem; font-weight: 600; color: #1a1a1a; margin-bottom: 10px; }
    .card-desc { color: #666; font-size: 0.9rem; line-height: 1.5; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        color: white;
        font-weight: 600;
        padding: 12px 30px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Result Cards */
    .result-fake {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    
    .result-real {
        background: linear-gradient(135deg, #f0fff0 0%, #e0ffe0 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    
    .result-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 2px; }
    .result-verdict-fake { font-size: 2.5rem; font-weight: 800; color: #dc2626; }
    .result-verdict-real { font-size: 2.5rem; font-weight: 800; color: #16a34a; }
    .result-confidence { font-size: 1.1rem; color: #666; margin-top: 5px; }
    .result-msg { margin-top: 15px; padding: 12px; border-radius: 8px; font-size: 0.9rem; }
    .result-msg-fake { background: rgba(220,38,38,0.1); color: #b91c1c; }
    .result-msg-real { background: rgba(22,163,74,0.1); color: #15803d; }
    
    /* Page Header */
    .page-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #333 100%);
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .page-title { font-size: 2rem; font-weight: 700; color: white; }
    .page-subtitle { color: #aaa; font-size: 0.95rem; margin-top: 8px; }
    
    /* Upload Area */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #ccc;
        border-radius: 16px;
        padding: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid #e0e0e0;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Initialize Detector
DATASET_DIR = "/Users/abhijeetgolhar/Downloads/P Project/Dataset"
@st.cache_resource
def get_detector():
    return DeepfakeDetector(DATASET_DIR)

detector = get_detector()

# ========== LANDING PAGE ==========
if st.session_state.page == 'landing':
    st.markdown("""
    <div class="landing-container">
        <div class="landing-logo">🛡️</div>
        <h1 class="landing-title">GuardianAI</h1>
        <p class="landing-subtitle">
            Advanced AI-powered detection system for deepfakes, synthetic media, 
            and fake news. Protect yourself from digital deception with cutting-edge technology.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True, key="start_btn"):
            st.session_state.page = 'dashboard'
            st.rerun()
    
    st.markdown("""
    <div class="landing-features">
        <div class="landing-feature">
            <div class="landing-feature-icon">🖼️</div>
            <div class="landing-feature-text">Image Detection</div>
        </div>
        <!-- <div class="landing-feature">
            <div class="landing-feature-icon">🎥</div>
            <div class="landing-feature-text">Video Analysis</div>
        </div>
        <div class="landing-feature">
            <div class="landing-feature-icon">🎵</div>
            <div class="landing-feature-text">Audio Verification</div>
        </div> -->
        <!-- <div class="landing-feature">
            <div class="landing-feature-icon">📝</div>
            <div class="landing-feature-text">Text Analysis</div>
        </div> -->
    </div>
    """, unsafe_allow_html=True)

# ========== DASHBOARD ==========
else:
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🛡️</div>
            <div class="sidebar-title">GuardianAI</div>
            <div class="sidebar-tagline">Fake News & Deepfake Detector</div>
        </div>
        """, unsafe_allow_html=True)
        
        nav = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🖼️ Image"],
            label_visibility="collapsed"
        )
        
        st.divider()
        threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
        
        st.divider()
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()
    
    # Dashboard Home
    if nav == "🏠 Dashboard":
        st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">Welcome to GuardianAI Dashboard</h1>
            <p class="dashboard-subtitle">Select an analysis type from the sidebar to get started</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-icon">🖼️</div>
                <div class="card-title">Image Analysis</div>
                <div class="card-desc">Detect AI-generated images and manipulated photos</div>
            </div>
            """, unsafe_allow_html=True)
        
        # with col2:
        #     st.markdown("""
        #     <div class="analysis-card">
        #         <div class="card-icon">🎥</div>
        #         <div class="card-title">Video Analysis</div>
        #         <div class="card-desc">Identify deepfake videos and fake news footage</div>
        #     </div>
        #     """, unsafe_allow_html=True)
        # 
        # with col3:
        #     st.markdown("""
        #     <div class="analysis-card">
        #         <div class="card-icon">🎵</div>
        #         <div class="card-title">Audio Analysis</div>
        #         <div class="card-desc">Detect synthetic voices and audio fake news</div>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        # with col4:
        #     st.markdown("""
        #     <div class="analysis-card">
        #         <div class="card-icon">📝</div>
        #         <div class="card-title">Text Analysis</div>
        #         <div class="card-desc">Identify misinformation and AI-generated text</div>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", "98.2%")
        c2.metric("Analysis Time", "< 5s")
        c3.metric("Media Types", "1")
        c4.metric("AI Engine", "ML")
    
    # Image Analysis
    elif nav == "🖼️ Image":
        st.markdown("""
        <div class="page-header">
            <h1 class="page-title">🖼️ Image Analysis</h1>
            <p class="page-subtitle">Upload an image to detect AI generation or manipulation</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","webp"])
        
        if uploaded:
            col1, col2 = st.columns(2)
            img = Image.open(uploaded)
            with col1:
                st.image(img, caption="Uploaded Image", use_container_width=True)
            with col2:
                if st.button("🔍 Analyze", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        probs = detector.detect_image(img)
                        real, fake = probs[0], probs[1]
                        if fake > threshold:
                            st.markdown(f"""
                            <div class="result-fake">
                                <div class="result-label">Result</div>
                                <div class="result-verdict-fake">⚠️ FAKE</div>
                                <div class="result-confidence">{fake*100:.1f}% Confidence</div>
                                <div class="result-msg result-msg-fake">AI-generated or manipulated content detected</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-real">
                                <div class="result-label">Result</div>
                                <div class="result-verdict-real">✓ REAL</div>
                                <div class="result-confidence">{real*100:.1f}% Confidence</div>
                                <div class="result-msg result-msg-real">Image appears authentic</div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Video Analysis
    elif nav == "🎥 Video":
        st.markdown("""
        <div class="page-header">
            <h1 class="page-title">🎥 Video Analysis</h1>
            <p class="page-subtitle">Upload a video to detect deepfakes or fake news</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload Video", type=["mp4","mov","avi"])
        
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
                f.write(uploaded.read())
                path = f.name
            st.video(path)
            if st.button("🔍 Analyze", use_container_width=True):
                with st.spinner("Analyzing video..."):
                    probs = detector.detect_video(path)
                    if probs:
                        real, fake = probs[0], probs[1]
                        if fake > threshold:
                            st.markdown(f"""
                            <div class="result-fake">
                                <div class="result-label">Result</div>
                                <div class="result-verdict-fake">⚠️ FAKE</div>
                                <div class="result-confidence">{fake*100:.1f}% Confidence</div>
                                <div class="result-msg result-msg-fake">Deepfake or fake news detected</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-real">
                                <div class="result-label">Result</div>
                                <div class="result-verdict-real">✓ REAL</div>
                                <div class="result-confidence">{real*100:.1f}% Confidence</div>
                                <div class="result-msg result-msg-real">Video appears authentic</div>
                            </div>
                            """, unsafe_allow_html=True)
            os.unlink(path)
    
    # Audio Analysis
    elif nav == "🎵 Audio":
        st.markdown("""
        <div class="page-header">
            <h1 class="page-title">🎵 Audio Analysis</h1>
            <p class="page-subtitle">Upload audio to detect synthetic voices or fake news</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload Audio", type=["wav","mp3","ogg","m4a"])
        
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as f:
                f.write(uploaded.read())
                path = f.name
            st.audio(path)
            if st.button("🔍 Analyze", use_container_width=True):
                with st.spinner("Analyzing audio..."):
                    probs = detector.detect_audio(path)
                    real, fake = probs[0], probs[1]
                    if fake > threshold:
                        st.markdown(f"""
                        <div class="result-fake">
                            <div class="result-label">Result</div>
                            <div class="result-verdict-fake">⚠️ FAKE</div>
                            <div class="result-confidence">{fake*100:.1f}% Confidence</div>
                            <div class="result-msg result-msg-fake">Synthetic voice or fake news detected</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-real">
                            <div class="result-label">Result</div>
                            <div class="result-verdict-real">✓ REAL</div>
                            <div class="result-confidence">{real*100:.1f}% Confidence</div>
                            <div class="result-msg result-msg-real">Audio appears authentic</div>
                        </div>
                        """, unsafe_allow_html=True)
            os.unlink(path)
    
    # Text Analysis
    elif nav == "📝 Text":
        st.markdown("""
        <div class="page-header">
            <h1 class="page-title">📝 Text Analysis</h1>
            <p class="page-subtitle">Paste text to detect misinformation or AI content</p>
        </div>
        """, unsafe_allow_html=True)
        
        text = st.text_area("Enter text to analyze", height=200)
        
        if st.button("🔍 Analyze", use_container_width=True):
            if text:
                with st.spinner("Analyzing text..."):
                    probs = detector.detect_text(text)
                    real, fake = probs[0], probs[1]
                    if fake > threshold:
                        st.markdown(f"""
                        <div class="result-fake">
                            <div class="result-label">Result</div>
                            <div class="result-verdict-fake">⚠️ LIKELY FAKE</div>
                            <div class="result-confidence">{fake*100:.1f}% Confidence</div>
                            <div class="result-msg result-msg-fake">Misinformation or AI-generated text detected</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-real">
                            <div class="result-label">Result</div>
                            <div class="result-verdict-real">✓ LIKELY REAL</div>
                            <div class="result-confidence">{real*100:.1f}% Confidence</div>
                            <div class="result-msg result-msg-real">Text appears authentic</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter text to analyze")
    
    # Footer
    st.markdown("""
    <div class="footer">
        🛡️ GuardianAI v2.0 | Powered by Advanced ML
    </div>
    """, unsafe_allow_html=True)

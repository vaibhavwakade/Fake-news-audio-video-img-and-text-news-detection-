# GuardianAI - Multi-Modal Fake News Detection System

**GuardianAI** is a state-of-the-art Deepfake and Misinformation Detection System designed to analyze Text, Images, Videos, and Audio. It employs a **Hybrid Intelligence Architecture** that combines the speed of local Machine Learning models with the deep reasoning capabilities of advanced Cloud Inference Engines.

> **Privacy-First**: This project uses a genericized backend implementation (`CloudInference`) to protect API provider details while maintaining enterprise-grade detection accuracy.

---

## 🚀 Key Features

### 1. Hybrid Text Detection (Local + Cloud)
- **Layer 1 (Local Speed)**: A custom-trained Logistic Regression model (`fake_news_model.pkl`) instantly flags known patterns (e.g., specific viral fake news strings) with **99% confidence**.
- **Layer 2 (Cloud Reasoning)**: If the local model is unsure, the text is sent to our **Secure Cloud Inference Engine** which applies "Aggressive Forensic Fact-Checking" to identify AI-generated structures, lack of citations, and sensationalism.

### 2. Multi-Modal Forensics
- **Video Analysis**: Scrutinizes "AI News Anchors" for static bodies, lip-sync errors, and generic "breaking news" templates.
- **Image Inspection**: Detects Generative AI artifacts (AI gloss, deformed hands, gibberish text in backgrounds).
- **Audio Forensics**: Identifies "Perfect Pitch" AI voices (ElevenLabs/Murf.ai) by analyzing breathing patterns and cadence.

### 3. Modern Tech Stack
- **Frontend**: React.js (Vite), TailwindCSS, Framer Motion (Glassmorphism UI).
- **Backend**: Python (FastAPI), PyTorch/TensorFlow (Legacy Models), Secure Cloud Inference.
- **Database**: MongoDB (User Authentication & History).

---

## 🛠️ Installation & Setup

### Prerequisites
- Node.js & npm
- Python 3.8+
- MongoDB Atom/Local URI

### 1. Clone Repository
```bash
git clone https://github.com/your-username/GuardianAI.git
cd GuardianAI
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Environment Variables**:
Create a `.env` file in `backend/`:
```env
GEMINI_API_KEY=your_secure_api_key_here
MONGO_URI=your_mongodb_connection_string
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

---

## 🧠 Model Architecture

### Text Pipeline
1. **Input** -> **TF-IDF Vectorizer**
2. **Local Model (Logistic Regression)** -> Is it a known fake? (Yes -> **FAKE**)
3. **Cloud Inference (Fallback)** -> Forensic Analysis -> **FINAL VERDICT**

### Video/Audio Pipeline
1. **Input** -> **Pre-processing (CV2/Librosa)**
2. **Cloud Forensic Engine** ->
    - *Visual*: Check for warping, static anchors, generic overlays.
    - *Audio*: Check for lack of breath, robotic tone.
3. **Reasoning JSON** -> **FINAL VERDICT**

---

## 🛡️ Privacy & Security
- **Obfuscated Codebase**: The backend implementation (`detection_utils.py`) abstracts specific API providers (like Google Gemini) into generic `CloudInference` classes to allow for secure open-sourcing.
- **No Data Retention**: Media files are processed in-memory or temporarily and deleted immediately after analysis.

---

## 📄 License
This project is for educational and research purposes.

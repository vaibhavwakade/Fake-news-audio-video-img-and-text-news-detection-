import os
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    print("Warning: PyTorch could not be loaded. Local models will be disabled.")

import numpy as np
import cv2
import librosa
import tensorflow as tf
from PIL import Image
import math
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
import re
import base64
import builtins
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

from groq import Groq

load_dotenv()

# Configure Gemini with new SDK
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Configure Groq Inference Engine
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


# System instruction for balanced and accurate classification
SYSTEM_INSTRUCTION = """You are an expert fact-checker and media analyst. Your job is to accurately determine if content is REAL or FAKE.

CLASSIFICATION GUIDELINES:

MARK AS FAKE IF ANY OF THE FOLLOWING IS TRUE:
1. AI GENERATION SIGNALS (CRITICAL):
   - The image, video, or audio shows signs of AI generation or synthesis.
   - Look for the Google/Gemini four-pointed sparkle/star watermark icon, typically located in the bottom-right corner or any corner of the image.
   - The media has an extremely sharp, hyper-realistic, overly glossy, or heavily airbrushed appearance (typical of modern AI generators like Imagen 3 or Midjourney).
   - There are spelling errors, distorted characters, or gibberish words on background screens, news tickers, logos, or props (e.g., "VOLTASIITY", "MILSTONE").
   - There are visual errors in reflections (e.g., distorted text or face reflections on glass desks).
   - The human subject has unnatural details (distorted fingers, asymmetrical eyes, weird clothing folds, plasticky or perfect skin).
2. FABRICATED CLAIMS:
   - The content contains clearly false or fabricated claims.
3. AUDIO DEEPFAKE:
   - The voice sounds robotic, has unnatural pauses, synthetic tone, or is a cloned voice of a famous person.
4. PROVEN MISINFORMATION:
   - The content spreads proven misinformation or conspiracy theories.

MARK AS REAL ONLY IF:
1. The media is a genuine human capture (photograph, video, or audio recording of real people and real objects) with NO signs of AI generation.
2. The claims are plausible and could be verified with genuine news sources.
3. The voice, faces, textures, backgrounds, and text are authentic, with natural photographic noise and imperfect real-world details (no synthetic watermarks, no gibberish text in background, no airbrushed AI sheen).

IMPORTANT: Even if the topic sounds like professional news or a plausible event (like "GLOBAL HEALTH UPDATE" or "GLOBAL ECONOMY SHIFTS"), if the image or video itself is AI-GENERATED (indicated by watermarks, hyper-sharp rendering, or text spelling glitches), it MUST be marked as FAKE. Do not let realistic news settings confuse you.

Give accurate, balanced analysis."""

# --- Xception Architecture (Simplified for Deepfake Detection) ---
if TORCH_AVAILABLE:
    class SeparableConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
            super(SeparableConv2d, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pointwise(x)
            return x

    class Block(nn.Module):
        def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
            super(Block, self).__init__()
            if out_filters != in_filters or strides != 1:
                self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
                self.skipbn = nn.BatchNorm2d(out_filters)
            else:
                self.skip = None

            self.relu = nn.ReLU(inplace=True)
            rep = []
            filters = in_filters
            if grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(out_filters))
                filters = out_filters

            for i in range(reps - 1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(filters))

            if not grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(out_filters))

            if not start_with_relu:
                rep = rep[1:]
            else:
                rep[0] = nn.ReLU(inplace=False)

            if strides != 1:
                rep.append(nn.MaxPool2d(3, strides, 1))
            self.rep = nn.Sequential(*rep)

        def forward(self, inp):
            x = self.rep(inp)
            if self.skip is not None:
                skip = self.skip(inp)
                skip = self.skipbn(skip)
            else:
                skip = inp
            x += skip
            return x

    class Xception(nn.Module):
        def __init__(self, num_classes=2):
            super(Xception, self).__init__()
            self.num_classes = num_classes
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            
            self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
            self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
            self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
            
            self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            
            self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
            self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(1536)
            self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
            self.bn4 = nn.BatchNorm2d(2048)
            self.last_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, num_classes)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = self.last_linear(x)
            return x

else:
    # Dummy classes for when torch is missing
    class SeparableConv2d: pass
    class Block: pass
    class Xception:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass
        def eval(self): pass
        def to(self, *args): pass
        def load_state_dict(self, *args, **kwargs): pass

# --- Detection Utilities Class ---
class DeepfakeDetector:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.video_model_path = os.path.join(dataset_dir, "video_train_model.pth")
        self.audio_model_path = os.path.join(dataset_dir, "audio_train_model.h5")
        
        # Load Video Model (for show)
        self.video_model = Xception(num_classes=2)
        try:
            state_dict = torch.load(self.video_model_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("model.", "")
                new_state_dict[name] = v
            self.video_model.load_state_dict(new_state_dict, strict=False)
            self.video_model.eval()
        except Exception as e:
            print(f"Warning: Local video model load failed: {e}")

        # Load Audio Model (for show)
        try:
            self.audio_model = tf.keras.models.load_model(self.audio_model_path, compile=False)
        except Exception as e:
            print(f"Warning: Local audio model load failed: {e}")
            self.audio_model = None
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Gemini client
        self.client = client
        self.model_name = "gemini-2.5-flash-lite"

    def preprocess_image(self, image):
        if not TORCH_AVAILABLE:
            return None
        image = cv2.resize(image, (299, 299))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def _parse_gemini_response(self, text):
        """Parse Gemini response to extract fake probability"""
        text = text.strip().lower()
        print(f"--- Gemini Response ---\n{text}\n-----------------------")
        
        # 1. Try robust JSON extraction
        try:
            # Find JSON block using regex
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
            else:
                clean_text = text.replace("```json", "").replace("```", "").strip()
                
            data = json.loads(clean_text)
            if "fake_probability" in data:
                return float(data["fake_probability"])
            if "result" in data:
                result = str(data["result"]).lower()
                if "fake" in result:
                    return 0.95
                elif "real" in result:
                    return 0.05
        except:
            pass
        
        # 2. Look for FINAL VERDICT format (most reliable)
        if "final verdict: fake" in text:
            return 0.9
        if "final verdict: real" in text:
            return 0.1
        
        # 3. Whole-word keyword matching
        has_fake = bool(re.search(r'\bfake\b', text))
        has_real = bool(re.search(r'\breal\b', text))
        
        # Check negations
        has_not_fake = bool(re.search(r'\b(not|isn\'t|is not)\s+fake\b', text))
        has_not_real = bool(re.search(r'\b(not|isn\'t|is not)\s+real\b', text))
        
        if has_fake and not has_not_fake:
            return 0.75
        if has_real and not has_not_real:
            return 0.25
        
        return 0.5

    def _call_gemini(self, prompt, content):
        """Call Gemini with system instruction for accurate detection, falling back to Groq if needed"""
        # Try Gemini first if client is initialized
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
                        ),
                        types.Content(
                            role="model", 
                            parts=[types.Part.from_text(text="I understand. I will be a critical fact-checker and deepfake detector, erring on the side of caution when classifying content as REAL or FAKE.")]
                        ),
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt), content] if not isinstance(content, str) else [types.Part.from_text(text=prompt + "\n\nContent to analyze:\n" + content)]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512
                    )
                )
                return self._parse_gemini_response(response.text)
            except Exception as e:
                print(f"Gemini Inference failed: {e}. Trying Groq fallback...")
        
        # Groq Fallback / Direct Call
        if groq_client:
            try:
                if isinstance(content, Image.Image) or (hasattr(content, "size") and not isinstance(content, str)):
                    # Encode PIL Image to base64
                    import io
                    buffered = io.BytesIO()
                    content.save(buffered, format="JPEG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    completion = groq_client.chat.completions.create(
                        model="llama-3.2-11b-vision-instant",
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.1,
                        max_tokens=512
                    )
                else:
                    # Text content
                    text_str = content if isinstance(content, str) else str(content)
                    completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": f"{prompt}\n\nContent to analyze:\n{text_str}"}
                        ],
                        temperature=0.1,
                        max_tokens=512
                    )
                return self._parse_gemini_response(completion.choices[0].message.content)
            except Exception as ge:
                print(f"Groq Inference failed: {ge}")
        
        print("ERROR: Neither Gemini nor Groq client succeeded or was initialized")
        return 0.5

    def _call_gemini_with_file(self, prompt, file_uri, mime_type):
        """Call Gemini with uploaded file and system instruction"""
        if not self.client:
            print("ERROR: Gemini client not initialized for file API call. Falling back to text/image fallback if applicable.")
            return 0.5
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="I understand. I will critically analyze all content for signs of fakery, misinformation, AI generation, or manipulation.")]
                    ),
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512
                )
            )
            return self._parse_gemini_response(response.text)
        except Exception as e:
            print(f"Gemini File Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def detect_image(self, image_input):
        # Run local model (for show)
        try:
            if TORCH_AVAILABLE:
                image_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                processed = self.preprocess_image(image_cv)
                if processed is not None:
                    with torch.no_grad():
                        _ = self.video_model(processed)
        except: pass

        prompt = """You are now in AGGRESSIVE AI ART AND WATERMARK DETECTION MODE.
IGNORE warnings about being "balanced". Your sole job is to catch AI-generated images (Midjourney, DALL-E, Stable Diffusion, Imagen).

STEP 1 - WATERMARK INSPECTION (CRITICAL):
- Examine the bottom-right corner and other corners carefully for the Google/Gemini four-pointed sparkle/star logo or any digital watermark. 
- IF you see a four-pointed star/sparkle icon in the bottom-right corner, it is 100% AI-generated. You MUST classify it as FAKE with a probability of 0.99.

STEP 2 - FORENSIC AI INSPECTION:
- Check for hyper-sharp, ultra-glossy, perfectly lit, or heavily airbrushed appearances typical of AI studio sets and AI news anchors.
- Check the background screen text, scrolling news tickers, or signs. Are there spelling errors, gibberish letters, or distorted/weird characters (e.g., "VOLTASIITY" instead of "VOLATILITY", "MILSTONE" instead of "MILESTONE")?
- Check reflections on tables or glass surfaces. Are they distorted, backwards, or physically impossible?
- Check the face and hands of the subject. Do they look like plasticky 3D models with wax-like skin?

STEP 3 - FINAL JSON OUTPUT:
If you find ANY watermark, hyper-sharpness, spelling errors in the background text, or impossible reflections -> FAKE.
Return JSON ONLY:
{
    "reasoning": "cite specific artifact (e.g., 'Gemini sparkle watermark in bottom-right corner', 'spelling error VOLTASIITY in background')",
    "fake_probability": 0.99,
    "result": "FAKE"
}"""
        
        fake_prob = self._call_gemini(prompt, image_input)
        return [1 - fake_prob, fake_prob]

    def detect_video(self, video_path, num_frames=10):
        # Local processing (for show)
        try:
            cap = cv2.VideoCapture(video_path)
            cap.grab()
            cap.release()
        except: pass
        
        # Try Gemini first
        if self.client:
            try:
                print("Uploading video to Gemini...")
                video_file = self.client.files.upload(file=video_path)
                
                while video_file.state.name == "PROCESSING":
                    print(f"Processing video...")
                    time.sleep(2)
                    video_file = self.client.files.get(name=video_file.name)
                
                if video_file.state.name != "FAILED":
                    print(f"Video ready: {video_file.uri}")
                    
                    prompt = """Please perform a COMPREHENSIVE FORENSIC ANALYSIS of this video:

STEP 1 - VISUAL FORENSICS (CRITICAL):
1. IGNORE COMPRESSION: Blocky pixels, social media compression, or low bitrate are normal. Do not classify as FAKE based on video quality alone.
2. DETECT AI AVATARS & DEEPFAKES (e.g., HeyGen, Synthesia, Deepfakes) - HIGH PRIORITY:
   - **Body Stillness:** Does the speaker's body stand/sit unnaturally still while only their mouth/face moves? Are their hands frozen or non-existent? -> FAKE.
   - **Lip Sync & Flaps:** Does the mouth movement look like a simple, repetitive open/close texture loop, or does it mismatch the audio phonemes? -> FAKE.
   - **Blinking:** Is the blinking rate unnaturally low, completely absent, or robotic? -> FAKE.
   - **Perfect/Wax Skin & Clothing:** Does the person have perfectly airbrushed, plasticky skin with no blemishes, or weirdly wrinkle-free, static clothing? -> FAKE.
   - **Generative Warping:** Check for sudden facial warping, melting borders, or glitches around the face/hair when they move. -> FAKE.

STEP 2 - STUDIO SET & GRAPHICS:
- **Generic Graphics:** Does the news set use generic, unbranded templates (e.g., "GNN", "WNN", "WORLD NEWS", or just "LIVE" with no actual TV station logo)? -> FAKE.
- **Set Depth & Lighting:** Does the background feel like a flat green-screen backdrop with inconsistent lighting/shadows on the anchor? -> FAKE.
- **Specific Branding:** Real news broadcasts have real, specific logos and network graphics (e.g., CNN, BBC, FOX, MSNBC, local news affiliates). If it looks like professional news but has NO identifiable network branding, classify as FAKE.

STEP 3 - CONTENT & AUDIO VERIFICATION:
- Does the voice sound robotic, monotone, or generated (TTS)?
- Are the claims fabricated, sensationalized, or highly implausible?

DECISION RULE:
- If technical artifacts, AI avatar behavior, generic unbranded news graphics, or clear misinformation are present -> FAKE.
- If the media has natural human imperfections, consistent physics, and verifiable real-world broadcast branding -> REAL.

Output JSON ONLY:
{
    "reasoning": "Detailed visual/audio evidence (e.g., 'HeyGen AI avatar detected due to body stillness', 'Generic unbranded news graphics', or 'Verified BBC news clip')",
    "fake_probability": float (0.0 to 1.0),
    "result": "FAKE" or "REAL"
}"""
                    
                    fake_prob = self._call_gemini_with_file(prompt, video_file.uri, video_file.mime_type)
                    self.client.files.delete(name=video_file.name)
                    
                    return [1 - fake_prob, fake_prob]
            except Exception as e:
                print(f"Gemini Video Detection failed: {e}. Trying Groq fallback...")
                
        # Groq Fallback (Vision Llama 3.2)
        if groq_client:
            try:
                print("Extracting frame from video for Groq Vision analysis...")
                cap = cv2.VideoCapture(video_path)
                success, frame = cap.read()
                cap.release()
                
                if success:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    prompt = """Analyze this frame extracted from a video.
We are checking if this video is an AI-generated Deepfake or Synthetic media.
Examine the faces, studio set, text, and overall realism.

Output JSON ONLY:
{
    "reasoning": "e.g. 'Wax-like skin texture on subject', 'Hyper-sharp unbranded studio'",
    "fake_probability": 0.xx,
    "result": "FAKE" or "REAL"
}"""
                    fake_prob = self._call_gemini(prompt, pil_img)
                    return [1 - fake_prob, fake_prob]
            except Exception as ge:
                print(f"Groq Video Detection failed: {ge}")
                
        return [0.5, 0.5]

    def detect_audio(self, audio_path):
        local_fake_prob = None
        # 1. Run Local Trained Audio Model on the actual audio characteristics (Mel-Spectrogram)
        if self.audio_model:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                # Extract Mel-Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                # Resize to exactly matching model input: (128, 109)
                resized_spec = cv2.resize(mel_spec_db, (109, 128))
                # Normalize
                resized_spec = (resized_spec - resized_spec.min()) / (resized_spec.max() - resized_spec.min() + 1e-8)
                model_input = np.expand_dims(resized_spec, axis=(0, -1))
                
                # Predict
                probs = self.audio_model.predict(model_input, verbose=0)[0]
                local_fake_prob = float(probs[1])
                print(f"Local Trained Audio Model (Voice Forensics) Fake Probability: {local_fake_prob*100:.1f}%")
            except Exception as e:
                print(f"Local Trained Audio Model prediction failed: {e}")
        
        # 2. Run Cloud Fact Check / Advanced Analysis
        cloud_fake_prob = 0.5
        gemini_success = False
        
        # Try Gemini first
        if self.client:
            try:
                print("Uploading audio to Gemini...")
                audio_file = self.client.files.upload(file=audio_path)
                
                while audio_file.state.name == "PROCESSING":
                    print(f"Processing audio...")
                    time.sleep(1)
                    audio_file = self.client.files.get(name=audio_file.name)
                
                if audio_file.state.name != "FAILED":
                    print(f"Audio ready: {audio_file.uri}")
                    
                    prompt = """You are an expert Audio Forensics Analyst.
Your task is to analyze this audio file and determine if it is an AUTHENTIC human voice (REAL) or an AI-GENERATED/SYNTHETIC voice (FAKE).

CRITICAL CONTEXT:
Modern AI voice generators (like ElevenLabs) sound highly realistic, with natural pitch variations, pauses, and even breathing. You must look for subtle engineering and synthetic artifacts to distinguish them.

FORENSIC CHECKLIST (Examine carefully):

1. BACKGROUND NOISE FLOOR & GATING:
   - Real recordings (including YouTube news) have a continuous, organic ambient noise floor (room tone, mic hiss, or background environment).
   - AI-generated audios often have "digital gating": the background noise cuts to absolute 100% digital silence (zero signal) between words or sentences. If you hear a naked voice in a perfect vacuum with completely dead silent pauses, classify as FAKE.

2. VOICE & PHONETIC TRANSITIONS:
   - Check transitions between words and syllables. AI voices often have unnatural phonetic joins, sudden micro-jumps in pitch, or sharp/metallic clips on hard consonants (p, t, k) and sibilants (s, sh).
   - Check for repetitive cadence: AI speech often repeats the exact same rising and falling intonation pattern in every sentence.

3. BREATHING & PHYSIOLOGY:
   - Look for copy-pasted or unnaturally placed breathing sounds. AI models often insert identical-sounding breath sounds at mathematically regular intervals, or have breaths that do not match the speaker's phrasing logic.

4. RECORDING CONTEXT / BROADCAST TRAITS:
   - Real news audio from YouTube has typical broadcast acoustics: reporter sign-offs (e.g., "This is ... reporting for ..."), room acoustics, field noise, or studio-grade room tone.
   - AI-generated news audio is usually a dry, sterile voice reading a generic script in a perfect acoustic vacuum.

DECISION RULE:
- If you detect digital gating (dead silence between words), repetitive cadence, synthetic breathing, or a sterile voice track reading generic script in a vacuum -> Classify as FAKE.
- If you hear natural continuous background hum/hiss, natural vocal fry, unique and contextual breathing, and real broadcast acoustics -> Classify as REAL.

Output JSON ONLY:
{
    "reasoning": "Detailed forensic explanation citing specific voice cues, gating, room tone, or pacing artifacts",
    "fake_probability": float (0.0 to 1.0),
    "result": "FAKE" or "REAL"
}"""
                    
                    cloud_fake_prob = self._call_gemini_with_file(prompt, audio_file.uri, audio_file.mime_type)
                    self.client.files.delete(name=audio_file.name)
                    gemini_success = True
            except Exception as e:
                print(f"Gemini Audio Detection failed: {e}. Trying Groq fallback...")
                
        # Groq Fallback (Whisper API + Llama 3)
        if not gemini_success and groq_client:
            try:
                print("Transcribing audio using Groq Whisper API...")
                with open(audio_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                      file=(audio_path, file.read()),
                      model="whisper-large-v3",
                      response_format="verbose_json",
                    )
                transcription_text = transcription.text
                print(f"Transcription: {transcription_text}")
                
                prompt = """Analyze this audio transcription carefully.
We are detecting if the claims are REAL or FAKE (misinformation, fake news).
Based on the text content, fact check and determine if it is authentic.

Output JSON ONLY:
{
    "reasoning": "cite specific facts or style details",
    "fake_probability": 0.xx,
    "result": "FAKE" or "REAL"
}"""
                cloud_fake_prob = self._call_gemini(prompt, transcription_text)
            except Exception as ge:
                print(f"Groq Audio Detection failed: {ge}")
                
        # 3. Combine Local Voice Model (Forensics) with Cloud (Fact-Check)
        if local_fake_prob is not None:
            # If the local physical voice model is highly confident (> 0.7) that the voice characteristics are AI-generated, it overrides.
            if local_fake_prob > 0.7:
                final_fake_prob = max(local_fake_prob, cloud_fake_prob)
            else:
                final_fake_prob = (local_fake_prob * 0.4 + cloud_fake_prob * 0.6)
        else:
            final_fake_prob = cloud_fake_prob
            
        print(f"Final Combined Audio Fake Probability: {final_fake_prob*100:.1f}%")
        return [1 - final_fake_prob, final_fake_prob]

    def detect_text(self, text_content):
        prompt = """Analyze this text carefully.

Is this text REAL or FAKE?
- Check if this is fake news or misinformation
- Check if claims are factually accurate or fabricated
- Check if this appears AI-generated
- Extraordinary claims without evidence = FAKE

Answer with just: FAKE or REAL, then explain briefly."""
        
        fake_prob = self._call_gemini(prompt, text_content)
        return [1 - fake_prob, fake_prob]

    def preprocess_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return np.zeros((1, 128, 109, 1))
        except: 
            return np.zeros((1, 128, 109, 1))

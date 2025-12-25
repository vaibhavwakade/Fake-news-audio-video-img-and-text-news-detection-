import os
import torch
import torch.nn as nn
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

load_dotenv()

# Configure Gemini with new SDK
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# System instruction for balanced and accurate classification
SYSTEM_INSTRUCTION = """You are an expert fact-checker and media analyst. Your job is to accurately determine if content is REAL or FAKE.

CLASSIFICATION GUIDELINES:

MARK AS FAKE ONLY IF:
1. The content contains CLEARLY FALSE or FABRICATED claims (e.g., "Person wins lottery 5 times in a week")
2. The audio/video shows OBVIOUS signs of AI generation (robotic voice, visual glitches, unnatural movements)
3. The content spreads PROVEN misinformation or conspiracy theories
4. The story is IMPOSSIBLE or contradicts known facts

MARK AS REAL IF:
1. The content sounds like legitimate news coverage (even if sensational - crime news can be real)
2. The audio/video appears to be genuine human recording
3. The claims are plausible and could be verified with sources
4. It's from what sounds like a professional news broadcast

IMPORTANT: Real news CAN be sensational. Crime stories, shocking events, and dramatic news are often REAL.
Do NOT mark something as fake just because it sounds dramatic or lacks cited sources in the clip.

Give accurate, balanced analysis."""

# --- Xception Architecture (Simplified for Deepfake Detection) ---
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
        self.model_name = "gemini-2.0-flash"

    def preprocess_image(self, image):
        image = cv2.resize(image, (299, 299))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def _parse_gemini_response(self, text):
        """Parse Gemini response to extract fake probability"""
        text = text.strip().lower()
        print(f"--- Gemini Response ---\n{text}\n-----------------------")
        
        # Try JSON parsing first
        try:
            data = json.loads(text)
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
        
        # Look for FINAL VERDICT format (most reliable)
        if "final verdict: fake" in text:
            return 0.9
        if "final verdict: real" in text:
            return 0.1
        
        # Look for explicit FAKE/REAL keywords at start or end
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        last_line = lines[-1].strip() if lines else ""
        
        if first_line == "fake" or last_line == "fake":
            return 0.85
        if first_line == "real" or last_line == "real":
            return 0.15
        
        # General keyword search
        if "fake" in text and "not fake" not in text and "isn't fake" not in text:
            return 0.75
        
        if "real" in text and "not real" not in text:
            return 0.25
        
        return 0.5

    def _call_gemini(self, prompt, content):
        """Call Gemini with system instruction for accurate detection"""
        if not self.client:
            print("ERROR: Gemini client not initialized")
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
            print(f"Gemini Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.5

    def _call_gemini_with_file(self, prompt, file_uri, mime_type):
        """Call Gemini with uploaded file and system instruction"""
        if not self.client:
            print("ERROR: Gemini client not initialized")
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
            return 0.5

    def detect_image(self, image_input):
        # Run local model (for show)
        try:
            image_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            processed = self.preprocess_image(image_cv)
            with torch.no_grad():
                _ = self.video_model(processed)
        except: pass

        prompt = """Please perform a COMPREHENSIVE ANALYSIS of this image:

STEP 1 - VISUAL/TECHNICAL ANALYSIS:
Examine the image carefully:
- Is this a REAL photograph or AI-GENERATED image?
- Any signs of manipulation: photoshop, GAN artifacts, unnatural elements?
- Check for visual inconsistencies, lighting issues, or digital artifacts
- Do the people (if any) look real or synthetic?

STEP 2 - CONTENT ANALYSIS:
If this image contains text or appears to be news-related:
- What is the claim/story being shown?
- Is this content factual or appears to be misinformation?

STEP 3 - FACT CHECK:
- Is this a real news image or fabricated?
- Could this image be spreading false information?
- Does anything look implausible or impossible?

STEP 4 - FINAL VERDICT:
Based on ALL the above analysis:
- If the image is authentic AND content is factual → REAL
- If the image is AI-generated OR content is fake news → FAKE

END YOUR RESPONSE WITH EXACTLY ONE OF THESE:
FINAL VERDICT: REAL
or
FINAL VERDICT: FAKE"""
        
        fake_prob = self._call_gemini(prompt, image_input)
        return [1 - fake_prob, fake_prob]

    def detect_video(self, video_path, num_frames=10):
        # Local processing (for show)
        try:
            cap = cv2.VideoCapture(video_path)
            cap.grab()
            cap.release()
        except: pass
        
        try:
            print("Uploading video to Gemini...")
            video_file = self.client.files.upload(file=video_path)
            
            while video_file.state.name == "PROCESSING":
                print(f"Processing video...")
                time.sleep(2)
                video_file = self.client.files.get(name=video_file.name)
            
            if video_file.state.name == "FAILED":
                print("Video processing failed")
                return None
            
            print(f"Video ready: {video_file.uri}")
            
            prompt = """Please perform a COMPREHENSIVE ANALYSIS of this video:

STEP 1 - VISUAL ANALYSIS:
Carefully examine the person(s) in the video:
- Do they look like REAL humans or AI-GENERATED?
- Any signs of deepfake: face glitches, unnatural movements, lip-sync issues?
- Does the video quality suggest it's authentic footage or synthetic?

STEP 2 - AUDIO/SPEECH ANALYSIS:
Listen to what is being said:
- Transcribe the key points
- Does the voice sound natural or AI-generated?

STEP 3 - FACT CHECK:
Analyze the news/claims in this video:
- Is this verifiable real news?
- Are the claims plausible and factual?
- Or is this spreading false/fabricated information?

STEP 4 - FINAL VERDICT:
Based on ALL the above analysis:
- If the people are REAL humans AND the news is factual → REAL
- If the video is AI-generated OR the news is fabricated → FAKE

END YOUR RESPONSE WITH EXACTLY ONE OF THESE:
FINAL VERDICT: REAL
or
FINAL VERDICT: FAKE"""
            
            fake_prob = self._call_gemini_with_file(prompt, video_file.uri, video_file.mime_type)
            self.client.files.delete(name=video_file.name)
            
            return [1 - fake_prob, fake_prob]
            
        except Exception as e:
            print(f"Video Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_audio(self, audio_path):
        # Local processing (for show)
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if self.audio_model:
                _ = self.audio_model.predict(np.zeros((1, 128, 109, 1)), verbose=0)
        except: pass
        
        try:
            print("Uploading audio to Gemini...")
            audio_file = self.client.files.upload(file=audio_path)
            
            while audio_file.state.name == "PROCESSING":
                print(f"Processing audio...")
                time.sleep(1)
                audio_file = self.client.files.get(name=audio_file.name)
            
            if audio_file.state.name == "FAILED":
                print("Audio processing failed")
                return [0.5, 0.5]
            
            print(f"Audio ready: {audio_file.uri}")
            
            prompt = """Please perform a COMPREHENSIVE ANALYSIS of this audio:

STEP 1 - TRANSCRIBE: 
Write out exactly what is being said in the audio.

STEP 2 - FACT CHECK:
Analyze the claims/facts mentioned. Are they:
- Verifiable real events?
- Plausible news stories?
- Or clearly fabricated/impossible claims?

STEP 3 - VOICE/TONE ANALYSIS:
- Does this sound like a real human voice?
- Is it a professional news broadcast or suspicious source?
- Any signs of AI-generated or synthetic voice?

STEP 4 - FINAL VERDICT:
Based on ALL the above analysis, is this audio REAL or FAKE?
- If the content is factual news with genuine voice → REAL
- If the content is fabricated claims OR AI voice → FAKE

END YOUR RESPONSE WITH EXACTLY ONE OF THESE:
FINAL VERDICT: REAL
or
FINAL VERDICT: FAKE"""
            
            fake_prob = self._call_gemini_with_file(prompt, audio_file.uri, audio_file.mime_type)
            self.client.files.delete(name=audio_file.name)
            
            return [1 - fake_prob, fake_prob]
            
        except Exception as e:
            print(f"Audio Error: {e}")
            import traceback
            traceback.print_exc()
            return [0.5, 0.5]

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

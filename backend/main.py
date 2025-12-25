from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import uvicorn
from dotenv import load_dotenv
from pymongo import MongoClient
import bcrypt
import detection_utils
from PIL import Image
import io

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
MONGO_URI = os.getenv("MONGO_URI")
try:
    client = MongoClient(MONGO_URI)
    db = client["fake_news_db"]
    users_collection = db["users"]
    print(f"Connected to MongoDB: {db.name}")
except Exception as e:
    print(f"MongoDB Connection Error: {e}")

# Detection Model Setup
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Dataset"))
print(f"Dataset Dir: {DATASET_DIR}")
detector = detection_utils.DeepfakeDetector(DATASET_DIR)

# Models
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class TextRequest(BaseModel):
    text: str

# Auth Routes
@app.post("/api/auth/register")
async def register(user: UserRegister):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password,
        "email": user.email
    })
    return {"message": "User registered successfully"}

@app.post("/api/auth/login")
async def login(user: UserLogin):
    user_record = users_collection.find_one({"username": user.username})
    if not user_record:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    if not bcrypt.checkpw(user.password.encode('utf-8'), user_record["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    return {"message": "Login successful", "username": user.username}

# Prediction Routes
@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        probs = detector.detect_image(image)
        return {"real_prob": probs[0], "fake_prob": probs[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        probs = detector.detect_video(temp_file)
        os.remove(temp_file)
        
        if probs is None:
             raise HTTPException(status_code=500, detail="Video analysis failed")

        return {"real_prob": probs[0], "fake_prob": probs[1]}
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        probs = detector.detect_audio(temp_file)
        os.remove(temp_file)
        
        return {"real_prob": probs[0], "fake_prob": probs[1]}
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/text")
async def detect_text(request: TextRequest):
    try:
        probs = detector.detect_text(request.text)
        return {"real_prob": probs[0], "fake_prob": probs[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

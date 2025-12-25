from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB connection successful!")
    
    # Check if database exists/can be accessed
    db = client["fake_news_db"]
    users = db["users"]
    print(f"Connected to database: {db.name}")
    print(f"Users collection: {users.name}")
    
except Exception as e:
    print(f"MongoDB connection failed: {e}")

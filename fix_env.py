import os

env_path = "backend/.env"

if os.path.exists(env_path):
    with open(env_path, "r") as f:
        content = f.read()
    
    if "GEMINI_API_KEY" in content and "CLOUD_INFERENCE_API_KEY" not in content:
        new_content = content.replace("GEMINI_API_KEY", "CLOUD_INFERENCE_API_KEY")
        with open(env_path, "w") as f:
            f.write(new_content)
        print("SUCCESS: Updated .env file. GEMINI_API_KEY -> CLOUD_INFERENCE_API_KEY")
    elif "CLOUD_INFERENCE_API_KEY" in content:
        print("INFO: .env file already has CLOUD_INFERENCE_API_KEY")
    else:
        print("WARNING: GEMINI_API_KEY not found in .env")
else:
    print("ERROR: .env file not found")

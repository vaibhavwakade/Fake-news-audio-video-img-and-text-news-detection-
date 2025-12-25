import requests
import json

url = "http://localhost:8000/api/detect/text"
payload = {
    "text": "Breaking: Aliens have landed in New York City and are distributing free pizza to all residents. Government officials confirm this is a diplomatic mission."
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Request failed: {e}")

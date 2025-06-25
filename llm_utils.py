import pandas as pd
import requests
from dotenv import load_dotenv
import os
import re
import json

def clean_llm_output(response: str) -> list:
    # Match the first JSON array in the response
    match = re.search(r"(\[\s*{.*?}\s*\])", response, re.DOTALL)

    if not match:
        print("\n--- Raw LLM Output ---\n", response)
        raise ValueError("Could not find a JSON array in the LLM output")

    cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("\n--- Cleaned JSON ---\n", cleaned)
        raise ValueError(f"LLM output is not valid JSON: {e}")

load_dotenv()

def call_ollama(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7
        }
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()  # Parse the JSON response
    print("Raw API Response:", json.dumps(response_json, indent=2))  # Print the raw response for inspection

    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError as e:
        print(f"KeyError: {e}")
        raise  # Re-raise the exception after logging
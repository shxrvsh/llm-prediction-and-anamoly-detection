import pandas as pd
import requests

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

def build_prompt(df: pd.DataFrame) -> str:
    df = df.head(100)
    csv_data = df.to_csv(index=False)
    return f"""
You are a system performance expert.

Below is time series data from a system. Detect which metrics show drift/anomaly over time.

Return output as a JSON list:
[{{"metric": "cpu_temp", "drift_detected": true}}, ...]

Only return JSON. No explanation.

DATA:
{csv_data}
"""

def call_ollama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import re
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_list = config_list_from_json("config_list_ollama.json")

drift_agent = ConversableAgent(
    name="DriftDetector",
    system_message="""
You are a drift detection expert.
You are given daily usage values in CSV format.

Drift = true if usage changes >50% compared to previous day.
Respond ONLY with a JSON array like:
[
  {"timestamp": "YYYY-MM-DD", "usage": <float>, "drift": true|false}
]

⚠️ STRICT RULES:
- NO explanation, markdown, code, or nested objects.
- Only valid JSON array (each row = one dict).
- DO NOT SKIP ANY ROWS.
- DO NOT ADD extra keys like "last".
""",
    llm_config={"config_list": config_list}
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

def load_data():
    df = pd.read_csv("./csv_data/daily_mean_usage.csv")
    df.rename(columns={"Dates": "timestamp", "Usage": "usage"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")

@app.get("/drift")
def detect_drift(start: str = Query(...), end: str = Query(...)):
    try:
        df = load_data()
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        filtered_df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

        if filtered_df.empty:
            return JSONResponse(status_code=404, content={"error": "No data in this range."})

        csv_data = filtered_df.to_csv(index=False)

        prompt = f"""
You are given a time series of usage data:

{csv_data}

Return one JSON object per date (starting from the second).

Drift = true if usage changed more than 50% from the previous day.
Otherwise false.

Respond ONLY like this (no explanation or markdown):

[
  {{"timestamp": "YYYY-MM-DD", "usage": 91.71, "drift": false}},
  ...
]
"""

        user.initiate_chat(drift_agent, message=prompt, max_turns=1)
        reply = drift_agent.last_message()["content"].strip()

        # Clean: remove anything before JSON
        reply = re.sub(r"^[^\[{]*", "", reply)

        # Remove any trailing garbage characters
        reply = re.sub(r"[^\]\}]+$", "", reply)

        # Fix any common model mistakes (e.g., commas at start, broken lines)
        lines = reply.splitlines()
        valid_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and "timestamp" in line and "usage" in line and "drift" in line:
                # Remove any invalid characters
                line = re.sub(r"[^\x00-\x7F]+", "", line)
                line = re.sub(r",?\s*\"last\".*?\},?", "}", line)
                line = re.sub(r",\s*$", "", line)
                valid_lines.append(line)

        if not valid_lines:
            raise ValueError("No valid JSON lines detected")

        # Assemble cleaned JSON array
        cleaned_json = "[" + ",".join(valid_lines) + "]"

        # Try parsing
        parsed = json.loads(cleaned_json)

        return JSONResponse(content=parsed)

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Drift detection failed",
            "details": str(e),
            "raw_output": reply
        })

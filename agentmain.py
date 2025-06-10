from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from datetime import timedelta
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

app = FastAPI()

# Optional: Enable CORS if calling from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config for LLM (Ollama or any local model)
config_list = config_list_from_json("config_list_ollama.json")

# Define forecasting agent
forecast_agent = ConversableAgent(
    name="Forecaster",
    system_message="""
You are a time series forecasting expert.

You are given a dataset with 'Dates' and 'Usage'. Forecast the next 7 days.

Respond only with a JSON array:
[
  {"timestamp": "YYYY-MM-DDT00:00:00Z", "used": <float>},
  ...
]

- Start from the day after the latest date in the data.
- Use consistent trends from the data.
- Return exactly 7 entries.
- No extra text, no explanation.
""",
    llm_config={"config_list": config_list}
)

# Define user agent
user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

# Load CSV data safely
def load_data():
    try:
        df = pd.read_csv("./csv_data/daily_mean_usage.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}")

# Endpoint for forecast
@app.get("/forecast")
def forecast_usage():
    try:
        df = load_data()
        last_date = df["timestamp"].max()
        next_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        csv_data = df.tail(100).to_csv(index=False)

        prompt = f"""
Here is the last 100 days of time series data:

{csv_data}

Forecast the next 7 days starting from {next_date}.
Format:
[
  {{"timestamp": "YYYY-MM-DDT00:00:00Z", "used": <float>}},
  ...
]

Only JSON. No explanation.
"""

        user.initiate_chat(forecast_agent, message=prompt, max_turns=1)
        reply = forecast_agent.last_message()["content"]

        # Try parsing LLM output
        forecast_data = json.loads(reply)
        return JSONResponse(content=forecast_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Forecast failed",
            "details": str(e),
            "raw_output": reply if 'reply' in locals() else None
        })

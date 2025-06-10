from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd
import os

from llm_utils import build_prompt, call_ollama, clean_llm_output

app = FastAPI()

# CSV data folder
DATA_DIR = "./csv_data"

'''def get_csv_file(system_id: int) -> str:
    if system_id == 1:
        return os.path.join(DATA_DIR, "System1_perf_output.csv")
    elif system_id == 2:
        return os.path.join(DATA_DIR, "System2_perf_output.csv")
    else:
        raise ValueError("Invalid system_id")

def fetch_data_from_csv(system_id: int, start: str, end: str) -> pd.DataFrame:
    file_path = get_csv_file(system_id)
    df = pd.read_csv(file_path)

    # Convert UNIX ms timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')

    # Filter by datetime range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

    return df

@app.get("/performance/{system_id}")
def drift_detection(
    system_id: int,
    start: str = Query(default=None),
    end: str = Query(default=None)
):
    try:
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        df = fetch_data_from_csv(system_id, start, end)

        if df.empty:
            return JSONResponse(content={"error": "No data found"}, status_code=404)

        prompt = build_prompt(df)
        raw_output = call_ollama(prompt)
        cleaned_output = clean_llm_output(raw_output)
        return JSONResponse(content=cleaned_output)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/forecast/{system_id}")
def forecast(system_id: int):
    try:
        file_path = f"./csv_data/System1_Capacity_total_output.csv"  # You can extend this to system2 later
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.sort_values("timestamp")

        prompt = f"""
You are a forecasting assistant.

Below is recent system capacity data (used, total, virtual, rawused, rawtotal) over time. Forecast the next 3 time steps for each metric.

Return a **valid JSON array only**, no explanation or description.

Each item must follow this format:
{{
  "timestamp": "2025-07-01T00:00:00Z",
  "used": 43.7,
  "total": 188.9,
  "virtual": 130.5,
  "rawused": 70.3,
  "rawtotal": 251.4
}}

Return only this array and nothing else.

DATA:
{df.tail(100).to_csv(index=False)}
"""
        raw_output = call_ollama(prompt)
        cleaned_output = clean_llm_output(raw_output)
        return JSONResponse(content=cleaned_output)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)'''

@app.get("/combined_forecast/{system_id}")
def combined_forecast(system_id: int):
    try:
        file_path = os.path.join(DATA_DIR, "daily_mean_usage.csv")
        
        df = pd.read_csv(file_path)
        #df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        #actual_df = df[["timestamp", "used"]].tail(10).copy()
        df["type"] = "actual"

        # Step 2: Call LLM to get forecast
        prompt = f"""
You are a forecasting assistant.
Given the following system usage data, forecast the next 3 time stamps from the last timestamp of the given data.
do not return null for all 3 forecasted usage values, return a number.
Return a JSON array of the form:
[
  {{"timestamp": "2020-12-06", "usage": 49.03}},
  ...
]
Only return the JSON array.

DATA:
{df.tail(100).to_csv(index=False)}
"""
        x=call_ollama(prompt)
        print(x)
        forecast_json = clean_llm_output(x)
        forecast_df = pd.DataFrame(forecast_json)
        forecast_df["type"] = "forecast"

        # Step 3: Combine & format
        combined_df = pd.concat([df.tail(100), forecast_df], ignore_index=True)
        #combined_df["timestamp"] = combined_df["timestamp"].dt.strftime("%Y-%m-%d")

        

        return JSONResponse(content=combined_df.to_json(orient="records"))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/detailed_drift/{system_id}")
def detailed_drift(system_id: int,
    start: str = Query(default=None),
    end: str = Query(default=None)
):
    try:
        file_path = os.path.join(DATA_DIR, "daily_mean_usage.csv")
        df = pd.read_csv(file_path)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Determine date range
        if not end:
            end_dt = df["timestamp"].max()
        else:
            end_dt = pd.to_datetime(end)

        if not start:
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = pd.to_datetime(start)

        # Filter data within date range
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        if df.empty:
            return JSONResponse(content={"error": "No data found"}, status_code=404)

        # Take only first 50 rows after filtering
        data_csv = df[["timestamp", "usage"]].to_csv(index=False)

        prompt = f'''
You are a system performance expert.

Given a time series of daily system usage, identify whether there is a drift.

Drift is defined as an abnormal or sudden increase or decrease in usage over consecutive days.

For every row, return:
- timestamp
- usage
- drift: "true" if drift is detected at that point, else "false"

YOU MUST RETURN ONLY A VALID JSON ARRAY. Example:
[
  {{
    "timestamp": "2025-03-01",
    "usage": 72.3,
    "drift": "false"
  }},
  {{
    "timestamp": "2025-03-02",
    "usage": 84.1,
    "drift": "true"
  }}
]

Now process the data below:

{data_csv}
'''

        raw_output = call_ollama(prompt)
        cleaned_output = clean_llm_output(raw_output)
        return JSONResponse(content=cleaned_output)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
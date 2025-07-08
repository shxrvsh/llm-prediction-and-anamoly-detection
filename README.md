<h1 align="center" id="title">Time Series Forecasting &amp; Drift Detection using LLM (Gemini) + AutoGen</h1>

<p id="description">This repository contains the code to perform forecasting and drift detection on time series data using: Gemini API for running LLM-based prompts. AutoGen framework to manage automated agent interactions. Both forecasting and drift tasks are performed via structured prompts with the LLM returning JSON-formatted results. Prompt templates for time series forecasting and trend drift detection No ML model training – fully LLM-driven logic. Can be connected to visualization tools like Apache Superset.</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the Repository</p>

```
git clone https://github.com/shxrvsh/llm-prediction-and-anamoly-detection.git  
```

<p>2. Install Dependencies</p>

```
pip install -r requirements.txt
```

<p>3. Set Your Gemini API Key in .env file</p>

```
GEMINI_API_KEY=your_api_key_here
```

<p>4. Run the LLM Service</p>

```
uvicorn main:app --reload
```

<p>5. Run the Forecasting agent Service</p>

```
uvicorn forecastagent:app --reload
```

<p>6. Run the Drift detection agent</p>

```
uvicorn driftagent:app --reload
```

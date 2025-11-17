# MLOps: California Housing Predictor (Flask + scikit‑learn)

Welcome to a tiny-but-mighty MLOps demo: a scikit‑learn pipeline that predicts California housing prices, wrapped in a Flask API so you can ship predictions faster than you can say “median income.”


**What’s inside**
- A trained `KNN` pipeline saved to `california_knn_pipeline.pkl`
- A lightweight Flask server in `app.py` exposing `POST /predict`
- Example requests in `requests.http` for quick testing in VS Code
- A training/exploration notebook: `assignment-9-mlops.ipynb`


**Quick Start**
- Requirements: Python 3.10+ (tested with 3.12), pip, and optionally a virtual environment.

1) Clone and install dependencies
```bash
pip install -r requirements.txt
```

2) (Optional) Use a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .\.venv\Scripts\activate  # on Windows PowerShell
pip install -r requirements.txt
```

3) Run the API
```bash
python app.py
# Server listens on http://0.0.0.0:5000
```

4) Send a prediction request (pick one)
```bash
# Using curl
curl -X POST http://localhost:5000/predict \
	-H 'Content-Type: application/json' \
	-d '{
				"MedInc": 3.5,
				"HouseAge": 25.0,
				"AveRooms": 5.0,
				"AveBedrms": 1.0,
				"Population": 1500.0,
				"AveOccup": 3.0,
				"Latitude": 37.5,
				"Longitude": -122.0
			}'

# Using VS Code REST Client (recommended): open requests.http and click "Send Request"
```


**API**
- `POST /predict`
	- Body: JSON with the following fields (floats)
		- `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
	- Response: `{ "prediction": <float> }`
	- Errors: returns `400` with `{ "error": "..." }` for invalid/missing input.

Note: The server infers column order from the pipeline’s preprocessor, so focus on sending all required fields with reasonable values. If you send only half the fields, the model will get confused and we all know confused KNNs make… interesting life choices.


**Project Structure**
```text
app.py                      # Flask API loading the pickled pipeline
california_knn_pipeline.pkl # Trained KNN pipeline (scikit-learn)
requests.http               # Ready-to-run HTTP requests for local testing
assignment-9-mlops.ipynb    # Notebook for data exploration/training
requirements.txt            # Python deps
```


**Training / Retraining**
- Explore or retrain in `assignment-9-mlops.ipynb`.
- Export a new pipeline to `california_knn_pipeline.pkl` (same filename) so the API loads it without code changes.
- Pro tip: keep the feature schema consistent, or update `app.py` to validate/reshape inputs accordingly.


**Some Fun Ways I Plan to Build On This**
- Productize like a pro:
	- Swap Flask → FastAPI for built-in OpenAPI docs and validation.
	- Add input schema validation with Pydantic/Marshmallow (no more mystery JSONs).
	- Add `/health` and `/version` endpoints so ops stops guessing.
- MLOps it up:
	- Track experiments and models with MLflow; register and serve versions.
	- Use DVC for data + model artifact versioning (because “final_final_v3.pkl” isn’t a strategy).
	- Automate training with a pipeline tool (Airflow, Prefect, Dagster, GitHub Actions CRON).
- Reliability and quality:
	- Write unit tests for preprocessing and a contract test for `/predict`.
	- Add CI (lint with ruff/flake8, format with black, run tests on PRs).
	- Add monitoring (latency, request volume, input drift, prediction distributions).
- Performance & scale:
	- Batch predict endpoint; add job queue (RQ/Celery) for heavier work.
	- Cache repeated requests with simple LRU/Redis (KNNs love a good cache).
	- Containerize with Docker and deploy to your platform of choice.
- UX for humans:
	- Tiny Streamlit/Gradio UI so PMs can click things (and stop asking for curl commands).
	- Add a simple front page with example payloads and copy button.


**Troubleshooting**
- “Model file not found”: ensure `california_knn_pipeline.pkl` exists in repo root.
- “ValueError about features”: double-check you’re sending all 8 fields listed above.
- Port already in use: stop the other service or run with another port: `FLASK_RUN_PORT=5050` (or set `port=5050` in `app.py`).
- Still stuck? Open an issue or ping a teammate with your best predictive meme.


If this helped you predict anything other than house prices, that was a happy accident. Ship responsibly, hydrate often, and may your MSE be ever low.
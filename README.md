# MLOps: California Housing Predictor (Flask + Streamlit + scikit‑learn)

Welcome to a comprehensive MLOps demo showcasing a production-ready scikit‑learn pipeline that predicts California housing prices. This project features both a Flask REST API for programmatic access and an interactive Streamlit web application for visual exploration and batch predictions.

## 🎯 What's Inside

- **Trained KNN Pipeline**: A scikit-learn K-Nearest Neighbors regression model saved to `california_knn_pipeline.pkl`
- **Flask REST API**: Lightweight API in `app.py` exposing `POST /predict` endpoint
- **Streamlit Web App**: Production-grade interactive UI in `streamlit_app.py` with:
  - Single prediction interface with adjustable sliders
  - Batch prediction with CSV upload
  - Performance metrics and visualization
  - Feature importance analysis
  - Model interpretability tools
- **Data Generator**: Synthetic data generation script (`populate_csv.py`) for creating realistic test datasets
- **Training Notebook**: Jupyter notebook (`assignment-9-mlops.ipynb`) for data exploration and model training
- **Example Requests**: Ready-to-use HTTP requests in `requests.http` for VS Code REST Client

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (tested with 3.12)
- pip
- (Optional) Virtual environment

### Installation

1. **Clone and install dependencies**
```bash
pip install -r requirements.txt
```

2. **(Optional) Use a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .\.venv\Scripts\activate  # on Windows PowerShell
pip install -r requirements.txt
```

### Running the Applications

#### Flask REST API

1. **Start the API server**
```bash
python app.py
# Server listens on http://0.0.0.0:5000
```

2. **Send a prediction request**
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

# Using VS Code REST Client: open requests.http and click "Send Request"
```

#### Streamlit Web Application

1. **Launch the Streamlit app**
```bash
streamlit run streamlit_app.py
# Automatically opens in your default browser at http://localhost:8501
```

2. **Use the interactive interface**
   - Adjust feature sliders in the sidebar for single predictions
   - Upload CSV files for batch predictions
   - Visualize model performance with interactive charts
   - Analyze feature importance with permutation-based methods

## 📊 Generating Test Data

Create realistic synthetic housing data for testing:

```bash
python populate_csv.py
# Generates housing_data.csv with 1M rows (~50MB)
```

The script creates data following the statistical distribution of the original California Housing dataset, including:
- Realistic feature distributions (lognormal, gamma, uniform)
- Non-linear relationships between features and target
- Geographic effects (coastal premium, Southern California pricing)

## 🔧 API Reference

### Flask API

**`POST /predict`**
- **Request Body**: JSON with the following float fields:
  - `MedInc`: Median income ($10k units)
  - `HouseAge`: Median house age (years)
  - `AveRooms`: Average rooms per household
  - `AveBedrms`: Average bedrooms per household
  - `Population`: Block group population
  - `AveOccup`: Average occupancy
  - `Latitude`: Geographic latitude
  - `Longitude`: Geographic longitude
- **Response**: `{ "prediction": <float> }` (in $100k units)
- **Errors**: Returns `400` with `{ "error": "..." }` for invalid/missing input

### Streamlit App Features

1. **Single Prediction**
   - Interactive sliders for all 8 features
   - Real-time prediction display
   - Feature descriptions and tooltips

2. **Batch Prediction**
   - CSV file upload
   - Data validation and preview
   - Downloadable predictions
   - Performance metrics (MAE, RMSE, R², MAPE)

3. **Model Analysis**
   - Prediction vs. actual scatter plots
   - Residual analysis
   - Permutation feature importance
   - Interactive visualizations with Plotly

4. **Documentation**
   - Model details and interpretation notes
   - Feature descriptions
   - Usage guidelines
   - CSV template download

## 📁 Project Structure

```text
.
├── app.py                          # Flask REST API
├── streamlit_app.py                # Streamlit web application
├── populate_csv.py                 # Synthetic data generator
├── california_knn_pipeline.pkl     # Trained KNN pipeline
├── requests.http                   # HTTP request examples
├── assignment-9-mlops.ipynb        # Training/exploration notebook
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔄 Training / Retraining

1. **Explore or retrain** in `assignment-9-mlops.ipynb`
2. **Export new pipeline** to `california_knn_pipeline.pkl` (same filename)
3. **Restart applications** to load the updated model
4. **Pro tip**: Keep the feature schema consistent, or update both `app.py` and `streamlit_app.py` to handle new features

## 🛠️ Future Enhancements

### Production Readiness
- **Swap Flask → FastAPI** for built-in OpenAPI docs, async support, and automatic validation
- **Add schema validation** with Pydantic/Marshmallow
- **Implement health checks** with `/health` and `/version` endpoints
- **Add authentication** for secure API access

### MLOps Improvements
- **Experiment tracking** with MLflow for model versioning and comparison
- **Data versioning** with DVC for reproducibility
- **Automated retraining** with Airflow/Prefect/Dagster
- **CI/CD pipeline** with GitHub Actions (lint, test, deploy)
- **Model monitoring** for drift detection and performance tracking

### Performance & Scale
- **Batch prediction endpoint** for efficient bulk processing
- **Job queue** (RQ/Celery) for async predictions
- **Caching layer** (Redis/LRU) for repeated requests
- **Containerization** with Docker for consistent deployments
- **Kubernetes deployment** for horizontal scaling

### User Experience
- **Enhanced Streamlit UI** with more interactive features
- **A/B testing** for model comparison
- **Prediction explanations** with SHAP/LIME
- **Custom themes** and branding
- **Mobile-responsive design**

### Testing & Quality
- **Unit tests** for preprocessing and prediction logic
- **Integration tests** for API endpoints
- **Contract tests** for API schema validation
- **Load testing** with Locust/K6
- **Performance benchmarking** for model inference time

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model file not found | Ensure `california_knn_pipeline.pkl` exists in repo root |
| ValueError about features | Double-check you're sending all 8 required fields |
| Port already in use | Stop other service or change port: `FLASK_RUN_PORT=5050` or `streamlit run streamlit_app.py --server.port 8502` |
| Streamlit doesn't open | Check if port 8501 is available, or manually navigate to displayed URL |
| CSV upload fails | Verify CSV contains all required columns with correct names |
| Memory error with large CSV | Process file in smaller chunks or reduce dataset size |

## 📝 Feature Descriptions

| Feature | Description | Unit |
|---------|-------------|------|
| `MedInc` | Median income in block group | $10,000s |
| `HouseAge` | Median age of houses in block group | Years |
| `AveRooms` | Average number of rooms per household | Count |
| `AveBedrms` | Average number of bedrooms per household | Count |
| `Population` | Block group population | Count |
| `AveOccup` | Average household occupancy | Count |
| `Latitude` | Block group latitude | Degrees |
| `Longitude` | Block group longitude | Degrees |

**Target Variable**: `MedHouseVal` - Median house value in $100,000s

## 🤝 Contributing

This is a learning/demo project, but suggestions are welcome! Areas for contribution:
- Additional model architectures (RandomForest, XGBoost, Neural Networks)
- Feature engineering improvements
- UI/UX enhancements for Streamlit app
- Docker/Kubernetes deployment configs
- Automated testing suite
- Documentation improvements

## 📄 License

This project is for educational purposes. The California Housing dataset is public domain.

## 🙏 Acknowledgments

- California Housing dataset from the 1990 Census
- scikit-learn for the ML pipeline
- Flask for the REST API
- Streamlit for the web interface
- Plotly for interactive visualizations

---

**Built with ❤️ by [@aueskinj](https://github.com/aueskinj)**

If this helped you predict anything other than house prices, that was a happy accident. Ship responsibly, hydrate often, and may your MSE be ever low. 🎯

# ✈️ AirFly — Smart Flight Delay Forecast

**Production-ready flight delay prediction system powered by machine learning**

AirFly is a polished web application that predicts airline delays using a sophisticated two-stage ML ensemble. Get instant delay forecasts, risk assessments, and batch predictions through an intuitive Streamlit interface.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

---

## 🎯 Features

### Core Capabilities
- **Single Flight Prediction**: Instant delay forecast with risk level (Low/Medium/High)
- **Batch Prediction**: Upload CSV files to analyze multiple flights at once
- **Route Context**: View 14-day rolling statistics for specific routes
- **Model Explainability**: Understand predictions with feature importance analysis
- **Interactive Visualizations**: Plotly-powered gauges, charts, and histograms

### Model Performance
- **Classifier Accuracy**: 92.7%
- **Overall MAE**: 16.3 minutes
- **Normal Delays MAE**: 12.5 minutes
- **Heavy Delays MAE**: 38.5 minutes
- **R² Score**: 0.604

### Technical Highlights
- Two-stage ensemble (classifier + dual regressors)
- 32 engineered features with leakage-safe rolling statistics
- Singleton model loading for fast predictions
- Comprehensive error handling and validation
- Full test coverage with pytest

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd airfly_data
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model artifacts are present**
   ```
   AirFly_Machine_Learning/model_artifacts/
   ├── clf_pipe.joblib
   ├── normal_reg_pipe.joblib
   ├── heavy_reg_pipe.joblib
   ├── features.json
   ├── metadata.json
   └── versions.json
   ```

### Running the Application

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Single Flight Prediction

1. Navigate to **🎯 Single Prediction** in the sidebar
2. Fill in flight details:
   - **Basic Info**: Airline, origin, destination
   - **Times**: Scheduled departure hour, actual departure/arrival hours
   - **Flight Metrics**: Distance, taxi times, air time
   - **Rolling Stats**: Historical averages (can use defaults for testing)
3. Click **🚀 Predict Delay**
4. View results:
   - Predicted delay in minutes
   - Risk badge (Low/Medium/High)
   - Interactive gauge chart
   - Prediction breakdown (normal vs heavy regime)
   - Feature importance (optional)

### Batch Prediction

1. Navigate to **📊 Batch Prediction**
2. Upload a CSV file with required columns (see format below)
3. Preview the first 5 rows
4. Click **🚀 Run Batch Prediction**
5. View summary statistics and visualizations
6. Download results as CSV

### CSV Format Specification

Your CSV must include these 32 columns:

**Numeric Features (28):**
- `TAXI_OUT`, `TAXI_IN`, `AIR_TIME`, `DISTANCE`
- `CRS_ELAPSED_TIME`, `ELAPSED_TIME`
- `ScheduledDepHour`, `DepartureHour`, `ArrivalHour`
- `month`, `dayofweek`, `is_weekend`
- `Route_rolling_mean_7d`, `Route_rolling_count_7d`
- `Route_rolling_mean_14d`, `Route_rolling_count_14d`
- `Airline_rolling_mean_7d`, `Airline_rolling_count_7d`
- `Airline_rolling_mean_14d`, `Airline_rolling_count_14d`
- `Airline_mean`, `ORIGIN_mean`, `DEST_mean`, `Route_mean`
- `Airline_count`, `ORIGIN_count`, `DEST_count`, `Route_count`

**Categorical Features (4):**
- `Airline` (e.g., "AA", "DL", "UA")
- `ORIGIN` (e.g., "ATL", "JFK", "LAX")
- `DEST` (e.g., "DFW", "ORD", "SFO")
- `Route` (e.g., "ATL_DFW")

**Example**: See `demo/sample_batch.csv` for a complete example with 10 flights.

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=backend --cov-report=term-missing
```

### Test Structure
- `tests/conftest.py`: Pytest fixtures and configuration
- `tests/test_predictor.py`: Comprehensive predictor tests
  - Initialization and singleton pattern
  - Single and batch predictions
  - Input validation
  - Feature importance
  - Edge cases

### Expected Test Results
All tests should pass. The test suite covers:
- ✅ Model loading and initialization
- ✅ Single prediction with sample data
- ✅ Batch prediction with CSV
- ✅ Input validation and error handling
- ✅ Risk level classification
- ✅ Feature importance calculation
- ✅ Edge cases (zero values, large values, single-row batches)

---

## 🏗️ Project Structure

```
airfly_data/
├── backend/
│   └── predictor.py              # Core prediction engine (singleton)
├── frontend/
│   └── app.py                    # Streamlit web application
├── demo/
│   ├── sample_input.json         # Sample single prediction input
│   └── sample_batch.csv          # Sample batch CSV (10 flights)
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   └── test_predictor.py         # Predictor unit tests
├── AirFly_Machine_Learning/
│   └── model_artifacts/          # Trained model files
│       ├── clf_pipe.joblib
│       ├── normal_reg_pipe.joblib
│       ├── heavy_reg_pipe.joblib
│       ├── features.json
│       ├── metadata.json
│       ├── versions.json
│       └── predictor_original.py # Original predictor (backup)
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions CI
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

### Two-Stage Ensemble Approach

**Stage 1: Heavy Delay Classifier**
- Predicts whether a flight will have a "heavy" delay (> 30 minutes)
- XGBoost classifier with 92.7% accuracy
- Precision: 87.4% | Recall: 58.2%

**Stage 2: Dual Regression Models**
- **Normal Regime**: Predicts delays for typical flights (MAE: 12.5 min)
- **Heavy Regime**: Predicts delays for problematic flights (MAE: 38.5 min)
- Uses log1p transformation to handle skewed distributions

**Final Prediction**
- Selects appropriate regressor based on classifier output
- Applies inverse transformation (expm1) to get minutes
- Classifies risk level: Low (<15 min), Medium (15-45 min), High (>45 min)

### Feature Engineering

Our model uses **32 carefully engineered features**:

1. **Flight Characteristics**: Distance, taxi times, air time, elapsed times
2. **Temporal Features**: Month, day of week, weekend flag, departure/arrival hours
3. **Rolling Statistics**: 7-day and 14-day rolling means and counts for routes and airlines
4. **Aggregate Features**: Historical means and counts for airlines, origins, destinations, routes

All features are **leakage-safe** — they don't use information unavailable at prediction time.

---

## ⚠️ Limitations & Assumptions

### What the Model Doesn't Include
- ❌ **Weather conditions** (major factor in delays)
- ❌ **Air traffic control (ATC)** delays and airspace congestion
- ❌ **Mechanical issues** or aircraft maintenance problems
- ❌ **Crew availability** or staffing issues
- ❌ **Real-time airport conditions**

### Best Use Cases
- ✅ Comparative analysis of flight options
- ✅ Risk assessment for travel planning
- ✅ Batch analysis of flight schedules
- ✅ Understanding delay patterns by route/airline

### Not Recommended For
- ❌ Real-time operational decisions (use live ATC data)
- ❌ Guaranteed delay predictions (treat as probabilistic estimates)
- ❌ Routes with very limited historical data
- ❌ Unprecedented events (e.g., pandemics, natural disasters)

---

## 📊 Acceptance Criteria

### Functional Requirements
- [x] Single flight prediction with all required features
- [x] Batch CSV upload and prediction
- [x] Risk level classification (Low/Medium/High)
- [x] Prediction breakdown (normal vs heavy regime)
- [x] Feature importance visualization
- [x] Route context with rolling statistics
- [x] Model metrics display
- [x] Explainability and FAQ section

### Technical Requirements
- [x] Singleton model loading (load once on startup)
- [x] Input validation with clear error messages
- [x] Non-negative delay predictions
- [x] Comprehensive test coverage (pytest)
- [x] CI/CD with GitHub Actions
- [x] Clean, documented code

### UI/UX Requirements
- [x] Polished gradient hero section
- [x] Animated airplane icon
- [x] Color-coded risk badges
- [x] Interactive Plotly visualizations
- [x] Responsive design
- [x] Clear navigation
- [x] Professional styling with custom CSS

---

## 🛠️ Development

### Adding New Features

1. **Backend**: Modify `backend/predictor.py`
2. **Frontend**: Edit `frontend/app.py`
3. **Tests**: Add tests to `tests/test_predictor.py`
4. **Run tests**: `pytest tests/ -v`

### Customizing the UI

Edit the CSS in `frontend/app.py` under the `st.markdown()` block. Key classes:
- `.hero-section`: Landing page hero
- `.risk-badge`: Risk level badges
- `.metric-card`: Metric display cards
- `.prediction-card`: Prediction result card

### Model Retraining

If you retrain the models:
1. Save new joblib files to `AirFly_Machine_Learning/model_artifacts/`
2. Update `metadata.json` with new metrics
3. Update `features.json` if feature list changes
4. Restart the Streamlit app

---

## 🐛 Troubleshooting

### Common Issues

**"Failed to load models"**
- Ensure model artifacts are in `AirFly_Machine_Learning/model_artifacts/`
- Check that all required files exist (clf_pipe.joblib, etc.)

**"Missing required features"**
- Verify your CSV has all 32 required columns
- Check column names match exactly (case-sensitive)
- See `demo/sample_batch.csv` for reference

**Tests failing**
- Run `pip install -r requirements.txt` to ensure dependencies are current
- Check Python version (3.9+ required)
- Verify model artifacts are accessible

**Streamlit not starting**
- Check port 8501 is not in use
- Try: `streamlit run frontend/app.py --server.port 8502`

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author

**Lalit Hire**

Built with ❤️ using Streamlit, scikit-learn, and XGBoost

---

## 🙏 Acknowledgments

- Dataset: Airline on-time performance data
- Models: scikit-learn, XGBoost
- UI: Streamlit, Plotly
- Testing: pytest

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Ready to predict delays?** Run `streamlit run frontend/app.py` and start forecasting! ✈️

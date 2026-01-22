# House Price Prediction – Advanced Regression Techniques

A production-ready machine learning system for predicting residential property prices in [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The project implements an automated training pipeline with hyperparameter optimisation, ensemble stacking, and a FastAPI-based inference service.

## Overview

This project tackles the Kaggle House Prices competition using a structured, modular approach. The solution employs gradient boosting methods (CatBoost, XGBoost, LightGBM), traditional regression models (ElasticNet), and tree-based ensembles (Random Forest) combined through a stacking regressor. The system features comprehensive preprocessing, target encoding, and feature engineering pipelines designed to handle missing data and maintain consistency between training and inference.

**Key Performance**: Validation RMSLE ≈ 0.11 (log-transformed SalePrice prediction)

## Project Structure

```
.
├── data/
│   ├── train.csv              # Training dataset (1460 samples, 81 features)
│   ├── test.csv               # Test dataset for competition submissions
│   ├── sample_submission.csv  # Submission format template
│   └── data_description.txt   # Feature documentation
│
├── models/
│   ├── final_model.joblib         # Active production model (stacking ensemble)
│   ├── preprocessor.joblib        # Fitted preprocessing pipeline
│   └── model_YYYYMMDD_HHMMSS.joblib  # Archived training versions
│
├── notebooks/
│   ├── house-prices-advanced-regression-techniques.ipynb  # EDA and experimentation
│   ├── Performance_Comparison.xlsx                        # Model evaluation metrics
│   └── heatmap.png                                        # Correlation analysis visualisation
│
├── src/
│   ├── model_utils.py      # Preprocessing pipeline and sklearn compatibility wrappers
│   ├── training_utils.py   # Optuna-based hyperparameter optimisation and stacking logic
│   ├── train.py            # End-to-end automated training pipeline
│   ├── predict.py          # Inference engine for single-house predictions
│   └── main.py             # FastAPI application for production deployment
│
├── requirements.txt        # Python dependencies
└── catboost_info/          # CatBoost training logs and artefacts
```

## Technical Architecture

### Preprocessing Pipeline (`HousePricePreprocessor`)

- **Missing Value Imputation**: Median-based for numerical features, "None" placeholder for categorical
- **Feature Engineering**: 20+ derived features including quality indices, total square footage, bathroom counts, kitchen ratios
- **Ordinal Encoding**: Quality features mapped to 0–5 scale (Po/Fa/TA/Gd/Ex/None)
- **Target Encoding**: Neighbourhood feature encoded using mean SalePrice (fitted on training data only)
- **One-Hot Encoding**: Categorical features (MSZoning, BsmtFinType1, FireplaceQu, QualityLevel, etc.)
- **Standardisation**: MaxAbs scaling applied to final 30 features (selected via permutation importance)
- **Column Alignment**: Ensures consistent feature set between training and inference (handles unseen categories)

### Model Training (`train.py`)

1. **Data Split**: 80/20 train-validation split with fixed random seed
2. **Target Transformation**: Log1p transformation applied to SalePrice to normalise distribution
3. **Hyperparameter Optimisation**: 200 Optuna trials per model (CatBoost, XGBoost, LightGBM, ElasticNet, Random Forest)
4. **Stacking Ensemble**: Meta-learner (Ridge Regression) combines base model predictions using 5-fold cross-validation
5. **Model Versioning**: Saves timestamped archives whilst maintaining `final_model.joblib` as the active production artefact

### Inference Service (`main.py`)

- **Framework**: FastAPI with Uvicorn ASGI server
- **Endpoints**:
  - `GET /`: Health check
  - `POST /predict`: Accepts JSON payload with house features, returns predicted price in USD
- **Model Loading**: Lazy initialisation on startup to avoid repeated I/O overhead
- **Error Handling**: HTTP 503 for model unavailability, HTTP 400 for malformed requests

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd HousePrices-AdvancedRegressionTechniques

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, 1.2GB disk space for dependencies

## Usage

### Training

Execute the full pipeline to retrain models with the latest data:

```bash
python src/train.py
```

**Output**: Updated `final_model.joblib` and `preprocessor.joblib` in `models/` directory
**Duration**: Approximately 15–20 minutes on standard hardware (depends on Optuna trial count)

### Inference (CLI)

Test predictions using the standalone script:

```bash
python src/predict.py
```

Modify the `sample_house` dictionary in `predict.py` to test different inputs.

### Inference (API)

Launch the production API server:

```bash
python src/main.py
```

The service becomes available at `http://localhost:8000`. Access interactive documentation at `http://localhost:8000/docs`.

**Example Request**:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "OverallQual": 7,
      "GrLivArea": 1500,
      "Neighborhood": "CollgCr",
      "YearBuilt": 2005,
      "TotalBsmtSF": 1000,
      "GarageCars": 2
    }
  }'
```

**Example Response**:

```json
{
  "prediction_usd": 185432.67,
  "status": "success"
}
```

## Model Details

### Base Learners

| Model          | Hyperparameters Tuned                          | Validation RMSLE |
|----------------|------------------------------------------------|------------------|
| CatBoost       | iterations, depth, learning_rate, l2_leaf_reg  | ~0.115           |
| XGBoost        | n_estimators, max_depth, eta, subsample        | ~0.117           |
| LightGBM       | num_leaves, learning_rate, feature_fraction    | ~0.118           |
| ElasticNet     | alpha, l1_ratio                                | ~0.125           |
| Random Forest  | n_estimators, max_depth, min_samples_split     | ~0.130           |

### Ensemble

- **Architecture**: Stacking Regressor with Ridge meta-learner
- **Cross-Validation**: 5-fold stratified CV for out-of-fold predictions
- **Final Validation RMSLE**: ~0.110

## Dependencies

Core libraries:
- `scikit-learn` 1.8.0 – Preprocessing, meta-learner
- `catboost` 1.2.8 – Gradient boosting
- `xgboost` 3.1.3 – Gradient boosting
- `lightgbm` 4.6.0 – Gradient boosting
- `optuna` 4.7.0 – Hyperparameter optimisation
- `fastapi` 0.128.0 – API framework
- `uvicorn` 0.40.0 – ASGI server
- `pandas` 2.3.3 – Data manipulation
- `numpy` 2.4.1 – Numerical computing

See `requirements.txt` for complete dependency list.

## Notes

- **Data Leakage Prevention**: Preprocessor fitted exclusively on training data; validation/test sets use learnt parameters
- **Sklearn 1.6+ Compatibility**: Custom `SklearnWrapper` class ensures third-party estimators comply with strict tag validation
- **Reproducibility**: Fixed random seeds (42) throughout training pipeline
- **Missing Features**: Inference handles incomplete inputs gracefully through median imputation and default categorical values

## Licence

This project is part of a personal portfolio and follows the Kaggle competition's terms of use.

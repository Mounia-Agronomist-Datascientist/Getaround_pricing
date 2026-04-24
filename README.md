# Getaround Pricing

Machine learning model to predict the optimal daily rental price for a car on the Getaround platform.

## Goal

Given a vehicle's characteristics (brand, mileage, engine power, fuel type, equipment...), predict its daily rental price.

## Project Structure
```
Getaround_pricing/
├── NOTEBOOKS/
│   ├── 01.EDA_pricing_optimization.ipynb   # Exploratory data analysis and data cleaning
│   ├── 02.Preprocessing.ipynb              # Feature engineering, encoding and scaling
│   └── 03.Model_training.ipynb             # Model training, evaluation and model selection
│
├── MODELS/
│   ├── preprocessing_pipeline.pkl          # Serialized Scikit-learn preprocessing pipeline
│   └── rental_price_predictor.joblib       # Trained XGBoost pricing model
│
└── environment.yml                         # Conda environment dependencies
```

## ML Pipeline
| Step | Details |
|---|---|
| Dataset | 4,843 vehicles, 14 features, target: `rental_price_per_day` |
| Cleaning | IQR outlier removal (mileage), winsorization (engine power), rare categories → "Other" |
| Preprocessing | `StandardScaler` (numerical), `OneHotEncoder` (categorical), `OrdinalEncoder` (boolean) |
| Final model | **XGBoost** — R² = 0.735, RMSE = 17.18 €, MAE = 10.08 € |

## Setup
```bash
conda env create -f environment.yml
conda activate getaround_pricing
Start MLflow before running the training notebook:

mlflow ui --port 5050
Then run the notebooks in order: 01 → 02 → 03.

API
The prediction API is deployed on HuggingFace Spaces (MouniaT/FastAPI_GetAround).

Endpoint: POST /predict

Request body:

{
  "model_key": "Citroën",
  "mileage": 50000,
  "engine_power": 120,
  "fuel": "diesel",
  "paint_color": "black",
  "car_type": "sedan",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": true,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": false,
  "winter_tires": false
}
Response:

{ "prediction": [98.5] }


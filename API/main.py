
# Imports -------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# Define the input data model -----------------------------
# This ensures FastAPI can validate incoming JSON data
class InputData(BaseModel):
    # Input is expected to be a list of lists of floats
    input: list[list[float]]


# Create the FastAPI application -------------------------
app = FastAPI(
    title="Rental Price Prediction API",
    description="""
    This API predicts rental prices based on a list of features and
    a pretrained model. Here is the list of features : 
    ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
       'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires', 'rental_price_per_day']

    ## Endpoints

    ### POST /predict
    - Accepts JSON input:
    {
        "input": [[Citroën, 13929, 210, petrol, grey, Other, True, True, False, False, True, True, True, 106]]

    - Returns JSON output:
    {
        "prediction": [120]
    }

    The automatic documentation is available at `/docs`.
    """,
    version="1.0"
)


# Load the preprocessing pipeline and the trained model ----
# Ensure the directory exists for Hugging Face deployment
os.makedirs("MODELS", exist_ok=True)

# Paths to the model  and the preprocessing pipeline saved with joblib
model_path = "MODELS/rental_price_predictor.joblib"
preprocessing_path = "MODELS/preprocessing_pipeline.pkl"

# Load the model
model = joblib.load(model_path)

# Laod the pipeline
preprocessor = joblib.load(preprocessing_path)


# Define the /predict endpoint ---------------------------
@app.post("/predict")
def predict(data: InputData):
    """
    Accepts JSON input with key 'input' containing a list of feature lists.
    Applies the same preprocessing pipeline than the one used for training the model.
    Returns predictions as a list under the key 'prediction'.
    """
    # Convert the input list of lists into a pandas DataFrame
    df_input = pd.DataFrame(data.input)

    # Apply preprocessing pipeline
    df_preprocessed = preprocessor.transform(df_input)

    # Predict using the loaded model
    predictions = model.predict(df_preprocessed)

    # Convert predictions to list and return as JSON
    return {"prediction": predictions.tolist()}


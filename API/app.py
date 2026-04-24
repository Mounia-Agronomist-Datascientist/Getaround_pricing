
# Imports -------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Union
from huggingface_hub import hf_hub_download


# Define the input data model -----------------------------
# This ensures FastAPI can validate incoming JSON data
class InputData(BaseModel):
    input: list[list[Union[str, float, int, bool]]]


# Create the FastAPI application -------------------------
app = FastAPI(
    title="Rental Price Prediction API",
    description="""
    This API predicts rental prices based on a list of features and
    a pretrained model. Here is the list of features : 
    ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
       'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires']

    ## Endpoints

    ### POST /predict
    - Accepts JSON input:
    ```json
    {
        "input": [["Citroën", 13929, 210, "petrol", "grey", "Other", true, true, false, false, true, true, true]]
    }
    ```
    - Returns JSON output:
    ```json
    {
        "prediction": [120.5]
    }
    ```

    The automatic documentation is available at `/docs`.
    """,
    version="1.0"
)


# Load the preprocessing pipeline and the trained model ----

# Download model from HF Hub if not already present locally
REPO_ID = "MouniaT/FastAPI_GetAround"
# Ensure the directory exists for Hugging Face deployment
os.makedirs("MODELS", exist_ok=True)

# Paths to the model and the preprocessing pipeline directly from HF
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="MODELS/rental_price_predictor.joblib",
    repo_type="space",
    local_dir="."
)

preprocessing_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="MODELS/preprocessing_pipeline.pkl",
    repo_type="space",
    local_dir="."
)

# Load the model
model = joblib.load(model_path)

# Load the pipeline
preprocessor = joblib.load(preprocessing_path)

# Retrieve column names from the fitted preprocessor
FEATURE_NAMES = preprocessor.feature_names_in_.tolist()
print("Columns expected by the preprocessing pipeline:")
print(f"Features expected ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# Define the /predict endpoint ---------------------------
@app.post("/predict")
def predict(data: InputData):
    """
    Accepts JSON input with key 'input' containing a list of feature lists.
    Applies the same preprocessing pipeline than the one used for training the model.
    Returns predictions as a list under the key 'prediction'.
    """

    # Convert input to DataFrame WITH the correct column names
    df_input = pd.DataFrame(data.input, columns=FEATURE_NAMES)

    # Cast boolean columns back to bool (JSON sends them as True/False 
    # but Pydantic may convert them to int or str)
    bool_columns = preprocessor.transformers_[2][2]  # 3rd transformer = boolean columns
    for col in bool_columns:
        df_input[col] = df_input[col].astype(bool)

    # Apply preprocessing pipeline
    df_preprocessed = preprocessor.transform(df_input)

    # Predict using the loaded model
    predictions = model.predict(df_preprocessed)

    # Convert predictions to list and return as JSON
    return {"prediction": predictions.tolist()}


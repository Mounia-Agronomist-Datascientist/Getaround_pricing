# This code needs to be run only once to be able to load  the binary files (trained model and preprocessing) into Hugging Face

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Upload the model
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="MODELS/rental_price_predictor.joblib",
    path_in_repo="MODELS/rental_price_predictor.joblib",
    repo_id="MouniaT/FastAPI_GetAround",
    repo_type="space"
)

# Upload the processor
api.upload_file(
    path_or_fileobj="API/MODELS/preprocessing_pipeline.pkl",
    path_in_repo="MODELS/preprocessing_pipeline.pkl",
    repo_id="MouniaT/FastAPI_GetAround",
    repo_type="space"
)
"""Central configuration: paths, column names and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
CLEANED_DIR = DATA_DIR / "cleaned"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "stroke_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DEFAULT_DATASET = RAW_DIR / "healthcare-dataset-stroke-data.csv"

ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")  # CSV and Excel only

TARGET = "stroke"
ID_COLUMN = "id"
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

CATEGORY_CHOICES = {
    "gender": ["Male", "Female"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
}

# Risk band: High >= decision threshold; Medium >= threshold * this fraction; else Low
RISK_MEDIUM_FRACTION = 0.5
RANDOM_STATE = 42

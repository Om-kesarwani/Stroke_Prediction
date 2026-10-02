"""Generate a synthetic dataset with the SAME columns as the Kaggle stroke dataset.

Use only for testing/demo when healthcare-dataset-stroke-data.csv is not at hand.
Usage: python scripts/make_sample_data.py [n_rows]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DEFAULT_DATASET  # noqa: E402


def make(n=5110, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(46, 22, n), 0.1, 82).round(1)
    gender = rng.choice(["Female", "Male", "Other"], n, p=[0.585, 0.414, 0.001])
    hyper = (rng.random(n) < np.clip((age - 20) / 400, 0.005, 0.3)).astype(int)
    heart = (rng.random(n) < np.clip((age - 30) / 600, 0.003, 0.2)).astype(int)
    married = np.where((age > 24) & (rng.random(n) < 0.8), "Yes", "No")
    work = rng.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"], n,
                      p=[0.57, 0.16, 0.13, 0.13, 0.01])
    work = np.where(age < 14, "children", np.where(work == "children", "Private", work))
    resid = rng.choice(["Urban", "Rural"], n)
    glucose = np.clip(rng.gamma(9, 12, n) + 8 * hyper, 55, 272).round(2)
    bmi = np.clip(rng.normal(28.5, 7.5, n), 11, 70).round(1)
    smoke = rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n,
                       p=[0.37, 0.17, 0.15, 0.31])

    logit = (-7.7 + 0.075 * age + 0.9 * hyper + 0.8 * heart + 0.006 * (glucose - 100)
             + 0.25 * (smoke == "smokes"))
    stroke = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "id": rng.choice(np.arange(100, 99999), n, replace=False), "gender": gender, "age": age,
        "hypertension": hyper, "heart_disease": heart, "ever_married": married,
        "work_type": work, "Residence_type": resid, "avg_glucose_level": glucose,
        "bmi": bmi, "smoking_status": smoke, "stroke": stroke,
    })
    df.loc[rng.random(n) < 0.039, "bmi"] = np.nan  # real data has ~3.9% missing BMI
    return df


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5110
    DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out = make(rows)
    out.to_csv(DEFAULT_DATASET, index=False)
    print(f"Wrote {len(out)} rows to {DEFAULT_DATASET}  (stroke rate {out.stroke.mean():.1%})")

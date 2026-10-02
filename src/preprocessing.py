"""Data-quality report, cleaning and the scikit-learn preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, ID_COLUMN,
                     NUMERIC_FEATURES, TARGET)


# ---------------------------------------------------------------- quality report
def quality_report(df: pd.DataFrame) -> dict:
    """Summarise data-quality problems and say what to change."""
    n = len(df)
    missing = df.isna().sum()
    missing = missing[missing > 0]
    issues: list[str] = []
    for col, cnt in missing.items():
        issues.append(f"'{col}' has {cnt} missing values ({cnt / n:.1%}) -> fill with the median (numeric) or most frequent value.")
    dup = int(df.duplicated().sum())
    if dup:
        issues.append(f"{dup} duplicate rows -> drop them.")
    if ID_COLUMN in df.columns:
        issues.append(f"'{ID_COLUMN}' is an identifier, not a medical feature -> exclude from training.")
    if "gender" in df.columns:
        rare = df["gender"].value_counts()
        rare = rare[rare < max(5, 0.005 * n)]
        for g, c in rare.items():
            issues.append(f"gender '{g}' appears only {c} time(s) -> too rare to learn from; drop or merge.")
    if "age" in df.columns and ((df["age"] < 0) | (df["age"] > 120)).any():
        issues.append("'age' has impossible values (<0 or >120) -> remove those rows.")
    if "bmi" in df.columns:
        out = ((df["bmi"] < 10) | (df["bmi"] > 70)).sum()
        if out:
            issues.append(f"{int(out)} BMI values outside 10-70 -> treat as outliers (clip or remove).")
    if TARGET in df.columns:
        rate = df[TARGET].mean()
        if rate < 0.2 or rate > 0.8:
            issues.append(f"Target '{TARGET}' is imbalanced ({rate:.1%} positive) -> use class weights and judge by recall/F1/ROC-AUC, not accuracy.")
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        issues.append(f"Required columns missing: {missing_cols}")
    return {
        "rows": n, "columns": df.shape[1], "duplicates": dup,
        "missing": missing.to_dict(), "dtypes": df.dtypes.astype(str).to_dict(),
        "issues": issues,
    }


# --------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (cleaned_df, log). BMI is left NaN here; the pipeline imputes it without leakage."""
    log: list[str] = []
    out = df.copy()
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"Dropped {before - len(out)} duplicate rows.")
    if "age" in out.columns:
        b = len(out)
        out = out[(out["age"] >= 0) & (out["age"] <= 120)]
        if len(out) != b:
            log.append(f"Removed {b - len(out)} rows with impossible age.")
    if "gender" in out.columns:
        b = len(out)
        out = out[out["gender"].isin(["Male", "Female"])]
        if len(out) != b:
            log.append(f"Removed {b - len(out)} rows with rare gender category.")
    if "bmi" in out.columns:
        bad = (out["bmi"] < 10) | (out["bmi"] > 70)
        if bad.any():
            out.loc[bad, "bmi"] = np.nan
            log.append(f"Set {int(bad.sum())} implausible BMI values to missing (imputed later).")
    for col in ["hypertension", "heart_disease", TARGET]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    out = out.reset_index(drop=True)
    log.append(f"Final shape: {out.shape[0]} rows x {out.shape[1]} columns.")
    return out, log


# --------------------------------------------------------------------- pipeline
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[FEATURES].copy(), df[TARGET].astype(int)

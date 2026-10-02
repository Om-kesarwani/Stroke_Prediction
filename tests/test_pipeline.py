import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_sample_data import make  # noqa: E402
from src import data_loader, model, preprocessing  # noqa: E402
from src.config import FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return make(n=2500, seed=1)


def test_quality_report_flags_known_issues(df):
    rep = preprocessing.quality_report(df)
    text = " ".join(rep["issues"])
    assert "bmi" in text and "imbalanced" in text and rep["rows"] == len(df)


def test_clean_removes_rare_gender_and_keeps_columns(df):
    cleaned, log = preprocessing.clean(df)
    assert set(cleaned["gender"]) <= {"Male", "Female"}
    assert list(cleaned.columns) == list(df.columns)
    assert log


def test_split_xy_requires_columns(df):
    with pytest.raises(ValueError):
        preprocessing.split_xy(df.drop(columns=["age"]))


def test_loader_accepts_csv_and_excel_rejects_pdf(df):
    csv = df.head(5).to_csv(index=False).encode()
    assert len(data_loader.read_table(csv, "a.csv")) == 5
    buf = io.BytesIO()
    df.head(5).to_excel(buf, index=False)
    assert len(data_loader.read_table(buf.getvalue(), "a.xlsx")) == 5
    with pytest.raises(data_loader.UnsupportedFileError):
        data_loader.read_table(b"%PDF", "a.pdf")


def test_save_upload_writes_file_and_rejects_pdf(tmp_path):
    p = data_loader.save_upload(b"a,b\n1,2\n", "My File (1).csv", folder=tmp_path)
    assert p.exists() and p.suffix == ".csv"
    with pytest.raises(data_loader.UnsupportedFileError):
        data_loader.save_upload(b"x", "x.pdf", folder=tmp_path)


def test_training_beats_chance_and_predicts(df, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(model, "METRICS_PATH", tmp_path / "m.json")
    cleaned, _ = preprocessing.clean(df)
    rep = model.train_and_compare(cleaned)
    assert rep["test_metrics"]["roc_auc"] > 0.65
    assert rep["test_metrics"]["recall"] > 0.3
    bundle = model.load_model()
    low = dict(gender="Female", age=20, hypertension=0, heart_disease=0, ever_married="No", work_type="Private",
               Residence_type="Urban", avg_glucose_level=85, bmi=22, smoking_status="never smoked")
    high = dict(low, age=80, hypertension=1, heart_disease=1, avg_glucose_level=220, smoking_status="smokes")
    assert model.predict_one(bundle, high)["probability"] > model.predict_one(bundle, low)["probability"]
    batch = model.predict_many(bundle, cleaned.head(20))
    assert {"risk_probability", "risk_band", "flagged"} <= set(batch.columns) and len(batch) == 20


def test_risk_band_logic():
    assert model.risk_band(0.9, 0.5) == "High"
    assert model.risk_band(0.3, 0.5) == "Medium"
    assert model.risk_band(0.1, 0.5) == "Low"


def test_model_handles_missing_bmi_at_predict_time(df, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(model, "METRICS_PATH", tmp_path / "m.json")
    cleaned, _ = preprocessing.clean(df)
    model.train_and_compare(cleaned)
    rec = {f: cleaned[f].iloc[0] for f in FEATURES}
    rec["bmi"] = float("nan")
    assert 0 <= model.predict_one(model.load_model(), rec)["probability"] <= 1

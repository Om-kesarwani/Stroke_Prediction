# Patient Health Outcomes Predictor (Stroke Risk)

A 100% Python application: **Streamlit** front end + **pandas / scikit-learn / Plotly** back end.
No React, Flask or HTML templates. Uploads: **CSV and Excel only**.

```
User/Admin -> Streamlit UI -> patient input -> preprocessing (sklearn Pipeline)
           -> ML model -> risk probability + band -> Plotly dashboard
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the pages in the sidebar in order: Upload Data -> Data Cleaning -> Train Model -> Predict -> Dashboard.

## Using the real dataset
`data/raw/healthcare-dataset-stroke-data.csv` is **synthetic demo data** so the app runs immediately.
Replace it with the real Kaggle file (same name) or upload the real file on page 1, then click **Train** on page 3.

## Folders
| Path | Purpose |
|---|---|
| `app.py`, `pages/` | Streamlit UI (home + 5 pages) |
| `src/data_loader.py` | read CSV/Excel, store uploads |
| `src/preprocessing.py` | quality report, cleaning, sklearn preprocessing pipeline |
| `src/model.py` | train/compare 3 models, threshold, save/load, predict |
| `src/charts.py` | Plotly figure builders |
| `data/uploads/` | every file uploaded from the UI (timestamped) |
| `data/cleaned/` | cleaned datasets saved from the UI |
| `models/` | trained model + metrics (created on training) |
| `tests/` | pytest suite (`python -m pytest -q`) |
| `PROGRESS.md` | build log / recovery notes |

## Modelling notes
- Target `stroke` is ~5% positive. A model that always says "no stroke" gets ~95% accuracy and catches nobody,
  so models are judged on **recall, precision, F1, ROC-AUC** and trained with **class weights**.
- The decision threshold is tuned on a validation split to favour recall (screening use). Risk bands are relative to it:
  High >= threshold, Medium >= half the threshold, else Low.
- Final metrics come from a test set never used for training or threshold selection.
- Missing BMI is imputed inside the pipeline (median of training data only - no leakage).

## Limitations
Educational project. Not a medical device; do not use for diagnosis or treatment decisions. Scores are ranking scores
from a class-weighted model, not literal probabilities of stroke.

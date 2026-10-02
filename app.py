"""Patient Health Outcomes Predictor - Streamlit entry point (100% Python).

Run:  streamlit run app.py
"""
import streamlit as st

from src import model, ui

ui.setup("Patient Health Outcomes Predictor")
ui.sidebar_status()

st.markdown(
    "Predict **stroke risk** from a patient's demographics and medical history, explore the data, "
    "and monitor model quality - all in one Python app."
)
ui.disclaimer()

raw = ui.get_raw()
c1, c2, c3 = st.columns(3)
c1.metric("Dataset", st.session_state.get("dataset_name", "none"))
c2.metric("Patients (rows)", f"{len(raw):,}" if raw is not None else "-")
metrics = model.load_metrics()
c3.metric("Model", metrics["best_model"] if metrics else "not trained")

st.subheader("How to use")
st.markdown(
    """
1. **Upload Data** - add a CSV or Excel file (saved in `data/uploads/`), or keep the default dataset.
2. **Data Cleaning** - see data-quality problems, what to change, and save a cleaned copy (`data/cleaned/`).
3. **Train Model** - train and compare three models; the best is saved.
4. **Predict** - enter one patient or score a whole file; see risk score, band and probability.
5. **Dashboard** - interactive Plotly charts of the data and of model performance.
"""
)

st.subheader("Pipeline")
st.code(
    "User/Admin -> Streamlit UI -> patient input -> preprocessing (sklearn Pipeline) "
    "-> ML model -> risk probability + band -> Plotly dashboard",
    language="text",
)
if raw is None:
    st.info("No default dataset found. Open **Upload Data** to begin.")

import streamlit as st

from src import charts, data_loader, model, ui
from src.config import CATEGORY_CHOICES, FEATURES

ui.setup("Predict Stroke Risk", "🔮")
ui.sidebar_status()
ui.disclaimer()

if not model.model_exists():
    st.warning("No trained model yet. Open **Train Model** first.")
    st.stop()
bundle = model.load_model()

single, batch = st.tabs(["Single patient", "Batch file (CSV / Excel)"])

with single:
    with st.form("patient"):
        a, b, c = st.columns(3)
        gender = a.selectbox("Gender", CATEGORY_CHOICES["gender"])
        age = b.number_input("Age", 0, 120, 55)
        married = c.selectbox("Ever married", CATEGORY_CHOICES["ever_married"])
        hyp = a.selectbox("Hypertension", ["No", "Yes"])
        heart = b.selectbox("Heart disease", ["No", "Yes"])
        work = c.selectbox("Work type", CATEGORY_CHOICES["work_type"])
        res = a.selectbox("Residence", CATEGORY_CHOICES["Residence_type"])
        glucose = b.number_input("Average glucose level (mg/dL)", 40.0, 400.0, 105.0)
        bmi = c.number_input("BMI", 10.0, 70.0, 27.0)
        smoke = a.selectbox("Smoking status", CATEGORY_CHOICES["smoking_status"])
        go_btn = st.form_submit_button("Predict", type="primary")
    if go_btn:
        rec = dict(gender=gender, age=age, hypertension=int(hyp == "Yes"), heart_disease=int(heart == "Yes"),
                   ever_married=married, work_type=work, Residence_type=res, avg_glucose_level=glucose,
                   bmi=bmi, smoking_status=smoke)
        res_ = model.predict_one(bundle, rec)
        left, right = st.columns([1, 1])
        left.plotly_chart(charts.risk_gauge(res_["probability"], res_["threshold"]), use_container_width=True)
        colour = {"Low": "success", "Medium": "warning", "High": "error"}[res_["band"]]
        getattr(right, colour)(f"**{res_['band']} risk** - model score {res_['probability']:.1%}")
        right.write(f"Decision threshold: {res_['threshold']:.0%}. "
                    + ("This patient is **flagged** for follow-up." if res_["flagged"] else "Not flagged."))
        right.caption("The score comes from a class-weighted model, so it is a ranking score, not a literal "
                      "percentage chance of stroke.")

with batch:
    st.write("Upload a CSV/Excel file containing these columns: " + ", ".join(f"`{f}`" for f in FEATURES))
    f = st.file_uploader("Patients file", type=["csv", "xlsx", "xls"], key="batch")
    if f is not None:
        try:
            df = data_loader.read_table(f.getvalue(), f.name)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                pred = model.predict_many(bundle, df)
                st.plotly_chart(charts.batch_risk_distribution(pred), use_container_width=True)
                st.dataframe(pred.sort_values("risk_probability", ascending=False), use_container_width=True)
                st.download_button("Download predictions (CSV)", pred.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

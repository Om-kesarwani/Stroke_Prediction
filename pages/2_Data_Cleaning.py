import streamlit as st

from src import charts, ui
from src.config import CLEANED_DIR
from src.preprocessing import clean, quality_report

ui.setup("Data Cleaning", "🧹")
ui.sidebar_status()
raw = ui.require_dataset()

rep = quality_report(raw)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{rep['rows']:,}")
c2.metric("Columns", rep["columns"])
c3.metric("Duplicate rows", rep["duplicates"])

st.subheader("What to change")
if rep["issues"]:
    for issue in rep["issues"]:
        st.warning(issue)
else:
    st.success("No obvious data-quality problems found.")

left, right = st.columns(2)
left.plotly_chart(charts.missing_values(raw), use_container_width=True)
if "stroke" in raw.columns:
    right.plotly_chart(charts.class_balance(raw), use_container_width=True)

with st.expander("Column types and summary statistics"):
    st.write(rep["dtypes"])
    st.dataframe(raw.describe(include="all").T, use_container_width=True)

st.subheader("Apply cleaning")
if st.button("Clean dataset and save", type="primary"):
    try:
        cleaned, log = clean(raw)
        st.session_state["clean_df"], st.session_state["clean_log"] = cleaned, log
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        name = st.session_state["dataset_name"].rsplit(".", 1)[0] + "_cleaned.csv"
        cleaned.to_csv(CLEANED_DIR / name, index=False)
        st.success(f"Saved `data/cleaned/{name}`.")
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")

if "clean_log" in st.session_state:
    for line in st.session_state["clean_log"]:
        st.write("•", line)
    st.caption("Remaining missing BMI values are filled inside the model pipeline (median from training data only).")
    st.dataframe(st.session_state["clean_df"].head(50), use_container_width=True)

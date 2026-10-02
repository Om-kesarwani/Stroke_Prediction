import streamlit as st

from src import charts, model, ui
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ui.setup("Dashboard", "📊")
ui.sidebar_status()
ui.require_dataset()
df = ui.get_clean()
if df is None:
    st.stop()
if "stroke" not in df.columns:
    st.error("This dashboard needs a 'stroke' column (0/1).")
    st.stop()

tab_data, tab_model = st.tabs(["Data explorer", "Model performance"])

with tab_data:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients", f"{len(df):,}")
    k2.metric("Stroke rate", f"{df['stroke'].mean():.1%}")
    k3.metric("Mean age", f"{df['age'].mean():.1f}")
    k4.metric("Hypertension", f"{df['hypertension'].mean():.1%}")

    a, b = st.columns(2)
    a.plotly_chart(charts.class_balance(df), use_container_width=True)
    b.plotly_chart(charts.age_band_rate(df), use_container_width=True)

    num = st.selectbox("Numeric feature", NUMERIC_FEATURES)
    st.plotly_chart(charts.numeric_distribution(df.dropna(subset=[num]), num), use_container_width=True)
    cat = st.selectbox("Categorical feature", CATEGORICAL_FEATURES + ["hypertension", "heart_disease"])
    st.plotly_chart(charts.stroke_rate_by_category(df, cat), use_container_width=True)
    st.plotly_chart(charts.correlation_heatmap(df), use_container_width=True)

with tab_model:
    report = st.session_state.get("report") or model.load_metrics()
    if not report:
        st.info("Train a model first (page 3) to see performance charts.")
    else:
        m = report["test_metrics"]
        st.subheader(f"{report['best_model']} - threshold {report['threshold']:.2f}")
        a, b = st.columns(2)
        a.plotly_chart(charts.roc_fig(report["curves"], m["roc_auc"]), use_container_width=True)
        b.plotly_chart(charts.confusion_matrix_fig(m["confusion"]), use_container_width=True)
        if report["importances"]:
            st.plotly_chart(charts.importance_fig(report["importances"]), use_container_width=True)
        st.plotly_chart(charts.model_comparison(report["comparison"]), use_container_width=True)

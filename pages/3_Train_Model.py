import pandas as pd
import streamlit as st

from src import charts, model, ui

ui.setup("Train Model", "🧠")
ui.sidebar_status()
ui.require_dataset()
st.write("Trains **Logistic Regression, Random Forest and Gradient Boosting** with class weighting, picks the best "
         "by validation ROC-AUC, and evaluates it once on a held-out test set.")

if st.button("Train and compare models", type="primary"):
    df = ui.get_clean()
    if df is not None:
        with st.spinner("Training..."):
            try:
                st.session_state["report"] = model.train_and_compare(df)
                st.success("Training finished - model saved to `models/`.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

report = st.session_state.get("report") or model.load_metrics()
if report:
    m = report["test_metrics"]
    st.subheader(f"Best model: {report['best_model']}")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Recall", "recall"), ("Precision", "precision"), ("F1", "f1"),
                                        ("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy")]):
        col.metric(label, f"{m[key]:.2f}")
    st.info("Accuracy is shown last on purpose: with ~5% positives, always answering 'no stroke' scores ~95% accuracy "
            "yet finds no one. Recall (stroke cases caught) matters most here.")
    a, b = st.columns(2)
    a.plotly_chart(charts.model_comparison(report["comparison"]), use_container_width=True)
    b.plotly_chart(charts.confusion_matrix_fig(m["confusion"]), use_container_width=True)
    table = pd.DataFrame(report["comparison"]).T[["recall", "precision", "f1", "roc_auc", "pr_auc", "threshold"]]
    st.dataframe(table.round(3), use_container_width=True)
else:
    st.caption("No trained model yet.")

"""Plotly figure builders (pure functions: DataFrame/metrics in, Figure out)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import TARGET

PALETTE = {"No stroke": "#4C78A8", "Stroke": "#E4572E"}
LAYOUT = dict(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))


def _label(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].map({0: "No stroke", 1: "Stroke"})


def class_balance(df: pd.DataFrame) -> go.Figure:
    counts = _label(df).value_counts().reset_index()
    counts.columns = ["Outcome", "Patients"]
    fig = px.pie(counts, names="Outcome", values="Patients", hole=0.55, color="Outcome",
                 color_discrete_map=PALETTE, title="Outcome balance")
    return fig.update_layout(**LAYOUT)


def numeric_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    d = df.assign(Outcome=_label(df))
    fig = px.histogram(d, x=col, color="Outcome", barmode="overlay", opacity=0.65, nbins=40,
                       histnorm="probability density", color_discrete_map=PALETTE,
                       title=f"{col} by outcome")
    return fig.update_layout(**LAYOUT)


def stroke_rate_by_category(df: pd.DataFrame, col: str) -> go.Figure:
    g = df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
    g["mean"] *= 100
    fig = px.bar(g, x=col, y="mean", text=g["count"].map(lambda c: f"n={c}"),
                 title=f"Stroke rate (%) by {col}", labels={"mean": "Stroke rate (%)"})
    fig.update_traces(marker_color="#E4572E")
    return fig.update_layout(**LAYOUT)


def age_band_rate(df: pd.DataFrame) -> go.Figure:
    bins = [0, 18, 30, 40, 50, 60, 70, 80, 121]
    band = pd.cut(df["age"], bins=bins, right=False)
    g = df.groupby(band, observed=True)[TARGET].mean().mul(100).reset_index()
    g["age"] = g["age"].astype(str)
    fig = px.line(g, x="age", y=TARGET, markers=True, title="Stroke rate (%) by age band",
                  labels={TARGET: "Stroke rate (%)", "age": "Age band"})
    return fig.update_layout(**LAYOUT)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num = df.select_dtypes("number").drop(columns=["id"], errors="ignore")
    fig = px.imshow(num.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Correlation of numeric columns")
    return fig.update_layout(**LAYOUT)


def missing_values(df: pd.DataFrame) -> go.Figure:
    m = df.isna().sum().reset_index()
    m.columns = ["Column", "Missing"]
    fig = px.bar(m, x="Column", y="Missing", title="Missing values per column")
    fig.update_traces(marker_color="#F2A541")
    return fig.update_layout(**LAYOUT)


def model_comparison(comparison: dict) -> go.Figure:
    rows = [{"Model": m, "Metric": k, "Score": v[k]}
            for m, v in comparison.items() for k in ("recall", "precision", "f1", "roc_auc")]
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Score", color="Model", barmode="group",
                 title="Model comparison on the held-out test set", range_y=[0, 1])
    return fig.update_layout(**LAYOUT)


def confusion_matrix_fig(cm: dict) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, x=["Predicted: no stroke", "Predicted: stroke"],
                    y=["Actual: no stroke", "Actual: stroke"], color_continuous_scale="Blues",
                    title="Confusion matrix (test set)")
    return fig.update_layout(**LAYOUT, coloraxis_showscale=False)


def roc_fig(curves: dict, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=curves["fpr"], y=curves["tpr"], mode="lines", name=f"Model (AUC {auc:.2f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey"))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate (recall)", **LAYOUT)
    return fig


def importance_fig(importances: dict) -> go.Figure:
    d = pd.DataFrame({"Feature": list(importances), "Importance": list(importances.values())})
    fig = px.bar(d.iloc[::-1], x="Importance", y="Feature", orientation="h", title="Top features")
    return fig.update_layout(**LAYOUT)


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%"},
        title={"text": "Predicted stroke risk score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, threshold * 50], "color": "#9BD08B"},
                         {"range": [threshold * 50, threshold * 100], "color": "#F6D673"},
                         {"range": [threshold * 100, 100], "color": "#E88D80"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
    return fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))


def batch_risk_distribution(pred: pd.DataFrame) -> go.Figure:
    counts = pred["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    counts.columns = ["Risk band", "Patients"]
    fig = px.bar(counts, x="Risk band", y="Patients", color="Risk band", title="Predicted risk bands",
                 color_discrete_map={"Low": "#9BD08B", "Medium": "#F6D673", "High": "#E88D80"})
    return fig.update_layout(**LAYOUT)

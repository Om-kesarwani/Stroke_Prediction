"""Small Streamlit helpers shared by all pages (session state + common widgets)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data_loader
from .preprocessing import clean

DISCLAIMER = ("Educational project - not a medical device. Predictions are statistical estimates from a public "
              "dataset and must not be used for diagnosis or treatment decisions.")


def setup(title: str, icon: str = "🩺"):
    st.set_page_config(page_title=f"{title} | Patient Outcome Predictor", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def disclaimer():
    st.caption(f"⚠️ {DISCLAIMER}")


def set_dataset(df: pd.DataFrame, name: str):
    st.session_state["raw_df"] = df
    st.session_state["dataset_name"] = name
    st.session_state.pop("clean_df", None)  # force re-clean for a new dataset


def get_raw() -> pd.DataFrame | None:
    """Active raw dataset; falls back to the default CSV in data/raw/ if present."""
    if "raw_df" not in st.session_state:
        try:
            set_dataset(data_loader.load_default(), data_loader.DEFAULT_DATASET.name)
        except FileNotFoundError:
            return None
    return st.session_state["raw_df"]


def get_clean() -> pd.DataFrame | None:
    raw = get_raw()
    if raw is None:
        return None
    if "clean_df" not in st.session_state:
        try:
            st.session_state["clean_df"], st.session_state["clean_log"] = clean(raw)
        except Exception as exc:  # malformed dataset
            st.error(f"Could not clean this dataset: {exc}")
            return None
    return st.session_state["clean_df"]


def require_dataset() -> pd.DataFrame:
    raw = get_raw()
    if raw is None:
        st.warning("No dataset loaded yet. Go to **Upload Data** and upload a CSV or Excel file.")
        st.stop()
    return raw


def sidebar_status():
    with st.sidebar:
        st.markdown("**Active dataset**")
        st.write(st.session_state.get("dataset_name", "none"))
        st.markdown("---")
        st.caption(DISCLAIMER)

import streamlit as st

from src import data_loader, ui
from src.config import UPLOAD_DIR

ui.setup("Upload Data", "📤")
ui.sidebar_status()
st.write("Upload a **CSV or Excel** file. The original file is stored in `data/uploads/` and becomes the active dataset.")

file = st.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls"])
if file is not None:
    try:
        data = file.getvalue()
        df = data_loader.read_table(data, file.name)
        path = data_loader.save_upload(data, file.name)
        ui.set_dataset(df, file.name)
        st.success(f"Saved to `{path.relative_to(path.parent.parent.parent)}` and set as the active dataset "
                   f"({len(df):,} rows, {df.shape[1]} columns).")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")

raw = ui.get_raw()
if raw is not None:
    st.subheader(f"Preview - {st.session_state['dataset_name']}")
    st.dataframe(raw.head(100), use_container_width=True)
    st.caption(f"{len(raw):,} rows x {raw.shape[1]} columns")

st.subheader("Previously uploaded files")
files = data_loader.list_uploads()
if not files:
    st.caption("Nothing uploaded yet.")
else:
    choice = st.selectbox("Re-open an earlier upload", files, format_func=lambda p: p.name)
    if st.button("Use this file"):
        ui.set_dataset(data_loader.read_table(choice), choice.name)
        st.rerun()
st.caption(f"Uploads folder: `{UPLOAD_DIR}`")

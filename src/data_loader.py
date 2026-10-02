"""Loading and storing datasets. Only CSV and Excel are supported."""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ALLOWED_EXTENSIONS, DEFAULT_DATASET, UPLOAD_DIR


class UnsupportedFileError(ValueError):
    """Raised when a file is not CSV or Excel."""


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def read_table(source, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV/Excel file from a path, bytes or a file-like object."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    if not is_allowed(name):
        raise UnsupportedFileError(f"Unsupported file type '{Path(name).suffix}'. Use CSV or Excel (.xlsx/.xls).")
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if Path(name).suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_excel(source)


def _safe_name(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("_") or "dataset"
    return stem + Path(filename).suffix.lower()


def save_upload(data: bytes, filename: str, folder: Path = UPLOAD_DIR) -> Path:
    """Store the raw uploaded bytes in the uploads folder (timestamped) and return the path."""
    if not is_allowed(filename):
        raise UnsupportedFileError("Only CSV and Excel files can be uploaded.")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{stamp}_{_safe_name(filename)}"
    path.write_bytes(data)
    return path


def list_uploads(folder: Path = UPLOAD_DIR) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and is_allowed(p.name)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_default() -> pd.DataFrame:
    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(
            f"{DEFAULT_DATASET.name} not found in data/raw/. Upload a dataset or run scripts/make_sample_data.py."
        )
    return read_table(DEFAULT_DATASET)

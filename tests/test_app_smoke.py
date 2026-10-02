"""Headless smoke test: every Streamlit page must run without raising an exception."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PAGES = ["app.py"] + [f"pages/{p.name}" for p in sorted((ROOT / "pages").glob("*.py"))]


@pytest.fixture(scope="module", autouse=True)
def _ensure_data():
    from scripts.make_sample_data import make
    from src.config import DEFAULT_DATASET
    if not DEFAULT_DATASET.exists():
        DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
        make().to_csv(DEFAULT_DATASET, index=False)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_train_button_then_predict():
    at = AppTest.from_file(str(ROOT / "pages/3_Train_Model.py"), default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    at2 = AppTest.from_file(str(ROOT / "pages/4_Predict.py"), default_timeout=60).run()
    assert not at2.exception, [e.value for e in at2.exception]
    at2.button[0].click().run()
    assert not at2.exception, [e.value for e in at2.exception]

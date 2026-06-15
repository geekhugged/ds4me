"""Smoke tests for the MLOps topic pages using streamlit.testing.v1.AppTest.

Run with: pytest streamlit_app/tests/test_mlops_pages.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_ml_lifecycle():
    at = AppTest.from_file(str(TOPICS_DIR / "ml_lifecycle.py"))
    at.run()
    assert not at.exception


def test_reproducibility():
    at = AppTest.from_file(str(TOPICS_DIR / "reproducibility.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run(timeout=60)
    assert not at.exception


def test_data_model_versioning():
    at = AppTest.from_file(str(TOPICS_DIR / "data_model_versioning.py"))
    at.run()
    assert not at.exception


def test_experiment_tracking():
    at = AppTest.from_file(str(TOPICS_DIR / "experiment_tracking.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run(timeout=60)
    assert not at.exception


def test_containerization():
    at = AppTest.from_file(str(TOPICS_DIR / "containerization.py"))
    at.run()
    assert not at.exception

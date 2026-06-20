"""Smoke tests for the third batch of new ML theory topic pages using
streamlit.testing.v1.AppTest.

Covers: dimensionality_reduction, feature_engineering, imbalanced_data,
statistical_significance.

Run with: pytest streamlit_app/tests/test_ml_theory_pages_3.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_dimensionality_reduction_loads():
    """Dimensionality reduction page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "dimensionality_reduction.py"))
    at.run(timeout=120)
    assert not at.exception


def test_dimensionality_reduction_method_switch():
    """Switching method radio (PCA / t-SNE) re-renders without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "dimensionality_reduction.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.radio) > 0
    for opt in at.radio[0].options:
        at.radio[0].set_value(opt).run(timeout=120)
        assert not at.exception


def test_dimensionality_reduction_noise_slider():
    """Changing a slider (noise / n_points / perplexity) re-renders correctly."""
    at = AppTest.from_file(str(TOPICS_DIR / "dimensionality_reduction.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.slider) > 0
    at.slider[0].set_value(at.slider[0].max).run(timeout=120)
    assert not at.exception


def test_feature_engineering_loads():
    """Feature engineering page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "feature_engineering.py"))
    at.run(timeout=120)
    assert not at.exception


def test_feature_engineering_method_switch():
    """Switching the scaling/encoding radio works for all options."""
    at = AppTest.from_file(str(TOPICS_DIR / "feature_engineering.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.radio) > 0
    for opt in at.radio[0].options:
        at.radio[0].set_value(opt).run(timeout=120)
        assert not at.exception


def test_imbalanced_data_loads():
    """Imbalanced data page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "imbalanced_data.py"))
    at.run(timeout=120)
    assert not at.exception


def test_imbalanced_data_strategy_switch():
    """Switching the imbalance-handling strategy radio works for all options."""
    at = AppTest.from_file(str(TOPICS_DIR / "imbalanced_data.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.radio) > 0
    for opt in at.radio[0].options:
        at.radio[0].set_value(opt).run(timeout=120)
        assert not at.exception


def test_imbalanced_data_threshold_slider():
    """Moving the classification threshold slider updates metrics without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "imbalanced_data.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.slider) > 0
    at.slider[-1].set_value(at.slider[-1].max).run(timeout=120)
    assert not at.exception
    at.slider[-1].set_value(at.slider[-1].min).run(timeout=120)
    assert not at.exception


def test_statistical_significance_loads():
    """Statistical significance page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "statistical_significance.py"))
    at.run(timeout=120)
    assert not at.exception


def test_statistical_significance_sample_size_slider():
    """Changing a slider (effect size / sample size / alpha) re-renders correctly."""
    at = AppTest.from_file(str(TOPICS_DIR / "statistical_significance.py"))
    at.run(timeout=120)
    assert not at.exception
    assert len(at.slider) > 0
    for i in range(len(at.slider)):
        at.slider[i].set_value(at.slider[i].max).run(timeout=120)
        assert not at.exception

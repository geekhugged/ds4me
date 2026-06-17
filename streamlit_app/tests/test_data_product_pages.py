"""Smoke tests for the Data Product topic pages using streamlit.testing.v1.AppTest.

Covers: north_star_aarrr_okr, funnel_analytics, ab_test_hypothesis,
significance_power_sample_size, retention_cohort.

Run with: pytest streamlit_app/tests/test_data_product_pages.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_north_star_aarrr_okr_loads():
    at = AppTest.from_file(str(TOPICS_DIR / "north_star_aarrr_okr.py"))
    at.run(timeout=60)
    assert not at.exception


def test_north_star_aarrr_okr_sliders():
    at = AppTest.from_file(str(TOPICS_DIR / "north_star_aarrr_okr.py"))
    at.run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(500_000).run(timeout=60)
    assert not at.exception


def test_north_star_aarrr_okr_radio():
    at = AppTest.from_file(str(TOPICS_DIR / "north_star_aarrr_okr.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 0
    at.radio[0].set_value(at.radio[0].options[1]).run(timeout=60)
    assert not at.exception


def test_funnel_analytics_loads():
    at = AppTest.from_file(str(TOPICS_DIR / "funnel_analytics.py"))
    at.run(timeout=60)
    assert not at.exception


def test_funnel_analytics_n_steps_slider():
    """Changing number of funnel steps rebuilds the per-step sliders."""
    at = AppTest.from_file(str(TOPICS_DIR / "funnel_analytics.py"))
    at.run(timeout=60)
    assert not at.exception
    # second slider is n_steps
    at.slider[1].set_value(3).run(timeout=60)
    assert not at.exception
    at.slider[1].set_value(6).run(timeout=60)
    assert not at.exception


def test_funnel_analytics_selectbox():
    at = AppTest.from_file(str(TOPICS_DIR / "funnel_analytics.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.selectbox) > 0
    options = at.selectbox[0].options
    at.selectbox[0].set_value(options[-1]).run(timeout=60)
    assert not at.exception


def test_ab_test_hypothesis_loads():
    at = AppTest.from_file(str(TOPICS_DIR / "ab_test_hypothesis.py"))
    at.run(timeout=60)
    assert not at.exception


def test_ab_test_hypothesis_inputs():
    at = AppTest.from_file(str(TOPICS_DIR / "ab_test_hypothesis.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.number_input) >= 4
    at.number_input[1].set_value(900).run(timeout=60)
    assert not at.exception


def test_ab_test_hypothesis_peeking_simulation():
    """Click the peeking simulation button with a small simulation budget."""
    at = AppTest.from_file(str(TOPICS_DIR / "ab_test_hypothesis.py"))
    at.run(timeout=60)
    assert not at.exception
    # sliders after the constructor section: alpha(0), n_sims(1), n_checks(2), n_final(3), p_true(4)
    slider_count = len(at.slider)
    at.slider[slider_count - 4].set_value(100).run(timeout=60)
    assert not at.exception
    at.button[0].click().run(timeout=120)
    assert not at.exception


def test_significance_power_sample_size_loads():
    at = AppTest.from_file(str(TOPICS_DIR / "significance_power_sample_size.py"))
    at.run(timeout=60)
    assert not at.exception


def test_significance_power_sample_size_sliders():
    at = AppTest.from_file(str(TOPICS_DIR / "significance_power_sample_size.py"))
    at.run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(2.5).run(timeout=60)
    assert not at.exception


def test_retention_cohort_loads():
    at = AppTest.from_file(str(TOPICS_DIR / "retention_cohort.py"))
    at.run(timeout=60)
    assert not at.exception


def test_retention_cohort_cohort_table_sliders():
    at = AppTest.from_file(str(TOPICS_DIR / "retention_cohort.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) >= 6
    at.slider[4].set_value(10).run(timeout=60)
    assert not at.exception


def test_retention_cohort_forecast_number_inputs():
    at = AppTest.from_file(str(TOPICS_DIR / "retention_cohort.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.number_input) >= 2
    at.number_input[0].set_value(200_000).run(timeout=60)
    assert not at.exception

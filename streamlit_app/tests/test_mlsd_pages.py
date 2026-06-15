"""Smoke tests for the ML System Design topic pages using streamlit.testing.v1.AppTest.

Run with: pytest streamlit_app/tests/test_mlsd_pages.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_mlsd_video_recsys():
    at = AppTest.from_file(str(TOPICS_DIR / "mlsd_video_recsys.py"))
    at.run()
    assert not at.exception

    # подвигать слайдеры (веса итогового скора, diversity, нагрузка)
    if len(at.slider) > 0:
        at.slider[0].set_value(50).run(timeout=60)
        assert not at.exception


def test_mlsd_ad_ctr():
    at = AppTest.from_file(str(TOPICS_DIR / "mlsd_ad_ctr.py"))
    at.run()
    assert not at.exception

    # изменить коэффициент downsampling и порог классификации
    if len(at.slider) > 0:
        at.slider[0].set_value(0.03).run(timeout=60)
        assert not at.exception


def test_mlsd_feed_ranking():
    at = AppTest.from_file(str(TOPICS_DIR / "mlsd_feed_ranking.py"))
    at.run()
    assert not at.exception

    if len(at.slider) > 0:
        at.slider[0].set_value(40).run(timeout=60)
        assert not at.exception


def test_mlsd_fraud_detection():
    at = AppTest.from_file(str(TOPICS_DIR / "mlsd_fraud_detection.py"))
    at.run()
    assert not at.exception

    # поменять порог классификации и cost-matrix
    if len(at.slider) > 0:
        at.slider[0].set_value(0.01).run(timeout=60)
        assert not at.exception


def test_mlsd_visual_search():
    at = AppTest.from_file(str(TOPICS_DIR / "mlsd_visual_search.py"))
    at.run()
    assert not at.exception

    # переключить категорию запроса
    if len(at.selectbox) > 0:
        options = at.selectbox[0].options
        at.selectbox[0].set_value(options[1]).run(timeout=60)
        assert not at.exception

    if len(at.slider) > 0:
        at.slider[0].set_value(60).run(timeout=60)
        assert not at.exception

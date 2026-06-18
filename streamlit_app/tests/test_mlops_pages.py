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


def test_git_github():
    at = AppTest.from_file(str(TOPICS_DIR / "git_github.py"))
    at.run()
    assert not at.exception

    # клик по первой кнопке (git commit — добавляет коммит в граф)
    at.button[0].click().run(timeout=60)
    assert not at.exception

    # выбрать ответ в тренажёре и проверить его
    if len(at.radio) > 0:
        at.radio[0].set_value(at.radio[0].options[0]).run(timeout=60)
        assert not at.exception

    # найти и кликнуть кнопку "Проверить ответ"
    for b in at.button:
        if b.label == "Проверить ответ":
            b.click().run(timeout=60)
            break
    assert not at.exception


def test_cli_basics():
    at = AppTest.from_file(str(TOPICS_DIR / "cli_basics.py"))
    at.run()
    assert not at.exception

    # выбрать команду и выполнить её через безопасный whitelist
    assert len(at.selectbox) > 0
    at.selectbox[0].set_value(at.selectbox[0].options[0]).run(timeout=60)
    assert not at.exception
    for b in at.button:
        if b.label == "Выполнить команду":
            b.click().run(timeout=60)
            break
    assert not at.exception

    # ввести ответ в тренажёре и проверить его
    if len(at.text_input) > 0:
        at.text_input[0].set_value("pwd").run(timeout=60)
        assert not at.exception
    for b in at.button:
        if b.label == "Проверить":
            b.click().run(timeout=60)
            break
    assert not at.exception

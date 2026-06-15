"""Smoke tests for the LeetCode demo pages using streamlit.testing.v1.AppTest.

Run with: pytest streamlit_app/tests/test_leetcode_pages.py
(from the streamlit_app directory, or adjust paths accordingly).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_two_sum():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_two_sum.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception


def test_longest_substring():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_longest_substring.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("3" in s.value for s in at.success)


def test_valid_parentheses_valid_case():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_valid_parentheses.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) == 1


def test_valid_parentheses_invalid_case():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_valid_parentheses.py"))
    at.run()
    at.text_input[0].set_value("([)]").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) >= 1


def test_number_of_islands():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_number_of_islands.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("3" in s.value for s in at.success)


def test_merge_intervals():
    at = AppTest.from_file(str(TOPICS_DIR / "leetcode_merge_intervals.py"))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("[1, 6]" in s.value for s in at.success)

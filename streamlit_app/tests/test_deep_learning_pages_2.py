"""Smoke tests for the second batch of Deep Learning topic pages
(weight initialization, regularization, vanishing gradients/ResNet, CNN, RNN/LSTM/GRU)
using streamlit.testing.v1.AppTest.

Run with: pytest streamlit_app/tests/test_deep_learning_pages_2.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_weight_initialization_loads():
    """Weight initialization page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "weight_initialization.py"))
    at.run(timeout=60)
    assert not at.exception


def test_weight_initialization_radio_scheme():
    """Switching init scheme radio re-renders correctly."""
    at = AppTest.from_file(str(TOPICS_DIR / "weight_initialization.py"))
    at.run(timeout=60)
    assert not at.exception
    for radio in at.radio:
        if "Xavier" in str(radio.options) or "He" in str(radio.options):
            for opt in radio.options:
                radio.set_value(opt).run(timeout=60)
                assert not at.exception
            break


def test_weight_initialization_slider_depth():
    """Changing depth slider on weight init page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "weight_initialization.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        at.slider[0].set_value(20).run(timeout=60)
        assert not at.exception


def test_dl_regularization_loads():
    """Regularization page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "dl_regularization.py"))
    at.run(timeout=60)
    assert not at.exception


def test_dl_regularization_dropout_slider():
    """Changing dropout rate slider re-renders without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "dl_regularization.py"))
    at.run(timeout=60)
    assert not at.exception
    for slider in at.slider:
        if hasattr(slider, "_value") and isinstance(slider.value, float) and 0.0 <= slider.value <= 0.8:
            slider.set_value(0.5).run(timeout=60)
            assert not at.exception
            break


def test_dl_regularization_weight_decay_slider():
    """Changing weight decay lambda slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "dl_regularization.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        at.slider[-1].set_value(50).run(timeout=60)
        assert not at.exception


def test_vanishing_gradients_loads():
    """Vanishing gradients / ResNet page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "vanishing_gradients_resnet.py"))
    at.run(timeout=60)
    assert not at.exception


def test_vanishing_gradients_checkbox_residual():
    """Toggling residual connections checkbox works."""
    at = AppTest.from_file(str(TOPICS_DIR / "vanishing_gradients_resnet.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.checkbox) > 0:
        current_val = at.checkbox[0].value
        at.checkbox[0].set_value(not current_val).run(timeout=60)
        assert not at.exception


def test_vanishing_gradients_radio_activation():
    """Switching activation radio button works."""
    at = AppTest.from_file(str(TOPICS_DIR / "vanishing_gradients_resnet.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.radio) > 0:
        options = at.radio[0].options
        for opt in options:
            at.radio[0].set_value(opt).run(timeout=60)
            assert not at.exception


def test_vanishing_gradients_depth_slider():
    """Changing network depth slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "vanishing_gradients_resnet.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        at.slider[0].set_value(60).run(timeout=60)
        assert not at.exception


def test_cnn_basics_loads():
    """CNN basics page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "cnn_basics.py"))
    at.run(timeout=60)
    assert not at.exception


def test_cnn_basics_image_selectbox():
    """Switching test image selectbox works."""
    at = AppTest.from_file(str(TOPICS_DIR / "cnn_basics.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.selectbox) > 0:
        options = at.selectbox[0].options
        for opt in options:
            at.selectbox[0].set_value(opt).run(timeout=60)
            assert not at.exception


def test_cnn_basics_kernel_selectbox():
    """Switching kernel selectbox (including custom kernel) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "cnn_basics.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.selectbox) > 1:
        options = at.selectbox[1].options
        for opt in options:
            at.selectbox[1].set_value(opt).run(timeout=60)
            assert not at.exception


def test_cnn_basics_pooling_radio():
    """Switching pooling type radio button works."""
    at = AppTest.from_file(str(TOPICS_DIR / "cnn_basics.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.radio) > 0:
        options = at.radio[0].options
        for opt in options:
            at.radio[0].set_value(opt).run(timeout=60)
            assert not at.exception


def test_rnn_lstm_gru_loads():
    """RNN/LSTM/GRU page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "rnn_lstm_gru.py"))
    at.run(timeout=60)
    assert not at.exception


def test_rnn_lstm_gru_seq_len_slider():
    """Changing sequence length slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "rnn_lstm_gru.py"))
    at.run(timeout=60)
    assert not at.exception
    for slider in at.slider:
        if hasattr(slider, "_value") and isinstance(slider.value, int) and 10 <= slider.value <= 100:
            slider.set_value(80).run(timeout=60)
            assert not at.exception
            break


def test_rnn_lstm_gru_forget_bias_slider():
    """Changing forget gate bias slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "rnn_lstm_gru.py"))
    at.run(timeout=60)
    assert not at.exception
    for slider in at.slider:
        if hasattr(slider, "_value") and isinstance(slider.value, float) and -2.0 <= slider.value <= 3.0:
            slider.set_value(2.0).run(timeout=60)
            assert not at.exception
            break


def test_rnn_lstm_gru_step_slider():
    """Changing the gate-view step slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "rnn_lstm_gru.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        at.slider[-1].set_value(2).run(timeout=60)
        assert not at.exception

"""Smoke tests for the Deep Learning topic pages using streamlit.testing.v1.AppTest.

Run with: pytest streamlit_app/tests/test_deep_learning_pages.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_mlp_loads():
    """MLP page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "mlp.py"))
    at.run(timeout=60)
    assert not at.exception


def test_mlp_run_button():
    """Clicking 'Запустить аппроксимацию' button on MLP page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "mlp.py"))
    at.run(timeout=60)
    assert not at.exception
    # Click the button that triggers numpy MLP training
    for b in at.button:
        if "аппроксимацию" in b.label.lower() or "запустить" in b.label.lower():
            b.click().run(timeout=120)
            break
    assert not at.exception


def test_mlp_slider_interaction():
    """Changing hidden layer size slider on MLP page re-renders without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "mlp.py"))
    at.run(timeout=60)
    assert not at.exception
    # Adjust first slider (n_input)
    if len(at.slider) > 0:
        at.slider[0].set_value(4).run(timeout=60)
        assert not at.exception


def test_activation_functions_loads():
    """Activation functions page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "activation_functions.py"))
    at.run(timeout=60)
    assert not at.exception


def test_activation_functions_toggle_derivatives():
    """Toggling the derivatives checkbox on activation functions page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "activation_functions.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.checkbox) > 0:
        # Toggle derivatives checkbox off
        current_val = at.checkbox[0].value
        at.checkbox[0].set_value(not current_val).run(timeout=60)
        assert not at.exception


def test_activation_functions_slider_saturation():
    """Moving the depth slider on activation functions page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "activation_functions.py"))
    at.run(timeout=60)
    assert not at.exception
    # There should be a slider for network depth (vanishing gradient section)
    # Find sliders and set one to a new value
    if len(at.slider) >= 2:
        at.slider[1].set_value(15).run(timeout=60)
        assert not at.exception


def test_backpropagation_loads():
    """Backpropagation page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "backpropagation.py"))
    at.run(timeout=60)
    assert not at.exception


def test_backpropagation_slider_interaction():
    """Changing weight sliders on backpropagation page recomputes gradients."""
    at = AppTest.from_file(str(TOPICS_DIR / "backpropagation.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        # Adjust input x slider
        at.slider[0].set_value(2.0).run(timeout=60)
        assert not at.exception


def test_backpropagation_radio_interaction():
    """Changing radio in gradient checking section works."""
    at = AppTest.from_file(str(TOPICS_DIR / "backpropagation.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.radio) > 0:
        # Try switching activation in vanishing gradient demo
        options = at.radio[0].options
        if len(options) > 1:
            at.radio[0].set_value(options[1]).run(timeout=60)
            assert not at.exception


def test_loss_functions_loads():
    """Loss functions page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "loss_functions.py"))
    at.run(timeout=60)
    assert not at.exception


def test_loss_functions_slider_interaction():
    """Adjusting the outlier strength slider on loss functions page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "loss_functions.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        at.slider[0].set_value(10.0).run(timeout=60)
        assert not at.exception


def test_loss_functions_radio_interaction():
    """Switching y_true radio button on loss functions page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "loss_functions.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.radio) > 0:
        options = at.radio[0].options
        for opt in options:
            at.radio[0].set_value(opt).run(timeout=60)
            assert not at.exception


def test_optimizers_loads():
    """Optimizers page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "optimizers.py"))
    at.run(timeout=60)
    assert not at.exception


def test_optimizers_slider_lr():
    """Changing SGD learning rate slider on optimizers page re-renders correctly."""
    at = AppTest.from_file(str(TOPICS_DIR / "optimizers.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.slider) > 0:
        # First slider is lr_sgd
        at.slider[0].set_value(0.005).run(timeout=60)
        assert not at.exception


def test_optimizers_step_count():
    """Changing number of steps slider on optimizers page works."""
    at = AppTest.from_file(str(TOPICS_DIR / "optimizers.py"))
    at.run(timeout=60)
    assert not at.exception
    # Find n_steps slider (typically has range 50-1000)
    for slider in at.slider:
        if hasattr(slider, "_value") and 50 <= (slider._value or 0) <= 1000:
            slider.set_value(200).run(timeout=60)
            assert not at.exception
            break

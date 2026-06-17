"""Smoke tests for the new ML theory topic pages using streamlit.testing.v1.AppTest.

Covers: classification_metrics, gradient_descent, regularization,
cross_validation, decision_trees.

Run with: pytest streamlit_app/tests/test_ml_theory_pages.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_classification_metrics_loads():
    """Classification metrics page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "classification_metrics.py"))
    at.run(timeout=60)
    assert not at.exception


def test_classification_metrics_threshold_slider():
    """Moving the classification threshold slider updates metrics without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "classification_metrics.py"))
    at.run(timeout=60)
    assert not at.exception
    # Sliders: separation, n_per_class, threshold (last one).
    assert len(at.slider) >= 3
    at.slider[-1].set_value(0.8).run(timeout=60)
    assert not at.exception
    at.slider[-1].set_value(0.2).run(timeout=60)
    assert not at.exception


def test_classification_metrics_separation_slider():
    """Changing class separation slider re-renders correctly."""
    at = AppTest.from_file(str(TOPICS_DIR / "classification_metrics.py"))
    at.run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(3.0).run(timeout=60)
    assert not at.exception


def test_gradient_descent_loads():
    """Gradient descent page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "gradient_descent.py"))
    at.run(timeout=60)
    assert not at.exception


def test_gradient_descent_optimizer_switch():
    """Switching optimizer radio (Batch GD / Momentum / Adam) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "gradient_descent.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 0
    options = at.radio[0].options
    for opt in options:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception


def test_gradient_descent_lr_slider():
    """Changing learning rate slider re-computes the trajectory."""
    at = AppTest.from_file(str(TOPICS_DIR / "gradient_descent.py"))
    at.run(timeout=60)
    assert not at.exception
    # lr is the 3rd slider (a_coef, b_coef, lr, momentum_beta, n_steps).
    assert len(at.slider) >= 3
    at.slider[2].set_value(0.5).run(timeout=60)
    assert not at.exception


def test_gradient_descent_steps_slider():
    """Changing number of steps slider works for all optimizers."""
    at = AppTest.from_file(str(TOPICS_DIR / "gradient_descent.py"))
    at.run(timeout=60)
    assert not at.exception
    at.slider[-1].set_value(60).run(timeout=60)
    assert not at.exception


def test_regularization_loads():
    """Regularization page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "regularization.py"))
    at.run(timeout=60)
    assert not at.exception


def test_regularization_method_switch():
    """Switching regularization method radio (Ridge/Lasso/ElasticNet) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "regularization.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 0
    options = at.radio[0].options
    for opt in options:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception


def test_regularization_lambda_slider():
    """Changing log10(lambda) slider updates coefficients plot."""
    at = AppTest.from_file(str(TOPICS_DIR / "regularization.py"))
    at.run(timeout=60)
    assert not at.exception
    # last slider is log_lambda
    at.slider[-1].set_value(1.0).run(timeout=60)
    assert not at.exception
    at.slider[-1].set_value(-2.0).run(timeout=60)
    assert not at.exception


def test_cross_validation_loads():
    """Cross-validation page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "cross_validation.py"))
    at.run(timeout=60)
    assert not at.exception


def test_cross_validation_k_folds_slider():
    """Changing number of folds slider re-renders the fold visualization."""
    at = AppTest.from_file(str(TOPICS_DIR / "cross_validation.py"))
    at.run(timeout=60)
    assert not at.exception
    # second slider is k_folds
    assert len(at.slider) >= 2
    at.slider[1].set_value(7).run(timeout=60)
    assert not at.exception


def test_cross_validation_degree_slider():
    """Changing polynomial degree slider for hyperparameter search works."""
    at = AppTest.from_file(str(TOPICS_DIR / "cross_validation.py"))
    at.run(timeout=120)
    assert not at.exception
    # degree slider index: n_samples(0), k_folds(1), n_data(2), noise(3), degree(4), k_cv(5)
    if len(at.slider) >= 5:
        at.slider[4].set_value(3).run(timeout=120)
        assert not at.exception


def test_decision_trees_loads():
    """Decision trees page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "decision_trees.py"))
    at.run(timeout=60)
    assert not at.exception


def test_decision_trees_dataset_switch():
    """Switching dataset radio (moons / linearly separable) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "decision_trees.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 0
    options = at.radio[0].options
    for opt in options:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception


def test_decision_trees_max_depth_slider():
    """Changing max_depth slider updates decision boundary and tree plot."""
    at = AppTest.from_file(str(TOPICS_DIR / "decision_trees.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) > 0
    at.slider[0].set_value(8).run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(1).run(timeout=60)
    assert not at.exception


def test_decision_trees_criterion_switch():
    """Switching split criterion (gini/entropy) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "decision_trees.py"))
    at.run(timeout=60)
    assert not at.exception
    if len(at.radio) > 1:
        options = at.radio[1].options
        for opt in options:
            at.radio[1].set_value(opt).run(timeout=60)
            assert not at.exception

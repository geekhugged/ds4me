"""Smoke tests for the second batch of new ML theory topic pages using
streamlit.testing.v1.AppTest.

Covers: ensembles, svm, naive_bayes, mle, clustering.

Run with: pytest streamlit_app/tests/test_ml_theory_pages_2.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

TOPICS_DIR = Path(__file__).resolve().parent.parent / "topics"


def test_ensembles_loads():
    """Ensembles page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "ensembles.py"))
    at.run(timeout=120)
    assert not at.exception


def test_ensembles_n_estimators_slider():
    """Changing number of estimators slider re-renders without error."""
    at = AppTest.from_file(str(TOPICS_DIR / "ensembles.py"))
    at.run(timeout=120)
    assert not at.exception
    # sliders: noise(0), n_estimators(1), max_depth_single(2), learning_rate(3)
    assert len(at.slider) >= 2
    at.slider[1].set_value(20).run(timeout=120)
    assert not at.exception


def test_ensembles_max_depth_slider():
    """Changing single tree max depth slider works."""
    at = AppTest.from_file(str(TOPICS_DIR / "ensembles.py"))
    at.run(timeout=120)
    assert not at.exception
    if len(at.slider) >= 3:
        at.slider[2].set_value(2).run(timeout=120)
        assert not at.exception


def test_svm_loads():
    """SVM page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "svm.py"))
    at.run(timeout=60)
    assert not at.exception


def test_svm_kernel_switch():
    """Switching kernel radio (linear/poly/rbf) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "svm.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 0
    # second radio is kernel choice
    kernel_radio = at.radio[1] if len(at.radio) > 1 else at.radio[0]
    for opt in kernel_radio.options:
        kernel_radio.set_value(opt).run(timeout=60)
        assert not at.exception


def test_svm_dataset_switch():
    """Switching dataset radio works."""
    at = AppTest.from_file(str(TOPICS_DIR / "svm.py"))
    at.run(timeout=60)
    assert not at.exception
    options = at.radio[0].options
    for opt in options:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception


def test_svm_c_and_gamma_sliders():
    """Changing C and gamma sliders works."""
    at = AppTest.from_file(str(TOPICS_DIR / "svm.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) >= 2
    at.slider[0].set_value(10.0).run(timeout=60)
    assert not at.exception
    at.slider[1].set_value(2.0).run(timeout=60)
    assert not at.exception


def test_naive_bayes_loads():
    """Naive Bayes page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "naive_bayes.py"))
    at.run(timeout=60)
    assert not at.exception


def test_naive_bayes_prior_slider():
    """Changing disease prior slider updates posterior calculation."""
    at = AppTest.from_file(str(TOPICS_DIR / "naive_bayes.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) > 0
    at.slider[0].set_value(0.3).run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(0.01).run(timeout=60)
    assert not at.exception


def test_naive_bayes_sensitivity_specificity_sliders():
    """Changing sensitivity/specificity sliders works."""
    at = AppTest.from_file(str(TOPICS_DIR / "naive_bayes.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) >= 3
    at.slider[1].set_value(0.99).run(timeout=60)
    assert not at.exception
    at.slider[2].set_value(0.6).run(timeout=60)
    assert not at.exception


def test_naive_bayes_gaussian_class_sliders():
    """Changing 1D gaussian classifier sliders (mu0, mu1, sigma, prior) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "naive_bayes.py"))
    at.run(timeout=60)
    assert not at.exception
    # sliders: prior(0), sensitivity(1), specificity(2), mu0(3), mu1(4), sigma(5), class_prior1(6)
    if len(at.slider) >= 7:
        at.slider[6].set_value(0.8).run(timeout=60)
        assert not at.exception


def test_mle_loads():
    """MLE page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "mle.py"))
    at.run(timeout=60)
    assert not at.exception


def test_mle_sample_size_slider():
    """Changing sample size slider re-fits the MLE estimate."""
    at = AppTest.from_file(str(TOPICS_DIR / "mle.py"))
    at.run(timeout=60)
    assert not at.exception
    # sliders: true_mu(0), true_sigma(1), m(2), seed(3)
    assert len(at.slider) >= 3
    at.slider[2].set_value(200).run(timeout=60)
    assert not at.exception
    at.slider[2].set_value(10).run(timeout=60)
    assert not at.exception


def test_mle_true_params_sliders():
    """Changing true mu/sigma sliders works."""
    at = AppTest.from_file(str(TOPICS_DIR / "mle.py"))
    at.run(timeout=60)
    assert not at.exception
    at.slider[0].set_value(-2.0).run(timeout=60)
    assert not at.exception
    at.slider[1].set_value(2.5).run(timeout=60)
    assert not at.exception


def test_clustering_loads():
    """Clustering page renders without exceptions."""
    at = AppTest.from_file(str(TOPICS_DIR / "clustering.py"))
    at.run(timeout=60)
    assert not at.exception


def test_clustering_algorithm_switch():
    """Switching algorithm radio (k-means / agglomerative / DBSCAN) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "clustering.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.radio) > 1
    options = at.radio[1].options
    for opt in options:
        # re-fetch the radio each iteration: switching algorithm changes the
        # number of sliders below it, which rebuilds the widget tree.
        at.radio[1].set_value(opt).run(timeout=60)
        assert not at.exception


def test_clustering_dataset_switch():
    """Switching dataset radio (blobs / moons) works."""
    at = AppTest.from_file(str(TOPICS_DIR / "clustering.py"))
    at.run(timeout=60)
    assert not at.exception
    options = at.radio[0].options
    for opt in options:
        at.radio[0].set_value(opt).run(timeout=60)
        assert not at.exception


def test_clustering_kmeans_k_slider():
    """Changing k slider for k-means works."""
    at = AppTest.from_file(str(TOPICS_DIR / "clustering.py"))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.slider) >= 2
    # second slider after noise should be k (since k-means is default algorithm)
    at.slider[1].set_value(5).run(timeout=60)
    assert not at.exception


def test_clustering_dbscan_params():
    """Switching to DBSCAN and adjusting eps/min_samples sliders works."""
    at = AppTest.from_file(str(TOPICS_DIR / "clustering.py"))
    at.run(timeout=60)
    assert not at.exception
    algo_radio = at.radio[1]
    algo_radio.set_value("DBSCAN").run(timeout=60)
    assert not at.exception
    if len(at.slider) >= 3:
        at.slider[1].set_value(0.5).run(timeout=60)
        assert not at.exception
        at.slider[2].set_value(10).run(timeout=60)
        assert not at.exception

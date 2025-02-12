import logging

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted
from hiclass import LocalClassifierPerLevel
from hiclass._calibration.Calibrator import _Calibrator
from hiclass.HierarchicalClassifier import make_leveled


@parametrize_with_checks([LocalClassifierPerLevel()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerLevel(
        local_classifier=LogisticRegression(), calibration_method="ivap"
    )
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.y_cal = np.array([["a", "b"], ["a", "c"]])
    digraph.X_cal = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPL")
    digraph.root_ = "a"
    digraph.sample_weight_ = None
    digraph.separator_ = "::HiClass::Separator::"
    digraph.masks_ = [
        [True, True],
        [True, True],
    ]
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for classifier in digraph_logistic_regression.local_classifiers_:
        assert isinstance(
            classifier,
            LogisticRegression,
        )


def test_initialize_local_calibrators(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    digraph_logistic_regression._initialize_local_calibrators()
    for calibrator in digraph_logistic_regression.local_calibrators_:
        assert isinstance(calibrator, _Calibrator)


def test_fit_digraph(digraph_logistic_regression):
    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    digraph_logistic_regression.n_jobs = 2
    digraph_logistic_regression.local_classifiers_ = classifiers
    digraph_logistic_regression._fit_digraph(local_mode=True)
    for classifier in digraph_logistic_regression.local_classifiers_:
        try:
            check_is_fitted(classifier)
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_calibrate_digraph(digraph_logistic_regression):
    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    digraph_logistic_regression.n_jobs = 2
    digraph_logistic_regression.local_classifiers_ = classifiers
    digraph_logistic_regression._fit_digraph(local_mode=True)

    calibrators = [
        _Calibrator(classifier)
        for classifier in digraph_logistic_regression.local_classifiers_
    ]
    digraph_logistic_regression.local_calibrators_ = calibrators
    digraph_logistic_regression._calibrate_digraph(local_mode=True)

    try:
        check_is_fitted(digraph_logistic_regression.local_calibrators_[1])
    except NotFittedError as e:
        pytest.fail(repr(e))
    assert 1


def test_fit_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    digraph_logistic_regression.n_jobs = 2
    digraph_logistic_regression.local_classifiers_ = classifiers

    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)
    for classifier in digraph_logistic_regression.local_classifiers_:
        try:
            check_is_fitted(classifier)
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_calibrate_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    digraph_logistic_regression.n_jobs = 2
    digraph_logistic_regression.local_classifiers_ = classifiers
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)

    calibrators = [
        _Calibrator(classifier)
        for classifier in digraph_logistic_regression.local_classifiers_
    ]
    digraph_logistic_regression.local_calibrators_ = calibrators
    digraph_logistic_regression._calibrate_digraph(local_mode=True, use_joblib=True)

    try:
        check_is_fitted(digraph_logistic_regression.local_calibrators_[1])
    except NotFittedError as e:
        pytest.fail(repr(e))
    assert 1


@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerLevel(
        local_classifier=LogisticRegression(),
        return_all_probabilities=True,
        calibration_method="ivap",
        probability_combiner=None,
    )

    digraph.separator_ = "::HiClass::Separator::"
    digraph.hierarchy_ = nx.DiGraph(
        [("r", "1"), ("r", "2"), ("1", "1.1"), ("1", "1.2"), ("2", "2.1"), ("2", "2.2")]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPL")
    digraph.max_levels_ = 2

    # for predict_proba
    tmp_labels = digraph._disambiguate(make_leveled(digraph.y_))
    digraph.max_level_dimensions_ = np.array(
        [len(np.unique(tmp_labels[:, level])) for level in range(tmp_labels.shape[1])]
    )
    digraph.global_classes_ = [
        np.unique(tmp_labels[:, level]).astype("str")
        for level in range(tmp_labels.shape[1])
    ]
    digraph.global_class_to_index_mapping_ = [
        {
            digraph.global_classes_[level][index]: index
            for index in range(len(digraph.global_classes_[level]))
        }
        for level in range(tmp_labels.shape[1])
    ]

    classes_ = [digraph.global_classes_[0]]
    for level in range(1, digraph.max_levels_):
        classes_.append(
            np.sort(
                np.unique(
                    [
                        label.split(digraph.separator_)[level]
                        for label in digraph.global_classes_[level]
                    ]
                )
            )
        )
    digraph.classes_ = classes_
    digraph.class_to_index_mapping_ = [
        {local_labels[index]: index for index in range(len(local_labels))}
        for local_labels in classes_
    ]

    digraph.dtype_ = "<U3"
    digraph.root_ = "r"
    digraph.masks_ = [
        [True, True, True, True],
        [True, True, True, True],
    ]
    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    classifiers[0].fit(digraph.X_, ["1", "1", "2", "2"])
    classifiers[1].fit(digraph.X_, ["1.1", "1.2", "2.1", "2.2"])
    digraph.local_classifiers_ = classifiers
    return digraph


def test_predict(fitted_logistic_regression):
    ground_truth = np.array([["2", "2.2"], ["2", "2.1"], ["1", "1.2"], ["1", "1.1"]])
    prediction = fitted_logistic_regression.predict([[7, 8], [5, 6], [3, 4], [1, 2]])
    assert_array_equal(ground_truth, prediction)


def test_predict_sparse(fitted_logistic_regression):
    ground_truth = np.array([["2", "2.2"], ["2", "2.1"], ["1", "1.2"], ["1", "1.1"]])
    prediction = fitted_logistic_regression.predict(
        csr_matrix([[7, 8], [5, 6], [3, 4], [1, 2]])
    )
    assert_array_equal(ground_truth, prediction)


def test_predict_proba(fitted_logistic_regression):
    proba = fitted_logistic_regression.predict_proba([[7, 8], [5, 6], [3, 4], [1, 2]])
    assert len(proba) == 2
    assert proba[0].shape == (4, 2)
    assert proba[1].shape == (4, 4)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )


def test_predict_proba_sparse(fitted_logistic_regression):
    proba = fitted_logistic_regression.predict_proba(
        csr_matrix([[7, 8], [5, 6], [3, 4], [1, 2]])
    )
    assert len(proba) == 2
    assert proba[0].shape == (4, 2)
    assert proba[1].shape == (4, 4)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )


def test_fit_predict():
    lcpl = LocalClassifierPerLevel(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    ground_truth = np.array(
        [["a", "b"], ["a::HiClass::Separator::b", "b::HiClass::Separator::c"]]
    )
    lcpl.fit(x, y)
    for level, classifier in enumerate(lcpl.local_classifiers_):
        try:
            check_is_fitted(classifier)
            assert_array_equal(ground_truth[level], classifier.classes_)
        except NotFittedError as e:
            pytest.fail(repr(e))
    predictions = lcpl.predict(x)
    assert_array_equal(y, predictions)


def test_fit_calibrate_predict_proba():
    lcpl = LocalClassifierPerLevel(
        local_classifier=LogisticRegression(),
        return_all_probabilities=True,
        calibration_method="ivap",
        probability_combiner="geometric",
    )

    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    ground_truth = np.array(
        [["a", "b"], ["a::HiClass::Separator::b", "b::HiClass::Separator::c"]]
    )
    lcpl.fit(x, y)
    lcpl.calibrate(x, y)
    for level, classifier in enumerate(lcpl.local_classifiers_):
        try:
            check_is_fitted(classifier)
            assert_array_equal(ground_truth[level], classifier.classes_)
        except NotFittedError as e:
            pytest.fail(repr(e))
    predictions = lcpl.predict(x)
    proba = lcpl.predict_proba(x)
    assert_array_equal(y, predictions)

    assert len(proba) == 2
    assert proba[0].shape == (2, 2)
    assert proba[1].shape == (2, 2)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )

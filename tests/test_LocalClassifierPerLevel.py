import logging

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from hiclass import LocalClassifierPerLevel


@parametrize_with_checks([LocalClassifierPerLevel()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerLevel(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPL")
    digraph.root_ = "a"
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

def test_fit_digraph_joblib_parallel(digraph_logistic_regression):
    from joblib import Parallel, delayed
    LocalClassifierPerLevel._has_ray = False

    classifiers = [
        LogisticRegression(),
        LogisticRegression(),
    ]
    digraph_logistic_regression.n_jobs = 2
    digraph_logistic_regression.local_classifiers_ = classifiers
    from joblib import Parallel, delayed, effective_n_jobs
    digraph_logistic_regression._fit_digraph(local_mode=True)
    for classifier in digraph_logistic_regression.local_classifiers_:
        try:
            check_is_fitted(classifier)
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_fit_1_class():
    lcpl = LocalClassifierPerLevel(local_classifier=LogisticRegression(), n_jobs=2)
    y = np.array([["1", "2"]])
    X = np.array([[1, 2]])
    ground_truth = np.array([["1", "2"]])
    lcpl.fit(X, y)
    prediction = lcpl.predict(X)
    assert_array_equal(ground_truth, prediction)






@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerLevel(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph(
        [("r", "1"), ("r", "2"), ("1", "1.1"), ("1", "1.2"), ("2", "2.1"), ("2", "2.2")]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPL")
    digraph.max_levels_ = 2
    digraph.dtype_ = "<U3"
    digraph.root_ = "r"
    digraph.separator_ = "::HiClass::Separator::"
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


@pytest.fixture
def empty_levels():
    X = [
        [1],
        [2],
        [3],
    ]
    y = [
        ["1"],
        ["2", "2.1"],
        ["3", "3.1", "3.1.2"],
    ]
    return X, y


def test_empty_levels(empty_levels):
    lcppn = LocalClassifierPerLevel()
    X, y = empty_levels
    lcppn.fit(X, y)
    predictions = lcppn.predict(X)
    ground_truth = [
        ["1", "", ""],
        ["2", "2.1", ""],
        ["3", "3.1", "3.1.2"],
    ]
    assert list(lcppn.hierarchy_.nodes) == [
        "1",
        "2",
        "2" + lcppn.separator_ + "2.1",
        "3",
        "3" + lcppn.separator_ + "3.1",
        "3" + lcppn.separator_ + "3.1" + lcppn.separator_ + "3.1.2",
        lcppn.root_,
    ]
    assert_array_equal(ground_truth, predictions)

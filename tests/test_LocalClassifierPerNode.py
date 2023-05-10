import logging

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from hiclass import LocalClassifierPerNode
from hiclass.BinaryPolicy import ExclusivePolicy
from hiclass.ConstantClassifier import ConstantClassifier


@parametrize_with_checks([LocalClassifierPerNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_with_policy():
    digraph = LocalClassifierPerNode(binary_policy="exclusive")
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.X_ = np.array([1, 2])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.sample_weight_ = None
    return digraph


def test_initialize_binary_policy(digraph_with_policy):
    digraph_with_policy._initialize_binary_policy()
    assert isinstance(digraph_with_policy.binary_policy_, ExclusivePolicy)


@pytest.fixture
def digraph_with_unknown_policy():
    digraph = LocalClassifierPerNode(binary_policy="unknown")
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
    return digraph


def test_initialize_unknown_binary_policy(digraph_with_unknown_policy):
    with pytest.raises(KeyError):
        digraph_with_unknown_policy._initialize_binary_policy()


@pytest.fixture
def digraph_with_object_policy():
    digraph = LocalClassifierPerNode(binary_policy=ExclusivePolicy)
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
    return digraph


def test_initialize_object_binary_policy(digraph_with_object_policy):
    with pytest.raises(ValueError):
        digraph_with_object_policy._initialize_binary_policy()


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.binary_policy_ = ExclusivePolicy(digraph.hierarchy_, digraph.X_, digraph.y_)
    digraph.sample_weight_ = None
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node != digraph_logistic_regression.root_:
            assert isinstance(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"],
                LogisticRegression,
            )
        else:
            with pytest.raises(KeyError):
                isinstance(
                    digraph_logistic_regression.hierarchy_.nodes[node]["classifier"],
                    LogisticRegression,
                )


def test_fit_digraph(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)
    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_fit_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)
    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_fit_1_class():
    lcpn = LocalClassifierPerNode(local_classifier=LogisticRegression(), n_jobs=2)
    y = np.array([["1", "2"]])
    X = np.array([[1, 2]])
    ground_truth = np.array([["1", "2"]])
    lcpn.fit(X, y)
    prediction = lcpn.predict(X)
    assert_array_equal(ground_truth, prediction)


def test_clean_up(digraph_logistic_regression):
    digraph_logistic_regression._clean_up()
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.X_ is None
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.y_ is None
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.binary_policy_ is None


@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph(
        [("r", "1"), ("r", "2"), ("1", "1.1"), ("1", "1.2"), ("2", "2.1"), ("2", "2.2")]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.max_levels_ = 2
    digraph.dtype_ = "<U3"
    digraph.root_ = "r"
    digraph.separator_ = "::HiClass::Separator::"
    classifiers = {
        "1": {"classifier": LogisticRegression()},
        "1.1": {"classifier": LogisticRegression()},
        "1.2": {"classifier": LogisticRegression()},
        "2": {"classifier": LogisticRegression()},
        "2.1": {"classifier": LogisticRegression()},
        "2.2": {"classifier": LogisticRegression()},
    }
    classifiers["1"]["classifier"].fit(digraph.X_, [1, 1, 0, 0])
    classifiers["1.1"]["classifier"].fit(digraph.X_, [1, 0, 0, 0])
    classifiers["1.2"]["classifier"].fit(digraph.X_, [0, 1, 0, 0])
    classifiers["2"]["classifier"].fit(digraph.X_, [0, 0, 1, 1])
    classifiers["2.1"]["classifier"].fit(digraph.X_, [0, 0, 1, 0])
    classifiers["2.2"]["classifier"].fit(digraph.X_, [0, 0, 0, 1])
    nx.set_node_attributes(digraph.hierarchy_, classifiers)
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
    lcpn = LocalClassifierPerNode(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcpn.fit(x, y)
    predictions = lcpn.predict(x)
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
    lcppn = LocalClassifierPerNode()
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


def test_fit_bert():
    bert = ConstantClassifier()
    lcpn = LocalClassifierPerNode(
        local_classifier=bert,
        bert=True,
    )
    X = ["Text 1", "Text 2"]
    y = ["a", "a"]
    lcpn.fit(X, y)
    check_is_fitted(lcpn)
    predictions = lcpn.predict(X)
    assert_array_equal(y, predictions)


def test_knn():
    knn = KNeighborsClassifier(
        n_neighbors=2,
    )
    lcpn = LocalClassifierPerNode(
        local_classifier=knn,
    )
    y = np.array([["a", "b"], ["a", "c"]])
    X = np.array([[1, 2], [3, 4]])
    lcpn.fit(X, y)
    check_is_fitted(lcpn)
    # predictions = lcpn.predict(X)
    # assert_array_equal(y, predictions)

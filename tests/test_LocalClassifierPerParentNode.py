import logging
import tempfile

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from hiclass import LocalClassifierPerParentNode
from hiclass.ConstantClassifier import ConstantClassifier


@parametrize_with_checks([LocalClassifierPerParentNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_logistic_regression_2d():
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.sample_weight_ = None
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression_2d):
    digraph_logistic_regression_2d._initialize_local_classifiers()
    for node in digraph_logistic_regression_2d.hierarchy_.nodes:
        if node == digraph_logistic_regression_2d.root_:
            assert isinstance(
                digraph_logistic_regression_2d.hierarchy_.nodes[node]["classifier"],
                LogisticRegression,
            )
        else:
            with pytest.raises(KeyError):
                isinstance(
                    digraph_logistic_regression_2d.hierarchy_.nodes[node]["classifier"],
                    LogisticRegression,
                )


@pytest.mark.parametrize("use_joblib", [True, False])
def test_fit_digraph_2d(digraph_logistic_regression_2d, use_joblib):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression_2d.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression_2d.hierarchy_, classifiers)
    digraph_logistic_regression_2d._fit_digraph(local_mode=True, use_joblib=use_joblib)
    try:
        check_is_fitted(
            digraph_logistic_regression_2d.hierarchy_.nodes["a"]["classifier"]
        )
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression_2d.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


@pytest.fixture
def digraph_logistic_regression_3d():
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("d", "e"),
            ("d", "f"),
        ]
    )
    digraph.y_ = np.array(
        [
            [["a", "b"], ["", ""]],
            [["a", "c"], ["", ""]],
            [["a", "b"], ["a", "c"]],
            [["d", "e"], ["d", "f"]],
        ]
    )
    digraph.X_ = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]
    )
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.sample_weight_ = None
    classifiers = {
        "a": {"classifier": LogisticRegression()},
        "d": {"classifier": LogisticRegression()},
    }
    digraph.n_jobs = 2
    nx.set_node_attributes(digraph.hierarchy_, classifiers)
    return digraph


@pytest.mark.parametrize("use_joblib", [True, False])
def test_fit_digraph_3d(digraph_logistic_regression_3d, use_joblib):
    digraph_logistic_regression_3d._fit_digraph(local_mode=True, use_joblib=use_joblib)
    try:
        for node in ["a", "d"]:
            check_is_fitted(
                digraph_logistic_regression_3d.hierarchy_.nodes[node]["classifier"]
            )
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c", "e", "f"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression_3d.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


def test_fit_1_class():
    lcppn = LocalClassifierPerParentNode(
        local_classifier=LogisticRegression(), n_jobs=2
    )
    y = np.array([["1", "2"]])
    X = np.array([[1, 2]])
    ground_truth = np.array([["1", "2"]])
    lcppn.fit(X, y)
    prediction = lcppn.predict(X)
    assert_array_equal(ground_truth, prediction)


@pytest.fixture
def digraph_2d():
    classifier = LocalClassifierPerParentNode()
    classifier.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    classifier.logger_ = logging.getLogger("HC")
    classifier.edge_list = tempfile.TemporaryFile()
    classifier.separator_ = "::HiClass::Separator::"
    return classifier


def test_get_parents(digraph_2d):
    ground_truth = np.array(["a", "b", "d", "e"])
    nodes = digraph_2d._get_parents()
    assert_array_equal(ground_truth, nodes)


@pytest.fixture
def x_and_y_arrays_2d():
    graph = LocalClassifierPerParentNode()
    graph.X_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    graph.y_ = np.array([["a", "b", "c"], ["a", "e", "f"], ["d", "g", "h"]])
    graph.hierarchy_ = nx.DiGraph(
        [("a", "b"), ("b", "c"), ("a", "e"), ("e", "f"), ("d", "g"), ("g", "h")]
    )
    graph.root_ = "r"
    graph.sample_weight_ = None
    return graph


def test_get_successors_2d_1(x_and_y_arrays_2d):
    x, y, weights = x_and_y_arrays_2d._get_successors("a")
    assert_array_equal(x_and_y_arrays_2d.X_[0:2], x)
    assert_array_equal(["b", "e"], y)
    assert weights is None


def test_get_successors_2d_2(x_and_y_arrays_2d):
    x, y, weights = x_and_y_arrays_2d._get_successors("d")
    assert_array_equal([x_and_y_arrays_2d.X_[-1]], x)
    assert_array_equal(["g"], y)
    assert weights is None


def test_get_successors_2d_3(x_and_y_arrays_2d):
    x, y, weights = x_and_y_arrays_2d._get_successors("b")
    assert_array_equal([x_and_y_arrays_2d.X_[0]], x)
    assert_array_equal(["c"], y)
    assert weights is None


@pytest.fixture
def x_and_y_arrays_3d():
    graph = LocalClassifierPerParentNode()
    graph.X_ = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            # Multi-label
            [10, 11, 12],
            [13, 14, 15],
        ]
    )
    graph.y_ = np.array(
        [
            [["a", "b", "c"], ["", "", ""]],
            [["a", "e", "f"], ["", "", ""]],
            [["d", "g", "h"], ["", "", ""]],
            # Multi-label
            [["a", "b", "c"], ["a", "e", "f"]],
            [["a", "b", "c"], ["d", "g", "h"]],
        ]
    )
    graph.hierarchy_ = nx.DiGraph(
        [("a", "b"), ("b", "c"), ("a", "e"), ("e", "f"), ("d", "g"), ("g", "h")]
    )
    graph.root_ = "r"
    graph.sample_weight_ = None
    return graph


def test_get_successors_3d_1(x_and_y_arrays_3d):
    x, y, weights = x_and_y_arrays_3d._get_successors("a")
    ground_truth_x = np.array(
        [[1, 2, 3], [4, 5, 6], [10, 11, 12], [10, 11, 12], [13, 14, 15]]
    )
    ground_truth_y = np.array(["b", "e", "b", "e", "b"])
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_get_successors_3d_2(x_and_y_arrays_3d):
    x, y, weights = x_and_y_arrays_3d._get_successors("d")
    ground_truth_x = x_and_y_arrays_3d.X_[[False, False, True, False, True]]
    ground_truth_y = np.array(["g", "g"])
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_get_successors_3d_3(x_and_y_arrays_3d):
    x, y, weights = x_and_y_arrays_3d._get_successors("b")
    ground_truth_x = x_and_y_arrays_3d.X_[[True, False, False, True, True]]
    ground_truth_y = np.array(["c", "c", "c"])
    assert_array_equal(ground_truth_x, x)
    assert ground_truth_y.shape == y.shape
    assert_array_equal(ground_truth_y, y)
    assert weights is None


@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph(
        [("r", "1"), ("r", "2"), ("1", "1.1"), ("1", "1.2"), ("2", "2.1"), ("2", "2.2")]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.max_levels_ = 2
    digraph.dtype_ = "<U3"
    digraph.root_ = "r"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.ndim_ = 2
    classifiers = {
        "r": {"classifier": LogisticRegression()},
        "1": {"classifier": LogisticRegression()},
        "2": {"classifier": LogisticRegression()},
    }
    classifiers["r"]["classifier"].fit(digraph.X_, ["1", "1", "2", "2"])
    classifiers["1"]["classifier"].fit(digraph.X_[:2], ["1.1", "1.2"])
    classifiers["2"]["classifier"].fit(digraph.X_[2:], ["2.1", "2.2"])
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
    lcppn = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcppn.fit(x, y)
    predictions = lcppn.predict(x)
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
    lcppn = LocalClassifierPerParentNode()
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


def test_bert():
    bert = ConstantClassifier()
    lcpn = LocalClassifierPerParentNode(
        local_classifier=bert,
        bert=True,
    )
    X = ["Text 1", "Text 2"]
    y = ["a", "a"]
    lcpn.fit(X, y)
    check_is_fitted(lcpn)
    predictions = lcpn.predict(X)
    assert_array_equal(y, predictions)

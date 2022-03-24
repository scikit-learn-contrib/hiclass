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


@parametrize_with_checks([LocalClassifierPerParentNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def ambiguous_node():
    graph = LocalClassifierPerParentNode()
    graph.y_ = np.array([["a", "b"], ["b", "c"]])
    return graph


def test_disambiguate(ambiguous_node):
    ground_truth = np.array(
        [["a", "a::HiClass::Separator::b"], ["b", "b::HiClass::Separator::c"]]
    )
    ambiguous_node._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node.y_)


@pytest.fixture
def graph_1d():
    graph = LocalClassifierPerParentNode()
    graph.y_ = np.array(["a", "b", "c", "d"])
    graph.logger_ = logging.getLogger("LCPPN")
    return graph


def test_create_digraph_1d(graph_1d):
    ground_truth = nx.DiGraph()
    ground_truth.add_nodes_from(np.array(["a", "b", "c", "d"]))
    graph_1d._create_digraph()
    assert nx.is_isomorphic(ground_truth, graph_1d.hierarchy_)
    assert list(ground_truth.nodes) == list(graph_1d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(graph_1d.hierarchy_.edges)


@pytest.fixture
def graph_1d_disguised_as_2d():
    graph = LocalClassifierPerParentNode()
    graph.y_ = np.array([["a"], ["b"], ["c"], ["d"]])
    graph.logger_ = logging.getLogger("LCPN")
    return graph


def test_create_digraph_1d_disguised_as_2d(graph_1d_disguised_as_2d):
    ground_truth = nx.DiGraph()
    ground_truth.add_nodes_from(np.array(["a", "b", "c", "d"]))
    graph_1d_disguised_as_2d._create_digraph()
    assert nx.is_isomorphic(ground_truth, graph_1d_disguised_as_2d.hierarchy_)
    assert list(ground_truth.nodes) == list(graph_1d_disguised_as_2d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(graph_1d_disguised_as_2d.hierarchy_.edges)


@pytest.fixture
def digraph_2d():
    dag_2d = LocalClassifierPerParentNode()
    dag_2d.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    dag_2d.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    dag_2d.logger_ = logging.getLogger("LCPPN")
    dag_2d.edge_list = tempfile.TemporaryFile()
    dag_2d.separator_ = "::HiClass::Separator::"
    return dag_2d


def test_create_digraph_2d(digraph_2d):
    ground_truth = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    digraph_2d._create_digraph()
    assert nx.is_isomorphic(ground_truth, digraph_2d.hierarchy_)
    assert list(ground_truth.nodes) == list(digraph_2d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(digraph_2d.hierarchy_.edges)


@pytest.fixture
def digraph_3d():
    dag_3d = LocalClassifierPerParentNode()
    dag_3d.y_ = np.arange(27).reshape((3, 3, 3))
    dag_3d.logger_ = logging.getLogger("LCPPN")
    return dag_3d


def test_create_digraph_3d(digraph_3d):
    with pytest.raises(ValueError):
        digraph_3d._create_digraph()


def test_convert_1d_y_to_2d(graph_1d):
    ground_truth = np.array([["a"], ["b"], ["c"], ["d"]])
    graph_1d._convert_1d_y_to_2d()
    assert_array_equal(ground_truth, graph_1d.y_)


def test_export_digraph(digraph_2d):
    ground_truth = b'"a","b",{}\n"b","c",{}\n"d","e",{}\n"e","f",{}\n'
    digraph_2d._export_digraph()
    digraph_2d.edge_list.seek(0)
    assert digraph_2d.edge_list.read() == ground_truth


@pytest.fixture
def cyclic_graph():
    graph = LocalClassifierPerParentNode()
    graph.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    graph.logger_ = logging.getLogger("LCPPN")
    return graph


def test_assert_digraph_is_dag(cyclic_graph):
    with pytest.raises(ValueError):
        cyclic_graph._assert_digraph_is_dag()


@pytest.fixture
def digraph_one_root():
    digraph = LocalClassifierPerParentNode()
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    return digraph


def test_add_artificial_root(digraph_one_root):
    digraph_one_root._add_artificial_root()
    successors = list(digraph_one_root.hierarchy_.successors("hiclass::root"))
    root = [
        node
        for node, in_degree in digraph_one_root.hierarchy_.in_degree()
        if in_degree == 0
    ]
    assert ["a"] == successors
    assert ["hiclass::root"] == root
    assert "hiclass::root" == digraph_one_root.root_


@pytest.fixture
def digraph_multiple_roots():
    digraph = LocalClassifierPerParentNode()
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("c", "d"), ("e", "f")])
    return digraph


def test_add_artificial_root_multiple_roots(digraph_multiple_roots):
    digraph_multiple_roots._add_artificial_root()
    successors = list(digraph_multiple_roots.hierarchy_.successors("hiclass::root"))
    root = [
        node
        for node, in_degree in digraph_multiple_roots.hierarchy_.in_degree()
        if in_degree == 0
    ]
    assert ["a", "c", "e"] == successors
    assert ["hiclass::root"] == root
    assert "hiclass::root" == digraph_multiple_roots.root_


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    return digraph


def test_initialize_local_classifiers_1(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node == digraph_logistic_regression.root_:
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


def test_initialize_local_classifiers_2(digraph_logistic_regression):
    digraph_logistic_regression.local_classifier = None
    digraph_logistic_regression._initialize_local_classifiers()
    assert isinstance(digraph_logistic_regression.local_classifier_, LogisticRegression)


def test_fit_digraph(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph()
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


def test_fit_digraph_parallel(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph_parallel(local_mode=True)
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
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


def test_get_parents(digraph_2d):
    ground_truth = np.array(["a", "b", "d", "e"])
    nodes = digraph_2d._get_parents()
    assert_array_equal(ground_truth, nodes)


@pytest.fixture
def x_and_y_arrays():
    graph = LocalClassifierPerParentNode()
    graph.X_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    graph.y_ = np.array([["a", "b", "c"], ["a", "e", "f"], ["d", "g", "h"]])
    graph.hierarchy_ = nx.DiGraph(
        [("a", "b"), ("b", "c"), ("a", "e"), ("e", "f"), ("d", "g"), ("g", "h")]
    )
    graph.root_ = "r"
    return graph


def test_get_successors(x_and_y_arrays):
    x, y = x_and_y_arrays._get_successors("a")
    assert_array_equal(x_and_y_arrays.X_[0:2], x)
    assert_array_equal(["b", "e"], y)
    x, y = x_and_y_arrays._get_successors("d")
    assert_array_equal([x_and_y_arrays.X_[-1]], x)
    assert_array_equal(["g"], y)
    x, y = x_and_y_arrays._get_successors("b")
    assert_array_equal([x_and_y_arrays.X_[0]], x)
    assert_array_equal(["c"], y)


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

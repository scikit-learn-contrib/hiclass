import logging
import networkx as nx
import numpy as np
import pytest
import tempfile
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from hiclass import LocalClassifierPerNode
from hiclass.BinaryPolicy import ExclusivePolicy


@parametrize_with_checks([LocalClassifierPerNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def graph_1d():
    graph = LocalClassifierPerNode()
    graph.y_ = np.array(["a", "b", "c", "d"])
    graph.logger_ = logging.getLogger("LCPN")
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
    graph = LocalClassifierPerNode()
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
    dag_2d = LocalClassifierPerNode()
    dag_2d.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    dag_2d.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    dag_2d.logger_ = logging.getLogger("LCPN")
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
    dag_3d = LocalClassifierPerNode()
    dag_3d.y_ = np.arange(27).reshape((3, 3, 3))
    dag_3d.logger_ = logging.getLogger("LCPN")
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
    graph = LocalClassifierPerNode()
    graph.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    graph.logger_ = logging.getLogger("LCPN")
    return graph


def test_assert_digraph_is_dag(cyclic_graph):
    with pytest.raises(ValueError):
        cyclic_graph._assert_digraph_is_dag()


@pytest.fixture
def digraph_with_policy():
    digraph = LocalClassifierPerNode(binary_policy="exclusive")
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.X_ = np.array([1, 2])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
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
def digraph_one_root():
    digraph = LocalClassifierPerNode()
    digraph.logger_ = logging.getLogger("LCPN")
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
    digraph = LocalClassifierPerNode()
    digraph.logger_ = logging.getLogger("LCPN")
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
    digraph = LocalClassifierPerNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.binary_policy_ = ExclusivePolicy(digraph.hierarchy_, digraph.X_, digraph.y_)
    return digraph


def test_initialize_local_classifiers_1(digraph_logistic_regression):
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


def test_initialize_local_classifiers_2(digraph_logistic_regression):
    digraph_logistic_regression.local_classifier = None
    digraph_logistic_regression._initialize_local_classifiers()
    assert isinstance(digraph_logistic_regression.local_classifier_, LogisticRegression)


def test_fit_digraph(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph()
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


def test_fit_digraph_parallel(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph_parallel(local_mode=True)
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

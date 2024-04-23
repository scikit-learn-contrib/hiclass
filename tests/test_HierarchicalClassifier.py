import logging
import tempfile

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression

from hiclass.HierarchicalClassifier import HierarchicalClassifier, make_leveled


@pytest.fixture
def ambiguous_node_str():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([["a", "b"], ["b", "c"]])
    return classifier


def test_disambiguate_str(ambiguous_node_str):
    ground_truth = np.array(
        [["a", "a::HiClass::Separator::b"], ["b", "b::HiClass::Separator::c"]]
    )
    ambiguous_node_str._disambiguate()
    ground_truth = np.array(
        [ambiguous_node_str.label_encoder_.transform(row) for row in ground_truth]
    )
    assert_array_equal(ground_truth, ambiguous_node_str.y_)


@pytest.fixture
def ambiguous_node_int():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([[1, 2], [2, 3]])
    return classifier


def test_disambiguate_int(ambiguous_node_int):
    ground_truth = np.array(
        [["1", "1::HiClass::Separator::2"], ["2", "2::HiClass::Separator::3"]]
    )
    ambiguous_node_int._disambiguate()
    ground_truth = np.array(
        [ambiguous_node_int.label_encoder_.transform(row) for row in ground_truth]
    )
    assert_array_equal(ground_truth, ambiguous_node_int.y_)


@pytest.fixture
def graph_1d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array(["a", "b", "c", "d"])
    classifier.logger_ = logging.getLogger("HC")
    return classifier


def test_create_digraph_1d(graph_1d):
    ground_truth = nx.DiGraph()
    ground_truth.add_nodes_from(np.array(["a", "b", "c", "d"]))
    graph_1d._create_digraph()
    assert nx.is_isomorphic(ground_truth, graph_1d.hierarchy_)
    assert list(ground_truth.nodes) == list(graph_1d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(graph_1d.hierarchy_.edges)


@pytest.fixture
def graph_1d_disguised_as_2d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([["a"], ["b"], ["c"], ["d"]])
    classifier.logger_ = logging.getLogger("HC")
    return classifier


def test_create_digraph_1d_disguised_as_2d(graph_1d_disguised_as_2d):
    ground_truth = nx.DiGraph()
    ground_truth.add_nodes_from(np.array(["a", "b", "c", "d"]))
    graph_1d_disguised_as_2d._create_digraph()
    assert nx.is_isomorphic(ground_truth, graph_1d_disguised_as_2d.hierarchy_)
    assert list(ground_truth.nodes) == list(graph_1d_disguised_as_2d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(graph_1d_disguised_as_2d.hierarchy_.edges)


@pytest.fixture
def digraph_2d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    classifier.logger_ = logging.getLogger("HC")
    classifier.edge_list = tempfile.TemporaryFile()
    classifier.separator_ = "::HiClass::Separator::"
    return classifier


def test_create_digraph_2d(digraph_2d):
    ground_truth = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    digraph_2d._create_digraph()
    assert nx.is_isomorphic(ground_truth, digraph_2d.hierarchy_)
    assert list(ground_truth.nodes) == list(digraph_2d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(digraph_2d.hierarchy_.edges)


@pytest.fixture
def digraph_3d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.arange(27).reshape((3, 3, 3))
    classifier.logger_ = logging.getLogger("HC")
    return classifier


def test_create_digraph_3d(digraph_3d):
    with pytest.raises(ValueError):
        digraph_3d._create_digraph()


def test_export_digraph(digraph_2d):
    ground_truth = b'"a","b",{}\n"b","c",{}\n"d","e",{}\n"e","f",{}\n'
    digraph_2d._export_digraph()
    digraph_2d.edge_list.seek(0)
    assert digraph_2d.edge_list.read() == ground_truth


@pytest.fixture
def cyclic_graph():
    classifier = HierarchicalClassifier()
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    classifier.logger_ = logging.getLogger("HC")
    return classifier


def test_assert_digraph_is_dag(cyclic_graph):
    with pytest.raises(ValueError):
        cyclic_graph._assert_digraph_is_dag()


def test_convert_1d_y_to_2d(graph_1d):
    ground_truth = np.array([["a"], ["b"], ["c"], ["d"]])
    graph_1d._convert_1d_y_to_2d()
    assert_array_equal(ground_truth, graph_1d.y_)


@pytest.fixture
def digraph_one_root():
    classifier = HierarchicalClassifier()
    classifier.logger_ = logging.getLogger("HC")
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    return classifier


def test_add_artificial_root(digraph_one_root):
    digraph_one_root._add_artificial_root()
    successors = list(digraph_one_root.hierarchy_.successors("hiclass::root"))
    assert ["a"] == successors
    assert "hiclass::root" == digraph_one_root.root_


@pytest.fixture
def digraph_multiple_roots():
    classifier = HierarchicalClassifier()
    classifier.logger_ = logging.getLogger("HC")
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("c", "d"), ("e", "f")])
    classifier.X_ = np.array([[1, 2], [3, 4], [5, 6]])
    classifier.y_ = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
    classifier.sample_weight_ = None
    return classifier


def test_add_artificial_root_multiple_roots(digraph_multiple_roots):
    digraph_multiple_roots._add_artificial_root()
    successors = list(digraph_multiple_roots.hierarchy_.successors("hiclass::root"))
    assert ["a", "c", "e"] == successors
    assert "hiclass::root" == digraph_multiple_roots.root_


def test_initialize_local_classifiers_2(digraph_multiple_roots):
    digraph_multiple_roots.local_classifier = None
    digraph_multiple_roots._initialize_local_classifiers()
    assert isinstance(digraph_multiple_roots.local_classifier_, LogisticRegression)


def test_clean_up(digraph_multiple_roots):
    digraph_multiple_roots._clean_up()
    with pytest.raises(AttributeError):
        assert digraph_multiple_roots.X_ is None
    with pytest.raises(AttributeError):
        assert digraph_multiple_roots.y_ is None


@pytest.fixture
def empty_levels():
    y = [
        ["a"],
        ["b", "c"],
        ["d", "e", "f"],
    ]
    return y


def test_make_leveled(empty_levels):
    ground_truth = np.array(
        [
            ["a", "", ""],
            ["b", "c", ""],
            ["d", "e", "f"],
        ]
    )
    result = make_leveled(empty_levels)
    assert_array_equal(ground_truth, result)


@pytest.fixture
def noniterable_y():
    y = [1, 2, 3]
    return y


def test_make_leveled_non_iterable_y(noniterable_y):
    assert noniterable_y == make_leveled(noniterable_y)


def test_fit_classifier():
    with pytest.raises(NotImplementedError):
        HierarchicalClassifier._fit_classifier(None, None)


def test_fit_digraph():
    with pytest.raises(NotImplementedError):
        HierarchicalClassifier._fit_digraph(None, None)


def test_pre_fit_bert():
    classifier = HierarchicalClassifier()
    classifier.logger_ = logging.getLogger("HC")
    classifier.bert = True
    x = [[0, 1], [2, 3]]
    y = [["a", "b"], ["c", "d"]]
    sample_weight = None
    classifier._pre_fit(x, y, sample_weight)

import logging
import networkx as nx
import numpy as np
import pytest
import tempfile
from numpy.testing import assert_array_equal

from hiclass.HierarchicalClassifier import HierarchicalClassifier


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

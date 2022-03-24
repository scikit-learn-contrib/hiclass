from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from hiclass import data
from hiclass.data import CATEGORICAL_ROOT_NODE

fixtures_loc = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_dag_edges():
    return [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (4, 7), (5, 7), (5, 8)]


@pytest.fixture
def small_dag_nodes():
    return list(range(1, 9))


@pytest.fixture
def small_dag(small_dag_edges):
    return nx.from_edgelist(small_dag_edges, create_using=nx.DiGraph)


def assert_graph_equals(graph, nodes, edges):
    assert len(graph.nodes) == len(nodes), "node length not equal"
    assert len(graph.edges) == len(edges), "edge length not equal"
    assert set(graph.nodes) == set(nodes), "there exist a variation between the nodes"
    assert set(graph.edges) == set(edges), "there exist a variation between the edges"


def test_graph_from_edge_pairs(small_dag_edges, small_dag_nodes):
    g = data.graph_from_edge_pairs(str(fixtures_loc / "small_dag_edgelist.csv"))
    assert_graph_equals(g, small_dag_nodes, small_dag_edges)


def test_graph_from_hierarchical_labels_simple():
    labels = np.array([[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8]], dtype=np.int32)
    g = data.graph_from_hierarchical_labels(labels)
    assert_graph_equals(
        g, [1, 2, 3, 4, 5, 6, 8], [(1, 2), (2, 3), (2, 4), (1, 5), (5, 6), (5, 8)]
    )


def test_graph_from_hierarchical_labels_simple_categorical():
    labels = np.array(
        [["1", "2", "3"], ["1", "2", "4"], ["1", "5", "6"], ["1", "5", "8"]]
    )
    g = data.graph_from_hierarchical_labels(
        labels, placeholder=data.PLACEHOLDER_LABEL_CAT
    )
    assert_graph_equals(
        g,
        ["1", "2", "3", "4", "5", "6", "8"],
        [("1", "2"), ("2", "3"), ("2", "4"), ("1", "5"), ("5", "6"), ("5", "8")],
    )


def test_graph_from_hierarchical_labels_uneven_hierarchy():
    labels = np.array(
        [[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8], [1, 3, -1]], dtype=np.int32
    )
    g = data.graph_from_hierarchical_labels(labels, placeholder=-1)
    assert_graph_equals(
        g,
        [1, 2, 3, 4, 5, 6, 8],
        [(1, 2), (1, 3), (2, 3), (2, 4), (1, 5), (5, 6), (5, 8)],
    )


def test_graph_from_hierarchical_labels_missing_root():
    labels = np.array([[2, 3], [2, 4], [5, 6], [5, 8]], dtype=np.int32)
    g = data.graph_from_hierarchical_labels(labels)
    assert_graph_equals(
        g, [2, 3, 4, 5, 6, 8, 9], [(9, 2), (2, 3), (2, 4), (9, 5), (5, 6), (5, 8)]
    )


def test_graph_from_hierarchical_labels_categorical_missing_root():
    labels = np.array([["2", "3"], ["2", "4"], ["5", "6"], ["5", "8"]])
    g = data.graph_from_hierarchical_labels(
        labels, placeholder=data.PLACEHOLDER_LABEL_CAT
    )
    assert_graph_equals(
        g,
        ["2", "3", "4", "5", "6", "8", CATEGORICAL_ROOT_NODE],
        [
            (CATEGORICAL_ROOT_NODE, "2"),
            ("2", "3"),
            ("2", "4"),
            (CATEGORICAL_ROOT_NODE, "5"),
            ("5", "6"),
            ("5", "8"),
        ],
    )


def test_graph_from_hierarchical_labels_single_row():
    labels = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    g = data.graph_from_hierarchical_labels(labels)
    assert_graph_equals(g, [1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])


def test_convert_categorical_labels_to_numeric():
    cat_labels = np.array([["a", "b", "c"], ["a", "c", ""]])
    placeholder = ""
    labels, translation = data.convert_categorical_labels_to_numeric(
        cat_labels, placeholder_label=placeholder
    )
    for k in translation.keys():
        cat_labels[cat_labels == k] = translation[k]
    assert labels.shape == (2, 3)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels[1, 2] == data.PLACEHOLDER_LABEL_NUMERIC


def test_convert_categorical_labels_to_numeric_flat():
    cat_labels = np.array([["b"], ["a"]])
    labels, translation = data.convert_categorical_labels_to_numeric(cat_labels)
    for k in translation.keys():
        cat_labels[cat_labels == k] = translation[k]
    cat_labels = cat_labels.astype(np.integer)
    assert labels.shape == (2, 1)
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.array_equal(cat_labels, labels)


def test_convert_categorical_labels_to_numeric_no_side_effects():
    cat_labels = np.array([["b"], ["a"]])
    cat_labels2 = cat_labels.copy()
    data.convert_categorical_labels_to_numeric(cat_labels)
    assert np.array_equal(cat_labels, cat_labels2)


def test_is_numeric_label():
    assert data.is_numeric_label(np.array([1]))
    assert data.is_numeric_label(np.array([1.0]))
    assert data.is_numeric_label(np.array([[1, 2, 3], [3.0, 2.0, 3.0]]))
    assert not data.is_numeric_label(np.array(["test"]))
    assert not data.is_numeric_label(np.array(["test", 1, 1.2]))
    assert not data.is_numeric_label(np.array([[1, 2, 3], ["test", 2, 3]]))


def test_find_root(small_dag):
    assert data.find_root(small_dag) == 1
    small_dag.add_edge(9, 2)
    assert data.find_root(small_dag) in [9, 1]


def test_flatten_labels():
    labels = np.array([[1, 2, 3], [4, 5, 6]])
    assert list(data.flatten_labels(labels)) == [3, 6]


def test_flatten_labels_categorical():
    labels = np.array([["1", "2", "3"], ["4", "5", "6"]])
    assert list(data.flatten_labels(labels, None)) == ["3", "6"]


def test_flatten_labels_none_placeholder():
    labels = np.array([[1, 2, 3], [4, 5, 6]])
    assert list(data.flatten_labels(labels, None)) == [3, 6]


def test_flatten_labels_undefined():
    labels = np.array([[1, 2, 3], [4, 5, -1], [6, -1, -1]])
    assert list(data.flatten_labels(labels, -1)) == [3, 5, 6]


def test_flatten_labels_already_flat():
    labels = np.array([1, 2])
    assert list(data.flatten_labels(labels)) == [1, 2]


def test_minimal_graph_depth(small_dag):
    x = data.minimal_graph_depth(small_dag)
    assert x == 4
    small_dag.add_edge(4, 8)
    x = data.minimal_graph_depth(small_dag)
    assert x == 3


def test_find_max_depth(small_dag):
    depth = data.find_max_depth(small_dag, 1)
    assert depth == 4
    depth = data.find_max_depth(small_dag, 2)
    assert depth == 3
    depth = data.find_max_depth(small_dag, 8)
    assert depth == 1

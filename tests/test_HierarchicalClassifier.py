import logging
import tempfile

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression

from hiclass.HierarchicalClassifier import HierarchicalClassifier, make_leveled


@pytest.fixture
def ambiguous_node_str_2d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([["a", "b"], ["b", "c"]])
    return classifier


def test_disambiguate_str_2d(ambiguous_node_str_2d):
    ground_truth = [
        ["a", "a::HiClass::Separator::b"],
        ["b", "b::HiClass::Separator::c"],
    ]
    ambiguous_node_str_2d._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_str_2d.y_)


@pytest.fixture
def ambiguous_node_str_3d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array(
        [
            [["a", "b"], ["", ""]],
            [["b", "c"], ["", ""]],
            [["d", "e"], ["f", "g"]],
        ]
    )
    return classifier


def test_disambiguate_str_3d(ambiguous_node_str_3d):
    ground_truth = [
        [["a", "a::HiClass::Separator::b"], ["", ""]],
        [["b", "b::HiClass::Separator::c"], ["", ""]],
        [["d", "d::HiClass::Separator::e"], ["f", "f::HiClass::Separator::g"]],
    ]
    ambiguous_node_str_3d._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_str_3d.y_)


@pytest.fixture
def ambiguous_node_int_2d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([[1, 2], [2, 3]])
    return classifier


def test_disambiguate_int_2d(ambiguous_node_int_2d):
    ground_truth = [
        ["1", "1::HiClass::Separator::2"],
        ["2", "2::HiClass::Separator::3"],
    ]
    ambiguous_node_int_2d._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_int_2d.y_)


@pytest.fixture
def ambiguous_node_int_3d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array(
        [
            [[1, 2], ["", ""]],
            [[2, 3], ["", ""]],
            [[4, 5], [6, 7]],
        ]
    )
    return classifier


def test_disambiguate_int_3d(ambiguous_node_int_3d):
    ground_truth = [
        [["1", "1::HiClass::Separator::2"], ["", ""]],
        [["2", "2::HiClass::Separator::3"], ["", ""]],
        [["4", "4::HiClass::Separator::5"], ["6", "6::HiClass::Separator::7"]],
    ]
    ambiguous_node_int_3d._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_int_3d.y_)


@pytest.fixture
def graph_1d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array(["a", "b", "c", "d"])
    classifier.logger_ = logging.getLogger("HC")
    return classifier


def test_create_digraph_1d(graph_1d):
    ground_truth = nx.DiGraph()
    ground_truth.add_nodes_from(["a", "b", "c", "d"])
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
    ground_truth.add_nodes_from(["a", "b", "c", "d"])
    graph_1d_disguised_as_2d._create_digraph()
    assert nx.is_isomorphic(ground_truth, graph_1d_disguised_as_2d.hierarchy_)
    assert list(ground_truth.nodes) == list(graph_1d_disguised_as_2d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(graph_1d_disguised_as_2d.hierarchy_.edges)


@pytest.fixture
def digraph_2d():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    classifier.logger_ = logging.getLogger("HC")
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
    classifier.y_ = np.array(
        [
            [["a", "b", "c"]],
            [["d", "e", "f"]],
        ]
    )
    classifier.logger_ = logging.getLogger("HC")
    classifier.separator_ = "::HiClass::Separator::"
    return classifier


def test_create_digraph_3d(digraph_3d):
    ground_truth = nx.DiGraph(
        [
            ("a", "b"),
            ("b", "c"),
            ("d", "e"),
            ("e", "f"),
        ]
    )
    digraph_3d._create_digraph()
    assert nx.is_isomorphic(ground_truth, digraph_3d.hierarchy_)
    assert list(ground_truth.nodes) == list(digraph_3d.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(digraph_3d.hierarchy_.edges)


@pytest.fixture
def digraph_3d_multi_label():
    classifier = HierarchicalClassifier()
    classifier.y_ = np.array(
        [
            [["a", "b", "c"], ["d", "e", "f"]],
            [["g", "h", "i"], ["j", "k", "l"]],
        ]
    )
    classifier.logger_ = logging.getLogger("HC")
    classifier.separator_ = "::HiClass::Separator::"
    return classifier


def test_create_digraph_3d_multi_label(digraph_3d_multi_label):
    ground_truth = nx.DiGraph(
        [
            ("a", "b"),
            ("b", "c"),
            ("d", "e"),
            ("e", "f"),
            ("g", "h"),
            ("h", "i"),
            ("j", "k"),
            ("k", "l"),
        ]
    )
    digraph_3d_multi_label._create_digraph()
    assert nx.is_isomorphic(ground_truth, digraph_3d_multi_label.hierarchy_)
    assert list(ground_truth.nodes) == list(digraph_3d_multi_label.hierarchy_.nodes)
    assert list(ground_truth.edges) == list(digraph_3d_multi_label.hierarchy_.edges)


def test_export_digraph(digraph_2d):
    digraph_2d.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    digraph_2d.edge_list = tempfile.TemporaryFile()
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
def empty_levels_2d():
    y = [
        ["a"],
        ["b", "c"],
        ["d", "e", "f"],
    ]
    return y


def test_make_leveled_2d(empty_levels_2d):
    ground_truth = [
        ["a", "", ""],
        ["b", "c", ""],
        ["d", "e", "f"],
    ]
    result = make_leveled(empty_levels_2d)
    assert_array_equal(ground_truth, result)


@pytest.fixture
def empty_levels_3d():
    y = [
        [["a"]],
        [["b", "c"]],
        [["d", "e", "f"]],
        [["g", "h", "i"], ["j", "k", "l"]],
        [["m", "n", "o"]],
    ]
    return y


def test_make_leveled_3d(empty_levels_3d):
    ground_truth = [
        # Labels that are the same as in the Single-Label Test case
        [["a", "", ""], ["", "", ""]],
        [["b", "c", ""], ["", "", ""]],
        [["d", "e", "f"], ["", "", ""]],
        # Multi-label Test cases
        [["g", "h", "i"], ["j", "k", "l"]],
        [["m", "n", "o"], ["", "", ""]],
    ]
    result = make_leveled(empty_levels_3d)
    assert_array_equal(ground_truth, result)


@pytest.fixture
def noniterable_y():
    y = [1, 2, 3]
    return y


def test_make_leveled_non_iterable_y(noniterable_y):
    result = make_leveled(noniterable_y)
    assert_array_equal(noniterable_y, result)


def test_fit_classifier():
    with pytest.raises(NotImplementedError):
        HierarchicalClassifier._fit_classifier(None, None)


def test_pre_fit_bert():
    classifier = HierarchicalClassifier()
    classifier.logger_ = logging.getLogger("HC")
    classifier.bert = True
    x = [[0, 1], [2, 3]]
    y = [["a", "b"], ["c", "d"]]
    sample_weight = None
    classifier._pre_fit(x, y, sample_weight)


@pytest.fixture
def separator_2d():
    hc = HierarchicalClassifier()
    hc.separator_ = "::HiClass::Separator::"
    y = np.array(
        [
            [
                "a",
                "a::HiClass::Separator::b",
                "a::HiClass::Separator::b::HiClass::Separator::c",
            ],
            [
                "d",
                "d::HiClass::Separator::e",
                "d::HiClass::Separator::e::HiClass::Separator::f",
            ],
            [
                "g",
                "g::HiClass::Separator::h",
                "g::HiClass::Separator::h::HiClass::Separator::i",
            ],
        ]
    )
    return hc, y


def test_remove_separator_2d(separator_2d):
    ground_truth = [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "h", "i"],
    ]
    hc, y = separator_2d
    hc._remove_separator(y)
    assert_array_equal(ground_truth, y)


@pytest.fixture
def separator_3d():
    hc = HierarchicalClassifier()
    hc.separator_ = "::HiClass::Separator::"
    y = np.array(
        [
            [
                [
                    "a",
                    "a::HiClass::Separator::b",
                    "a::HiClass::Separator::b::HiClass::Separator::c",
                ]
            ],
            [
                [
                    "d",
                    "d::HiClass::Separator::e",
                    "d::HiClass::Separator::e::HiClass::Separator::f",
                ]
            ],
            [
                [
                    "g",
                    "g::HiClass::Separator::h",
                    "g::HiClass::Separator::h::HiClass::Separator::i",
                ]
            ],
        ]
    )
    return hc, y


def test_remove_separator_3d(separator_3d):
    ground_truth = [
        [["a", "b", "c"]],
        [["d", "e", "f"]],
        [["g", "h", "i"]],
    ]
    hc, y = separator_3d
    hc._remove_separator(y)
    assert_array_equal(ground_truth, y)


@pytest.fixture
def separator_3d_multi_label():
    hc = HierarchicalClassifier()
    hc.separator_ = "::HiClass::Separator::"
    # TODO: Fix this test case, which is being recognized as a 2D array instead of a 3D array
    y = np.array(
        [
            [["a", "a::HiClass::Separator::b"], ["a::HiClass::Separator::c"]],
            [["d", "d::HiClass::Separator::e"], ["d::HiClass::Separator::f"]],
            [["g", "g::HiClass::Separator::h"], ["g::HiClass::Separator::i"]],
        ]
    )
    return hc, y


def test_remove_separator_3d_multi_label(separator_3d_multi_label):
    ground_truth = [
        [["a", "b"], ["a", "c"]],
        [["d", "e"], ["d", "f"]],
        [["g", "h"], ["g", "i"]],
    ]
    hc, y = separator_3d_multi_label
    hc._remove_separator(y)
    assert_array_equal(ground_truth, y)

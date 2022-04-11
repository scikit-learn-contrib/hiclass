import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hiclass.HierarchicalClassifier import HierarchicalClassifier


@pytest.fixture
def ambiguous_node_str():
    graph = HierarchicalClassifier()
    graph.y_ = np.array([["a", "b"], ["b", "c"]])
    return graph


def test_disambiguate_str(ambiguous_node_str):
    ground_truth = np.array(
        [["a", "a::HiClass::Separator::b"], ["b", "b::HiClass::Separator::c"]]
    )
    ambiguous_node_str._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_str.y_)


@pytest.fixture
def ambiguous_node_int():
    graph = HierarchicalClassifier()
    graph.y_ = np.array([[1, 2], [2, 3]])
    return graph


def test_disambiguate_int(ambiguous_node_int):
    ground_truth = np.array(
        [["1", "1::HiClass::Separator::2"], ["2", "2::HiClass::Separator::3"]]
    )
    ambiguous_node_int._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node_int.y_)

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hiclass.HierarchicalClassifier import HierarchicalClassifier


@pytest.fixture
def ambiguous_node():
    graph = HierarchicalClassifier()
    graph.y_ = np.array([["a", "b"], ["b", "c"]])
    return graph


def test_disambiguate(ambiguous_node):
    ground_truth = np.array(
        [["a", "a::HiClass::Separator::b"], ["b", "b::HiClass::Separator::c"]]
    )
    ambiguous_node._disambiguate()
    assert_array_equal(ground_truth, ambiguous_node.y_)

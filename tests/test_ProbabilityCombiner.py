import numpy as np
import pytest
from unittest.mock import Mock
import networkx as nx
from numpy.testing import assert_array_almost_equal, assert_array_equal
import math

from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass.probability_combiner import MultiplyCombiner


@pytest.fixture
def one_sample_probs_with_hierarchy():
    hierarchy = nx.DiGraph()
    root_node = "hiclass::root"
    hierarchy.add_node(root_node)
    classes_ = [[], [], []]

    for i in range(3):
        # first level
        first_level_node = f'level_0_node_{i}'
        classes_[0].append(first_level_node)
        hierarchy.add_node(first_level_node)
        hierarchy.add_edge(root_node, first_level_node)
        for j in range(3):
            # second level
            second_level_node = f'level_1_node_{i}:{j}'
            classes_[1].append(second_level_node)
            hierarchy.add_node(second_level_node)
            hierarchy.add_edge(first_level_node, second_level_node)
            for k in range(2):
                # third level
                third_level_node = f'level_2_node_{i}:{j}:{k}'
                classes_[2].append(third_level_node)
                hierarchy.add_node(third_level_node)
                hierarchy.add_edge(second_level_node, third_level_node)

    probs = [
        np.array([[0.3, 0.5, 0.2]]), # level 0
        np.array([[0.19, 0.10, 0.16, 0.14, 0.07, 0.08, 0.19, 0.03, 0.04]]), # level 1
        np.array([[0.03, 0.17, 0.07, 0.01, 0.00, 0.06, 0.17, 0.02, 0.01, 0.07, 0.20, 0.01, 0.10, 0.00, 0.05, 0.02, 0.00, 0.01]]) # level 2
    ]
    assert all([np.sum(probs[level], axis=1) == 1 for level in range(3)])

    return hierarchy, probs, classes_

def test_multiply_combiner(one_sample_probs_with_hierarchy):

    hierarchy, probs, classes = one_sample_probs_with_hierarchy
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = classes
    classifier.max_levels_ = len(classes)
    classifier.class_to_index_mapping_ = [{classifier.classes_[level][index]: index for index in range(len(classifier.classes_[level]))} for level in range(classifier.max_levels_)]
    classifier.hierarchy_ = hierarchy

    combiner = MultiplyCombiner(classifier=classifier)
    combined_probs = combiner.combine(probs)

    # check combined probability of first node for both levels
    assert math.isclose(combined_probs[1][0][0], 0.0569, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][0], probs[0][0][0] * probs[1][0][0], abs_tol=1e-4)
    
    assert math.isclose(combined_probs[2][0][0], 0.0017, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][0], probs[0][0][0] * probs[1][0][0] * probs[2][0][0], abs_tol=1e-4)

    # check combined probability of last node for both levels
    assert math.isclose(combined_probs[1][0][-1], 0.008, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][-1], probs[0][0][-1] * probs[1][0][-1], abs_tol=1e-4)
    
    assert math.isclose(combined_probs[2][0][-1], 8e-5, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][-1], probs[0][0][-1] * probs[1][0][-1] * probs[2][0][-1], abs_tol=1e-4)

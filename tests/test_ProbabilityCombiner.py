import numpy as np
import pytest
from unittest.mock import Mock
import networkx as nx
from numpy.testing import assert_array_almost_equal, assert_array_equal
import math
from scipy.stats import gmean

from hiclass.HierarchicalClassifier import make_leveled

from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass.probability_combiner import MultiplyCombiner, ArithmeticMeanCombiner, GeometricMeanCombiner


def _disambiguate(y, separator):
    if y.ndim == 2:
        new_y = []
        for i in range(y.shape[0]):
            row = [str(y[i, 0])]
            for j in range(1, y.shape[1]):
                parent = str(row[-1])
                child = str(y[i, j])
                row.append(parent + separator + child)
            new_y.append(np.asarray(row, dtype=np.str_))
        return np.array(new_y)
    return y

@pytest.fixture
def one_sample_probs_with_hierarchy():
    hierarchy = nx.DiGraph()
    root_node = "hiclass::root"
    hierarchy.add_node(root_node)
    separator = "::HiClass::Separator::"
    y_ = []

    for i in range(3):
        # first level
        first_level_node = f'node_{i}'
        hierarchy.add_node(first_level_node)
        hierarchy.add_edge(root_node, first_level_node)
        for j in range(3):
            # second level
            second_level_node = f'node_{i}:{j}'
            second_level_label = _disambiguate(np.array([[first_level_node, second_level_node]]), separator)[0][1]
            hierarchy.add_node(second_level_label)
            hierarchy.add_edge(first_level_node, second_level_label)
            for k in range(2):
                # third level
                third_level_node = f'node_{i}:{j}:{k}'
                third_level_label = _disambiguate(np.array([[first_level_node, second_level_node, third_level_node]]), separator)[0][2]
                y_.append([first_level_node, second_level_node, third_level_node])
                hierarchy.add_node(third_level_label)
                hierarchy.add_edge(second_level_label, third_level_label)

    probs = [
        np.array([[0.3, 0.5, 0.2]]), # level 0
        np.array([[0.19, 0.10, 0.16, 0.14, 0.07, 0.08, 0.19, 0.03, 0.04]]), # level 1
        np.array([[0.03, 0.17, 0.07, 0.01, 0.00, 0.06, 0.17, 0.02, 0.01, 0.07, 0.20, 0.01, 0.10, 0.00, 0.05, 0.02, 0.00, 0.01]]) # level 2
    ]
    assert all([np.sum(probs[level], axis=1) == 1 for level in range(3)])

    y_ = np.array(y_)
    y_ = make_leveled(y_)
    y_ = _disambiguate(y_, separator)

    global_classes_ = [np.unique(y_[:, level]).astype("str") for level in range(y_.shape[1])]

    classes_ = [global_classes_[0]]
    for level in range(1, len(y_[1])):
        classes_.append(np.sort(np.unique([label.split(separator)[level] for label in global_classes_[level]])))

    class_to_index_mapping_ = [{local_labels[index]: index for index in range(len(local_labels))} for local_labels in classes_]

    return hierarchy, probs, global_classes_, classes_, class_to_index_mapping_

def test_multiply_combiner(one_sample_probs_with_hierarchy):
    hierarchy, probs, global_classes, classes_, class_to_index_mapping_ = one_sample_probs_with_hierarchy
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.global_classes_ = global_classes
    classifier.max_levels_ = len(global_classes)
    classifier.classes_ = classes_
    classifier.class_to_index_mapping_ = class_to_index_mapping_
    
    classifier.hierarchy_ = hierarchy
    classifier.separator_ = "::HiClass::Separator::"

    combiner = MultiplyCombiner(classifier=classifier, normalize=False)
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

def test_arithmetic_mean_combiner(one_sample_probs_with_hierarchy):
    hierarchy, probs, global_classes, classes_, class_to_index_mapping_ = one_sample_probs_with_hierarchy
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    
    classifier.global_classes_ = global_classes
    classifier.max_levels_ = len(global_classes)
    classifier.classes_ = classes_
    classifier.class_to_index_mapping_ = class_to_index_mapping_
    classifier.hierarchy_ = hierarchy
    classifier.separator_ = "::HiClass::Separator::"

    combiner = ArithmeticMeanCombiner(classifier=classifier, normalize=False)
    combined_probs = combiner.combine(probs)

    # check combined probability of first node for both levels
    assert math.isclose(combined_probs[1][0][0], 0.245, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][0], (probs[0][0][0] + probs[1][0][0]) / 2, abs_tol=1e-4)

    assert math.isclose(combined_probs[2][0][0], 0.1733, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][0], (probs[0][0][0] + probs[1][0][0] + probs[2][0][0]) / 3, abs_tol=1e-4)

    # check combined probability of last node for both levels
    assert math.isclose(combined_probs[1][0][-1], 0.12, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][-1], (probs[0][0][-1] + probs[1][0][-1]) / 2, abs_tol=1e-4)
    
    assert math.isclose(combined_probs[2][0][-1], 0.0833, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][-1], (probs[0][0][-1] + probs[1][0][-1] + probs[2][0][-1]) / 3, abs_tol=1e-4)

def test_geometric_mean_combiner(one_sample_probs_with_hierarchy):
    hierarchy, probs, global_classes, classes_, class_to_index_mapping_ = one_sample_probs_with_hierarchy
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.separator_ = "::HiClass::Separator::"
    classifier.global_classes_ = global_classes
    classifier.classes_ = classes_

    classifier.max_levels_ = len(global_classes)
    classifier.class_to_index_mapping_ = class_to_index_mapping_
    classifier.hierarchy_ = hierarchy

    combiner = GeometricMeanCombiner(classifier=classifier, normalize=False)
    combined_probs = combiner.combine(probs)

    # check combined probability of first node for both levels
    assert math.isclose(combined_probs[1][0][0], 0.2387, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][0], gmean([probs[0][0][0], probs[1][0][0]]), abs_tol=1e-4)

    assert math.isclose(combined_probs[2][0][0], 0.1195, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][0], gmean([probs[0][0][0], probs[1][0][0], probs[2][0][0]]), abs_tol=1e-4)

    # check combined probability of last node for both levels
    assert math.isclose(combined_probs[1][0][-1], 0.0894, abs_tol=1e-4)
    assert math.isclose(combined_probs[1][0][-1], gmean([probs[0][0][-1], probs[1][0][-1]]), abs_tol=1e-4)
    
    assert math.isclose(combined_probs[2][0][-1], 0.0430, abs_tol=1e-4)
    assert math.isclose(combined_probs[2][0][-1], gmean([probs[0][0][-1], probs[1][0][-1], probs[2][0][-1]]), abs_tol=1e-4)

def test_mean_combiner_with_normalization(one_sample_probs_with_hierarchy):
    hierarchy, probs, global_classes, classes_, class_to_index_mapping_ = one_sample_probs_with_hierarchy
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.separator_ = "::HiClass::Separator::"
    classifier.global_classes_ = global_classes
    classifier.classes_ = classes_

    classifier.max_levels_ = len(global_classes)
    classifier.class_to_index_mapping_ = class_to_index_mapping_
    classifier.hierarchy_ = hierarchy

    combiners = [MultiplyCombiner(classifier), ArithmeticMeanCombiner(classifier), GeometricMeanCombiner(classifier)]

    for combiner in combiners:
        combined_probs = combiner.combine(probs)

        assert len(combined_probs) == 3
        assert combined_probs[0].shape == (1, 3)
        assert combined_probs[1].shape == (1, 9)
        assert combined_probs[2].shape == (1, 18)
        assert_array_almost_equal(np.sum(combined_probs[0], axis=1), np.ones(len(combined_probs[0])), decimal=10)
        assert_array_almost_equal(np.sum(combined_probs[1], axis=1), np.ones(len(combined_probs[1])), decimal=10)
        assert_array_almost_equal(np.sum(combined_probs[2], axis=1), np.ones(len(combined_probs[2])), decimal=10)

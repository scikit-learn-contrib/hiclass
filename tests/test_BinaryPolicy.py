from pathlib import Path
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hiclass.BinaryPolicy import (
    BinaryPolicy,
    ExclusivePolicy,
    LessExclusivePolicy,
    ExclusiveSiblingsPolicy,
    InclusivePolicy,
    LessInclusivePolicy,
    SiblingsPolicy,
)

fixtures_loc = Path(__file__).parent / "fixtures"


@pytest.fixture
def digraph():
    return nx.DiGraph(
        [
            ("r", "1"),
            ("r", "2"),
            ("1", "1.1"),
            ("1", "1.2"),
            ("2", "2.1"),
            ("2", "2.2"),
            ("2.1", "2.1.1"),
            ("2.1", "2.1.2"),
            ("2.2", "2.2.1"),
            ("2.2", "2.2.2"),
        ]
    )


@pytest.fixture
def digraph_numeric():
    return nx.DiGraph(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (5, 7),
            (5, 8),
            (6, 9),
            (6, 10),
        ]
    )


@pytest.fixture
def features_1d():
    return np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
    )


@pytest.fixture
def features_2d():
    return np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]
    )


@pytest.fixture
def features_sparse():
    return csr_matrix(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]
    )


@pytest.fixture
def labels_1d():
    return np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
    )


@pytest.fixture
def labels_2d():
    return np.array(
        [
            ["1", "1.1"],
            ["1", "1.2"],
            ["2", "2.1"],
            ["2", "2.2"],
            ["2.1", "2.1.1"],
            ["2.1", "2.1.2"],
            ["2.2", "2.2.1"],
            ["2.2", "2.2.2"],
        ]
    )


@pytest.fixture
def labels_2d_numeric():
    return np.array(
        [
            [1, 3],
            [1, 4],
            [2, 5],
            [2, 6],
            [5, 7],
            [5, 8],
            [6, 9],
            [6, 10],
        ]
    )


def test_binary_policy_positive_examples(digraph, features_1d, labels_2d):
    policy = BinaryPolicy(digraph, features_1d, labels_2d)
    with pytest.raises(NotImplementedError):
        policy.positive_examples("1")


def test_binary_policy_negative_examples(digraph, features_1d, labels_2d):
    policy = BinaryPolicy(digraph, features_1d, labels_2d)
    with pytest.raises(NotImplementedError):
        policy.negative_examples("1")


def test_exclusive_policy_positive_examples_1d_1(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [True, False, False, False, False, False, False, False]
    result = policy.positive_examples(1)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_2d_1(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.positive_examples("1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_1d_2(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [False, True, False, False, False, False, False, False]
    result = policy.positive_examples(2)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_2d_2(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, False, False, False, False]
    result = policy.positive_examples("2")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_1d_3(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [False, False, False, False, True, False, False, False]
    result = policy.positive_examples(5)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_2d_3(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, False, True, True, False, False]
    result = policy.positive_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_1d_1(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [False, True, True, True, True, True, True, True]
    result = policy.negative_examples(1)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_2d_1(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_1d_2(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [True, False, True, True, True, True, True, True]
    result = policy.negative_examples(2)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_2d_2(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, True, True, True, True]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_1d_3(digraph, features_2d, labels_1d):
    policy = ExclusivePolicy(digraph, features_2d, labels_1d)
    ground_truth = [True, True, True, True, False, True, True, True]
    result = policy.negative_examples(5)
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_2d_3(digraph, features_1d, labels_2d):
    policy = ExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, True, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_numeric_1(
    digraph_numeric, features_2d, labels_2d_numeric
):
    policy = LessExclusivePolicy(digraph_numeric, features_2d, labels_2d_numeric)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples(1)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_1d_1(
    digraph_numeric, features_1d, labels_1d
):
    policy = LessExclusivePolicy(digraph_numeric, features_1d, labels_1d)
    ground_truth = [False, True, False, False, True, True, True, True]
    result = policy.negative_examples(1)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_1(digraph, features_1d, labels_2d):
    policy = LessExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_1d_2(
    digraph_numeric, features_1d, labels_1d
):
    policy = LessExclusivePolicy(digraph_numeric, features_1d, labels_1d)
    ground_truth = [True, False, True, True, False, False, False, False]
    result = policy.negative_examples(2)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_numeric_2(
    digraph_numeric, features_1d, labels_2d_numeric
):
    policy = LessExclusivePolicy(digraph_numeric, features_1d, labels_2d_numeric)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples(2)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_2(digraph, features_1d, labels_2d):
    policy = LessExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_numeric_3(
    digraph_numeric, features_1d, labels_2d_numeric
):
    policy = LessExclusivePolicy(digraph_numeric, features_1d, labels_2d_numeric)
    ground_truth = [True, True, False, True, False, False, True, True]
    result = policy.negative_examples(5)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_1d_3(
    digraph_numeric, features_1d, labels_1d
):
    policy = LessExclusivePolicy(digraph_numeric, features_1d, labels_1d)
    ground_truth = [True, True, True, True, False, True, False, False]
    result = policy.negative_examples(5)
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2d_3(digraph, features_1d, labels_2d):
    policy = LessExclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, True, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_1(digraph, features_1d, labels_2d):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, False, False, False, False]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_2(digraph, features_1d, labels_2d):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_3(digraph, features_1d, labels_2d):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, False, True, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_1(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.positive_examples("1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_2(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.positive_examples("2")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_3(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, False, True, True, False, False]
    result = policy.positive_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_1(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_2(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_3(digraph, features_1d, labels_2d):
    policy = InclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_1(digraph, features_1d, labels_2d):
    policy = LessInclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_2(digraph, features_1d, labels_2d):
    policy = LessInclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_3(digraph, features_1d, labels_2d):
    policy = LessInclusivePolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, True, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_1(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, True, True, True, True, True, True]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_2(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [True, True, False, False, False, False, False, False]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_3(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth = [False, False, False, True, False, False, True, True]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_siblings_get_binary_examples_1d_1(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth_x = [1, 2, 3, 4, 5, 6, 7, 8]
    ground_truth_y = [1, 1, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_1d_2(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth_x = [3, 4, 5, 6, 7, 8, 1, 2]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_1d_3(digraph, features_1d, labels_2d):
    policy = SiblingsPolicy(digraph, features_1d, labels_2d)
    ground_truth_x = [3, 5, 6, 4, 7, 8]
    ground_truth_y = [1, 1, 1, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_1(digraph, features_2d, labels_2d):
    policy = SiblingsPolicy(digraph, features_2d, labels_2d)
    ground_truth_x = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
    ]
    ground_truth_y = [1, 1, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_2(digraph, features_2d, labels_2d):
    policy = SiblingsPolicy(digraph, features_2d, labels_2d)
    ground_truth_x = [
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [1, 2],
        [3, 4],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_3(digraph, features_2d, labels_2d):
    policy = SiblingsPolicy(digraph, features_2d, labels_2d)
    ground_truth_x = [[5, 6], [9, 10], [11, 12], [7, 8], [13, 14], [15, 16]]
    ground_truth_y = [1, 1, 1, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_1(digraph, features_sparse, labels_2d):
    policy = SiblingsPolicy(digraph, features_sparse, labels_2d)
    ground_truth_x = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
    ]
    ground_truth_y = [1, 1, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_2(digraph, features_sparse, labels_2d):
    policy = SiblingsPolicy(digraph, features_sparse, labels_2d)
    ground_truth_x = [
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [1, 2],
        [3, 4],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_3(digraph, features_sparse, labels_2d):
    policy = SiblingsPolicy(digraph, features_sparse, labels_2d)
    ground_truth_x = [[5, 6], [9, 10], [11, 12], [7, 8], [13, 14], [15, 16]]
    ground_truth_y = [1, 1, 1, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None

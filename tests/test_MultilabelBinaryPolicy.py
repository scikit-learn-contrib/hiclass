from pathlib import Path
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hiclass.MultiLabelBinaryPolicy import (
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
            9,
            10,
            11,
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
            [17, 18],
            [19, 20],
            [21, 22],
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
            [17, 18],
            [19, 20],
            [21, 22],
        ]
    )


@pytest.fixture
def labels():
    return np.array(
        [
            # labels that are the same as in the Single-Label Test case
            [["1", "1.1"], ["", ""]],
            [["1", "1.2"], ["", ""]],
            [["2", "2.1"], ["", ""]],
            [["2", "2.2"], ["", ""]],
            [["2.1", "2.1.1"], ["", ""]],
            [["2.1", "2.1.2"], ["", ""]],
            [["2.2", "2.2.1"], ["", ""]],
            [["2.2", "2.2.2"], ["", ""]],
            # Multi-label Test cases
            [["1", "1.1"], ["1", "1.2"]],
            [["1", "1.1"], ["2", "2.1"]],
            [["2.1", "2.1.1"], ["2.2", "2.2.2"]],
        ]
    )


def test_binary_policy_positive_examples(digraph, features_1d, labels):
    policy = BinaryPolicy(digraph, features_1d, labels)
    with pytest.raises(NotImplementedError):
        policy.positive_examples("1")


def test_binary_policy_negative_examples(digraph, features_1d, labels):
    policy = BinaryPolicy(digraph, features_1d, labels)
    with pytest.raises(NotImplementedError):
        policy.negative_examples("1")


def test_exclusive_policy_positive_examples_1(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        False,
        False,
        False,
    ]
    result = policy.positive_examples("1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_2(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        True,
        False,
    ]
    result = policy.positive_examples("1.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_positive_examples_3(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        False,
        True,
        False,
    ]
    result = policy.positive_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_1(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        True,
        True,
        True,
    ]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_2(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("1.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_policy_negative_examples_3(digraph, features_1d, labels):
    policy = ExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        True,
        False,
        True,
    ]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_1(digraph, features_1d, labels):
    policy = LessExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_2(digraph, features_1d, labels):
    policy = LessExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        False,
        False,
    ]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_less_exclusive_policy_negative_examples_3(digraph, features_1d, labels):
    policy = LessExclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        # Mutli-label Test cases
        True,
        True,
        False,
    ]
    result = policy.negative_examples("2.1.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_1(digraph, features_1d, labels):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        False, 
        False,
    ]
    result = policy.negative_examples("1.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_2(digraph, features_1d, labels):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        False,
        False,
        False,
    ]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_exclusive_siblings_policy_negative_examples_3(digraph, features_1d, labels):
    policy = ExclusiveSiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("2.1.2")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_1(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        True,
        False,
    ]
    result = policy.positive_examples("1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_2(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        True,
        True,
    ]
    result = policy.positive_examples("2")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_positive_examples_3(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        # Mutli-label Test cases
        False,
        True,
        True,
    ]
    result = policy.positive_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_1(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_2(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        False,
        False,
    ]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_inclusive_policy_negative_examples_3(digraph, features_1d, labels):
    policy = InclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        # Mutli-label Test cases
        True,
        False,
        False,
    ]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_1(digraph, features_1d, labels):
    policy = LessInclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_2(digraph, features_1d, labels):
    policy = LessInclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        False,
        False,
    ]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_less_inclusive_policy_negative_examples_3(digraph, features_1d, labels):
    policy = LessInclusivePolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        # Mutli-label Test cases
        True,
        True,
        False,
    ]
    result = policy.negative_examples("2.1.1")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_1(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        # Mutli-label Test cases
        False,
        True,
        True,
    ]
    result = policy.negative_examples("1")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_2(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        # Mutli-label Test cases
        True,
        True,
        False,
    ]
    result = policy.negative_examples("2")
    assert_array_equal(ground_truth, result)


def test_siblings_policy_negative_examples_3(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth = [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        # Mutli-label Test cases
        False,
        False,
        True,
    ]
    result = policy.negative_examples("2.1")
    assert_array_equal(ground_truth, result)


def test_siblings_get_binary_examples_1d_1(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth_x = [
        1,
        2,
        9,
        10,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
    ]  # TODO: 10 is both positive and negative example
    ground_truth_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_1d_2(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth_x = [
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        1,
        2,
        9,
        10,
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_1d_3(digraph, features_1d, labels):
    policy = SiblingsPolicy(digraph, features_1d, labels)
    ground_truth_x = [3, 5, 6, 10, 11, 4, 7, 8, 11]
    ground_truth_y = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_1(digraph, features_2d, labels):
    policy = SiblingsPolicy(digraph, features_2d, labels)
    ground_truth_x = [
        [1, 2],
        [3, 4],
        [17, 18],
        [19, 20],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [19, 20],
        [21, 22],
    ]
    ground_truth_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_2(digraph, features_2d, labels):
    policy = SiblingsPolicy(digraph, features_2d, labels)
    ground_truth_x = [
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [19, 20],
        [21, 22],
        [1, 2],
        [3, 4],
        [17, 18],
        [19, 20],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_2d_3(digraph, features_2d, labels):
    policy = SiblingsPolicy(digraph, features_2d, labels)
    ground_truth_x = [
        [5, 6],
        [9, 10],
        [11, 12],
        [19, 20],
        [21, 22],
        [7, 8],
        [13, 14],
        [15, 16],
        [21, 22],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x)
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_1(digraph, features_sparse, labels):
    policy = SiblingsPolicy(digraph, features_sparse, labels)
    ground_truth_x = [
        [1, 2],
        [3, 4],
        [17, 18],
        [19, 20],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [19, 20],
        [21, 22],
    ]
    ground_truth_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("1")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_2(digraph, features_sparse, labels):
    policy = SiblingsPolicy(digraph, features_sparse, labels)
    ground_truth_x = [
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [19, 20],
        [21, 22],
        [1, 2],
        [3, 4],
        [17, 18],
        [19, 20],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None


def test_siblings_get_binary_examples_sparse_3(digraph, features_sparse, labels):
    policy = SiblingsPolicy(digraph, features_sparse, labels)
    ground_truth_x = [
        [5, 6],
        [9, 10],
        [11, 12],
        [19, 20],
        [21, 22],
        [7, 8],
        [13, 14],
        [15, 16],
        [21, 22],
    ]
    ground_truth_y = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    x, y, weights = policy.get_binary_examples("2.1")
    assert_array_equal(ground_truth_x, x.todense())
    assert_array_equal(ground_truth_y, y)
    assert weights is None

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from hiclass import Policies

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


def test_siblings_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    assert list(samples) == [True] * 8


def test_siblings_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    assert list(samples) == [False] * 8


def test_siblings_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_siblings_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    assert list(samples) == [False, True, False, True, True, True, True, True]


def test_siblings_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_siblings_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    true_answer = [False] * 8
    true_answer[7] = True
    assert list(samples) == true_answer


def test_siblings_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True]
    assert list(samples) == true_answer


def test_siblings_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, True, False]


def test_siblings_policy_positive_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True] * 2
    assert list(samples) == true_answer


def test_siblings_policy_negative_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.SiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, True, False] * 2


def test_inclusive_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    assert list(samples) == [True] * 8


def test_inclusive_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    assert list(samples) == [False] * 8


def test_inclusive_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_inclusive_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    assert list(samples) == [False, True, False, True, True, True, True, True]


def test_inclusive_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_inclusive_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    assert list(samples) == [False, False, True, False, False, True, False, True]


def test_inclusive_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True]
    assert list(samples) == true_answer


def test_inclusive_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, False, False]


def test_inclusive_policy_positive_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True] * 2
    assert list(samples) == true_answer


def test_inclusive_policy_negative_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.InclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, False, False] * 2


def test_exclusive_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    true_answer = [False] * 8
    true_answer[0] = True
    assert list(samples) == true_answer


def test_exclusive_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    true_answer = [True] * 8
    true_answer[0] = False
    assert list(samples) == true_answer


def test_exclusive_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_exclusive_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    true_answer = [True] * 8
    true_answer[2] = False
    assert list(samples) == true_answer


def test_exclusive_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_exclusive_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    true_answer = [True] * 8
    true_answer[6] = False
    assert list(samples) == true_answer


def test_exclusive_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False] * 8
    true_answer[1] = True
    assert list(samples) == true_answer


def test_exclusive_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True] * 8
    true_answer[1] = False
    assert list(samples) == true_answer


def test_exclusive_policy_positive_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False] * 16
    true_answer[1] = True
    true_answer[9] = True
    assert list(samples) == true_answer


def test_exclusive_policy_negative_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.ExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True] * 16
    true_answer[1] = False
    true_answer[9] = False
    assert list(samples) == true_answer


def test_less_exclusive_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    true_answer = [False] * 8
    true_answer[0] = True
    assert list(samples) == true_answer


def test_less_exclusive_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    assert list(samples) == [False] * 8


def test_less_exclusive_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_less_exclusive_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    true_answer = [True] * 8
    true_answer[2] = False
    assert list(samples) == true_answer


def test_less_exclusive_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_less_exclusive_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    true_answer = [True] * 8
    true_answer[6] = False
    assert list(samples) == true_answer


def test_less_exclusive_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False] * 8
    true_answer[1] = True
    assert list(samples) == true_answer


def test_less_exclusive_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True, False, True, True, False, False, False, False]
    assert list(samples) == true_answer


def test_less_exclusive_policy_positive_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False] * 16
    true_answer[1] = True
    true_answer[9] = True
    assert list(samples) == true_answer


def test_less_exclusive_policy_negative_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.LessExclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True, False, True, True, False, False, False, False] * 2
    assert list(samples) == true_answer


def test_less_inclusive_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    true_answer = [True] * 8
    assert list(samples) == true_answer


def test_less_inclusive_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    assert list(samples) == [False] * 8


def test_less_inclusive_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_less_inclusive_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    true_answer = [True] * 8
    true_answer[2] = False
    assert list(samples) == true_answer


def test_less_inclusive_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_less_inclusive_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    true_answer = [True] * 8
    true_answer[6] = False
    assert list(samples) == true_answer


def test_less_inclusive_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True]
    assert list(samples) == true_answer


def test_less_inclusive_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True, False, True, True, False, False, False, False]
    assert list(samples) == true_answer


def test_less_inclusive_policy_positive_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, True, True, True, True] * 2
    assert list(samples) == true_answer


def test_less_inclusive_policy_negative_samples_middle_node_duplicate_labels(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.LessInclusivePolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    true_answer = [True, False, True, True, False, False, False, False] * 2
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_positive_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(1)
    true_answer = [False] * 8
    true_answer[0] = True
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_negative_samples_root(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(1)
    assert list(samples) == [False] * 8


def test_exclusive_siblings_policy_positive_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(3)
    true_answer = [False] * 8
    true_answer[2] = True
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_negative_samples_leaf(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(3)
    assert list(samples) == [False, True, False, True, False, False, False, False]


def test_exclusive_siblings_policy_positive_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(7)
    true_answer = [False] * 8
    true_answer[6] = True
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_negative_samples_leaf_multiple_parents(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(7)
    true_answer = [False] * 8
    true_answer[7] = True
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_positive_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False] * 8
    true_answer[1] = True
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_negative_samples_middle_node(small_dag):
    labels = np.arange(1, 9)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, False, False]


def test_exclusive_siblings_policy_positive_samples_middle_node_duplicate_labels(
    small_dag,
):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.positive_samples(2)
    true_answer = [False, True, False, False, False, False, False, False] * 2
    assert list(samples) == true_answer


def test_exclusive_siblings_policy_negative_samples_middle_node_duplicate_labels(
    small_dag,
):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.ExclusiveSiblingsPolicy(small_dag, labels)
    samples = policy.negative_samples(2)
    assert list(samples) == [False, False, True, True, False, False, False, False] * 2


def test_policy_positive_samples(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.Policy(small_dag, labels)
    with pytest.raises(NotImplementedError):
        assert policy.positive_samples(2)


def test_policy_negative_samples(small_dag):
    labels = np.array(list(range(1, 9)) * 2)
    policy = Policies.Policy(small_dag, labels)
    with pytest.raises(NotImplementedError):
        assert policy.negative_samples(2)

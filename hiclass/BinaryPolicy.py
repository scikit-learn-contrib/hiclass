"""Classes to create binary policies for positive and negative samples. Used by the :code:LocalClassifierPerNode."""
from abc import ABC

from scipy.sparse import vstack, csr_matrix
import networkx as nx
import numpy as np


class BinaryPolicy(ABC):
    """
    Abstract class used for all binary policies.

    Every policy should implement the methods positive_examples and negative_examples.
    """

    def __init__(
        self, digraph: nx.DiGraph, X: np.ndarray, y: np.ndarray, sample_weight=None
    ):
        """
        Initialize a BinaryPolicy with the required data.

        Parameters
        ----------
        digraph : nx.DiGraph
            DiGraph which is used for inferring nodes relationships.
        X : np.ndarray
            Features which will be used for fitting a model.
        y : np.ndarray
            Labels which will be assigned to the different samples.
            Has to be 2D array.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        """
        self.digraph = digraph
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.axis = {
            2: 1,
            3: (2, 1),
        }

    def positive_examples(self, node) -> np.ndarray:
        """
        Gather all positive examples corresponding to the given node.

        Parameters
        ----------
        node
            Node for which the positive examples should be searched.

        Returns
        -------
        positive_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        raise NotImplementedError

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_samples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        raise NotImplementedError

    def _get_descendants(self, node, inclusive: bool = True):
        """
        Gather all descendants for a given node.

        Parameters
        ----------
        node
            Node for which the descendants should be obtained.
        inclusive : bool, default=True
            True if the given node should be included in the list of descendants.

        Returns
        -------
        descendants : set
            Set of descendants for a given node.
        """
        descendants = set()
        if inclusive:
            descendants.add(node)
        for successor in nx.dfs_successors(self.digraph, node).values():
            descendants.update(successor)
        return descendants

    def _get_siblings(self, node):
        """
        Gather all siblings for a given node.

        Parameters
        ----------
        node
            Node for which the siblings should be obtained.

        Returns
        -------
        siblings : set
            Set of siblings for a given node.
        """
        parents = self.digraph.predecessors(node)
        parents = list(parents)
        siblings = set()
        for parent in parents:
            siblings.update(self.digraph.successors(parent))
        siblings.discard(node)
        return siblings

    def get_binary_examples(self, node) -> tuple:
        """
        Gather all positive and negative examples for a given node.

        Parameters
        ----------
        node
            Node for which the positive and negative examples should be searched.

        Returns
        -------
        X : np.ndarray
            The subset with positive and negative features.
        y : np.ndarray
            The subset with positive and negative labels.
        """
        positive_examples = self.positive_examples(node)
        negative_examples = np.logical_and(
            np.logical_not(positive_examples), self.negative_examples(node)
        )
        positive_x = self.X[positive_examples]
        negative_x = self.X[negative_examples]
        positive_weights = (
            self.sample_weight[positive_examples]
            if self.sample_weight is not None
            else None
        )
        negative_weights = (
            self.sample_weight[negative_examples]
            if self.sample_weight is not None
            else None
        )
        if isinstance(self.X, np.ndarray):
            X = np.concatenate([positive_x, negative_x])
            sample_weights = (
                np.concatenate([positive_weights, negative_weights])
                if self.sample_weight is not None
                else None
            )
            y = np.zeros(len(X))
            y[: len(positive_x)] = 1
        elif isinstance(self.X, csr_matrix):
            X = vstack([positive_x, negative_x])
            sample_weights = (
                vstack([positive_weights, negative_weights])
                if self.sample_weight is not None
                else None
            )
            y = np.zeros(X.shape[0])
            y[: positive_x.shape[0]] = 1
        return X, y, sample_weights

    def _get_axis(self):
        return self.axis[self.y.ndim]


class ExclusivePolicy(BinaryPolicy):
    """Implement the exclusive policy of the referenced paper."""

    def positive_examples(self, node) -> np.ndarray:
        """
        Gather all positive examples corresponding to the given node.

        This only includes examples for the given node.

        Parameters
        ----------
        node
            Node for which the positive examples should be searched.

        Returns
        -------
        positive_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        positive_examples = np.isin(self.y, node).any(axis=self._get_axis())
        return positive_examples

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes all examples except the positive ones.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        negative_examples = np.logical_not(self.positive_examples(node))
        return negative_examples


class LessExclusivePolicy(ExclusivePolicy):
    """Implement the less exclusive policy of the referenced paper."""

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes all examples except the examples for the current
        node and its children.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        descendants = self._get_descendants(node, inclusive=True)
        negative_examples = np.logical_not(
            np.isin(self.y, list(descendants)).any(axis=self._get_axis())
        )
        return negative_examples


class ExclusiveSiblingsPolicy(ExclusivePolicy):
    """Implement the exclusive siblings policy of the referenced paper."""

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes examples for all nodes that have the same parent as the given node.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        siblings = self._get_siblings(node)
        negative_examples = np.isin(self.y, list(siblings)).any(axis=self._get_axis())
        return negative_examples


class InclusivePolicy(BinaryPolicy):
    """Implement the inclusive policy of the referenced paper."""

    def positive_examples(self, node) -> np.ndarray:
        """
        Gather all positive examples corresponding to the given node.

        This includes examples for the given node and its descendants.

        Parameters
        ----------
        node
            Node for which the positive examples should be searched.

        Returns
        -------
        positive_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        descendants = self._get_descendants(node, inclusive=True)
        positive_examples = np.isin(self.y, list(descendants)).any(
            axis=self._get_axis()
        )
        return positive_examples

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes all examples, except the examples for the given node, its descendants and successors.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        descendants = self._get_descendants(node, inclusive=True)
        ancestors = nx.ancestors(self.digraph, node)
        descendants_and_ancestors = set.union(descendants, ancestors)
        negative_examples = np.logical_not(
            np.isin(self.y, list(descendants_and_ancestors)).any(axis=self._get_axis())
        )
        return negative_examples


class LessInclusivePolicy(InclusivePolicy):
    """Implement the less inclusive policy of the referenced paper."""

    # TODO: inherit negatives_examples from LessExclusivePolicy
    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes all examples except the examples for the current
        node and its children.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        descendants = self._get_descendants(node, inclusive=True)
        negative_examples = np.logical_not(
            np.isin(self.y, list(descendants)).any(axis=self._get_axis())
        )
        return negative_examples


class SiblingsPolicy(InclusivePolicy):
    """Implement the siblings policy of the referenced paper."""

    def negative_examples(self, node) -> np.ndarray:
        """
        Gather all negative examples corresponding to the given node.

        This includes all examples for nodes that have the same ancestors as the given node,
        as well as their descendants.

        Parameters
        ----------
        node
            Node for which the negative examples should be searched.

        Returns
        -------
        negative_examples : np.ndarray
            A mask for which examples are included (True) and which are not.
        """
        siblings = self._get_siblings(node)
        negative_classes = set()
        for sibling in siblings:
            negative_classes.update(self._get_descendants(sibling, inclusive=True))
        negative_examples = np.isin(self.y, list(negative_classes)).any(
            axis=self._get_axis()
        )
        return negative_examples


IMPLEMENTED_POLICIES = {
    "exclusive": ExclusivePolicy,
    "less_exclusive": LessExclusivePolicy,
    "exclusive_siblings": ExclusiveSiblingsPolicy,
    "inclusive": InclusivePolicy,
    "less_inclusive": LessInclusivePolicy,
    "siblings": SiblingsPolicy,
}

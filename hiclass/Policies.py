"""Classes to create binary data policies for positive and negative samples. Used by the :code:LocalClassifierPerNode."""
from abc import ABC

import networkx as nx
import numpy as np

from hiclass.data import NODE_TYPE


class Policy(ABC):
    """
    Abstract class used for all policies.

    Every policy should implement positive_samples and negative_samples methods.
    """

    def __init__(self, label_graph: nx.DiGraph, labels: np.ndarray):
        """
        Initialize a Policy with the required data.

        Parameters
        ----------
        label_graph : nx.DiGraph
            Graph which is used for inferring label relationships.
        labels : np.array
            Labels which will be assigned to the different samples.
            Has to be 1d array of integers.
        """
        self.graph = label_graph
        self.data = labels

    def positive_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all positive labels corresponding to the given label.

        Parameters
        ----------
        label : int or str
            Label for which the positive samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        raise NotImplementedError

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        raise NotImplementedError

    def _get_descendants(self, label: NODE_TYPE, inclusive: bool = True):
        """
        Gather all descendants for a given label.

        Parameters
        ----------
        label : int or str
            Label for which the descendants should be obtained.
        inclusive : bool, default=True
            True if the given label should be included in the list of descendants.

        Returns
        -------
        descendants : set
            Set of descendants for a given label.
        """
        descendants = set()
        if inclusive:
            descendants.add(label)
        for labels in nx.dfs_successors(self.graph, label).values():
            descendants.update(labels)
        return descendants

    def _get_siblings(self, label: NODE_TYPE):
        """
        Gather all siblings for a given label.

        Parameters
        ----------
        label : int or str
            Label for which the siblings should be obtained.

        Returns
        -------
        siblings : set
            Set of siblings for a given label.
        """
        parents = self.graph.predecessors(label)
        parents = list(parents)
        siblings = set()
        for parent in parents:
            siblings.update(self.graph.successors(parent))
        if label in siblings:
            siblings.remove(label)
        return siblings


class ExclusivePolicy(Policy):
    """Represent the exclusive policy of the referenced paper."""

    def positive_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all positive labels corresponding to the given label.

        This only includes the given label.

        Parameters
        ----------
        label : int or str
            Label for which the positive samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        return np.isin(self.data, label)

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels except the given label.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        negative_classes = np.logical_not(self.positive_samples(label))
        return negative_classes


class LessExclusivePolicy(ExclusivePolicy):
    """Represent the less exclusive policy of the referenced paper."""

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels except the given label and its children.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        negative_classes = set(self.graph.nodes)
        descendants = self._get_descendants(label)
        negative_classes.difference_update(descendants)
        return np.isin(self.data, list(negative_classes))


class ExclusiveSiblingsPolicy(ExclusivePolicy):
    """
    Represent the exclusive siblings policy of the referenced paper.

    Siblings are here defined as all classes that have the same distance to the root node (since DAGs are supported).
    """

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels that have the same distance from the root as the given label.
        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        siblings = self._get_siblings(label)
        return np.isin(self.data, list(siblings))


class InclusivePolicy(Policy):
    """Represent the inclusive policy of the referenced paper."""

    def positive_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all positive labels corresponding to the given label.

        This includes the label itself and all its descendants.

        Parameters
        ----------
        label : int or str
            Label for which the positive samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        descendants = self._get_descendants(label)
        return np.isin(self.data, list(descendants))

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels except the given label, its descendants and ancestors.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        negative_classes = set(self.graph.nodes)
        descendants = self._get_descendants(label)
        negative_classes.difference_update(descendants)
        ancestors = nx.ancestors(self.graph, label)
        negative_classes.difference_update(ancestors)
        return np.isin(self.data, list(negative_classes))


class SiblingsPolicy(InclusivePolicy):
    """
    Represent the siblings policy of the referenced paper.

    Siblings are defined here as all classes that have the same distance to the root node (since DAGs are supported).
    """

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels that have the same distance from the root as the given label, as well as all their
        children. In DAGs, this can include classes of the positive sample.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        siblings = self._get_siblings(label)
        negative_classes = set()
        for sibling in siblings:
            negative_classes.update(self._get_descendants(sibling))
        return np.isin(self.data, list(negative_classes))


class LessInclusivePolicy(InclusivePolicy):
    """Represent the less inclusive policy of the referenced paper."""

    def negative_samples(self, label: NODE_TYPE) -> np.ndarray:
        """
        Gather all negative labels corresponding to the given label.

        This includes all labels except the given label and its children.

        Parameters
        ----------
        label : int or str
            Label for which the negative samples should be searched.

        Returns
        -------
        positive_samples : np.array
            A mask for which labels are included (True) and which are not.
        """
        negative_classes = np.logical_not(self.positive_samples(label))
        return negative_classes


IMPLEMENTED_POLICIES = {
    "exclusive": ExclusivePolicy,
    "less_exclusive": LessExclusivePolicy,
    "exclusive_siblings": ExclusiveSiblingsPolicy,
    "inclusive": InclusivePolicy,
    "siblings": SiblingsPolicy,
    "less_inclusive": LessInclusivePolicy,
}

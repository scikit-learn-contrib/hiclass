"""Abstract class defining the structure of a probability combiner."""
import abc
import numpy as np
from typing import List
from collections import defaultdict
from networkx.exception import NetworkXError
from hiclass import HierarchicalClassifier


class ProbabilityCombiner(abc.ABC):
    """Abstract class defining the structure of a probability combiner."""

    def __init__(
        self, classifier: HierarchicalClassifier, normalize: bool = True
    ) -> None:
        """Initialize probability combiner object."""
        self.classifier = classifier
        self.normalize = normalize

    @abc.abstractmethod
    def combine(self, proba: List[np.ndarray]) -> List[np.ndarray]:
        """Combine probabilities over multiple levels."""
        ...

    def _normalize(self, proba: List[np.ndarray]):
        return [
            np.nan_to_num(
                level_probabilities / level_probabilities.sum(axis=1, keepdims=True)
            )
            for level_probabilities in proba
        ]

    def _find_predecessors(self, level: int):
        predecessors = defaultdict(list)
        for node in self.classifier.global_classes_[level]:
            try:
                predecessor = list(self.classifier.hierarchy_.predecessors(node))[0]
            except NetworkXError:
                # skip empty levels
                continue

            predecessor_name = str(predecessor).split(self.classifier.separator_)[-1]
            node_name = str(node).split(self.classifier.separator_)[-1]

            predecessors[node_name].append(predecessor_name)
        return predecessors

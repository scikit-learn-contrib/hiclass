"""Abstract class defining the structure of a probability combiner."""

import abc
import numpy as np
from typing import List
from collections import defaultdict
from networkx.exception import NetworkXError
from hiclass import HierarchicalClassifier
from hiclass._hiclass_utils import _normalize_probabilities


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
        return _normalize_probabilities(proba)

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

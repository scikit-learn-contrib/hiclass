import abc
import numpy as np
from typing import List
from collections import defaultdict
from networkx.exception import NetworkXError

class ProbabilityCombiner(abc.ABC):

    def __init__(self, classifier, normalize=True) -> None:
        self.classifier = classifier
        self.normalize = normalize

    @abc.abstractmethod
    def combine(self, proba: List[np.ndarray]) -> List[np.ndarray]:
        ...
    
    def _normalize(self, proba):
        return [
            np.nan_to_num(level_probabilities / level_probabilities.sum(axis=1, keepdims=True)) 
                for level_probabilities in proba
        ]
    
    def _find_predecessors(self, level):
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

import abc
import numpy as np
from typing import List

class ProbabilityCombiner(abc.ABC):

    def __init__(self, classifier) -> None:
        self.classifier = classifier

    @abc.abstractmethod
    def combine(self, proba: List[np.ndarray]) -> List[np.ndarray]:
        ...

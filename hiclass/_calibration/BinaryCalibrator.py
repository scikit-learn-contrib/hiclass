import abc
import numpy as np


class _BinaryCalibrator(abc.ABC):
    @abc.abstractmethod
    def fit(
        self, y: np.ndarray, scores: np.ndarray, X: np.ndarray = None
    ):  # pragma: no cover
        ...

    @abc.abstractmethod
    def predict_proba(
        self, scores: np.ndarray, X: np.ndarray = None
    ):  # pragma: no cover
        ...

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

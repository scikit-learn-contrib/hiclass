import abc


class _BinaryCalibrator(abc.ABC):

    @abc.abstractmethod
    def fit(self, y, scores, X=None):  # pragma: no cover
        ...

    @abc.abstractmethod
    def predict_proba(self, scores, X=None):  # pragma: no cover
        ...

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

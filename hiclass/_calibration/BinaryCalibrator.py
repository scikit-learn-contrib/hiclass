import abc


class _BinaryCalibrator(abc.ABC):

    @abc.abstractmethod
    def fit(self, y, scores, X=None):  # pragma: no cover
        ...

    @abc.abstractmethod
    def predict_proba(self, scores, X=None):  # pragma: no cover
        ...

from hiclass._calibration.BinaryCalibrator import _BinaryCalibrator
from sklearn.calibration import _SigmoidCalibration
from sklearn.utils.validation import check_is_fitted


class _PlattScaling(_BinaryCalibrator):
    name = "PlattScaling"

    def __init__(self) -> None:
        self._is_fitted = False
        self.platt_scaling = _SigmoidCalibration()

    def fit(self, y, scores, X=None):
        self.platt_scaling.fit(scores, y)
        self._is_fitted = True
        return self

    def predict_proba(self, scores, X=None):
        check_is_fitted(self)
        return self.platt_scaling.predict(scores)

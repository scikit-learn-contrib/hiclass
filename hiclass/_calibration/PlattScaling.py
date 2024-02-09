from hiclass._calibration.BinaryCalibrator import _BinaryCalibrator
from sklearn.calibration import _SigmoidCalibration
from sklearn.exceptions import NotFittedError


class _PlattScaling(_BinaryCalibrator):
    name = "PlattScaling"

    def __init__(self) -> None:
        self.fitted = False
        self.platt_scaling = _SigmoidCalibration()

    def fit(self, y, scores, X=None):
        self.platt_scaling.fit(scores, y)
        self.fitted = True
        return self

    def predict_proba(self, scores, X=None):
        if not self.fitted:
            raise NotFittedError(f"This {self.name} calibrator is not fitted yet. Call 'fit' with appropriate arguments before using this calibrator.")
        return self.platt_scaling.predict(scores)

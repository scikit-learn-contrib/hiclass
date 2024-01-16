from calibration.BinaryCalibrator import BinaryCalibrator
from sklearn.isotonic import IsotonicRegression as SkLearnIR
from sklearn.exceptions import NotFittedError



class IsotonicRegression(BinaryCalibrator):
    name = "IsotonicRegression"

    def __init__(self, params={}) -> None:
        self.fitted = False
        if "out_of_bounds" not in params:
            params["out_of_bounds"] = "clip"
        self.isotonic_regression = SkLearnIR(**params)

    def fit(self, y, scores, X=None):
        self.isotonic_regression.fit(scores, y)
        self.fitted = True
        return self

    def predict_proba(self, scores, X=None):
        if not self.fitted:
            raise NotFittedError(f"This {self.name} calibrator is not fitted yet. Call 'fit' with appropriate arguments before using this calibrator.")
        return self.isotonic_regression.predict(scores)

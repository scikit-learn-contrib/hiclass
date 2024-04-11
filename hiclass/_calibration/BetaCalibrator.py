from hiclass._calibration.BinaryCalibrator import _BinaryCalibrator
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.linear_model import LogisticRegression


class _BetaCalibrator(_BinaryCalibrator):
    name = "BetaCalibrator"

    def __init__(self) -> None:
        super().__init__()
        self.skip_calibration = False

    def fit(self, y: np.ndarray, scores: np.ndarray, X: np.ndarray = None):
        unique_labels = len(np.unique(y))
        if unique_labels < 2:
            self.skip_calibration = True
            self._is_fitted = True
            return self

        scores_1 = np.log(scores)
        scores_2 = -np.log(1 - scores)
        feature_matrix = np.column_stack((scores_1, scores_2))

        lr = LogisticRegression()
        lr.fit(feature_matrix, y)
        self.a, self.b = lr.coef_.flatten()
        self.c = lr.intercept_[0]

        self._is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray, X: np.ndarray = None):
        check_is_fitted(self)
        if self.skip_calibration:
            return scores
        return 1 / (
            1
            + 1
            / (
                np.exp(self.c)
                * (np.power(scores, self.a) / np.power((1 - scores), self.b))
            )
        )

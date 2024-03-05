"""Constant classifier if there is only one class in the training set."""

import numpy as np


class ConstantClassifier:
    """A classifier that always returns the only label seen during fit."""

    def fit(self, X, y, sample_weight=None):
        """
        Fit a constant classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples, n_levels)
            The target values, i.e., hierarchical class labels for classification.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 1:
            raise ValueError(
                f"Labels should have only one class to fit, but instead found {len(self.classes_)}"
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for the given data.

        Parameters
        ----------
        X : np.ndarray of shape(n_samples, ...)
            Data that should be predicted. Only the number of samples matters.

        Returns
        -------
        output : np.ndarray
            1 for the previously seen class.
        """
        return np.vstack([1] * X.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for the given data.

        Parameters
        ----------
        X : np.ndarray of shape(n_samples, ...)
            Data that should be predicted. Only the number of samples matters.

        Returns
        -------
        output : np.ndarray
            1 for the previously seen class.
        """
        return np.vstack([self.classes_] * X.shape[0])

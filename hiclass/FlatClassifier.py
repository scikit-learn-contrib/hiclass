"""
Flat classifier approach, used for comparison purposes.

Implementation by @lpfgarcia
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class FlatClassifier(BaseEstimator):
    """
    Assign a single flat classifier that learns the flat hierarchy.

    A flat classifier in this context concatenates all levels in the hierarchy into a single label.

    Examples
    --------
    >>> from hiclass import FlatClassifier
    >>> y = [['1', '1.1'], ['2', '2.1']]
    >>> X = [[1, 2], [3, 4]]
    >>> flat = FlatClassifier()
    >>> flat.fit(X, y)
    >>> flat.predict(X)
    array([['1', '1.1'],
        ['2', '2.1']])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
    ):
        """
        Initialize a flat classifier.

        Parameters
        ----------
        local_classifier : BaseEstimator, default=LogisticRegression
            The scikit-learn model used for the flat classification. Needs to have fit, predict and clone methods.
        """
        self.local_classifier = local_classifier

    def fit(self, X, y, sample_weight=None):
        """
        Fit a flat classifier.

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
        # Convert from hierarchical labels to flat labels
        self.separator_ = "::HiClass::Separator::"
        y = [self.separator_.join(i) for i in y]

        # Fit flat classifier
        self.local_classifier.fit(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict classes for the given data.

        Hierarchical labels are returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Predict and remove separator
        predictions = [
            i.split(self.separator_) for i in self.local_classifier.predict(X)
        ]

        return np.array(predictions)

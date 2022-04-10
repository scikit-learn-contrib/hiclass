"""
Local classifier per level approach.

Numeric and string output labels are both handled.
"""
import logging
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LocalClassifierPerLevel(BaseEstimator):
    """
    Assign local classifiers to each level of the hierarchy, except the root node.

    A local classifier per level is a local hierarchical classifier that fits one local multi-class classifier
    for each level of the class hierarchy, except for the root node.
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
    ):
        """
        Initialize a local classifier per level.

        Parameters
        ----------
        local_classifier : BaseEstimator, default=LogisticRegression
            The local_classifier used to create the collection of local classifiers. Needs to have fit, predict and
            clone methods.
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
            See https://verboselogs.readthedocs.io/en/latest/readme.html#overview-of-logging-levels
            for more information.
        edge_list : str, default=None
            Path to write the hierarchy built.
        replace_classifiers : bool, default=True
            Turns on (True) the replacement of a local classifier with a constant classifier when trained on only
            a single unique class.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Only :code:`fit` is parallelized.
        """
        self.local_classifier = local_classifier
        self.verbose = verbose
        self.edge_list = edge_list
        self.replace_classifiers = replace_classifiers
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit a local classifier per level.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples, n_levels)
            The target values, i.e., hierarchical class labels for classification.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        # and convert them to np.ndarray if need be
        self.X_, self.y_ = check_X_y(X, y, multi_output=True, accept_sparse="csr")

        # Create and configure logger
        self._create_logger()

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
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    def _create_logger(self):
        # Create logger
        self.logger_ = logging.getLogger("LCPL")
        self.logger_.setLevel(self.verbose)

        # Create console handler and set verbose level
        ch = logging.StreamHandler()
        ch.setLevel(self.verbose)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        self.logger_.addHandler(ch)

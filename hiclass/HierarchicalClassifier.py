"""Shared code for all classifiers."""
import abc
import logging
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y


class HierarchicalClassifier(abc.ABC):
    """Abstract class for the local hierarchical classifiers.

    Offers mostly utility methods and common data initialization.
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
        Initialize a local hierarchical classifier.

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
        Fit a local hierarchical classifier.

        Needs to be subclassed by other classifiers as it only offers common methods.

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

    def _create_logger(self):
        # Create logger
        self.logger_ = logging.getLogger(self.classifier_abbreviation)
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

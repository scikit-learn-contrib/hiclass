"""
Local classifier per level approach.

Numeric and string output labels are both handled.
"""
from copy import deepcopy

import numpy as np
import ray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier


@ray.remote
def _parallel_fit(lcpl, level):
    classifier = lcpl.local_classifiers_[level]
    X = lcpl.X_
    y = lcpl.y_[:, level]
    unique_y = np.unique(y)
    if len(unique_y) == 1 and lcpl.replace_classifiers:
        classifier = ConstantClassifier()
    classifier.fit(X, y)
    return classifier


class LocalClassifierPerLevel(BaseEstimator, HierarchicalClassifier):
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
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPL",
        )

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
        # Execute common methods necessary before fitting
        super()._pre_fit(X, y)

        # Fit local classifiers in DAG
        super().fit(X, y)

        # TODO: Store the classes seen during fit

        # TODO: Add function to allow user to change local classifier

        # TODO: Add parameter to receive hierarchy as parameter in constructor

        # TODO: Add support to empty labels in some levels

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

        # Input validation
        X = check_array(X, accept_sparse="csr")

        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # TODO: Add threshold to stop prediction halfway if need be

        self.logger_.info("Predicting")

        for level, classifier in enumerate(self.local_classifiers_):
            self.logger_.info(f"Predicting level {level}")
            if level == 0:
                y[:, level] = classifier.predict(X)
            else:
                all_probabilities = classifier.predict_proba(X)
                successors = np.array(
                    [
                        list(self.hierarchy_.successors(node))
                        for node in y[:, level - 1]
                    ],
                    dtype=object,
                )
                classes_masks = np.array(
                    [
                        np.isin(classifier.classes_, successors[i])
                        for i in range(len(successors))
                    ]
                )
                probabilities = np.array(
                    [
                        all_probabilities[i, classes_masks[i]]
                        for i in range(len(classes_masks))
                    ],
                    dtype=object,
                )
                highest_probabilities = [
                    np.argmax(probabilities[i], axis=0)
                    for i in range(len(probabilities))
                    if len(probabilities[i] > 0)
                ]
                classes = np.array(
                    [
                        classifier.classes_[classes_masks[i]]
                        for i in range(len(classes_masks))
                    ],
                    dtype=object,
                )
                classes = classes[self.masks_[level]]
                y[self.masks_[level], level] = np.array(
                    [
                        classes[i][highest_probabilities[i]]
                        for i in range(len(highest_probabilities))
                    ]
                )

        # Convert back to 1D if there is only 1 column to pass all sklearn's checks
        if self.max_levels_ == 1:
            y = y.flatten()

        # Remove separator from predictions
        if y.ndim == 2:
            for i in range(y.shape[0]):
                for j in range(1, y.shape[1]):
                    y[i, j] = y[i, j].split(self.separator_)[-1]

        return y

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        self.local_classifiers_ = [
            deepcopy(self.local_classifier_) for _ in range(self.y_.shape[1])
        ]
        self.masks_ = [None for _ in range(self.y_.shape[1])]

    def _fit_digraph(self):
        self.logger_.info("Fitting local classifiers")
        for level, classifier in enumerate(self.local_classifiers_):
            self.logger_.info(
                f"Fitting local classifier for level '{level + 1}' ({level + 1}/{len(self.local_classifiers_)})"
            )
            X = self.X_
            y = self.y_[:, level]

            # Detect empty leaf nodes
            leaves = np.array([str(i).split(self.separator_)[-1] for i in y])
            mask = leaves != ""

            # Remove rows with empty leaf nodes
            X = X[mask]
            y = y[mask]

            # Store mask for current level
            self.masks_[level] = mask

            unique_y = np.unique(y)
            if len(unique_y) == 1 and self.replace_classifiers:
                self.logger_.warning(
                    f"Fitting ConstantClassifier for level '{level + 1}'"
                )
                self.local_classifiers_[level] = ConstantClassifier()
                classifier = self.local_classifiers_[level]
            classifier.fit(X, y)

    def _fit_digraph_parallel(self, local_mode: bool = False):
        self.logger_.info("Fitting local classifiers")
        ray.init(num_cpus=self.n_jobs, local_mode=local_mode, ignore_reinit_error=True)
        lcpl = ray.put(self)
        results = [
            _parallel_fit.remote(lcpl, level)
            for level in range(len(self.local_classifiers_))
        ]
        classifiers = ray.get(results)
        for level, classifier in enumerate(classifiers):
            self.local_classifiers_[level] = classifier

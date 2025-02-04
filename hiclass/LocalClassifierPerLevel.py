"""
Local classifier per level approach.

Numeric and string output labels are both handled.
"""

import hashlib
import pickle
from copy import deepcopy
from os.path import exists

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass._calibration.Calibrator import _Calibrator

from hiclass.probability_combiner import (
    init_strings as probability_combiner_init_strings,
)

try:
    import ray
except ImportError:
    _has_ray = False
else:
    _has_ray = True


class LocalClassifierPerLevel(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each level of the hierarchy, except the root node.

    A local classifier per level is a local hierarchical classifier that fits one local multi-class classifier
    for each level of the class hierarchy, except for the root node.

    Examples
    --------
    >>> from hiclass import LocalClassifierPerLevel
    >>> y = [['1', '1.1'], ['2', '2.1']]
    >>> X = [[1, 2], [3, 4]]
    >>> lcpl = LocalClassifierPerLevel()
    >>> lcpl.fit(X, y)
    >>> lcpl.predict(X)
    array([['1', '1.1'],
       ['2', '2.1']])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
        calibration_method: str = None,
        return_all_probabilities: bool = False,
        probability_combiner: str = "multiply",
        tmp_dir: str = None,
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
            If :code:`Ray` is installed it is used, otherwise it defaults to :code:`Joblib`.
        bert : bool, default=False
            If True, skip scikit-learn's checks and sample_weight passing for BERT.
        calibration_method : {"ivap", "cvap", "platt", "isotonic", "beta"}, str, default=None
            If set, use the desired method to calibrate probabilities returned by predict_proba().
        return_all_probabilities : bool, default=False
            If True, return probabilities for all levels. Otherwise, return only probabilities for the last level.
        probability_combiner: {"geometric", "arithmetic", "multiply", None}, str, default="multiply"
            Specify the rule for combining probabilities over multiple levels:

            - `geometric`: Each levels probabilities are calculated by taking the geometric mean of itself and its predecessors;
            - `arithmetic`: Each levels probabilities are calculated by taking the arithmetic mean of itself and its predecessors;
            - `multiply`: Each levels probabilities are calculated by multiplying itself with its predecessors.
            - `None`: No aggregation.
        tmp_dir : str, default=None
            Temporary directory to persist local classifiers that are trained. If the job needs to be restarted,
            it will skip the pre-trained local classifier found in the temporary directory.
        """
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPL",
            bert=bert,
            calibration_method=calibration_method,
            tmp_dir=tmp_dir,
        )
        self.return_all_probabilities = return_all_probabilities
        self.probability_combiner = probability_combiner

        if (
            self.probability_combiner
            and self.probability_combiner not in probability_combiner_init_strings
        ):
            raise ValueError(
                f"probability_combiner must be one of {', '.join(probability_combiner_init_strings)} or None."
            )

    def fit(self, X, y, sample_weight=None):
        """
        Fit a local classifier per level.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
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
        # Execute common methods necessary before fitting
        super()._pre_fit(X, y, sample_weight)
        self.local_calibrators_ = None

        # Fit local classifiers in DAG
        super().fit(X, y)

        # TODO: Store the classes seen during fit

        # TODO: Add function to allow user to change local classifier

        # TODO: Add parameter to receive hierarchy as parameter in constructor

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
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # TODO: Add threshold to stop prediction halfway if need be

        self.logger_.info("Predicting")

        # Predict first level
        classifier = self.local_classifiers_[0]
        y[:, 0] = classifier.predict(X).flatten()

        self._predict_remaining_levels(X, y)

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y

    def predict_proba(self, X):
        """
        Predict class probabilities for the given data.

        Hierarchical labels are returned.
        If return_all_probabilities=True: Returns the probabilities for each level.
        Else: Returns the probabilities for the lowest level.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        T : ndarray of shape (n_samples,n_classes) or List[ndarray(n_samples,n_classes)]
            The predicted probabilities of the lowest levels or of all levels.
        """
        # Check if fit has been called
        check_is_fitted(self)

        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        if not self.calibration_method:
            self.logger_.info(
                "It is not recommended to use predict_proba() without calibration"
            )

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        self.logger_.info("Predicting Probability")

        # Predict first level
        classifier = self.local_classifiers_[0]
        calibrator = (
            self.local_calibrators_[0]
            if hasattr(self, "self.local_calibrators_")
            else None
        )

        # use classifier as a fallback if no calibrator is available
        calibrator = calibrator or classifier
        proba = calibrator.predict_proba(X)
        y[:, 0] = calibrator.classes_[np.argmax(proba, axis=1)]

        level_probability_list = [proba] + self._predict_proba_remaining_levels(X, y)
        level_probability_list = self._combine_and_reorder(level_probability_list)

        # combine probabilities
        if self.probability_combiner:
            probability_combiner_ = self._create_probability_combiner(
                self.probability_combiner
            )
            self.logger_.info(
                f"Combining probabilities using {type(probability_combiner_).__name__}"
            )
            level_probability_list = probability_combiner_.combine(
                level_probability_list
            )

        return (
            level_probability_list
            if self.return_all_probabilities
            else level_probability_list[-1]
        )

    def _predict_proba_remaining_levels(self, X, y):
        level_probability_list = []
        for level in range(1, y.shape[1]):
            classifier = self.local_classifiers_[level]
            calibrator = (
                self.local_calibrators_[level]
                if hasattr(self, "self.local_calibrators_")
                else None
            )
            # use classifier as a fallback if no calibrator is available
            calibrator = calibrator or classifier
            probabilities = calibrator.predict_proba(X)
            level_probability_list.append(probabilities)
        return level_probability_list

    def _predict_remaining_levels(self, X, y):
        for level in range(1, y.shape[1]):
            classifier = self.local_classifiers_[level]
            probabilities = classifier.predict_proba(X)
            classes = self.local_classifiers_[level].classes_
            probabilities_dict = [dict(zip(classes, prob)) for prob in probabilities]
            successors = self._get_successors(y[:, level - 1])
            successors_prob = self._get_successors_probability(
                probabilities_dict, successors
            )
            index_max_probability = [
                np.argmax(prob) if len(prob) > 0 else None for prob in successors_prob
            ]
            y[:, level] = [
                (
                    successors_list[index_max_probability[i]]
                    if index_max_probability[i] is not None
                    else ""
                )
                for i, successors_list in enumerate(successors)
            ]

    @staticmethod
    def _get_successors_probability(probabilities_dict, successors):
        successors_probability = [
            np.array(
                [probabilities_dict[i][successor] for successor in successors_list]
            )
            for i, successors_list in enumerate(successors)
        ]
        return successors_probability

    def _get_successors(self, level):
        successors = [
            (
                list(self.hierarchy_.successors(node))
                if self.hierarchy_.has_node(node)
                else []
            )
            for node in level
        ]
        return successors

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        self.local_classifiers_ = [
            deepcopy(self.local_classifier_) for _ in range(self.y_.shape[1])
        ]
        self.masks_ = [None for _ in range(self.y_.shape[1])]

    def _initialize_local_calibrators(self):
        super()._initialize_local_calibrators()
        self.local_calibrators_ = [
            _Calibrator(estimator=local_classifier, method=self.calibration_method)
            for local_classifier in self.local_classifiers_
        ]

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")

        def logging_wrapper(func, level, separator, max_level):
            self.logger_.info(f"fitting level {level + 1}/{max_level}")
            return func(self, level, separator)

        if self.n_jobs > 1:
            if _has_ray and not use_joblib:
                if not ray.is_initialized:
                    ray.init(
                        num_cpus=self.n_jobs,
                        local_mode=local_mode,
                        ignore_reinit_error=True,
                    )
                lcpl = ray.put(self)
                _parallel_fit = ray.remote(self._fit_classifier)
                results = [
                    _parallel_fit.remote(lcpl, level, self.separator_)
                    for level in range(len(self.local_classifiers_))
                ]
                classifiers = ray.get(results)
            else:
                classifiers = Parallel(n_jobs=self.n_jobs)(
                    delayed(logging_wrapper)(
                        self._fit_classifier,
                        level,
                        self.separator_,
                        len(self.local_classifiers_),
                    )
                    for level in range(len(self.local_classifiers_))
                )
        else:
            classifiers = [
                logging_wrapper(
                    self._fit_classifier,
                    level,
                    self.separator_,
                    len(self.local_classifiers_),
                )
                for level in range(len(self.local_classifiers_))
            ]
        for level, classifier in enumerate(classifiers):
            self.local_classifiers_[level] = classifier

    def _calibrate_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local calibrators")

        def logging_wrapper(func, level, separator, max_level):
            self.logger_.info(f"calibrating level {level + 1}/{max_level}")
            return func(self, level, separator)

        if self.n_jobs > 1:
            if _has_ray and not use_joblib:
                if not ray.is_initialized:
                    ray.init(
                        num_cpus=self.n_jobs,
                        local_mode=local_mode,
                        ignore_reinit_error=True,
                    )
                lcpl = ray.put(self)
                _parallel_fit = ray.remote(self._fit_calibrator)
                results = [
                    _parallel_fit.remote(lcpl, level, self.separator_)
                    for level in range(len(self.local_calibrators_))
                ]
                calibrators = ray.get(results)

            else:
                calibrators = Parallel(n_jobs=self.n_jobs)(
                    delayed(logging_wrapper)(
                        self._fit_calibrator,
                        level,
                        self.separator_,
                        len(self.local_calibrators_),
                    )
                    for level in range(len(self.local_calibrators_))
                )
        else:
            calibrators = [
                logging_wrapper(
                    self._fit_calibrator,
                    level,
                    self.separator_,
                    len(self.local_calibrators_),
                )
                for level in range(len(self.local_calibrators_))
            ]

        for level, calibrator in enumerate(calibrators):
            self.local_calibrators_[level] = calibrator

    @staticmethod
    def _fit_classifier(self, level, separator):
        classifier = self.local_classifiers_[level]
        if self.tmp_dir:
            md5 = hashlib.md5(str(level).encode("utf-8")).hexdigest()
            filename = f"{self.tmp_dir}/{md5}.sav"
            if exists(filename):
                try:
                    (_, classifier) = pickle.load(open(filename, "rb"))
                    self.logger_.info(
                        f"Loaded trained model for local classifier {level} from file {filename}"
                    )
                    return classifier
                except (pickle.UnpicklingError, EOFError):
                    self.logger_.error(f"Could not load model from file {filename}")
        self.logger_.info(f"Training local classifier {level}")
        X, y, sample_weight = self._remove_empty_leaves(
            separator, self.X_, self.y_[:, level], self.sample_weight_
        )

        unique_y = np.unique(y)
        if len(unique_y) == 1 and self.replace_classifiers:
            classifier = ConstantClassifier()
        if not self.bert:
            try:
                classifier.fit(X, y, sample_weight)
            except TypeError:
                classifier.fit(X, y)
        else:
            classifier.fit(X, y)
        self._save_tmp(level, classifier)
        return classifier

    @staticmethod
    def _fit_calibrator(self, level, separator):
        try:
            calibrator = self.local_calibrators_[level]
        except IndexError:
            self.logger_.info("no calibrator for " + "level: " + str(level))
            return None

        X, y, _ = self._remove_empty_leaves(
            separator, self.X_cal, self.y_cal[:, level], None
        )
        if len(y) == 0 or len(np.unique(y)) < 2:
            self.logger_.info(
                f"No calibration samples to fit calibrator for level: {str(level)}"
            )
            return None
        calibrator.fit(X, y)
        return calibrator

    @staticmethod
    def _remove_empty_leaves(separator, X, y, sample_weight):
        # Detect rows where leaves are not empty
        leaves = np.array([str(i).split(separator)[-1] for i in y])
        mask = leaves != ""
        X = X[mask]
        y = y[mask]
        sample_weight = sample_weight[mask] if sample_weight is not None else None
        return X, y, sample_weight

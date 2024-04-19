"""
Local classifier per node approach.

Numeric and string output labels are both handled.
"""

import hashlib
import pickle
from copy import deepcopy
from os.path import exists

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass import BinaryPolicy
from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass._calibration.Calibrator import _Calibrator

from hiclass.probability_combiner import (
    init_strings as probability_combiner_init_strings,
)

from hiclass._hiclass_utils import _normalize_probabilities


class LocalClassifierPerNode(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each node of the graph, except the root node.

    A local classifier per node is a local hierarchical classifier that fits one local binary classifier
    for each node of the class hierarchy, except for the root node.

    Examples
    --------
    >>> from hiclass import LocalClassifierPerNode
    >>> y = [['1', '1.1'], ['2', '2.1']]
    >>> X = [[1, 2], [3, 4]]
    >>> lcpn = LocalClassifierPerNode()
    >>> lcpn.fit(X, y)
    >>> lcpn.predict(X)
    array([['1', '1.1'],
       ['2', '2.1']])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        binary_policy: str = "siblings",
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
        calibration_method: str = None,
        return_all_probabilities: bool = False,
        probability_combiner: str = "geometric",
        tmp_dir: str = None,
    ):
        """
        Initialize a local classifier per node.

        Parameters
        ----------
        local_classifier : BaseEstimator, default=LogisticRegression
            The local_classifier used to create the collection of local classifiers. Needs to have fit, predict and
            clone methods.
        binary_policy : {"exclusive", "less_exclusive", "exclusive_siblings", "inclusive", "less_inclusive", "siblings"}, str, default="siblings"
            Specify the rule for defining positive and negative training examples, using one of the following options:

            - `exclusive`: Positive examples belong only to the class being considered. All classes are negative examples, except for the selected class;
            - `less_exclusive`: Positive examples belong only to the class being considered. All classes are negative examples, except for the selected class and its descendants;
            - `exclusive_siblings`: Positive examples belong only to the class being considered. All sibling classes are negative examples;
            - `inclusive`: Positive examples belong only to the class being considered and its descendants. All classes are negative examples, except for the selected class, its descendants and ancestors;
            - `less_inclusive`: Positive examples belong only to the class being considered and its descendants. All classes are negative examples, except for the selected class and its descendants;
            - `siblings`: Positive examples belong only to the class being considered and its descendants. All siblings and their descendant classes are negative examples.

            See :ref:`Training Policies` for more information about the different policies.
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
        probability_combiner: {"geometric", "arithmetic", "multiply"}, str, default="geometric"
            Specify the rule for combining probabilities over multiple levels:

            - `geometric`: Each levels probabilities are calculated by taking the geometric mean of itself and its predecessors;
            - `arithmetic`: Each levels probabilities are calculated by taking the arithmetic mean of itself and its predecessors;
            - `multiply`: Each levels probabilities are calculated by multiplying itself with its predecessors.
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
            classifier_abbreviation="LCPN",
            bert=bert,
            calibration_method=calibration_method,
            tmp_dir=tmp_dir,
        )
        self.binary_policy = binary_policy
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
        Fit a local classifier per node.

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

        # Initialize policy
        self.binary_policy_ = self._initialize_binary_policy(calibration=False)

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

        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)

        self.logger_.info("Predicting")

        for predecessor, successors in bfs:
            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
            else:
                mask = np.isin(y, predecessor).any(axis=1)
                subset_x = X[mask]
            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for i, successor in enumerate(successors):
                    successor_name = str(successor).split(self.separator_)[-1]
                    self.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]
                    probabilities[:, i] = classifier.predict_proba(subset_x)[
                        :, positive_index
                    ][:, 0]
                highest_probability = np.argmax(probabilities, axis=1)
                prediction = []
                for i in highest_probability:
                    prediction.append(successors[i])
                level = nx.shortest_path_length(
                    self.hierarchy_, self.root_, predecessor
                )
                prediction = np.array(prediction)
                y[mask, level] = prediction

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

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        if not self.calibration_method:
            self.logger_.info(
                "It is not recommended to use predict_proba() without calibration"
            )
        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)
        self.logger_.info("Predicting Probability")

        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)
        level_probability_list = []
        last_level = -1

        for predecessor, successors in bfs:
            level = nx.shortest_path_length(self.hierarchy_, self.root_, predecessor)
            level_dimension = self.max_level_dimensions_[level]

            if last_level != level:
                last_level = level
                cur_level_probabilities = np.zeros((X.shape[0], level_dimension))
                level_probability_list.append(cur_level_probabilities)

            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
            else:
                mask = np.isin(y, self.global_classes_[level - 1]).any(axis=1)
                subset_x = X[mask]

            if subset_x.shape[0] > 0:
                local_probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for i, successor in enumerate(successors):
                    self.logger_.info(
                        f"Predicting probabilities for node '{str(successor)}'"
                    )
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    # use classifier as a fallback if no calibrator is available
                    calibrator = (
                        self.hierarchy_.nodes[successor].get("calibrator", classifier)
                        or classifier
                    )
                    positive_index = np.where(calibrator.classes_ == 1)[0]
                    proba = calibrator.predict_proba(subset_x)[:, positive_index][:, 0]
                    local_probabilities[:, i] = proba
                    class_index = self.global_class_to_index_mapping_[level][
                        str(successor)
                    ]
                    level_probability_list[-1][mask, class_index] = proba

                highest_local_probability = np.argmax(local_probabilities, axis=1)
                path_prediction = []
                for i in highest_local_probability:
                    path_prediction.append(successors[i])
                path_prediction = np.array(path_prediction)

                y[mask, level] = path_prediction

        y = self._convert_to_1d(y)
        self._remove_separator(y)

        # normalize probabilities
        level_probability_list = _normalize_probabilities(level_probability_list)

        # combine probabilities horizontally
        level_probability_list = self._combine_and_reorder(level_probability_list)

        # combine probabilities vertically
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

    def _initialize_binary_policy(self, calibration=False):
        if isinstance(self.binary_policy, str):
            self.logger_.info(f"Initializing {self.binary_policy} binary policy")
            try:
                if calibration:
                    binary_policy_ = BinaryPolicy.IMPLEMENTED_POLICIES[
                        self.binary_policy.lower()
                    ](self.hierarchy_, self.X_cal, self.y_cal, None)
                else:
                    binary_policy_ = BinaryPolicy.IMPLEMENTED_POLICIES[
                        self.binary_policy.lower()
                    ](self.hierarchy_, self.X_, self.y_, self.sample_weight_)
                return binary_policy_
            except KeyError:
                self.logger_.error(
                    f"Policy {self.binary_policy} not implemented. Available policies are:\n"
                    + f"{list(BinaryPolicy.IMPLEMENTED_POLICIES.keys())}"
                )
                raise KeyError(f"Policy {self.binary_policy} not implemented.")
        else:
            self.logger_.error("Binary policy is not a string")
            raise ValueError(
                f"Binary policy type must str, not {type(self.binary_policy)}."
            )

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        local_classifiers = {}
        for node in self.hierarchy_.nodes:
            # Skip only root node
            if node != self.root_:
                local_classifiers[node] = {
                    "classifier": deepcopy(self.local_classifier_)
                }
        nx.set_node_attributes(self.hierarchy_, local_classifiers)

    def _initialize_local_calibrators(self):
        super()._initialize_local_calibrators()
        local_calibrators = {}
        for node in self.hierarchy_.nodes:
            # Skip only root node
            if node != self.root_:
                # get classifier from node
                local_classifier = self.hierarchy_.nodes[node]["classifier"]
                local_calibrators[node] = {
                    "calibrator": _Calibrator(
                        estimator=local_classifier, method=self.calibration_method
                    )
                }
        nx.set_node_attributes(self.hierarchy_, local_calibrators)

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        self._fit_node_classifier(nodes, local_mode, use_joblib)

    def _calibrate_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local calibrators")
        self.cal_binary_policy_ = self._initialize_binary_policy(calibration=True)
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        self._fit_node_calibrator(nodes, local_mode, use_joblib)

    @staticmethod
    def _fit_classifier(self, node):
        classifier = self.hierarchy_.nodes[node]["classifier"]
        if self.tmp_dir:
            md5 = hashlib.md5(node.encode("utf-8")).hexdigest()
            filename = f"{self.tmp_dir}/{md5}.sav"
            if exists(filename):
                (_, classifier) = pickle.load(open(filename, "rb"))
                self.logger_.info(
                    f"Loaded trained model for local classifier {node.split(self.separator_)[-1]} from file {filename}"
                )
                return classifier
        self.logger_.info(f"Training local classifier {node}")
        X, y, sample_weight = self.binary_policy_.get_binary_examples(node)
        unique_y = np.unique(y)
        if len(unique_y) == 1 and self.replace_classifiers:
            self.logger_.info("adding constant classifier")
            classifier = ConstantClassifier()
        if not self.bert:
            try:
                classifier.fit(X, y, sample_weight)
            except TypeError:
                classifier.fit(X, y)
        else:
            classifier.fit(X, y)
        self._save_tmp(node, classifier)
        return classifier

    @staticmethod
    def _fit_calibrator(self, node):
        try:
            calibrator = self.hierarchy_.nodes[node]["calibrator"]
        except KeyError:
            self.logger_.info("no calibrator for " + "node: " + str(node))
            return None
        X, y, _ = self.cal_binary_policy_.get_binary_examples(node)
        if len(y) == 0 or len(np.unique(y)) < 2:
            self.logger_.info(
                f"No calibration samples to fit calibrator for node: {str(node)}"
            )
            return None
        calibrator.fit(X, y)
        return calibrator

    def _clean_up(self):
        super()._clean_up()
        if hasattr(self, "binary_policy_"):
            del self.binary_policy_
        if hasattr(self, "cal_binary_policy_"):
            del self.cal_binary_policy_

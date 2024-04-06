"""
Local classifier per parent node approach.

Numeric and string output labels are both handled.
"""

from copy import deepcopy

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass._calibration.Calibrator import _Calibrator

from hiclass.probability_combiner import init_strings as probability_combiner_init_strings

class LocalClassifierPerParentNode(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each parent node of the graph.

    A local classifier per parent node is a local hierarchical classifier that fits one multi-class classifier
    for each parent node of the class hierarchy.

    Examples
    --------
    >>> from hiclass import LocalClassifierPerParentNode
    >>> y = [['1', '1.1'], ['2', '2.1']]
    >>> X = [[1, 2], [3, 4]]
    >>> lcppn = LocalClassifierPerParentNode()
    >>> lcppn.fit(X, y)
    >>> lcppn.predict(X)
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
        probability_combiner: str = "geometric"
    ):
        """
        Initialize a local classifier per parent node.

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
        calibration_method : {"ivap", "cvap", "platt", "isotonic"}, str, default=None
            If set, use the desired method to calibrate probabilities returned by predict_proba().
        return_all_probabilities : bool, default=False
            If True, return probabilities for all levels. Otherwise, return only probabilities for the last level.
        probability_combiner: {"geometric", "arithmetic", "multiply"}, str, default="geometric"
            Specify the rule for combining probabilities over multiple levels:

            - `geometric`: Each levels probabilities are calculated by taking the geometric mean of itself and its predecessors;
            - `arithmetic`: Each levels probabilities are calculated by taking the arithmetic mean of itself and its predecessors;
            - `multiply`: Each levels probabilities are calculated by multiplying itself with its predecessors.
        """
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPPN",
            bert=bert,
            calibration_method=calibration_method,
        )
        self.return_all_probabilities = return_all_probabilities
        self.probability_combiner = probability_combiner

        if self.probability_combiner and self.probability_combiner not in probability_combiner_init_strings:
            raise ValueError(f"probability_combiner must be one of {', '.join(probability_combiner_init_strings)} or None.")

    def fit(self, X, y, sample_weight=None):
        """
        Fit a local classifier per parent node.

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
        # Execute common methods necessary before fitting
        super()._pre_fit(X, y, sample_weight)

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
        classifier = self.hierarchy_.nodes[self.root_]["classifier"]
        y[:, 0] = classifier.predict(X).flatten()

        self._predict_remaining_levels(X, y)

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        if not self.calibration_method:
            self.logger_.info("It is not recommended to use predict_proba() without calibration")
        
        self.logger_.info("Predicting Probability")

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # Predict first level
        classifier = self.hierarchy_.nodes[self.root_]["classifier"]
        # use classifier as a fallback if no calibrator is available
        calibrator = self.hierarchy_.nodes[self.root_].get("calibrator", classifier) or classifier
        proba = calibrator.predict_proba(X)

        y[:, 0] = calibrator.classes_[np.argmax(proba, axis=1)]
        level_probability_list = [proba] + self._predict_proba_remaining_levels(X, y)
    
        level_probability_list = self._combine_and_reorder(level_probability_list)

        # combine probabilities
        if self.probability_combiner:
            probability_combiner_ = self._create_probability_combiner(self.probability_combiner)
            self.logger_.info(f"Combining probabilities using {type(probability_combiner_).__name__}")
            level_probability_list = probability_combiner_.combine(level_probability_list)
        
        return level_probability_list if self.return_all_probabilities else level_probability_list[-1]

    def _predict_proba_remaining_levels(self, X, y):
        level_probability_list = []
        for level in range(1, y.shape[1]):
            predecessors = set(y[:, level - 1])
            predecessors.discard("")
            level_dimension = self.max_level_dimensions_[level]
            cur_level_probabilities = np.zeros((X.shape[0], level_dimension))

            for predecessor in predecessors:
                mask = np.isin(y[:, level - 1], self.global_classes_[level-1])
                predecessor_x = X[mask]
                if predecessor_x.shape[0] > 0:
                    successors = list(self.hierarchy_.successors(predecessor))
                    if len(successors) > 0:
                        classifier = self.hierarchy_.nodes[predecessor]["classifier"]
                        # use classifier as a fallback if no calibrator is available
                        calibrator = self.hierarchy_.nodes[predecessor].get("calibrator", classifier) or classifier

                        proba = calibrator.predict_proba(predecessor_x)
                        y[mask, level] = calibrator.classes_[np.argmax(proba, axis=1)]

                        for successor in successors:
                            class_index = self.global_class_to_index_mapping_[level][str(successor)]

                            proba_index = np.where(calibrator.classes_ == successor)[0][0]
                            cur_level_probabilities[mask, class_index] = proba[:, proba_index]

            level_probability_list.append(cur_level_probabilities)
        
        # normalize probabilities
        level_probability_list = [
            np.nan_to_num(level_probabilities / level_probabilities.sum(axis=1, keepdims=True)) 
                for level_probabilities in level_probability_list
        ]

        return level_probability_list
        
    def _predict_remaining_levels(self, X, y):
        for level in range(1, y.shape[1]):
            predecessors = set(y[:, level - 1])
            predecessors.discard("")
            for predecessor in predecessors:
                mask = np.isin(y[:, level - 1], predecessor)
                predecessor_x = X[mask]
                if predecessor_x.shape[0] > 0:
                    successors = list(self.hierarchy_.successors(predecessor))
                    if len(successors) > 0:
                        classifier = self.hierarchy_.nodes[predecessor]["classifier"]
                        y[mask, level] = classifier.predict(predecessor_x).flatten()

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        local_classifiers = {}
        nodes = self._get_parents()
        for node in nodes:
            local_classifiers[node] = {"classifier": deepcopy(self.local_classifier_)}
        nx.set_node_attributes(self.hierarchy_, local_classifiers)

    def _initialize_local_calibrators(self):
        super()._initialize_local_calibrators()
        local_calibrators = {}
        nodes = self._get_parents()
        for node in nodes:
            local_classifier = self.hierarchy_.nodes[node]["classifier"]
            local_calibrators[node] = {
                "calibrator": _Calibrator(estimator=local_classifier, method=self.calibration_method)
            }
        nx.set_node_attributes(self.hierarchy_, local_calibrators)

    def _get_parents(self):
        nodes = []
        for node in self.hierarchy_.nodes:
            # Skip only leaf nodes
            successors = list(self.hierarchy_.successors(node))
            if len(successors) > 0:
                nodes.append(node)
        return nodes

    def _get_successors(self, node, calibration=False):
        successors = list(self.hierarchy_.successors(node))
        mask = np.isin(self.y_cal, successors).any(axis=1) if calibration else np.isin(self.y_, successors).any(axis=1)
        X = self.X_cal[mask] if calibration else self.X_[mask]
        y = []
        masked_labels = self.y_cal[mask] if calibration else self.y_[mask]
        for row in masked_labels:
            if node == self.root_:
                y.append(row[0])
                self.logger_.info(y)
            else:
                y.append(row[np.where(row == node)[0][0] + 1])
        y = np.array(y)
        sample_weight = None if calibration else (
            self.sample_weight_[mask] if self.sample_weight_ is not None else None
        )
        return X, y, sample_weight

    @staticmethod
    def _fit_classifier(self, node):
        classifier = self.hierarchy_.nodes[node]["classifier"]
        # get children examples
        X, y, sample_weight = self._get_successors(node)
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
        return classifier

    @staticmethod
    def _fit_calibrator(self, node):
        try:
            calibrator = self.hierarchy_.nodes[node]["calibrator"]
        except KeyError:
            self.logger_.info("no calibrator for " + "node: " + str(node))
            return None
        X, y, _ = self._get_successors(node, calibration=True)
        if len(y) == 0 or len(np.unique(y)) < 2:
            self.logger_.info(f"No calibration samples to fit calibrator for node: {str(node)}")
            return None
        calibrator.fit(X, y)
        return calibrator
        
    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = self._get_parents()
        self._fit_node_classifier(nodes, local_mode, use_joblib)
    
    def _calibrate_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local calibrators")
        nodes = self._get_parents()
        self._fit_node_calibrator(nodes, local_mode, use_joblib)

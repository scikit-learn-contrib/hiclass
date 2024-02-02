"""
Local classifier per parent node approach.

Numeric and string output labels are both handled.
"""

from copy import deepcopy
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.MultiLabelHierarchicalClassifier import (
    MultiLabelHierarchicalClassifier,
    make_leveled,
)


class MultiLabelLocalClassifierPerParentNode(
    BaseEstimator, MultiLabelHierarchicalClassifier
):
    """
    Assign local classifiers to each parent node of the graph.

    A local classifier per parent node is a local hierarchical classifier that fits one multi-class classifier
    for each parent node of the class hierarchy.

    Examples
    --------
    >>> from hiclass import MultiLabelLocalClassifierPerParentNode
    >>> y = [[['1', '1.1'], ['2', '2.1']]]
    >>> X = [[1, 2]]
    >>> mllcppn = MultiLabelLocalClassifierPerParentNode()
    >>> mllcppn.fit(X, y)
    >>> mllcppn.predict(X)
    array([[['1', '1.1'],
       ['2', '2.1']]])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        tolerance: float = None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        r"""
        Initialize a multi-label local classifier per parent node.

        Parameters
        ----------
        local_classifier : BaseEstimator, default=LogisticRegression
            The local_classifier used to create the collection of local classifiers. Needs to have fit, predict and
            clone methods.
        tolerance : float, default=None
            The tolerance used to determine multi-labels. If set to None, only the child class with highest probability is predicted.
            Otherwise, all child classes with :math:`probability >= max\_prob - tolerance` are predicted.
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
        """
        super().__init__(
            local_classifier=local_classifier,
            tolerance=tolerance,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPPN",
            bert=bert,
        )

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

    def predict(self, X, tolerance: float = None):
        r"""
        Predict classes for the given data.

        Hierarchical labels are returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        tolerance : float, default=None
            The tolerance used to determine multi-labels.
            If set to None, only the child class with highest probability is predicted.
            Overrides the tolerance set in the constructor.
            Otherwise, all child classes with :math:`probability >= max\_prob - tolerance` are predicted.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        # Check if fit has been called
        check_is_fitted(self)
        _tolerance = (tolerance if tolerance is not None else self.tolerance) or 0.0

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr")
        else:
            X = np.array(X)

        # TODO: Add threshold to stop prediction halfway if need be

        self.logger_.info("Predicting")

        y = [[] for _ in range(X.shape[0])]
        # Predict first level
        classifier = self.hierarchy_.nodes[self.root_]["classifier"]

        probabilities = classifier.predict_proba(X)
        for probs, ls in zip(probabilities, y):
            prediction = classifier.classes_[
                np.greater_equal(probs, np.max(probs) - _tolerance)
            ]
            for pred in prediction:
                ls.append([pred])

        y = self._predict_remaining_(X, y, _tolerance)
        y = make_leveled(y)
        y = np.array(y, dtype=self.dtype_)

        # TODO: this only needed for the sklearn check_estimator_sparse_data test to pass :(
        # However it also breaks a bunch of other tests
        # y = self._convert_to_1d(
        #   self._convert_to_2d(y)
        # )
        self._remove_separator(y)

        return y

    def _get_mask_and_indices(self, y, node):
        mask = np.zeros(len(y), dtype=bool)
        indicies = defaultdict(lambda: [])
        for i, multi_label in enumerate(y):
            for j, label in enumerate(multi_label):
                if label[-1] == node:
                    # if label[-1].split(self.separator_)[-1] == node:
                    mask[i] = True
                    indicies[i].append(j)
        return mask, indicies

    def _get_nodes_to_predict(self, y):
        last_predictions = set()
        for multi_label in y:
            for label in multi_label:
                last_predictions.add(label[-1])

        nodes_to_predict = []
        for node in last_predictions:
            if node in self.hierarchy_.nodes and self.hierarchy_.nodes[node].get(
                "classifier"
            ):
                nodes_to_predict.append(node)

        return nodes_to_predict

    def _predict_remaining_(self, X, y, tolerance):
        nodes_to_predict = self._get_nodes_to_predict(y)
        while nodes_to_predict:
            for node in nodes_to_predict:
                classifier = self.hierarchy_.nodes[node]["classifier"]
                mask, indices = self._get_mask_and_indices(y, node)
                subset_x = X[mask]
                probabilities = classifier.predict_proba(subset_x)
                for probs, (i, ls) in zip(probabilities, indices.items()):
                    prediction = classifier.classes_[
                        np.greater_equal(probs, np.max(probs) - tolerance)
                    ]
                    for j in ls:
                        y[i][j].append(prediction[0])
                        for pred in prediction[1:]:
                            _old_y = y[i][j][:-1].copy()
                            y[i].insert(j + 1, _old_y + [pred])
            nodes_to_predict = self._get_nodes_to_predict(y)
        return y

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        local_classifiers = {}
        nodes = self._get_parents()
        for node in nodes:
            local_classifiers[node] = {"classifier": deepcopy(self.local_classifier_)}
        nx.set_node_attributes(self.hierarchy_, local_classifiers)

    def _get_parents(self):
        nodes = []
        for node in self.hierarchy_.nodes:
            # Skip only leaf nodes
            successors = list(self.hierarchy_.successors(node))
            if len(successors) > 0:
                nodes.append(node)
        return nodes

    def _get_successors(self, node):
        successors = list(self.hierarchy_.successors(node))
        mask = np.isin(self.y_, successors).any(axis=(2, 1))
        y = []
        if isinstance(self.X_, csr_matrix):
            X = csr_matrix((0, self.X_.shape[1]), dtype=self.X_.dtype)
        else:
            X = []
        sample_weight = [] if self.sample_weight_ is not None else None
        for i in range(self.y_.shape[0]):
            if mask[i]:
                row = self.y_[i]
                labels = np.unique(
                    row[np.isin(row, successors)]
                )  # We do not want to double count the same row, e.g [["a", "b"], ["a", "c"]] should only count once for the root classifier with y label "a"
                y.extend(labels)
                for _ in range(labels.shape[0]):
                    if isinstance(self.X_, csr_matrix):
                        X = vstack([X, self.X_[i]])
                    else:
                        X.append(self.X_[i])
                    if self.sample_weight_ is not None:
                        sample_weight.append(self.sample_weight_[i])
        y = np.array(y)
        if isinstance(self.X_, np.ndarray):
            X = np.array(X)
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

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = self._get_parents()
        self._fit_node_classifier(nodes, local_mode, use_joblib)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimator_sparse_data": "Multi-label multi-output prediction format is not support in sklearn"
            },
        }

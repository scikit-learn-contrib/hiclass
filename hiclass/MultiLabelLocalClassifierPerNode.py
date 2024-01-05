"""
Local classifier per node approach.

Numeric and string output labels are both handled.
"""
from copy import deepcopy

import functools
import networkx as nx
import numpy as np

from hiclass import BinaryPolicy
from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.MultiLabelHierarchicalClassifier import (
    MultiLabelHierarchicalClassifier,
    make_leveled,
)

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# monkeypatching check_array to accept 3 dimensional arrays
import sklearn.utils.validation

# TODO: Move to MultiLabelHierarchicalClassifier (Parent Class)
sklearn.utils.validation.check_array = functools.partial(
    sklearn.utils.validation.check_array, allow_nd=True
)


class MultiLabelLocalClassifierPerNode(BaseEstimator, MultiLabelHierarchicalClassifier):
    """
    Assign local classifiers to each node of the graph, except the root node.

    A local classifier per node is a local hierarchical classifier that fits one local binary classifier
    for each node of the class hierarchy, except for the root node.

    Examples
    --------
    >>> from hiclass import MultiLabelLocalClassifierPerNode.py
    >>> y = [[['1', '1.1'], ['', '']], [['1', '1.1'], ['2', '2.1']]]
    >>> X = [[1, 2], [3, 4]]
    >>> lcpn = MultiLabelLocalClassifierPerNode.py()
    >>> lcpn.fit(X, y)
    >>> lcpn.predict(X)
    array([[['1', '1.1']],
       [['2', '2.1']]])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        binary_policy: str = "siblings",
        tolerance: float = None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        r"""
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
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPN",
            bert=bert,
        )
        self.binary_policy = binary_policy
        self.tolerance = tolerance

    def fit(self, X, y, sample_weight=None):
        """
        Fit a local classifier per node.

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

        # Initialize policy
        self._initialize_binary_policy()

        # Fit local classifiers in DAG
        super().fit(X, y)

        # TODO: Store the classes seen during fit

        # TODO: Add function to allow user to change local classifier

        # TODO: Add parameter to receive hierarchy as parameter in constructor

        # Return the classifier
        return self

    def predict(self, X, tolerance: float = None) -> np.ndarray:
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
            X = sklearn.utils.validation.check_array(
                X, accept_sparse="csr"
            )  # TODO: Decide allow_nd True or False
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = [[[]] for _ in range(X.shape[0])]

        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)

        self.logger_.info("Predicting")

        for predecessor, successors in bfs:
            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
                y_row_indices = [[i, [0]] for i in range(X.shape[0])]
            else:
                # get indices of rows that have the predecessor
                y_row_indices = []
                for row in range(X.shape[0]):
                    # create list of indices
                    _t = [z for z, l in enumerate(y[row]) if l[-1] == predecessor]

                    # y_row_indices is a list of lists, each list contains the index of the row and a list of column indices
                    y_row_indices.append([row, _t])

                # Filter
                mask = [True if l[1] else False for l in y_row_indices]
                y_row_indices = [l for l in y_row_indices if l[1]]
                subset_x = X[mask]

            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for row, successor in enumerate(successors):
                    successor_name = str(successor).split(self.separator_)[-1]
                    self.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]
                    probabilities[:, row] = classifier.predict_proba(subset_x)[
                        :, positive_index
                    ][:, 0]

                # get indices of probabilities that are within tolerance of max

                highest_probabilities = np.max(probabilities, axis=1).reshape(-1, 1)
                indices_probabilities_within_tolerance = np.argwhere(
                    np.greater_equal(probabilities, highest_probabilities - _tolerance)
                )

                prediction = [[] for _ in range(subset_x.shape[0])]
                for row, column in indices_probabilities_within_tolerance:
                    prediction[row].append(successors[column])

                k = 0  # index of prediction
                for row, col_list in y_row_indices:
                    for j in col_list:
                        if not prediction[k]:
                            y[row][j].append("")
                        else:
                            for pi, _suc in enumerate(prediction[k]):
                                if pi == 0:
                                    y[row][j].append(_suc)
                                else:
                                    # in case of mulitple predictions, copy the previous prediction up to (but not including) the last prediction and add the new one
                                    _old_y = y[row][j][:-1].copy()
                                    y[row].insert(j + 1, _old_y + [_suc])
                    k += 1

        y = make_leveled(y)
        self._remove_separator(y)
        y = np.array(y, dtype=self.dtype_)

        return y

    def _initialize_binary_policy(self):
        if isinstance(self.binary_policy, str):
            self.logger_.info(f"Initializing {self.binary_policy} binary policy")
            try:
                self.binary_policy_ = BinaryPolicy.IMPLEMENTED_POLICIES[
                    self.binary_policy.lower()
                ](self.hierarchy_, self.X_, self.y_, self.sample_weight_)
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

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        self._fit_node_classifier(nodes, local_mode, use_joblib)

    @staticmethod
    def _fit_classifier(self, node):
        classifier = self.hierarchy_.nodes[node]["classifier"]
        X, y, sample_weight = self.binary_policy_.get_binary_examples(node)
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

    def _clean_up(self):
        super()._clean_up()
        del self.binary_policy_

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimator_sparse_data": "Multi-label multi-output prediction format is not support in sklearn"
            },
        }

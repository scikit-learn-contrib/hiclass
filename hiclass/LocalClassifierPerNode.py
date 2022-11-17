"""
Local classifier per node approach.

Numeric and string output labels are both handled.
"""
from copy import deepcopy

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass import BinaryPolicy
from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier

from collections import defaultdict


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

            # TODO : Phrasing?
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
        """
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPN",
        )
        self.binary_policy = binary_policy

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

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # TODO: Add threshold to stop prediction halfway if need be

        self.logger_.info("Predicting")

        for level in range(y.shape[1]):
            
            predecessors = set(y[:, level - 1]) if level >= 1 else set([self.root_]) # in case of level 0 the predecessor is the root node
            predecessors.discard("")

            for predecessor in predecessors:
                mask = np.isin(y[:, level - 1], predecessor) if level >= 1 else np.ones(y.shape[0], dtype=bool)
                predecessor_x = X[mask]
                
                if predecessor_x.shape[0] > 0:
                    successors = list(self.hierarchy_.successors(predecessor))
                    if len(successors) > 0:
                        probabilities = np.zeros((predecessor_x.shape[0], len(successors)))
                        for i,successor in enumerate(successors):
                            classifier = self.hierarchy_.nodes[successor]["classifier"]
                            positive_index = np.where(classifier.classes_ == 1)[0]
                            probabilities[:,i] = classifier.predict_proba(predecessor_x)[:, positive_index][:,0]
                        
                        highest_probability_index = np.argmax(probabilities, axis=1)

                        predictions = np.array([successors[i] for i in highest_probability_index])
                        y[mask, level] = predictions
                        
        y = self._convert_to_1d(y)
        
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
        classifier.fit(X, y, sample_weight)
        return classifier

    def _clean_up(self):
        super()._clean_up()
        del self.binary_policy_

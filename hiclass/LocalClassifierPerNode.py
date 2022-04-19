"""
Local classifier per node approach.

Numeric and string output labels are both handled.
"""
from copy import deepcopy

import networkx as nx
import numpy as np
import ray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass import BinaryPolicy
from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier


@ray.remote
def _parallel_fit(lcpn, node):
    classifier = lcpn.hierarchy_.nodes[node]["classifier"]
    X, y = lcpn.binary_policy_.get_binary_examples(node)
    unique_y = np.unique(y)
    if len(unique_y) == 1 and lcpn.replace_classifiers:
        classifier = ConstantClassifier()
    classifier.fit(X, y)
    return classifier


class LocalClassifierPerNode(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each node of the graph, except the root node.

    A local classifier per node is a local hierarchical classifier that fits one local binary classifier
    for each node of the class hierarchy, except for the root node.
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
        binary_policy : str, default="siblings"
            Rules for defining positive and negative training examples.
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
            classifier_abbreviation="LCPN",
        )
        self.binary_policy = binary_policy

    def fit(self, X, y):
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

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Execute common methods necessary before fitting
        super()._pre_fit(X, y)

        # Initialize policy
        self._initialize_binary_policy()

        # Fit local classifiers in DAG
        super().fit(X, y)

        # TODO: Store the classes seen during fit

        # TODO: Add function to allow user to change local classifier

        # TODO: Add parameter to receive hierarchy as parameter in constructor

        # TODO: Add support to empty labels in some levels

        # Delete unnecessary variables
        self._clean_up()

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
                if prediction.ndim == 2 and prediction.shape[1] == 1:
                    prediction = prediction.flatten()
                y[mask, level] = prediction

        # Convert back to 1D if there is only 1 column to pass all sklearn's checks
        if self.max_levels_ == 1:
            y = y.flatten()

        # Remove separator from predictions
        if y.ndim == 2:
            for i in range(y.shape[0]):
                for j in range(1, y.shape[1]):
                    y[i, j] = y[i, j].split(self.separator_)[-1]

        return y

    def _initialize_binary_policy(self):
        if isinstance(self.binary_policy, str):
            self.logger_.info(f"Initializing {self.binary_policy} binary policy")
            try:
                self.binary_policy_ = BinaryPolicy.IMPLEMENTED_POLICIES[
                    self.binary_policy.lower()
                ](self.hierarchy_, self.X_, self.y_)
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

    def _fit_digraph_parallel(self, local_mode: bool = False):
        self.logger_.info("Fitting local classifiers")
        ray.init(num_cpus=self.n_jobs, local_mode=local_mode, ignore_reinit_error=True)
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        lcpn = ray.put(self)
        results = [_parallel_fit.remote(lcpn, node) for node in nodes]
        classifiers = ray.get(results)
        for classifier, node in zip(classifiers, nodes):
            self.hierarchy_.nodes[node]["classifier"] = classifier

    def _fit_digraph(self):
        self.logger_.info("Fitting local classifiers")
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        for index, node in enumerate(nodes):
            node_name = str(node).split(self.separator_)[-1]
            self.logger_.info(
                f"Fitting local classifier for node '{node_name}' ({index + 1}/{len(nodes)})"
            )
            classifier = self.hierarchy_.nodes[node]["classifier"]
            X, y = self.binary_policy_.get_binary_examples(node)
            unique_y = np.unique(y)
            if len(unique_y) == 1 and self.replace_classifiers:
                node_name = str(node).split(self.separator_)[-1]
                self.logger_.warning(
                    f"Fitting ConstantClassifier for node '{node_name}'"
                )
                self.hierarchy_.nodes[node]["classifier"] = ConstantClassifier()
                classifier = self.hierarchy_.nodes[node]["classifier"]
            classifier.fit(X, y)

    def _clean_up(self):
        self.logger_.info("Cleaning up variables that can take a lot of disk space")
        del self.X_
        del self.y_
        del self.binary_policy_

"""
Local classifier per parent node approach.

Numeric and string output labels are both handled.
"""

import logging
import networkx as nx
import numpy as np
import ray
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier


@ray.remote
def _parallel_fit(lcppn, node):
    classifier = lcppn.hierarchy_.nodes[node]["classifier"]
    # get children examples
    X, y = lcppn._get_successors(node)
    unique_y = np.unique(y)
    if len(unique_y) == 1 and lcppn.replace_classifiers:
        classifier = ConstantClassifier()
    classifier.fit(X, y)
    return classifier


class LocalClassifierPerParentNode(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each parent node of the graph.

    A local classifier per parent node is a local hierarchical classifier that fits one multi-class classifier
    for each parent node of the class hierarchy.
    """

    def fit(self, X, y):
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

        # Avoids creating more columns in prediction if edges are a->b and b->c,
        # which would generate the prediction a->b->c
        self._disambiguate()

        # Create DAG from self.y_ and store to self.hierarchy_
        self._create_digraph()

        # If user passes edge_list, then export
        # DAG to CSV file to visualize with Gephi
        self._export_digraph()

        # Assert that graph is directed acyclic
        self._assert_digraph_is_dag()

        # If y is 1D, convert to 2D for binary policies
        self._convert_1d_y_to_2d()

        # Detect root(s) and add artificial root to DAG
        self._add_artificial_root()

        # Initialize local classifiers in DAG
        self._initialize_local_classifiers()

        # Fit local classifiers in DAG
        if self.n_jobs > 1:
            self._fit_digraph_parallel()
        else:
            self._fit_digraph()

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
                classifier = self.hierarchy_.nodes[predecessor]["classifier"]
                prediction = classifier.predict(subset_x)
                level = nx.shortest_path_length(
                    self.hierarchy_, self.root_, predecessor
                )
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

    def _create_logger(self):
        # Create logger
        self.logger_ = logging.getLogger("LCPPN")
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

    def _disambiguate(self):
        self.separator_ = "::HiClass::Separator::"
        if self.y_.ndim == 2:
            new_y = []
            for i in range(self.y_.shape[0]):
                row = [self.y_[i, 0]]
                for j in range(1, self.y_.shape[1]):
                    row.append(str(row[-1]) + self.separator_ + str(self.y_[i, j]))
                new_y.append(np.asarray(row, dtype=np.str_))
            self.y_ = np.array(new_y)

    def _create_digraph(self):
        # Create DiGraph
        self.hierarchy_ = nx.DiGraph()

        # Save dtype of y_
        self.dtype_ = self.y_.dtype

        # 1D disguised as 2D
        if self.y_.ndim == 2 and self.y_.shape[1] == 1:
            self.logger_.info("Converting y to 1D")
            self.y_ = self.y_.flatten()

        # Check dimension of labels
        if self.y_.ndim == 2:
            # 2D labels
            # Create max_levels variable
            self.max_levels_ = self.y_.shape[1]
            rows, columns = self.y_.shape
            self.logger_.info(f"Creating digraph from {rows} 2D labels")
            for row in range(rows):
                for column in range(columns - 1):
                    self.hierarchy_.add_edge(
                        self.y_[row, column], self.y_[row, column + 1]
                    )

        elif self.y_.ndim == 1:
            # 1D labels
            # Create max_levels_ variable
            self.max_levels_ = 1
            self.logger_.info(f"Creating digraph from {self.y_.size} 1D labels")
            for label in self.y_:
                self.hierarchy_.add_node(label)

        else:
            # Unsuported dimension
            self.logger_.error(f"y with {self.y_.ndim} dimensions detected")
            raise ValueError(
                f"Creating graph from y with {self.y_.ndim} dimensions is not supported"
            )

    def _convert_1d_y_to_2d(self):
        # This conversion is necessary for the binary policies
        if self.y_.ndim == 1:
            self.y_ = np.reshape(self.y_, (-1, 1))

    def _export_digraph(self):
        # Check if edge_list is set
        if self.edge_list:
            # Add quotes to all nodes in case the text has commas
            mapping = {}
            for node in self.hierarchy_:
                mapping[node] = '"{}"'.format(node.split(self.separator_)[-1])
            hierarchy = nx.relabel_nodes(self.hierarchy_, mapping, copy=True)
            # Export DAG to CSV file
            self.logger_.info(f"Writing edge list to file {self.edge_list}")
            nx.write_edgelist(hierarchy, self.edge_list, delimiter=",")

    def _assert_digraph_is_dag(self):
        # Assert that graph is directed acyclic
        if not nx.is_directed_acyclic_graph(self.hierarchy_):
            self.logger_.error("Cycle detected in graph")
            raise ValueError("Graph is not directed acyclic")

    def _add_artificial_root(self):
        # Detect root(s)
        roots = [
            node for node, in_degree in self.hierarchy_.in_degree() if in_degree == 0
        ]
        self.logger_.info(f"Detected {len(roots)} roots")

        # Add artificial root as predecessor to root(s) detected
        self.root_ = "hiclass::root"
        for old_root in roots:
            self.hierarchy_.add_edge(self.root_, old_root)

    def _initialize_local_classifiers(self):
        # Create a deep copy of the local classifier specified
        # for each node in the hierarchy and save to attribute "classifier"
        self.logger_.info("Initializing local classifiers")
        local_classifiers = {}
        if self.local_classifier is None:
            self.local_classifier_ = LogisticRegression()
        else:
            self.local_classifier_ = self.local_classifier
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
        mask = np.isin(self.y_, successors).any(axis=1)
        X = self.X_[mask]
        y = []
        for row in self.y_[mask]:
            if node == self.root_:
                y.append(row[0])
            else:
                y.append(row[np.where(row == node)[0][0] + 1])
        y = np.array(y)
        return X, y

    def _fit_digraph_parallel(self, local_mode: bool = False):
        self.logger_.info("Fitting local classifiers")
        ray.init(num_cpus=self.n_jobs, local_mode=local_mode, ignore_reinit_error=True)
        nodes = self._get_parents()
        lcppn = ray.put(self)
        results = [_parallel_fit.remote(lcppn, node) for node in nodes]
        classifiers = ray.get(results)
        for classifier, node in zip(classifiers, nodes):
            self.hierarchy_.nodes[node]["classifier"] = classifier

    def _fit_digraph(self):
        self.logger_.info("Fitting local classifiers")
        nodes = self._get_parents()
        for index, node in enumerate(nodes):
            node_name = str(node).split(self.separator_)[-1]
            self.logger_.info(
                f"Fitting local classifier for node '{node_name}' ({index + 1}/{len(nodes)})"
            )
            classifier = self.hierarchy_.nodes[node]["classifier"]
            # get children examples
            X, y = self._get_successors(node)
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

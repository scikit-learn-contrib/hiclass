"""
Local classifier per parent node approach.

Numeric and string output labels are both handled.
"""

import hashlib
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from os.path import exists
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted

from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier


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
        tmp_dir: str = None,
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
            classifier_abbreviation="LCPPN",
            bert=bert,
            tmp_dir=tmp_dir,
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
        sample_weight = (
            self.sample_weight_[mask] if self.sample_weight_ is not None else None
        )
        return X, y, sample_weight

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
        # get children examples
        X, y, sample_weight = self._get_successors(node)
        unique_y = np.unique(y)
        if len(unique_y) == 1 and self.replace_classifiers:
            classifier = ConstantClassifier()
        if not self.bert:
            try:
                label_encoder = LabelEncoder()
                label_encoder.fit(y)
                y = label_encoder.transform(y)
                classifier.fit(X, y, sample_weight)
            except TypeError:
                classifier.fit(X, y)
        else:
            classifier.fit(X, y)
        self._save_tmp(node, classifier)
        return classifier

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = self._get_parents()
        self._fit_node_classifier(nodes, local_mode, use_joblib)

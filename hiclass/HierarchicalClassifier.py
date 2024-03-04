"""Shared code for all classifiers."""
import abc
import logging

import networkx as nx
import numpy as np
import scipy
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted

try:
    import ray
except ImportError:
    _has_ray = False
else:
    _has_ray = True


def make_leveled(y):
    """
    Add empty cells if columns' length differs.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_levels)
        The target values, i.e., hierarchical class labels for classification.

    Returns
    -------
    leveled_y : array-like of shape (n_samples, n_levels)
        The leveled target values, i.e., hierarchical class labels for classification.

    Notes
    -----
    If rows are not iterable, returns the current y without modifications.

    Examples
    --------
    >>> from hiclass.HierarchicalClassifier import make_leveled
    >>> y = [['a'], ['b', 'c']]
    >>> make_leveled(y)
    array([['a', ''],
       ['b', 'c']])
    """
    try:
        depth = max([len(row) for row in y])
    except TypeError:
        return y
    y = np.array(y, dtype=object)
    leveled_y = [[i for i in row] + [""] * (depth - len(row)) for row in y]
    return np.array(leveled_y)


class HierarchicalClassifier(abc.ABC):
    """Abstract class for the local hierarchical classifiers.

    Offers mostly utility methods and common data initialization.
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
        classifier_abbreviation: str = "",
        calibration_method: str = None,
    ):
        """
        Initialize a local hierarchical classifier.

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
        classifier_abbreviation : str, default=""
            The abbreviation of the local hierarchical classifier to be displayed during logging.
        calibration_method : {"ivap", "cvap", "platt", "isotonic"}, str, default=None
            If set, use the desired method to calibrate probabilities returned by predict_proba().
        """
        self.local_classifier = local_classifier
        self.verbose = verbose
        self.edge_list = edge_list
        self.replace_classifiers = replace_classifiers
        self.n_jobs = n_jobs
        self.bert = bert
        self.classifier_abbreviation = classifier_abbreviation
        self.calibration_method = calibration_method

    def fit(self, X, y, sample_weight=None):
        """
        Fit a local hierarchical classifier.

        Needs to be subclassed by other classifiers as it only offers common methods.

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
        # Fit local classifiers in DAG
        self._fit_digraph()

        # Delete unnecessary variables
        if not self.calibration_method == "cvap":
            self._clean_up()

    def _pre_fit(self, X, y, sample_weight):
        # Check that X and y have correct shape
        # and convert them to np.ndarray if need be

        if not self.bert:
            self.X_, self.y_ = self._validate_data(
                X, y, multi_output=True, accept_sparse="csr", allow_nd=True
            )
        else:
            self.X_ = np.array(X)
            self.y_ = np.array(y)

        if sample_weight is not None:
            self.sample_weight_ = _check_sample_weight(sample_weight, X)
        else:
            self.sample_weight_ = None

        self.y_ = make_leveled(self.y_)
        # Avoids creating more columns in prediction if edges are a->b and b->c,
        # which would generate the prediction a->b->c
        self.y_ = self._disambiguate(self.y_)

        if self.y_.ndim > 1:
            self.max_level_dimensions_ = np.array([len(np.unique(self.y_[:, level])) for level in range(self.y_.shape[1])])
            self.classes_ = [np.unique(self.y_[:, level]).astype("str") for level in range(self.y_.shape[1])]
            self.class_to_index_mapping_ = [{self.classes_[level][index]: index for index in range(len(self.classes_[level]))} for level in range(self.y_.shape[1])]
        else:
            self.max_level_dimensions_ = np.array([len(np.unique(self.y_))])
            self.classes_ = [np.unique(self.y_).astype("str")]
            self.class_to_index_mapping_ = [{self.classes_[0][index] : index for index in range(len(self.classes_[0]))}]

        # Create and configure logger
        self._create_logger()

        # Avoids creating more columns in prediction if edges are a->b and b->c,
        # which would generate the prediction a->b->c
        #self.y_ = self._disambiguate(self.y_)

        # Create DAG from self.y_ and store to self.hierarchy_
        self._create_digraph()

        # If user passes edge_list, then export
        # DAG to CSV file to visualize with Gephi
        self._export_digraph()

        # Assert that graph is directed acyclic
        self._assert_digraph_is_dag()

        # If y is 1D, convert to 2D for binary policies
        self.y_ = self._convert_1d_y_to_2d(self.y_)

        # Detect root(s) and add artificial root to DAG
        self._add_artificial_root()

        # Initialize local classifiers in DAG
        self._initialize_local_classifiers()

    def calibrate(self, X, y):
        """
        Fit a local calibrator per node.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The calibration input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples, n_levels)
            The target values, i.e., hierarchical class labels for classification.

        Returns
        -------
        self : object
            Calibrated estimator.
        """
        if not self.calibration_method:
            raise ValueError("No calibration method specified")

        # check if fitted
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)
        
        if self.calibration_method == "cvap":
            # combine train and calibration dataset for cross validation
            if isinstance(self.X_, scipy.sparse._csr.csr_matrix):
                self.X_cross_val = scipy.sparse.vstack([self.X_, X])
                self.logger_.info(f"CV Dataset X: {str(type(self.X_cross_val))} {str(self.X_cross_val.shape)}")
            else:
                self.X_cross_val = np.hstack([self.X_, X])
                self.logger_.info(f"CV Dataset X: {str(type(self.X_cross_val))} {str(self.X_cross_val.shape)}")
            self.y_cross_val = np.vstack([self.y_, y])
            self.y_cross_val = make_leveled(self.y_cross_val)
            self.y_cross_val = self._disambiguate(self.y_cross_val)
            self.y_cross_val = self._convert_1d_y_to_2d(self.y_cross_val)
        else:
            self.X_cal = X
            self.y_cal = y

            self.y_cal = make_leveled(self.y_cal)
            self.y_cal = self._disambiguate(self.y_cal)
            self.y_cal = self._convert_1d_y_to_2d(self.y_cal)

        self.cal_binary_policy_ = self._initialize_binary_policy(calibration=True)
        self.logger_.info("Calibrating")

        # Create a calibrator for each local classifier
        self._initialize_local_calibrators()
        self._calibrate_digraph()

        self._clean_up()
        return self

    def _predict_ood():
        pass

    def _create_logger(self):
        # Create logger
        self.logger_ = logging.getLogger(self.classifier_abbreviation)
        self.logger_.setLevel(self.verbose)

        # Create console handler and set verbose level
        if not self.logger_.hasHandlers():
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

    def _disambiguate(self, y):
        self.separator_ = "::HiClass::Separator::"
        if y.ndim == 2:
            new_y = []
            for i in range(y.shape[0]):
                row = [str(y[i, 0])]
                for j in range(1, y.shape[1]):
                    parent = str(row[-1])
                    child = str(y[i, j])
                    row.append(parent + self.separator_ + child)
                new_y.append(np.asarray(row, dtype=np.str_))
            return np.array(new_y)
        return y

    def _create_digraph(self):
        # Create DiGraph
        self.hierarchy_ = nx.DiGraph()

        # Save dtype of y_
        self.dtype_ = self.y_.dtype

        self._create_digraph_1d()

        self._create_digraph_2d()

        if self.y_.ndim > 2:
            # Unsuported dimension
            self.logger_.error(f"y with {self.y_.ndim} dimensions detected")
            raise ValueError(
                f"Creating graph from y with {self.y_.ndim} dimensions is not supported"
            )

    def _create_digraph_1d(self):
        # Flatten 1D disguised as 2D
        if self.y_.ndim == 2 and self.y_.shape[1] == 1:
            self.logger_.info("Converting y to 1D")
            self.y_ = self.y_.flatten()
        if self.y_.ndim == 1:
            # Create max_levels_ variable
            self.max_levels_ = 1
            self.logger_.info(f"Creating digraph from {self.y_.size} 1D labels")
            for label in self.y_:
                self.hierarchy_.add_node(label)

    def _create_digraph_2d(self):
        if self.y_.ndim == 2:
            # Create max_levels variable
            self.max_levels_ = self.y_.shape[1]
            rows, columns = self.y_.shape
            self.logger_.info(f"Creating digraph from {rows} 2D labels")
            for row in range(rows):
                for column in range(columns - 1):
                    parent = self.y_[row, column].split(self.separator_)[-1]
                    child = self.y_[row, column + 1].split(self.separator_)[-1]
                    if parent != "" and child != "":
                        # Only add edge if both parent and child are not empty
                        self.hierarchy_.add_edge(
                            self.y_[row, column], self.y_[row, column + 1]
                        )
                    elif parent != "" and column == 0:
                        self.hierarchy_.add_node(parent)

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

    def _convert_1d_y_to_2d(self, y):
        return np.reshape(y, (-1, 1)) if y.ndim == 1 else y

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
        if self.local_classifier is None:
            self.local_classifier_ = LogisticRegression()
        else:
            self.local_classifier_ = self.local_classifier

        @abc.abstractmethod
        def _initialize_local_calibrators(self):
            raise NotImplementedError("Method should be implemented in the LCPN and LCPPN")

    def _convert_to_1d(self, y):
        # Convert predictions to 1D if there is only 1 column
        if self.max_levels_ == 1:
            y = y.flatten()
        return y

    def _remove_separator(self, y):
        # Remove separator from predictions
        if y.ndim == 2:
            for i in range(y.shape[0]):
                for j in range(1, y.shape[1]):
                    y[i, j] = y[i, j].split(self.separator_)[-1]

    def _fit_node_classifier(
        self, nodes, local_mode: bool = False, use_joblib: bool = False
    ):
        if self.n_jobs > 1:
            if _has_ray and not use_joblib:
                ray.init(
                    num_cpus=self.n_jobs,
                    local_mode=local_mode,
                    ignore_reinit_error=True,
                )
                lcppn = ray.put(self)
                _parallel_fit = ray.remote(self._fit_classifier)
                results = [_parallel_fit.remote(lcppn, node) for node in nodes]
                classifiers = ray.get(results)
            else:
                classifiers = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fit_classifier)(self, node) for node in nodes
                )

        else:
            classifiers = []
            for idx, node in enumerate(nodes):
                self.logger_.info(f"fitting node {idx+1}/{len(nodes)}: {str(node)}")
                classifiers.append(self._fit_classifier(self, node))
            #classifiers = [self._fit_classifier(self, node) for node in nodes]
        for classifier, node in zip(classifiers, nodes):
            self.hierarchy_.nodes[node]["classifier"] = classifier

    def _fit_node_calibrator(self, nodes, local_mode: bool = False, use_joblib: bool = False):
        # TODO: add support for multithreading
        #calibrators = [self._fit_calibrator(self, node) for node in nodes]
        calibrators = []
        for idx, node in enumerate(nodes):
            self.logger_.info(f"calibrating node {idx+1}/{len(nodes)}: {str(node)}")
            calibrators.append(self._fit_calibrator(self, node))

        for calibrator, node in zip(calibrators, nodes):
            if calibrator:
                self.hierarchy_.nodes[node]["calibrator"] = calibrator

    @staticmethod
    def _fit_classifier(self, node):
        raise NotImplementedError("Method should be implemented in the LCPN and LCPPN")

    @staticmethod
    def _fit_calibrator(self, node):
        raise NotImplementedError("Method should be implemented in the LCPN and LCPPN")

    def _clean_up(self):
        self.logger_.info("Cleaning up variables that can take a lot of disk space")
        if hasattr(self, 'X_'):
            del self.X_
        if hasattr(self, 'y_'):
            del self.y_
        if hasattr(self, 'sample_weight') and self.sample_weight_ is not None:
            del self.sample_weight_
        if hasattr(self, 'X_cal'):
            del self.X_cal
        if hasattr(self, 'y_cal'):
            del self.y_cal
        if hasattr(self, 'y_cross_val'):
            del self.y_cross_val
        if hasattr(self, 'X_cross_val'):
            del self.X_cross_val

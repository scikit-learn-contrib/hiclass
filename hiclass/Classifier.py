"""Shared code for all classifiers."""
import abc
import logging
import multiprocessing
import sys
from itertools import starmap
from typing import Any, Callable, Iterator, List, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier

import hiclass.data as data_utils
from hiclass import Policies
from hiclass.data import (
    CATEGORICAL_ROOT_NODE,
    NODE_TYPE,
    SEPARATOR,
    minimal_per_node_depth,
)


class Classifier(abc.ABC):
    """Abstract class for the local hierarchical classifiers.

    Offers mostly utility methods and common data initialization. This includes getting and setting classifiers on nodes
    as well as calculating and accepting the class hierarchy.
    """

    def __init__(
        self,
        n_jobs: int = 1,
        verbose: int = 0,
        local_classifier: Any = BaseEstimator(),
        hierarchy: Optional[nx.DiGraph] = None,
        unique_taxonomy: bool = True,
    ):
        """
        Initialize a classifier.

        Parameters
        ----------
        n_jobs : int, default=1
            The number of jobs to run in parallel. Only :code:`fit` is parallelized.
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
            See https://verboselogs.readthedocs.io/en/latest/readme.html#overview-of-logging-levels
            for more information.
        local_classifier : BaseEstimator instance
            The local_classifier used to create the collection of local classifiers. Needs to have fit, predict and
            clone methods.
        hierarchy : nx.DiGraph, default=None
            Label hierarchy used in prediction and fitting. If None, it will be inferred during training.
        unique_taxonomy : bool, default=True
            True if the elements in the hierarchy have unique names, otherwise it can have unexpected behaviour.
            For example, a->b->c and d->b->e could have different meanings for b, so in that case unique_taxonomy
            should be set to false.
        """
        self.n_jobs = n_jobs
        self.log = logging.getLogger(__name__)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(verbose)
        sh.setFormatter(
            logging.Formatter("[%(levelname)s]\t [%(name)s]\t : %(message)s")
        )
        self.log.addHandler(sh)
        self.local_classifier = local_classifier
        self.hierarchy = hierarchy
        self.unique_taxonomy = unique_taxonomy
        self.placeholder_label = None
        self.max_depth = None
        if self.hierarchy is not None:
            self._initialize_hierarchy_information()
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        placeholder_label: Optional[NODE_TYPE] = None,
        replace_classifiers: bool = True,
    ):
        """
        Fits all classifiers.

        Needs to be subclassed by other classifiers as it only offers hierarchy methods.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training input samples.
        Y : np.array of shape (n_samples, n_levels)
            The hierarchical labels.
        placeholder_label : int or str, default=None
            Label that corresponds to "no label available for this data point". Defaults will be used if not passed.
        replace_classifiers : bool, default=True
            Turns on (True) the replacement of a local classifier with a constant classifier when trained on only
            a single unique class.
        """
        if placeholder_label is not None:
            self.placeholder_label = placeholder_label
        if not self.unique_taxonomy:
            Y = self._make_unique(Y)
        if self.hierarchy is None:
            self.hierarchy = data_utils.graph_from_hierarchical_labels(
                Y, self.placeholder_label
            )
            self._initialize_hierarchy_information()

    def predict(self, X: np.ndarray, threshold: float = 0):
        """
        Predict classes for the given data.

        Hierarchical labels are returned. If threshold is specified, prediction can end early.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input samples.
        threshold : float, default=None
            Minimum confidence score to continue prediction for children nodes.
        Returns
        -------
        y : np.array of shape (n_samples, n_levels)
            The predicted classes.
        """
        final_prediction = self._init_prediction(X)
        self._specialize_prediction(
            self.root,
            depth=0,
            data=X,
            threshold=threshold,
            final_prediction=final_prediction,
            selected_rows=np.ones(np.shape(X)[0], dtype=np.bool),
        )
        final_prediction = self._delete(CATEGORICAL_ROOT_NODE, final_prediction)
        if not self.unique_taxonomy:
            final_prediction = self._remove_separator(final_prediction)
        return final_prediction

    def _remove_separator(self, prediction):
        for i in range(np.shape(prediction)[0]):
            for j in range(1, np.shape(prediction)[1]):
                prediction[i, j] = prediction[i, j].split(SEPARATOR)[-1]
        return prediction

    def _make_unique(self, Y):
        new_Y = []
        for i in range(np.shape(Y)[0]):
            row = [Y[i, 0]]
            for j in range(1, np.shape(Y)[1]):
                row.append(row[-1] + SEPARATOR + Y[i, j])
            new_Y.append(np.asarray(row, dtype=np.str_))
        return np.array(new_Y)

    def _delete(self, character, prediction):
        final_prediction = []
        for i in range(np.shape(prediction)[0]):
            row = prediction[i]
            row = np.array(row[row != character]).flatten()
            final_prediction.append(row)
        final_prediction = np.asarray(final_prediction, dtype=object)
        return final_prediction

    def _init_prediction(self, X: np.ndarray):
        return np.full(
            (np.shape(X)[0], self.max_depth),
            self.placeholder_label,
            dtype=object if self.categorical_hierarchy else np.int32,
        )

    def _fit_classifier(self, fit_fct: Callable, fit_args: Iterator[Tuple]) -> None:
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = pool.starmap(fit_fct, fit_args)
        else:
            results = list(starmap(fit_fct, fit_args))

        for descriptor, classifier, warning in results:
            self.set_classifier(descriptor, classifier)
            if warning:
                self.log.warning(warning)

    def _specialize_prediction(
        self,
        local_identifier: NODE_TYPE,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
        selected_rows: np.ndarray,
    ) -> None:
        nodes_to_predict = self._get_nodes_to_predict(local_identifier)
        additional_prediction_args = self._choose_required_prediction_args(
            local_identifier, nodes_to_predict, selected_rows
        )

        if self._stopping_criterion(nodes_to_predict):
            return

        prediction = self._create_prediction(data, *additional_prediction_args)
        most_likely_labels, rows_under_threshold = self._analyze_prediction(
            prediction, threshold
        )

        self._continue_prediction(
            nodes_to_predict,
            selected_rows,
            most_likely_labels,
            rows_under_threshold,
            depth + 1,
            data,
            threshold,
            final_prediction,
        )

    def _get_nodes_to_predict(self, local_identifier: NODE_TYPE) -> List[NODE_TYPE]:
        raise NotImplementedError

    def _choose_required_prediction_args(
        self,
        local_identifier: NODE_TYPE,
        nodes_to_predict: List[NODE_TYPE],
        selected_rows: np.ndarray,
    ) -> Tuple:
        raise NotImplementedError

    def _stopping_criterion(self, value: Any) -> bool:
        raise NotImplementedError

    def _create_prediction(self, data: np.ndarray, *args) -> np.ndarray:
        raise NotImplementedError

    def _predict_with_classifier(self, classifier: Any, data: np.ndarray):
        if hasattr(classifier, "predict_proba"):
            return classifier.predict_proba(data)
        else:
            self.log.warning(
                "Local classifier does not support .predict_proba(). Trying with .predict()"
            )
            return classifier.predict(data)

    @staticmethod
    def _analyze_prediction(
        predictions: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        max_predictions = np.max(predictions, axis=0)
        most_likely_labels = np.argmax(predictions, axis=0)

        under_threshold = max_predictions < threshold
        return most_likely_labels, under_threshold

    def _continue_prediction(
        self,
        nodes_to_predict: List[NODE_TYPE],
        selected_rows: np.ndarray,
        most_likely_labels: np.ndarray,
        rows_under_threshold: np.ndarray,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _initialize_hierarchy_information(self) -> None:
        self.root = data_utils.find_root(self.hierarchy)
        if self.max_depth is None:
            self.max_depth = data_utils.find_max_depth(self.hierarchy, self.root)
        self.categorical_hierarchy = isinstance(self.root, str)
        if self.placeholder_label is None:
            if self.categorical_hierarchy:
                self.placeholder_label = data_utils.PLACEHOLDER_LABEL_CAT
            else:
                self.placeholder_label = data_utils.PLACEHOLDER_LABEL_NUMERIC
        self._initialize_classifiers()

    def _initialize_classifiers(self) -> None:
        depth_per_node = minimal_per_node_depth(self.hierarchy)
        max_considered_depth = max(depth_per_node.values())
        self.classes_ = [[] for _ in range(max_considered_depth)]
        for node, level in depth_per_node.items():
            self.classes_[level - 1].append(node)
        for level in range(len(self.classes_)):
            self.classes_[level] = np.array(self.classes_[level])

    def _copy_classifier(self):
        if is_classifier(self.local_classifier):
            return clone(self.local_classifier)
        else:
            try:
                return self.local_classifier.clone()
            except AttributeError:
                error_msg = (
                    "Assigned local classifier is not made by sklearn. Trying to copy by calling .clone()"
                    " failed."
                )
                self.log.error(error_msg)
                raise AttributeError(error_msg)

    def _copy_and_assign_classifier(self, node: NODE_TYPE) -> None:
        classifier = self._copy_classifier()
        self.hierarchy.nodes[node]["classifier"] = classifier

    def get_classifier(self, descriptor: Any) -> Any:
        """
        Return the local classifier associated to the given descriptor.

        Parameters
        ----------
        descriptor : Any
            The descriptor for which the local classifier should be returned.

        Returns
        -------
        classifier : Any
            The local classifier that is assigned to the given descriptor.
        """
        raise NotImplementedError

    def set_classifier(self, descriptor: Any, classifier: Any) -> None:
        """
        Set the local classifier for a given descriptor.

        Parameters
        ----------
        descriptor : Any
            The descriptor for which the local classifier should be set.
        classifier : Any
            The local classifier that will be assigned to the given descriptor.
        """
        raise NotImplementedError


class NodeClassifier(Classifier, abc.ABC):
    """Abstract class for classifiers that have their local classifiers linked to nodes."""

    def __init__(
        self,
        n_jobs: int = 1,
        verbose: int = 0,
        local_classifier: Any = BaseEstimator(),
        hierarchy: Optional[nx.DiGraph] = None,
        unique_taxonomy: bool = True,
        policy: Union[str, Type[Policies.Policy]] = "siblings",
    ):
        """
        Initialize a classifier.

        Extends the superclass by adding a data policy used in fitting.

        Parameters
        ----------
        n_jobs : int, default=1
            The number of jobs to run in parallel. Only :code:`fit` is parallelized.
        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
            See https://verboselogs.readthedocs.io/en/latest/readme.html#overview-of-logging-levels
            for more information.
        local_classifier : BaseEstimator instance
            The local classifier used to create the collection of local classifiers. Needs to have fit, predict and
            clone methods.
        hierarchy : nx.DiGraph, default=None
            Label hierarchy used in prediction and fitting. If None, it will be inferred during training.
        unique_taxonomy : bool, default=True
            True if the elements in the hierarchy have unique names, otherwise it can have unexpected behaviour.
            For example, a->b->c and d->b->e could have different meanings for b, so in that case unique_taxonomy
            should be set to false.
        policy : Policy, default="siblings"
            Rules for defining positive and negative training samples.
        """
        self.policy = policy
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            local_classifier=local_classifier,
            hierarchy=hierarchy,
            unique_taxonomy=unique_taxonomy,
        )

    def _specialize_prediction(
        self,
        local_identifier: NODE_TYPE,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
        selected_rows: np.ndarray,
    ) -> None:
        final_prediction[selected_rows, depth] = local_identifier
        super(NodeClassifier, self)._specialize_prediction(
            local_identifier, depth, data, threshold, final_prediction, selected_rows
        )

    def _get_nodes_to_predict(self, local_identifier: NODE_TYPE) -> List[NODE_TYPE]:
        return list(self.hierarchy.successors(local_identifier))

    def _choose_required_prediction_args(
        self,
        local_identifier: NODE_TYPE,
        nodes_to_predict: List[NODE_TYPE],
        selected_rows: np.ndarray,
    ) -> Tuple[np.ndarray, NODE_TYPE]:
        return selected_rows, local_identifier

    def _stopping_criterion(self, nodes_to_predict: List[NODE_TYPE]) -> bool:
        return len(nodes_to_predict) == 0

    def _continue_prediction(
        self,
        nodes_to_predict: List[NODE_TYPE],
        selected_rows: np.ndarray,
        most_likely_labels: np.ndarray,
        rows_under_threshold: np.ndarray,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
    ) -> None:
        if np.all(rows_under_threshold):
            return

        for child in nodes_to_predict:
            corresponding_data = np.copy(selected_rows)
            corresponding_data[selected_rows] = np.logical_and(
                most_likely_labels == nodes_to_predict.index(child),
                np.logical_not(rows_under_threshold),
            )
            if any(corresponding_data):
                self._specialize_prediction(
                    child,
                    depth,
                    data,
                    threshold,
                    final_prediction,
                    corresponding_data,
                )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        placeholder_label: Optional[NODE_TYPE] = None,
        replace_classifiers: bool = True,
    ) -> None:
        """
        Fit all classifiers.

        Extends superclass method and needs to be subclassed by other classifiers. Adds label flattening and policy
        initialization.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training input samples.
        Y : np.array of shape (n_samples, n_levels)
            The hierarchical labels.
        placeholder_label : int or str, default=None
            Label that corresponds to "no label available for this data point". Defaults will be used if not passed.
        replace_classifiers : bool, default=True
            Turns on (True) the replacement of a local classifier with a constant classifier when trained on only
            a single unique class.
        """
        super().fit(X, Y, placeholder_label, replace_classifiers)

        if not self.unique_taxonomy:
            Y = self._make_unique(Y)

        labels = data_utils.flatten_labels(Y, self.placeholder_label)
        if type(self.policy) == str:
            try:
                self.policy = Policies.IMPLEMENTED_POLICIES[self.policy.lower()](
                    self.hierarchy, labels
                )
            except KeyError:
                raise_error_and_log(
                    KeyError,
                    self.log,
                    f"Policy {self.policy} not implemented. Available policies are:\n"
                    f"{list(Policies.IMPLEMENTED_POLICIES.keys())}",
                )
        else:
            self.policy = self.policy(self.hierarchy, labels)

    def _initialize_classifiers(self) -> None:
        super()._initialize_classifiers()
        for node in self._get_nodes_with_classifier():
            self._copy_and_assign_classifier(node)

    def _get_nodes_with_classifier(self) -> List[NODE_TYPE]:
        raise NotImplementedError

    def _get_node(self, label: NODE_TYPE):
        if type(label) == str or (
            hasattr(label, "dtype") and label.dtype.type is np.str_
        ):
            if self.categorical_hierarchy:
                return self.hierarchy.nodes[label]
            raise_error_and_log(
                ValueError,
                self.log,
                "String label was given, but label hierarchy is numeric.",
            )
        else:
            if self.categorical_hierarchy:
                raise_error_and_log(
                    ValueError,
                    self.log,
                    "Numeric label was given, but label hierarchy is categorical.",
                )
            else:
                return self.hierarchy.nodes[label]

    def get_classifier(self, descriptor: NODE_TYPE) -> Any:
        """
        Return the local classifier associated to the given label.

        Parameters
        ----------
        descriptor : int or str
            The label for which the local classifier should be returned.

        Returns
        -------
        classifier : Any
            The local classifier that is assigned to the given label.
        """
        return self._get_node(descriptor)["classifier"]

    def set_classifier(self, descriptor: NODE_TYPE, classifier: Any) -> None:
        """
        Set the local classifier for a given label.

        Parameters
        ----------
        descriptor : int or str
            The label for which the local classifier should be set.
        classifier : Any
            The local classifier that will be assigned to the given label.
        """
        self._get_node(descriptor)["classifier"] = classifier


class ConstantClassifier:
    """A classifier that returns 1 for a specified label for all samples during prediction."""

    def __init__(self, class_to_predict: int, num_classes: int) -> None:
        """
        Initialize the classifier.

        Parameters
        ----------
        class_to_predict : int
            The index of the label that should be predicted with a probability of one. Needs to be within `num_classes`
        num_classes : int
            The amount of labels that should be predicted by the classifier.
        """
        self.value = np.zeros(num_classes, dtype=int)
        self.value[class_to_predict] = 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict X with previously set parameters.

        Parameters
        ----------
        X : np.ndarray of shape(n_samples, ...)
            Data that should be predicted. Only the number of samples matters.

        Returns
        -------
        output : np.ndarray
            1 for the previously set label and 0 for all others for all samples in X.
        """
        return np.vstack([self.value] * np.shape(X)[0])


def fit_or_replace(
    replace_classifier: bool,
    classifier: Any,
    data: np.ndarray,
    labels: np.ndarray,
    out_dim: int,
):
    """
    Fit a given local classifier on data and labels and replace it if there is only a single label to predict.

    Parameters
    ----------
    replace_classifier : bool
        Only possibly replace the classifier if this is True. Otherwise always try to fit the classifier.
    classifier : Any
        Local classifier to be fitted.
    data : np.ndarray
        Data used for fitting.
    labels : np.ndarray
        Classes used for fitting.
    out_dim : int
        Number of columns that the classifier should predict.

    Returns
    -------
    classifier : Any,
        Either the fitted classifier or a ConstantClassifier.
    warning_msg : str or None
        None if a fitted classifier is returned, a warning message otherwise.
    """
    unique_labels = np.unique(labels)
    warning = None
    if replace_classifier and len(unique_labels) == 1:
        warning = "One or more classifiers could not be fitted since all data points have the same label."
        warning = warning + ' These were replaced with a "constant classifier".'
        classifier = ConstantClassifier(unique_labels[0], out_dim)
    else:
        classifier.fit(data, labels)
    return classifier, warning


class DuplicateFilter:
    """A filter that removes duplicate messages from logging."""

    def __init__(self):
        """Initialize filter."""
        self.msgs = set()

    def filter(self, record: Any) -> bool:
        """
        Filter messages.

        Parameters
        ----------
        record : logged message
            Text to be filtered.

        Returns
        -------
        rv : bool
            True if message was not found in the filter, false otherwise.
        """
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


def raise_error_and_log(
    error: Type[Exception], logger: logging.Logger, msg: str
) -> None:
    """
    Log a message and throw it as an error.

    Parameters
    ----------
    error : Exception
        The type of error to throw.
    logger : logging.Logger
        The logger to use for logging.
    msg : str
        Message to throw and to log.
    """
    logger.error(msg)
    raise error(msg)

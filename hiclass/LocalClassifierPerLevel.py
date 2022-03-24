"""
Local classifier per level approach.

Numeric and string output labels are both handled.
"""
from itertools import repeat
from typing import Any, Tuple, Type, Union, List, Optional

import numpy as np

from hiclass.data import NODE_TYPE
from hiclass.Classifier import (
    Classifier,
    DuplicateFilter,
    fit_or_replace,
    ConstantClassifier,
    raise_error_and_log,
)


class DefaultProbability:
    """Defines probability calculations used to return probabilities for all local classifiers."""

    def predict_proba(self, classifiers: List[Any], X: np.ndarray) -> List:
        """
        Compute prediction probabilities.

        Parameters
        ----------
        classifiers : List of classifiers
            The local classifiers that should be used for the prediction
        X : np.array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : np.ndarray of shape (n_levels, n_samples)
            Prediction probabilities for all classes in each hierarchical level.
        """
        probabilities = []
        for classifier in classifiers:
            probabilities.append(classifier.predict_proba(X))
        return probabilities


IMPLEMENTED_PROBABILITIES = {"default": DefaultProbability}


def _fit_classifier(
    level: int,
    classifier: Any,
    level_nodes: np.array,
    data: np.ndarray,
    labels: np.ndarray,
    placeholder: NODE_TYPE,
    replace_classifier: bool,
) -> Tuple[int, Any, str]:
    """Fits a local classifier for a given level and data and metadata.

    Parameters
    ----------
    level : int
        The level that the classifier should be assigned to.
    classifier : Any
        The classifier that is fitted.
    level_nodes : np.array
        A list of nodes from the hierarchy that belong to the given `level`.
    data : np.array
        The data that is being used for training the classifier.
    labels : np.array
        The labels corresponding to the given data. Must have the same number of rows as `data`.
    placeholder : int or str
        A placeholder descriptor for non-existing labels in the `labels` argument.
    replace_classifier : bool
        Turns on (True) the replacement of classifiers with a constant classifier when trained on only
        a single unique label.

    Returns
    -------
    level : int
        Level for which the classifier was fitted.
    classifier : Any
        The classifier that was fitted for the given node.
    warning : str
        The warnings raised.
    """
    shifted_level = level - 1
    relevant_data = labels[:, shifted_level] != placeholder
    classes = np.zeros_like(relevant_data, dtype=int)

    mapping = dict(zip(level_nodes, range(0, len(level_nodes))))
    for node in level_nodes:
        positive_samples = labels[:, shifted_level] == node
        classes[positive_samples] = mapping[node]

    classifier, warning = fit_or_replace(
        replace_classifier,
        classifier,
        data[relevant_data],
        classes[relevant_data],
        len(mapping),
    )
    return level, classifier, warning


class LocalClassifierPerLevel(Classifier):
    """
    Assign local classifiers for each class hierarchy level.

    A local classifier per level is a local hierarchical classifier that fits one local multi-class classifier
    for each level of the hierarchy. In case of a DAG, nodes are assigned their highest possible level, with the root
    being the highest level.
    """

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        placeholder_label: Optional[NODE_TYPE] = None,
        replace_classifiers: bool = True,
    ) -> None:
        """
        Fit the local classifiers.

        Parameters
        ----------
        X : np.array of shape(n_samples, n_features)
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

        duplicate_filter = DuplicateFilter()
        self.log.addFilter(duplicate_filter)  # removes possible duplicate warnings

        if all(Y[:, 0] == self.root):
            Y = Y[:, 1:]
        if any(Y[:, 0] == self.root):
            error_msg = (
                "Some labels contain the root node and some do not. Classifier can only be fit when "
                "labels are consistent."
            )
            raise_error_and_log(ValueError, self.log, error_msg)

        fit_args = zip(
            range(0, len(self.classes_)),
            self.classifiers,
            self.classes_,
            repeat(X),
            repeat(Y),
            repeat(placeholder_label),
            repeat(replace_classifiers),
        )

        next(fit_args)  # skip the root node
        self._fit_classifier(_fit_classifier, fit_args)
        self.classifiers[0] = ConstantClassifier(0, 1)

        self.log.removeFilter(duplicate_filter)  # delete duplicate filter

    def predict_proba(
        self, X: np.ndarray, algorithm: Union[str, Type[DefaultProbability]] = "default"
    ) -> List:
        """
        Compute prediction probabilities.

        Parameters
        ----------
        X : np.array of shape(n_samples, n_features)
            The input samples.
        algorithm : str or Probability
            The algorithm to use for calculating probabilities.

        Returns
        -------
        probabilities : np.ndarray of shape (n_levels, n_samples)
            Prediction probabilities for all classes in each hierarchical level.
        """
        if type(algorithm) == str:
            try:
                probability_algorithm = IMPLEMENTED_PROBABILITIES[algorithm.lower()]()
            except KeyError:
                error_msg = (
                    f"Probability algorithm {algorithm} not implemented. Available algorithms are:\n"
                    f"{list(IMPLEMENTED_PROBABILITIES.keys())}"
                )
                raise_error_and_log(KeyError, self.log, error_msg)
        else:
            probability_algorithm = algorithm
        return probability_algorithm.predict_proba(self.classifiers, X)

    def _ensure_possible_predictions(
        self, final_prediction: np.ndarray, predictions: np.ndarray, depth: int
    ) -> np.ndarray:
        """Zero prediction results that are not possible."""
        if depth == 0:
            return predictions
        previous_predictions = np.unique(final_prediction[:, depth - 1])
        corrected_predictions = np.zeros_like(predictions)
        possible_children = [
            list(self.hierarchy.successors(node)) for node in previous_predictions
        ]
        for previous_prediction, possible_predictions in zip(
            previous_predictions, possible_children
        ):
            relevant_rows = np.where(
                final_prediction[:, depth - 1] == previous_prediction
            )[0][:, None]
            possible_predictions = np.array(
                [
                    np.where(self.classes_[depth] == node)[0][0]
                    for node in possible_predictions
                ]
            )
            corrected_predictions[relevant_rows, possible_predictions] = predictions[
                relevant_rows, possible_predictions
            ]
        return corrected_predictions

    def _specialize_prediction(
        self,
        local_identifier: NODE_TYPE,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
        selected_rows: np.ndarray,
    ) -> None:
        local_identifier = depth
        if self._stopping_criterion(depth):
            return

        nodes_to_predict = self._get_nodes_to_predict(local_identifier)
        additional_prediction_args = self._choose_required_prediction_args(
            local_identifier, nodes_to_predict, selected_rows
        )

        predictions = self._create_prediction(data, *additional_prediction_args)
        predictions = self._ensure_possible_predictions(
            final_prediction, predictions, depth
        ).T

        most_likely_labels, rows_under_threshold = self._analyze_prediction(
            predictions, threshold
        )
        selected_rows = np.logical_and(
            selected_rows, np.logical_not(rows_under_threshold)
        )

        self._continue_prediction(
            nodes_to_predict,
            selected_rows,
            most_likely_labels,
            rows_under_threshold,
            depth,
            data,
            threshold,
            final_prediction,
        )

    def _get_nodes_to_predict(self, local_identifier: int) -> np.ndarray:
        return self.classes_[local_identifier]

    def _choose_required_prediction_args(
        self,
        local_identifier: NODE_TYPE,
        nodes_to_predict: np.ndarray,
        selected_rows: np.ndarray,
    ) -> Any:
        return [local_identifier]

    def _stopping_criterion(self, value: int) -> bool:
        return value >= len(self.classes_)

    def _create_prediction(self, data: np.ndarray, depth: int) -> np.ndarray:
        classifier = self.get_classifier(depth)
        return self._predict_with_classifier(classifier, data)

    def _continue_prediction(
        self,
        nodes_to_predict: np.ndarray,
        selected_rows: np.ndarray,
        most_likely_labels: np.ndarray,
        rows_under_threshold: np.ndarray,
        depth: int,
        data: np.ndarray,
        threshold: float,
        final_prediction: np.ndarray,
    ):
        if np.all(rows_under_threshold):
            return

        for label in nodes_to_predict:
            corresponding_data = np.copy(selected_rows)
            label_idx = np.where(nodes_to_predict == label)[0][0]
            corresponding_data[selected_rows] = (most_likely_labels == label_idx)[
                selected_rows
            ]
            final_prediction[corresponding_data, depth] = label

        self._specialize_prediction(
            depth + 1, depth + 1, data, threshold, final_prediction, selected_rows
        )

    def _initialize_classifiers(self) -> None:
        super()._initialize_classifiers()
        self.classifiers = [self._copy_classifier() for _ in self.classes_]

    def get_classifier(self, descriptor: int) -> Any:
        """
        Return the local classifier associated to the given hierarchy level.

        Raise IndexError if the level is invalid.

        Parameters
        ----------
        descriptor : int or str
            the descriptor for which the local classifier should be returned.

        Returns
        -------
        classifier : Any
            The local classifier that is assigned to the given descriptor.
        """
        if descriptor < 0 or descriptor >= len(self.classes_):
            raise IndexError("Hierarchy level does not exist.")
        return self.classifiers[descriptor]

    def set_classifier(self, descriptor: int, classifier: Any) -> None:
        """
        Set the local classifier for a given hierarchy level.

        Parameters
        ----------
        descriptor : int or str
            the descriptor for which the local classifier should be set.
        classifier : Any
            The local classifier that will be assigned to the given label.
        """
        self.classifiers[descriptor] = classifier

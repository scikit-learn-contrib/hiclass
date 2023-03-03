"""Helper functions to compute hierarchical evaluation metrics."""
import numpy as np
from sklearn.utils import check_array

from hiclass.HierarchicalClassifier import make_leveled


def _validate_input(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_pred = make_leveled(y_pred)
    y_true = make_leveled(y_true)
    y_true = check_array(y_true, dtype=None)
    y_pred = check_array(y_pred, dtype=None)
    return y_true, y_pred


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical precision score.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    average: {"micro", "macro"}, str, default="micro"
        This parameter determines the type of averaging performed during the computation:

        - `micro`: The precision is computed by summing over all individual instances, :math:`\displaystyle{hP = \frac{\sum_{i=1}^{n}| \alpha_i \cap \beta_i |}{\sum_{i=1}^{n}| \alpha_i |}}`, where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors, with summations computed over all test examples.
        - `macro`: The precision is computed for each instance and then averaged, :math:`\displaystyle{hP = \frac{\sum_{i=1}^{n}hP_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors.

    Returns
    -------
    precision : float
        What proportion of positive identifications was actually correct?
    """
    average_functions = {
        "micro": _precision_micro,
        "macro": _precision_macro,
    }
    return average_functions[average](y_true, y_pred)


def _precision_micro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        predicted_set = set(prediction)
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    precision = sum_intersection / sum_prediction_and_ancestors
    return precision


def _precision_macro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    sum_precisions = 0
    for ground_truth, predicted in zip(y_true, y_pred):
        sample_precision = _precision_micro([ground_truth], [predicted])
        sum_precisions = sum_precisions + sample_precision
    return sum_precisions / len(y_true)


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical recall score.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    average: {"micro", "macro"}, str, default="micro"
        This parameter determines the type of averaging performed during the computation:

        - `micro`: The recall is computed by summing over all individual instances, :math:`\displaystyle{hR = \frac{\sum_{i=1}^{n}|\alpha_i \cap \beta_i|}{\sum_{i=1}^{n}|\beta_i|}}`, where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors, with summations computed over all test examples.
        - `macro`: The recall is computed for each instance and then averaged, :math:`\displaystyle{hR = \frac{\sum_{i=1}^{n}hR_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors.

    Returns
    -------
    recall : float
        What proportion of actual positives was identified correctly?
    """
    average_functions = {
        "micro": _recall_micro,
        "macro": _recall_macro,
    }
    return average_functions[average](y_true, y_pred)


def _recall_micro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        predicted_set = set(prediction)
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def _recall_macro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    sum_recalls = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        sample_recall = _recall_micro([ground_truth], [prediction])
        sum_recalls = sum_recalls + sample_recall
    return sum_recalls / len(y_true)


def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical f-score.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    average: {"micro", "macro"}, str, default="micro"
        This parameter determines the type of averaging performed during the computation:

        - `micro`: The f-score is computed by summing over all individual instances, :math:`\displaystyle{hF = \frac{2 \times hP \times hR}{hP + hR}}`, where :math:`hP` is the hierarchical precision and :math:`hR` is the hierarchical recall.
        - `macro`: The f-score is computed for each instance and then averaged, :math:`\displaystyle{hF = \frac{\sum_{i=1}^{n}hF_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors.
    Returns
    -------
    f1 : float
        Weighted average of the precision and recall
    """
    average_functions = {
        "micro": _f_score_micro,
        "macro": _f_score_macro,
    }
    return average_functions[average](y_true, y_pred)


def _f_score_micro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec)
    return f1


def _f_score_macro(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _validate_input(y_true, y_pred)
    sum_f_scores = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        sample_f_score = _f_score_micro([ground_truth], [prediction])
        sum_f_scores = sum_f_scores + sample_f_score
    return sum_f_scores / len(y_true)

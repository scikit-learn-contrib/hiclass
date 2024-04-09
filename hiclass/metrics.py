"""Helper functions to compute hierarchical evaluation metrics."""

from typing import Union, List
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.preprocessing import LabelEncoder

from hiclass.HierarchicalClassifier import make_leveled
from hiclass import HierarchicalClassifier


def _validate_input(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_pred = make_leveled(y_pred)
    y_true = make_leveled(y_true)
    y_true = check_array(y_true, dtype=None, ensure_2d=False, allow_nd=True)
    y_pred = check_array(y_pred, dtype=None, ensure_2d=False, allow_nd=True)
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
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _precision_micro,
        "macro": _precision_macro,
    }
    return functions[average](y_true, y_pred)


def _precision_micro(y_true: np.ndarray, y_pred: np.ndarray):
    precision_micro = {
        1: _precision_micro_1d,
        2: _precision_micro_2d,
        3: _precision_micro_3d,
    }
    return precision_micro[y_true.ndim](y_true, y_pred)


def _precision_micro_1d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set([ground_truth])
        ground_truth_set.discard("")
        predicted_set = set([prediction])
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    return sum_intersection / sum_prediction_and_ancestors


def _precision_micro_2d(y_true: np.ndarray, y_pred: np.ndarray):
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
    return sum_intersection / sum_prediction_and_ancestors


def _precision_micro_3d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for row_ground_truth, row_prediction in zip(y_true, y_pred):
        ground_truth_set = set()
        predicted_set = set()
        for ground_truth, prediction in zip(row_ground_truth, row_prediction):
            ground_truth_set.update(ground_truth)
            predicted_set.update(prediction)
        ground_truth_set.discard("")
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    return sum_intersection / sum_prediction_and_ancestors


def _precision_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return _compute_macro(y_true, y_pred, _precision_micro)


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
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _recall_micro,
        "macro": _recall_macro,
    }
    return functions[average](y_true, y_pred)


def _recall_micro(y_true: np.ndarray, y_pred: np.ndarray):
    recall_micro = {
        1: _recall_micro_1d,
        2: _recall_micro_2d,
        3: _recall_micro_3d,
    }
    return recall_micro[y_true.ndim](y_true, y_pred)


def _recall_micro_1d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set([ground_truth])
        ground_truth_set.discard("")
        predicted_set = set([prediction])
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def _recall_micro_2d(y_true: np.ndarray, y_pred: np.ndarray):
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


def _recall_micro_3d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for row_ground_truth, row_prediction in zip(y_true, y_pred):
        ground_truth_set = set()
        predicted_set = set()
        for ground_truth, prediction in zip(row_ground_truth, row_prediction):
            ground_truth_set.update(ground_truth)
            predicted_set.update(prediction)
        ground_truth_set.discard("")
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
    return _compute_macro(y_true, y_pred, _recall_micro)


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
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _f_score_micro,
        "macro": _f_score_macro,
    }
    return functions[average](y_true, y_pred)


def _f_score_micro(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec)


def _f_score_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return _compute_macro(y_true, y_pred, _f_score_micro)


def _compute_macro(y_true: np.ndarray, y_pred: np.ndarray, _micro_function):
    overall_sum = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        sample_score = _micro_function(np.array([ground_truth]), np.array([prediction]))
        overall_sum = overall_sum + sample_score
    return overall_sum / len(y_true)


def _prepare_data(classifier: HierarchicalClassifier, y_true: np.ndarray, y_prob: np.ndarray, level: int, y_pred: np.ndarray = None):
    classifier_classes = np.array(classifier.classes_[level]).astype("str")
    y_true = make_leveled(y_true)
    y_true = classifier._disambiguate(y_true)
    y_true = np.array(list(map(lambda x: x[level], y_true)))
    y_true = np.array([label.split(classifier.separator_)[level] for label in y_true])

    if y_pred is not None:
        y_pred = make_leveled(y_pred)
        y_pred = classifier._disambiguate(y_pred)
        y_pred = np.array(list(map(lambda x: x[level], y_pred)))
        y_pred = np.array([label.split(classifier.separator_)[level] for label in y_pred])

    unique_labels = np.unique(y_true).astype("str")
    # add labels not seen in the training process
    new_labels = np.sort(np.union1d(unique_labels, classifier_classes))

    new_y_prob = np.zeros((y_prob.shape[0], len(new_labels)), dtype=np.float32)
    for idx, label in enumerate(new_labels):
        if label in classifier_classes:
            old_idx = np.where(classifier_classes == label)[0][0]
            new_y_prob[:, idx] = y_prob[:, old_idx]

    return y_true, y_pred, new_labels, new_y_prob


_calibration_aggregations = ["average", "sum", "None"]


def _aggregate_scores(scores, agg):
    if agg == 'average':
        return np.mean(scores)
    if agg == 'sum':
        return np.sum(scores)
    if agg is None or agg == 'None':
        return scores


def _validate_args(agg, y_prob, level):
    if agg and agg not in _calibration_aggregations:
        raise ValueError(f"{agg} is not a valid aggregation function.")
    if isinstance(y_prob, list) and len(y_prob) == 0:
        raise ValueError("y_prob is empty.")
    if (isinstance(y_prob, list) and len(y_prob) == 1 or isinstance(y_prob, np.ndarray)) and level is None:
        raise ValueError("If y_prob is not a list of probabilities the level must be specified.")
    if isinstance(y_prob, list) and len(y_prob) == 1:
        return y_prob[0]
    return y_prob


def multiclass_brier_score(classifier: HierarchicalClassifier, y_true: np.ndarray, y_prob: Union[np.ndarray | List], agg='average', level=None):
    """Compute the brier score for two or more classes.

    Parameters
    ----------
    classifier : HierarchicalClassifier
        The classifier used.
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_prob : np.array of shape (n_samples, n_unique_labels_per_level) or List[np.array((n_samples, n_unique_labels_per_level))]
        Predicted probabilities.
    agg: {"average", "sum", None}, str, default="average"
        This parameter determines the type of averaging performed during the computation if y_prob contains probabilities for multiple levels:

        - `average`: Calculate the average brier score over all levels.
        - `sum`: Calculate the summed brier score over all levels.
        - None: Don't aggregate results. Returns a list of brier scores.
    level : int, default=None
        Specifies the level of y_prob if y_prob is not a list of numpy arrays.
    Returns
    -------
    brier_score : float or List[float]
        Brier score of predicted probabilities.
    """
    y_prob = _validate_args(agg, y_prob, level)
    if isinstance(y_prob, list):
        scores = []
        for level in range(make_leveled(y_true).shape[1]):
            scores.append(_multiclass_brier_score(classifier, y_true, y_prob[level], level))
        return _aggregate_scores(scores, agg)
    return _multiclass_brier_score(classifier, y_true, y_prob, level)


def log_loss(classifier: HierarchicalClassifier, y_true: np.ndarray, y_prob: Union[np.ndarray | List], agg='average', level=None):
    """Compute the log loss of predicted probabilities.

    Parameters
    ----------
    classifier : HierarchicalClassifier
        The classifier used.
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_prob : np.array of shape (n_samples, n_unique_labels_per_level) or List[np.array((n_samples, n_unique_labels_per_level))]
        Predicted probabilities.
    agg: {"average", "sum", None}, str, default="average"
        This parameter determines the type of averaging performed during the computation if y_prob contains probabilities for multiple levels:

        - `average`: Calculate the average brier score over all levels.
        - `sum`: Calculate the summed brier score over all levels.
        - None: Don't aggregate results. Returns a list of brier scores.
    level : int, default=None
        Specifies the level of y_prob if y_prob is not a list of numpy arrays.
    Returns
    -------
    log_loss : float or List[float]
        Log loss of predicted probabilities.
    """
    y_prob = _validate_args(agg, y_prob, level)
    if isinstance(y_prob, list):
        scores = []
        for level in range(make_leveled(y_true).shape[1]):
            scores.append(_log_loss(classifier, y_true, y_prob[level], level))
        return _aggregate_scores(scores, agg)
    return _multiclass_brier_score(classifier, y_true, y_prob, level)


def expected_calibration_error(classifier: HierarchicalClassifier, y_true, y_prob: Union[np.ndarray | List], y_pred, n_bins=10, agg='average', level=None):
    """Compute the expected calibration error.

    Parameters
    ----------
    classifier : HierarchicalClassifier
        The classifier used.
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_prob : np.array of shape (n_samples, n_unique_labels_per_level) or List[np.array((n_samples, n_unique_labels_per_level))]
        Predicted probabilities.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    n_bins : int, default=10
        Number of bins to calculate the metric.
    agg: {"average", "sum", None}, str, default="average"
        This parameter determines the type of averaging performed during the computation if y_prob contains probabilities for multiple levels:

        - `average`: Calculate the average brier score over all levels.
        - `sum`: Calculate the summed brier score over all levels.
        - None: Don't aggregate results. Returns a list of brier scores.
    level : int, default=None
        Specifies the level of y_prob if y_prob is not a list of numpy arrays.
    Returns
    -------
    expected_calibration_error : float or List[float]
        Expected calibration error of predicted probabilities.
    """
    y_prob = _validate_args(agg, y_prob, level)
    if isinstance(y_prob, list):
        scores = []
        for level in range(make_leveled(y_true).shape[1]):
            scores.append(_expected_calibration_error(classifier, y_true, y_prob[level], y_pred, level, n_bins))
        return _aggregate_scores(scores, agg)
    return _expected_calibration_error(classifier, y_true, y_prob, y_pred, level, n_bins)


def static_calibration_error(classifier, y_true, y_prob: Union[np.ndarray | List], y_pred, n_bins=10, agg='average', level=None):
    """Compute the static calibration error.

    Parameters
    ----------
    classifier : HierarchicalClassifier
        The classifier used.
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_prob : np.array of shape (n_samples, n_unique_labels_per_level) or List[np.array((n_samples, n_unique_labels_per_level))]
        Predicted probabilities.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    n_bins : int, default=10
        Number of bins to calculate the metric.
    agg: {"average", "sum", None}, str, default="average"
        This parameter determines the type of averaging performed during the computation if y_prob contains probabilities for multiple levels:

        - `average`: Calculate the average brier score over all levels.
        - `sum`: Calculate the summed brier score over all levels.
        - None: Don't aggregate results. Returns a list of brier scores.
    level : int, default=None
        Specifies the level of y_prob if y_prob is not a list of numpy arrays.
    Returns
    -------
    static_calibration_error : float or List[float]
        Static calibration error of predicted probabilities.
    """
    y_prob = _validate_args(agg, y_prob, level)
    if isinstance(y_prob, list):
        scores = []
        for level in range(make_leveled(y_true).shape[1]):
            scores.append(_static_calibration_error(classifier, y_true, y_prob[level], y_pred, level, n_bins=n_bins))
        return _aggregate_scores(scores, agg)
    return _static_calibration_error(classifier, y_true, y_prob, y_pred, level, n_bins=n_bins)


def adaptive_calibration_error(classifier, y_true, y_prob: Union[np.ndarray | List], y_pred, n_ranges=10, agg='average', level=None):
    """Compute the adaptive calibration error.

    Parameters
    ----------
    classifier : HierarchicalClassifier
        The classifier used.
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_prob : np.array of shape (n_samples, n_unique_labels_per_level) or List[np.array((n_samples, n_unique_labels_per_level))]
        Predicted probabilities.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    n_ranges : int, default=10
        Number of ranges to calculate the metric.
    agg: {"average", "sum", None}, str, default="average"
        This parameter determines the type of averaging performed during the computation if y_prob contains probabilities for multiple levels:

        - `average`: Calculate the average brier score over all levels.
        - `sum`: Calculate the summed brier score over all levels.
        - None: Don't aggregate results. Returns a list of brier scores.
    level : int, default=None
        Specifies the level of y_prob if y_prob is not a list of numpy arrays.
    Returns
    -------
    adaptive_calibration_error : float or List[float]
        Adaptive calibration error of predicted probabilities.
    """
    y_prob = _validate_args(agg, y_prob, level)
    if isinstance(y_prob, list):
        scores = []
        for level in range(make_leveled(y_true).shape[1]):
            scores.append(_adaptive_calibration_error(classifier, y_true, y_prob[level], y_pred, level, n_ranges=n_ranges))
        return _aggregate_scores(scores, agg)
    return _adaptive_calibration_error(classifier, y_true, y_prob, y_pred, level, n_ranges=n_ranges)


def _multiclass_brier_score(classifier: HierarchicalClassifier, y_true: np.ndarray, y_prob: np.ndarray, level: int):
    y_true, _, labels, y_prob = _prepare_data(classifier, y_true, y_prob, level)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_true_encoded = label_encoder.transform(y_true)
    return (1 / y_prob.shape[0]) * np.sum(np.sum(np.square(y_prob - np.eye(y_prob.shape[1])[y_true_encoded]), axis=1))


def _log_loss(classifier: HierarchicalClassifier, y_true: np.ndarray, y_prob: np.ndarray, level: int):
    y_true, _, labels, y_prob = _prepare_data(classifier, y_true, y_prob, level)
    return sk_log_loss(y_true, y_prob, labels=labels)


def _expected_calibration_error(classifier: HierarchicalClassifier, y_true, y_prob, y_pred, level, n_bins=10):
    y_true, y_pred, labels, y_prob = _prepare_data(classifier, y_true, y_prob, level, y_pred)

    n = len(y_true)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    y_prob = np.max(y_prob, axis=1)
    stacked = np.column_stack([y_prob, y_pred_encoded, y_true_encoded])

    # calculate equally sized bins
    _, bin_edges = np.histogram(stacked, bins=n_bins, range=(0, 1))
    bin_indices = np.digitize(stacked, bin_edges)[:, 0]

    # add bin index to each data point
    data = np.column_stack([stacked, bin_indices])

    # create bin mask
    masks = (data[:, -1, None] == range(1, n_bins + 1)).T

    # create actual bins
    bins = [data[masks[i]] for i in range(n_bins)]

    # calculate ECE
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    ece = 0
    for i in range(n_bins):
        acc[i] = 1 / (bins[i].shape[0]) * np.sum((bins[i][:, 1] == bins[i][:, 2])) if bins[i].shape[0] != 0 else 0
        conf[i] = 1 / (bins[i].shape[0]) * np.sum(bins[i][:, 0]) if bins[i].shape[0] != 0 else 0
        ece += (bins[i].shape[0] / n) * abs(acc[i] - conf[i]) if bins[i].shape[0] != 0 else 0
    return ece


def _static_calibration_error(classifier, y_true, y_prob, y_pred, level, n_bins=10):
    y_true, y_pred, labels, y_prob = _prepare_data(classifier, y_true, y_prob, level, y_pred)

    n_samples, n_classes = y_prob.shape
    assert n_classes > 2

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    class_error = np.zeros(n_classes)

    for k in range(n_classes):
        class_scores = y_prob[:, k]
        stacked = np.column_stack([class_scores, y_pred_encoded, y_true_encoded])

        # create bins
        _, bin_edges = np.histogram(stacked, bins=n_bins, range=(0, 1))
        bin_indices = np.digitize(stacked, bin_edges)[:, 0]

        # add bin index to each data point
        data = np.column_stack([stacked, bin_indices])

        # create bin mask
        masks = (data[:, -1, None] == range(1, n_bins + 1)).T

        # create actual bins
        bins = [data[masks[i]] for i in range(n_bins)]

        # calculate per class calibration error
        acc = np.zeros(n_bins)
        conf = np.zeros(n_bins)
        error = 0
        for i in range(n_bins):
            acc[i] = 1 / (bins[i].shape[0]) * np.sum((bins[i][:, 1] == bins[i][:, 2])) if bins[i].shape[0] != 0 else 0
            conf[i] = 1 / (bins[i].shape[0]) * np.sum(bins[i][:, 0]) if bins[i].shape[0] != 0 else 0
            error += (bins[i].shape[0] / n_samples) * abs(acc[i] - conf[i]) if bins[i].shape[0] != 0 else 0

        class_error[k] = error

    return np.mean(class_error)


def _adaptive_calibration_error(classifier, y_true, y_prob, y_pred, level, n_ranges=10):
    y_true, y_pred, labels, y_prob = _prepare_data(classifier, y_true, y_prob, level, y_pred)

    _, n_classes = y_prob.shape
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    class_error = np.zeros(n_classes)

    for k in range(n_classes):
        class_scores = y_prob[:, k]

        # sort by score probability
        idx = np.argsort([class_scores])[0]
        class_scores, ordered_y_pred_labels, ordered_y_true = class_scores[idx], y_pred_encoded[idx], y_true_encoded[idx]
        stacked = np.column_stack([np.array(range(len(class_scores))), class_scores, ordered_y_pred_labels, ordered_y_true])

        bin_edges = np.floor(np.linspace(0, len(class_scores), n_ranges + 1, endpoint=True)).astype(int)
        _, bin_edges = np.histogram(stacked, bins=bin_edges, range=(0, len(class_scores)))
        bin_indices = np.digitize(stacked, bin_edges)[:, 0]

        # add bin index to each data point
        data = np.column_stack([stacked, bin_indices])

        # create bin mask
        masks = (data[:, -1, None] == range(1, n_ranges + 1)).T

        # create actual bins
        bins = [data[masks[i]] for i in range(n_ranges)]

        # calculate per class calibration error
        acc = np.zeros(n_ranges)
        conf = np.zeros(n_ranges)
        error = 0
        for i in range(n_ranges):
            acc[i] = 1 / (bins[i].shape[0]) * np.sum((bins[i][:, 2] == bins[i][:, 3])) if bins[i].shape[0] != 0 else 0
            conf[i] = 1 / (bins[i].shape[0]) * np.sum(bins[i][:, 1]) if bins[i].shape[0] != 0 else 0
            error += abs(acc[i] - conf[i]) if bins[i].shape[0] != 0 else 0

        class_error[k] = error

    return (1 / (n_classes * n_ranges)) * np.sum(class_error)

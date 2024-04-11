import numpy as np
import pytest
from pytest import approx
import math
from numpy.testing import assert_array_almost_equal, assert_array_equal
from unittest.mock import Mock

from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass.metrics import (
    precision,
    recall,
    f1,
    _multiclass_brier_score,
    _log_loss,
    _expected_calibration_error,
    _static_calibration_error,
    _adaptive_calibration_error,
    multiclass_brier_score,
    log_loss,
    expected_calibration_error,
    static_calibration_error,
    adaptive_calibration_error,
)


# TODO: add tests for 3D dataframe (not sure if it's possible to have 3D dataframes)


def test_unmatched_lengths_1d_list():
    y_true = [1, 2, 3]
    y_pred = [1, 2]
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_2d_list():
    y_true = [[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8]]
    y_pred = [[1, 2, 3], [1, 2, 4]]
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_3d_list():
    y_true = [
        [["human", "mermaid"], ["fish", "mermaid"]],
        [["human", "minotaur"], ["bull", "minotaur"]],
    ]
    y_pred = [
        [["human", "mermaid"], ["fish", "mermaid"]],
    ]
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_1d_np_array():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_2d_np_array():
    y_true = np.array([[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8]])
    y_pred = np.array([[1, 2, 3], [1, 2, 4]])
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
        ]
    )
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_precision_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 5, 6]
    assert 0.5 == precision(y_true, y_pred, "micro")
    assert 1 == precision(y_true, y_true, "micro")


def test_precision_micro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert 0.5 == precision(y_true, y_pred, "micro")
    assert 1 == precision(y_true, y_true, "micro")


def test_precision_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 5, 6])
    assert 0.5 == precision(y_true, y_pred, "micro")
    assert 1 == precision(y_true, y_true, "micro")


def test_precision_micro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert 0.5 == precision(y_true, y_pred, "micro")
    assert 1 == precision(y_true, y_true, "micro")


def test_precision_micro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.8333 == approx(precision(y_true, y_pred, "micro"), rel=1e-3)
    assert 1 == precision(y_true, y_true, "micro")


def test_precision_macro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 5, 6, 7]
    assert 0.25 == precision(y_true, y_pred, "macro")
    assert 1 == precision(y_true, y_true, "macro")


def test_precision_macro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 5, 6, 7], [1, 2, 3, 4]]
    assert 0.375 == precision(y_true, y_pred, "macro")
    assert 1 == precision(y_true, y_true, "macro")


def test_precision_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 5, 6, 7])
    assert 0.25 == precision(y_true, y_pred, "macro")
    assert 1 == precision(y_true, y_true, "macro")


def test_precision_macro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert 0.375 == precision(y_true, y_pred, "macro")
    assert 1 == precision(y_true, y_true, "macro")


def test_precision_macro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.8333 == approx(precision(y_true, y_pred, "macro"), rel=1e-3)
    assert 1 == precision(y_true, y_true, "macro")


def test_recall_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 5, 6, 7]
    assert 0.25 == recall(y_true, y_pred, "micro")


def test_recall_micro_2d_list():
    y_true = [[1, 2], [1, 2]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert 1 == recall(y_true, y_pred, "micro")
    assert 0.5 == recall(y_pred, y_true, "micro")


def test_recall_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 5, 6, 7])
    assert 0.25 == recall(y_true, y_pred, "micro")


def test_recall_micro_2d_np_array():
    y_true = np.array([[1, 2], [1, 2]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert 1 == recall(y_true, y_pred, "micro")
    assert 0.5 == recall(y_pred, y_true, "micro")


def test_recall_micro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.8333 == approx(recall(y_true, y_pred, "micro"), rel=1e-3)
    assert 1 == recall(y_true, y_true, "micro")


def test_recall_macro_1d_list():
    y_true = [1, 2, 3, 5]
    y_pred = [1, 5, 6, 7]
    assert 0.25 == recall(y_true, y_pred, "macro")


def test_recall_macro_2d_list():
    y_true = [[1, 2], [1, 2]]
    y_pred = [[1, 5, 6, 7], [1, 2, 3, 4]]
    assert 0.75 == recall(y_true, y_pred, "macro")
    assert 0.375 == recall(y_pred, y_true, "macro")


def test_recall_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 5])
    y_pred = np.array([1, 5, 6, 7])
    assert 0.25 == recall(y_true, y_pred, "macro")


def test_recall_macro_2d_np_array():
    y_true = np.array([[1, 2], [1, 2]])
    y_pred = np.array([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert 0.75 == recall(y_true, y_pred, "macro")
    assert 0.375 == recall(y_pred, y_true, "macro")


def test_recall_macro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.8333 == approx(recall(y_true, y_pred, "macro"), rel=1e-3)
    assert 1 == recall(y_true, y_true, "macro")


def test_f1_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 5, 6]
    assert 0.5 == f1(y_true, y_pred, "micro")


def test_f1_micro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert 0.5 == f1(y_true, y_pred, "micro")


def test_f1_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 5, 6])
    assert 0.5 == f1(y_true, y_pred, "micro")


def test_f1_micro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert 0.5 == f1(y_true, y_pred, "micro")


def test_f1_micro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.9090 == approx(f1(y_true, y_pred, "micro"), rel=1e-3)
    assert 1 == f1(y_true, y_true, "micro")


def test_f1_macro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 4]
    assert 1 == f1(y_true, y_pred, "macro")


def test_f1_macro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 5, 6], [1, 2, 3]]
    assert 0.4285714 == approx(f1(y_true, y_pred, "macro"))


def test_f1_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    assert 1 == f1(y_true, y_pred, "macro")


def test_f1_macro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 5, 6], [1, 2, 3]])
    assert 0.4285714 == approx(f1(y_true, y_pred, "macro"))


def test_f1_macro_3d_np_array():
    y_true = np.array(
        [
            [["human", "mermaid"], ["fish", "mermaid"]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    y_pred = np.array(
        [
            [["human", "mermaid"], ["", ""]],
            [["human", "minotaur"], ["bull", "minotaur"]],
        ]
    )
    assert 0.9 == approx(f1(y_true, y_pred, "macro"), rel=1e-3)
    assert 1 == f1(y_true, y_true, "macro")


def test_empty_levels_2d_list_1():
    y_true = [["2", "3"], ["1"], ["4", "5", "6"]]
    y_pred = [["1"], ["2", "3"], ["4", "5", "6"]]
    assert 0.5 == f1(y_true, y_pred)
    assert 1 == f1(y_true, y_true)


def test_empty_levels_2d_list_2():
    y_true = [["1"], ["2", "3"], ["4", "5", "6"]]
    y_pred = [["1"], ["2", "3"], ["4", "5", "6"]]
    assert 1 == f1(y_true, y_pred)
    assert 1 == f1(y_true, y_true)


@pytest.fixture
def uncertainty_data():
    prob = [
        np.array(
            [
                [0.88, 0.06, 0.06],
                [0.22, 0.48, 0.30],
                [0.33, 0.33, 0.34],
                [0.49, 0.40, 0.11],
                [0.23, 0.03, 0.74],
                [0.21, 0.67, 0.12],
                [0.34, 0.34, 0.32],
                [0.02, 0.77, 0.21],
                [0.44, 0.42, 0.14],
                [0.85, 0.13, 0.02],
            ]
        )
    ]

    assert_array_equal(np.sum(prob[0], axis=1), np.ones(len(prob[0])))

    y_pred = np.array([[0], [1], [2], [0], [2], [1], [0], [1], [0], [0]])
    y_true = np.array([[0], [2], [0], [0], [2], [1], [1], [1], [0], [0]])

    return prob, y_pred, y_true


@pytest.fixture
def uncertainty_data_multi_level():
    prob = [
        np.array([[0.88, 0.06, 0.06], [0.22, 0.48, 0.30], [0.33, 0.33, 0.34]]),
        np.array([[0.88, 0.06, 0.06], [0.22, 0.48, 0.30], [0.33, 0.33, 0.34]]),
    ]

    assert_array_equal(np.sum(prob[0], axis=1), np.ones(len(prob[0])))

    y_pred = np.array([[0, 3], [1, 4], [2, 5]])
    y_true = np.array([[0, 3], [2, 5], [0, 4]])

    return prob, y_pred, y_true


def test_local_brier_score(uncertainty_data):
    prob, _, y_true = uncertainty_data
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2]]
    classifier.separator_ = "::HiClass::Separator::"

    brier_score = _multiclass_brier_score(classifier, y_true, prob[0], level=0)
    assert math.isclose(brier_score, 0.34852, abs_tol=1e-4)


def test_brier_score_multi_level(uncertainty_data_multi_level):
    prob, _, y_true = uncertainty_data_multi_level
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"
    brier_score_avg = multiclass_brier_score(classifier, y_true, prob, agg="average")
    brier_score_sum = multiclass_brier_score(classifier, y_true, prob, agg="sum")
    brier_score_per_level = multiclass_brier_score(classifier, y_true, prob, agg=None)
    assert math.isclose(brier_score_avg, 0.48793, abs_tol=1e-4)
    assert math.isclose(brier_score_sum, 0.97586, abs_tol=1e-4)
    assert math.isclose(brier_score_per_level[0], 0.48793, abs_tol=1e-4)
    assert math.isclose(brier_score_per_level[1], 0.48793, abs_tol=1e-4)


def test_brier_score_single_level(uncertainty_data_multi_level):
    prob, _, y_true = uncertainty_data_multi_level
    prob = prob[1]
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"

    brier_score_1 = multiclass_brier_score(classifier, y_true, prob, level=1)
    brier_score_2 = multiclass_brier_score(classifier, y_true, [prob], level=1)
    assert math.isclose(brier_score_1, 0.48793, abs_tol=1e-4)
    assert math.isclose(brier_score_2, 0.48793, abs_tol=1e-4)


def test_local_log_loss(uncertainty_data):
    prob, _, y_true = uncertainty_data
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2]]
    classifier.separator_ = "::HiClass::Separator::"

    log_loss = _log_loss(classifier, y_true, prob[0], level=0)
    assert math.isclose(log_loss, 0.61790, abs_tol=1e-4)


def test_log_loss_multi_level(uncertainty_data_multi_level):
    prob, _, y_true = uncertainty_data_multi_level
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"
    log_loss_avg = log_loss(classifier, y_true, prob, agg="average")
    log_loss_sum = log_loss(classifier, y_true, prob, agg="sum")
    log_loss_per_level = log_loss(classifier, y_true, prob, agg=None)
    assert math.isclose(log_loss_avg, 0.81348, abs_tol=1e-4)
    assert math.isclose(log_loss_sum, 1.62697, abs_tol=1e-4)
    assert math.isclose(log_loss_per_level[0], 0.81348, abs_tol=1e-4)
    assert math.isclose(log_loss_per_level[1], 0.81348, abs_tol=1e-4)


def test_log_loss_single_level(uncertainty_data_multi_level):
    prob, _, y_true = uncertainty_data_multi_level
    prob = prob[1]
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"

    log_loss_1 = log_loss(classifier, y_true, prob, level=1)
    log_loss_2 = log_loss(classifier, y_true, [prob], level=1)
    assert math.isclose(log_loss_1, 0.48793, abs_tol=1e-4)
    assert math.isclose(log_loss_2, 0.48793, abs_tol=1e-4)


def test_local_expected_calibration_error(uncertainty_data):
    prob, y_pred, y_true = uncertainty_data
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2]]
    classifier.separator_ = "::HiClass::Separator::"

    ece = _expected_calibration_error(
        classifier, y_true, prob[0], y_pred, level=0, n_bins=3
    )
    assert math.isclose(ece, 0.118, abs_tol=1e-4)


def test_expected_calibration_error_multi_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"
    ece_avg = expected_calibration_error(
        classifier, y_true, prob, y_pred, agg="average"
    )
    ece_sum = expected_calibration_error(classifier, y_true, prob, y_pred, agg="sum")
    ece_per_level = expected_calibration_error(
        classifier, y_true, prob, y_pred, agg=None
    )
    assert math.isclose(ece_avg, 0.31333, abs_tol=1e-4)
    assert math.isclose(ece_sum, 0.62666, abs_tol=1e-4)
    assert math.isclose(ece_per_level[0], 0.31333, abs_tol=1e-4)
    assert math.isclose(ece_per_level[1], 0.31333, abs_tol=1e-4)


def test_expected_calibration_error_single_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    prob = prob[1]
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"
    ece_1 = expected_calibration_error(classifier, y_true, prob, y_pred, level=1)
    ece_2 = expected_calibration_error(classifier, y_true, [prob], y_pred, level=1)
    assert math.isclose(ece_1, 0.31333, abs_tol=1e-4)
    assert math.isclose(ece_2, 0.31333, abs_tol=1e-4)


def test_local_static_calibration_error(uncertainty_data):
    prob, y_pred, y_true = uncertainty_data
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2]]
    classifier.separator_ = "::HiClass::Separator::"

    sce = _static_calibration_error(
        classifier, y_true, prob[0], y_pred, level=0, n_bins=3
    )
    assert math.isclose(sce, 0.3889, abs_tol=1e-3)


def test_static_calibration_error_multi_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"

    sce_avg = static_calibration_error(classifier, y_true, prob, y_pred, agg="average")
    sce_sum = static_calibration_error(classifier, y_true, prob, y_pred, agg="sum")
    sce_per_level = static_calibration_error(classifier, y_true, prob, y_pred, agg=None)
    assert math.isclose(sce_avg, 0.44444, abs_tol=1e-4)
    assert math.isclose(sce_sum, 0.88888, abs_tol=1e-4)
    assert math.isclose(sce_per_level[0], 0.44444, abs_tol=1e-4)
    assert math.isclose(sce_per_level[1], 0.44444, abs_tol=1e-4)


def test_static_calibration_error_single_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    prob = prob[1]
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"
    sce_1 = static_calibration_error(classifier, y_true, prob, y_pred, level=1)
    sce_2 = static_calibration_error(classifier, y_true, [prob], y_pred, level=1)
    assert math.isclose(sce_1, 0.44444, abs_tol=1e-4)
    assert math.isclose(sce_2, 0.44444, abs_tol=1e-4)


def test_local_adaptive_calibration_error(uncertainty_data):
    prob, y_pred, y_true = uncertainty_data
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2]]
    classifier.separator_ = "::HiClass::Separator::"

    ace = _adaptive_calibration_error(
        classifier, y_true, prob[0], y_pred, level=0, n_ranges=3
    )
    assert math.isclose(ace, 0.44, abs_tol=1e-3)


def test_adaptive_calibration_error_multi_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"

    ace_avg = adaptive_calibration_error(
        classifier, y_true, prob, y_pred, agg="average"
    )
    ace_sum = adaptive_calibration_error(classifier, y_true, prob, y_pred, agg="sum")
    ace_per_level = adaptive_calibration_error(
        classifier, y_true, prob, y_pred, agg=None
    )
    assert math.isclose(ace_avg, 0.13333, abs_tol=1e-4)
    assert math.isclose(ace_sum, 0.26666, abs_tol=1e-4)
    assert math.isclose(ace_per_level[0], 0.13333, abs_tol=1e-4)
    assert math.isclose(ace_per_level[1], 0.13333, abs_tol=1e-4)


def test_adaptive_calibration_error_single_level(uncertainty_data_multi_level):
    prob, y_pred, y_true = uncertainty_data_multi_level
    prob = prob[1]
    obj = HierarchicalClassifier()
    classifier = Mock(spec=obj)
    classifier._disambiguate = obj._disambiguate
    classifier.classes_ = [[0, 1, 2], [3, 4, 5]]
    classifier.separator_ = "::HiClass::Separator::"

    ace_1 = adaptive_calibration_error(classifier, y_true, prob, y_pred, level=1)
    ace_2 = adaptive_calibration_error(classifier, y_true, [prob], y_pred, level=1)
    assert math.isclose(ace_1, 0.13333, abs_tol=1e-4)
    assert math.isclose(ace_2, 0.13333, abs_tol=1e-4)

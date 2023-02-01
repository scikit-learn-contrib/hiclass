import numpy as np
import pytest
from hiclass import metrics


def test_unmatched_lengths():
    y_true = np.array([[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8]], dtype=np.int32)
    y_pred = np.array([[1, 2, 3], [1, 2, 4]], dtype=np.int32)
    with pytest.raises(AssertionError):
        metrics.precision(y_true, y_pred)


def test_precision():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert metrics.precision(y_true, y_pred) == 0.5
    assert metrics.precision(y_true, y_true) == 1


def test_recall():
    y_true = np.array([[1, 2], [1, 2]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert metrics.recall(y_true, y_pred) == 1
    assert metrics.recall(y_pred, y_true) == 0.5


def test_f1():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert metrics.f1(y_true, y_pred) == 0.5


def test_empty_levels_1():
    y_true = [["2", "3"], ["1"], ["4", "5", "6"]]
    y_pred = [["1", "", ""], ["2", "3", ""], ["4", "5", "6"]]
    assert metrics.f1(y_true, y_pred) == 0.5
    assert metrics.f1(y_true, y_true) == 1


def test_empty_levels_2():
    y_true = [["1"], ["2", "3"], ["4", "5", "6"]]
    y_pred = [["1", "", ""], ["2", "3", ""], ["4", "5", "6"]]
    assert metrics.f1(y_true, y_pred) == 1
    assert metrics.f1(y_true, y_true) == 1

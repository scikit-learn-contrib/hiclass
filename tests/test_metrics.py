import numpy as np
import pandas as pd
import pytest
from pytest import approx

from hiclass.metrics import precision, recall, f1


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


def test_unmatched_lengths_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3])
    y_pred = pd.DataFrame([1, 2])
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_unmatched_lengths_2d_dataframe():
    y_true = pd.DataFrame([[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 8]])
    y_pred = pd.DataFrame([[1, 2, 3], [1, 2, 4]])
    with pytest.raises(AssertionError):
        precision(y_true, y_pred)


def test_precision_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 5, 6]
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_micro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 5, 6])
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_micro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_micro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 4])
    y_pred = pd.DataFrame([1, 2, 5, 6])
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_micro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = pd.DataFrame([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert precision(y_true, y_pred, "micro") == 0.5
    assert precision(y_true, y_true, "micro") == 1


def test_precision_macro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 5, 6, 7]
    assert precision(y_true, y_pred, "macro") == 0.25
    assert precision(y_true, y_true, "macro") == 1


def test_precision_macro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 5, 6, 7], [1, 2, 3, 4]]
    assert precision(y_true, y_pred, "macro") == 0.375
    assert precision(y_true, y_true, "macro") == 1


def test_precision_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 5, 6, 7])
    assert precision(y_true, y_pred, "macro") == 0.25
    assert precision(y_true, y_true, "macro") == 1


def test_precision_macro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert precision(y_true, y_pred, "macro") == 0.375
    assert precision(y_true, y_true, "macro") == 1


def test_precision_macro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 4])
    y_pred = pd.DataFrame([1, 5, 6, 7])
    assert precision(y_true, y_pred, "macro") == 0.25
    assert precision(y_true, y_true, "macro") == 1


def test_precision_macro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = pd.DataFrame([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert precision(y_true, y_pred, "macro") == 0.375
    assert precision(y_true, y_true, "macro") == 1


def test_recall_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 5, 6, 7]
    assert recall(y_true, y_pred, "micro") == 0.25


def test_recall_micro_2d_list():
    y_true = [[1, 2], [1, 2]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert recall(y_true, y_pred, "micro") == 1
    assert recall(y_pred, y_true, "micro") == 0.5


def test_recall_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 5, 6, 7])
    assert recall(y_true, y_pred, "micro") == 0.25


def test_recall_micro_2d_np_array():
    y_true = np.array([[1, 2], [1, 2]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert recall(y_true, y_pred, "micro") == 1
    assert recall(y_pred, y_true, "micro") == 0.5


def test_recall_micro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 5])
    y_pred = pd.DataFrame([1, 5, 6, 7])
    assert recall(y_true, y_pred, "micro") == 0.25


def test_recall_micro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2], [1, 2]])
    y_pred = pd.DataFrame([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert recall(y_true, y_pred, "micro") == 1
    assert recall(y_pred, y_true, "micro") == 0.5


def test_recall_macro_1d_list():
    y_true = [1, 2, 3, 5]
    y_pred = [1, 5, 6, 7]
    assert recall(y_true, y_pred, "macro") == 0.25


def test_recall_macro_2d_list():
    y_true = [[1, 2], [1, 2]]
    y_pred = [[1, 5, 6, 7], [1, 2, 3, 4]]
    assert recall(y_true, y_pred, "macro") == 0.75
    assert recall(y_pred, y_true, "macro") == 0.375


def test_recall_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 5])
    y_pred = np.array([1, 5, 6, 7])
    assert recall(y_true, y_pred, "macro") == 0.25


def test_recall_macro_2d_np_array():
    y_true = np.array([[1, 2], [1, 2]])
    y_pred = np.array([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert recall(y_true, y_pred, "macro") == 0.75
    assert recall(y_pred, y_true, "macro") == 0.375


def test_recall_macro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 5])
    y_pred = pd.DataFrame([1, 5, 6, 7])
    assert recall(y_true, y_pred, "macro") == 0.25


def test_recall_macro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2], [1, 2]])
    y_pred = pd.DataFrame([[1, 5, 6, 7], [1, 2, 3, 4]])
    assert recall(y_true, y_pred, "macro") == 0.75
    assert recall(y_pred, y_true, "macro") == 0.375


def test_f1_micro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 5, 6]
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_micro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 2, 5, 6], [1, 2, 3, 4]]
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_micro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 5, 6])
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_micro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_micro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 4])
    y_pred = pd.DataFrame([1, 2, 5, 6])
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_micro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = pd.DataFrame([[1, 2, 5, 6], [1, 2, 3, 4]])
    assert f1(y_true, y_pred, "micro") == 0.5


def test_f1_macro_1d_list():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 4]
    assert f1(y_true, y_pred, "macro") == 1


def test_f1_macro_2d_list():
    y_true = [[1, 2, 3, 4], [1, 2, 5, 6]]
    y_pred = [[1, 5, 6], [1, 2, 3]]
    assert 0.4285714 == approx(f1(y_true, y_pred, "macro"))


def test_f1_macro_1d_np_array():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    assert f1(y_true, y_pred, "macro") == 1


def test_f1_macro_2d_np_array():
    y_true = np.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = np.array([[1, 5, 6], [1, 2, 3]])
    assert 0.4285714 == approx(f1(y_true, y_pred, "macro"))


def test_f1_macro_1d_dataframe():
    y_true = pd.DataFrame([1, 2, 3, 4])
    y_pred = pd.DataFrame([1, 2, 3, 4])
    assert f1(y_true, y_pred, "macro") == 1


def test_f1_macro_2d_dataframe():
    y_true = pd.DataFrame([[1, 2, 3, 4], [1, 2, 5, 6]])
    y_pred = pd.DataFrame([[1, 5, 6], [1, 2, 3]])
    assert 0.4285714 == approx(f1(y_true, y_pred, "macro"))


def test_empty_levels_2d_list_1():
    y_true = [["2", "3"], ["1"], ["4", "5", "6"]]
    y_pred = [["1", "", ""], ["2", "3", ""], ["4", "5", "6"]]
    assert f1(y_true, y_pred) == 0.5
    assert f1(y_true, y_true) == 1


def test_empty_levels_2d_dataframe_1():
    y_true = pd.DataFrame([["2", "3"], ["1"], ["4", "5", "6"]]).fillna("")
    y_pred = pd.DataFrame([["1", "", ""], ["2", "3", ""], ["4", "5", "6"]])
    assert f1(y_true, y_pred) == 0.5
    assert f1(y_true, y_true) == 1


def test_empty_levels_2d_list_2():
    y_true = [["1"], ["2", "3"], ["4", "5", "6"]]
    y_pred = [["1", "", ""], ["2", "3", ""], ["4", "5", "6"]]
    assert f1(y_true, y_pred) == 1
    assert f1(y_true, y_true) == 1


def test_empty_levels_2d_dataframe_2():
    y_true = pd.DataFrame([["1"], ["2", "3"], ["4", "5", "6"]]).fillna("")
    y_pred = pd.DataFrame([["1", "", ""], ["2", "3", ""], ["4", "5", "6"]])
    assert f1(y_true, y_pred) == 1
    assert f1(y_true, y_true) == 1

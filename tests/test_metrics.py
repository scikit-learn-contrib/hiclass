import numpy as np
import pytest
from pytest import approx

from hiclass.metrics import precision, recall, f1


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

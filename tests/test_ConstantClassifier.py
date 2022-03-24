import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hiclass.ConstantClassifier import ConstantClassifier


def test_fit_1():
    X = [1, 2, 3]
    y = ["a", "a", "a"]
    classifier = ConstantClassifier()
    classifier.fit(X, y)
    assert classifier.classes_ == "a"


def test_fit_2():
    X = [1, 2, 3]
    y = ["a", "b", "c"]
    classifier = ConstantClassifier()
    with pytest.raises(ValueError):
        classifier.fit(X, y)


def test_predict_proba():
    X = np.array([1, 2, 3])
    classifier = ConstantClassifier()
    predict_proba = classifier.predict_proba(X)
    ground_truth = np.array([[1], [1], [1]])
    assert_array_equal(ground_truth, predict_proba)


def test_predict():
    X = np.array([1, 2, 3])
    classifier = ConstantClassifier()
    classifier.classes_ = "a"
    predictions = classifier.predict(X)
    ground_truth = np.array([["a"], ["a"], ["a"]])
    assert_array_equal(ground_truth, predictions)

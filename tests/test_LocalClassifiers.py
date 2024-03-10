import os
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pyfakefs.fake_filesystem_unittest import Patcher
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
    LocalClassifierPerParentNode,
)
from hiclass.ConstantClassifier import ConstantClassifier

classifiers = [
    LocalClassifierPerLevel,
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
]


@pytest.mark.parametrize("classifier", classifiers)
def test_fit_1_class(classifier):
    clf = classifier(local_classifier=LogisticRegression(), n_jobs=2)
    y = np.array([["1", "2"]])
    X = np.array([[1, 2]])
    ground_truth = np.array([["1", "2"]])
    clf.fit(X, y)
    prediction = clf.predict(X)
    assert_array_equal(ground_truth, prediction)


@pytest.fixture
def empty_levels():
    X = [
        [1],
        [2],
        [3],
    ]
    y = np.array(
        [
            ["1"],
            ["2", "2.1"],
            ["3", "3.1", "3.1.2"],
        ],
        dtype=object,
    )
    return X, y


@pytest.mark.parametrize("classifier", classifiers)
def test_empty_levels(empty_levels, classifier):
    clf = classifier()
    X, y = empty_levels
    clf.fit(X, y)
    predictions = clf.predict(X)
    ground_truth = [
        ["1", "", ""],
        ["2", "2.1", ""],
        ["3", "3.1", "3.1.2"],
    ]
    assert list(clf.hierarchy_.nodes) == [
        "1",
        "2",
        "2" + clf.separator_ + "2.1",
        "3",
        "3" + clf.separator_ + "3.1",
        "3" + clf.separator_ + "3.1" + clf.separator_ + "3.1.2",
        clf.root_,
    ]
    assert_array_equal(ground_truth, predictions)


@pytest.mark.parametrize("classifier", classifiers)
def test_fit_bert(classifier):
    bert = ConstantClassifier()
    clf = classifier(
        local_classifier=bert,
        bert=True,
    )
    X = ["Text 1", "Text 2"]
    y = ["a", "a"]
    clf.fit(X, y)
    check_is_fitted(clf)
    predictions = clf.predict(X)
    assert_array_equal(y, predictions)


@pytest.mark.parametrize("classifier", classifiers)
def test_knn(classifier):
    knn = KNeighborsClassifier(
        n_neighbors=2,
    )
    clf = classifier(
        local_classifier=knn,
    )
    y = np.array([["a", "b"], ["a", "c"]])
    X = np.array([[1, 2], [3, 4]])
    clf.fit(X, y)
    check_is_fitted(clf)
    # predictions = lcpn.predict(X)
    # assert_array_equal(y, predictions)


@pytest.mark.parametrize("classifier", classifiers)
def test_fit_multiple_dim_input(classifier):
    clf = classifier()
    X = np.random.rand(1, 275, 3)
    y = np.array([["a", "b", "c"]])
    clf.fit(X, y)
    check_is_fitted(clf)


@pytest.mark.parametrize("classifier", classifiers)
def test_predict_multiple_dim_input(classifier):
    clf = classifier()
    X = np.random.rand(1, 275, 3)
    y = np.array([["a", "b", "c"]])
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions is not None


@pytest.mark.parametrize("classifier", classifiers)
def test_tmp_dir(classifier):
    clf = classifier(tmp_dir=".")
    with Patcher() as patcher:
        x = np.array([[1, 2], [3, 4]])
        y = np.array([["a", "b"], ["c", "d"]])
        clf.fit(x, y)
        if isinstance(clf, LocalClassifierPerLevel):
            filename = "cfcd208495d565ef66e7dff9f98764da.sav"
            expected_name = 0
        else:
            filename = "0cc175b9c0f1b6a831c399e269772661.sav"
            expected_name = "a"
        assert patcher.fs.exists(filename)
        (name, classifier) = pickle.load(open(filename, "rb"))
        assert expected_name == name
        check_is_fitted(classifier)
        clf.fit(x, y)

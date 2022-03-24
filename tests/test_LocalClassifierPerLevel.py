from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from hiclass.Classifier import ConstantClassifier
from hiclass.LocalClassifierPerLevel import LocalClassifierPerLevel
from hiclass.data import PLACEHOLDER_LABEL_CAT, PLACEHOLDER_LABEL_NUMERIC

fixtures_loc = Path(__file__).parent / "fixtures"


class TestClassifier:
    """Predict the first class with `predict_value` , the second class with `1-predict_value`, the rest with 0."""

    def __init__(self, predict_value: float = 1):
        self.X = None
        self.Y = None
        self.predicted_x = None
        self.prediction = None
        self.p = predict_value
        super().__init__()

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_classes = len(np.unique(Y))
        self.prediction = [0] * num_classes
        self.prediction[0] = self.p
        if num_classes > 1:
            self.prediction[1] = 1 - self.p

    def predict_proba(self, X):
        self.predicted_x = X
        return np.vstack([self.prediction] * len(X))

    def clone(self):
        return self.__class__(self.p)


class TestClassifierGranular(TestClassifier):
    def predict_proba(self, X):
        self.predicted_x = X
        if len(X) > 1:
            return np.vstack(
                ([self.prediction] * (len(X) - 1), [self.prediction[::-1]])
            )
        else:
            return np.array([self.prediction[::-1]])


class ErrorClassifier:
    def fit(self, X, Y):
        raise ValueError("TestMessage")

    def clone(self):
        return ErrorClassifier()


@pytest.fixture()
def data():
    return np.array([[1, 2], [2, 3], [3, 4], [4, 5]])


@pytest.fixture
def labels(request):
    if request.param == "integer":
        return np.array(
            [[1, 2, 3], [1, 2, 4], [1, 2, PLACEHOLDER_LABEL_NUMERIC], [1, 5, 6]]
        )
    if request.param == "categorical":
        return np.array(
            [
                ["1", "2", "3"],
                ["1", "2", "4"],
                ["1", "2", PLACEHOLDER_LABEL_CAT],
                ["1", "5", "6"],
            ]
        )
    raise ValueError("invalid internal test config")


@pytest.fixture()
def hierarchy(request):
    if request.param == "integer":
        return nx.from_edgelist(
            [(1, 2), (2, 3), (2, 4), (1, 5), (5, 6)], create_using=nx.DiGraph
        )
    if request.param == "categorical":
        return nx.from_edgelist(
            [("1", "2"), ("2", "3"), ("2", "4"), ("1", "5"), ("5", "6")],
            create_using=nx.DiGraph,
        )
    if request.param is None:
        return None
    raise ValueError("invalid internal test config")


def param_combination():
    return [
        ("integer", -1, "integer"),
        ("integer", -1, None),
        ("categorical", "", "categorical"),
        ("categorical", "", None),
    ]


def param_combination_hierarchy():
    return [("integer", -1, "integer"), ("categorical", "", "categorical")]


def test_predict_not_fitted(data):
    lcpl = LocalClassifierPerLevel()
    # TODO raise proper error here and insert it in test
    with pytest.raises(Exception):
        lcpl.predict(data)


@pytest.mark.parametrize(
    "labels, placeholder, num_jobs",
    [
        ("integer", -1, 0),
        ("categorical", "", 0),
        ("integer", -1, 2),
        ("categorical", "", 2),
    ],
    indirect=["labels"],
)
def test_fit_all_initialized(data, labels, placeholder, num_jobs):
    lcpl = LocalClassifierPerLevel(local_classifier=TestClassifier(), n_jobs=num_jobs)
    lcpl.fit(data, labels, placeholder_label=placeholder)
    with pytest.raises(IndexError):
        lcpl.get_classifier(-1)
    for i in range(3):
        lcpl.get_classifier(i)
    with pytest.raises(IndexError):
        lcpl.get_classifier(4)


@pytest.mark.parametrize(
    "labels, placeholder, hierarchy",
    param_combination(),
    indirect=["labels", "hierarchy"],
)
def test_fit_proper_data(data, labels, placeholder, hierarchy):
    lcpl = LocalClassifierPerLevel(
        local_classifier=TestClassifier(1), hierarchy=hierarchy
    )
    lcpl.fit(data, labels, placeholder_label=placeholder, replace_classifiers=False)
    assert np.array_equal(lcpl.get_classifier(1).X, data)
    assert list(lcpl.get_classifier(1).Y) == [0, 0, 0, 1]

    assert np.array_equal(lcpl.get_classifier(2).X, np.vstack((data[:2], data[3:])))
    assert list(lcpl.get_classifier(2).Y) == [0, 1, 2]


def test_fit_classifier_cannot_be_fit(data):
    labels = np.array([[1, 2], [1, 3], [1, 2], [1, 2]])
    lcpl = LocalClassifierPerLevel(local_classifier=ErrorClassifier())
    with pytest.raises(ValueError, match="TestMessage"):
        lcpl.fit(data, labels)


def test_fit_classifier_labels_no_root(data):
    labels = np.array([[2], [3], [2], [2]])
    lcpl = LocalClassifierPerLevel(local_classifier=TestClassifier(1))
    lcpl.fit(data, labels)
    lcpl.predict(data)


def test_fit_mixed_labels_error(data):
    labels = np.array([[1], [3], [2], [2]])
    hierarchy = nx.from_edgelist([(1, 2), (1, 3)], create_using=nx.DiGraph)
    lcpl = LocalClassifierPerLevel(
        local_classifier=TestClassifier(1), hierarchy=hierarchy
    )
    with pytest.raises(ValueError):
        lcpl.fit(data, labels)


def test_string_data_policy_proper_initialization():
    lcpl = LocalClassifierPerLevel(local_classifier=TestClassifier(1))
    lcpl.fit(
        np.array([[1, 2], [2, 3]]),
        np.array([[1, 2], [1, 3]]),
        replace_classifiers=False,
    )
    assert list(lcpl.get_classifier(1).Y) == [0, 1]


def test_predict_simple(data):
    labels = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 2, 8]])
    lcpl = LocalClassifierPerLevel(local_classifier=RandomForestClassifier())
    lcpl.fit(data, labels, replace_classifiers=False)
    result = lcpl.predict(data, threshold=0)
    assert np.array_equal(result, [[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 2, 8]])


@pytest.mark.parametrize(
    "labels, placeholder, hierarchy",
    param_combination(),
    indirect=["labels", "hierarchy"],
)
def test_predict_threshold(data, labels, placeholder, hierarchy):
    lcpl = LocalClassifierPerLevel(
        local_classifier=TestClassifier(0.6), hierarchy=hierarchy
    )
    lcpl.fit(
        data, labels, placeholder_label=placeholder, replace_classifiers=False
    )  # fit to initialize hierarchy
    lcpl.set_classifier(1, TestClassifier(0.8))
    lcpl.fit(data, labels, placeholder_label=placeholder, replace_classifiers=False)
    result = lcpl.predict(data, threshold=0)
    assert np.array_equal(result, [[labels[0][0], labels[0][1], labels[0][2]]] * 4)
    result = lcpl.predict(data, threshold=0.9)
    assert np.array_equal(result, [[labels[0][0], placeholder, placeholder]] * 4)
    result = lcpl.predict(data, threshold=0.7)
    assert np.array_equal(result, [[labels[0][0], labels[0][1], placeholder]] * 4)


@pytest.mark.parametrize(
    "labels, placeholder, hierarchy",
    param_combination_hierarchy(),
    indirect=["labels", "hierarchy"],
)
def test_predict_discard_invalid_combinations(data, labels, placeholder, hierarchy):
    expected_result = [[labels[3][0], labels[3][1], placeholder]] * 3 + [labels[0]]
    lcpl = LocalClassifierPerLevel(
        local_classifier=TestClassifier(0.6), hierarchy=hierarchy
    )
    lcpl.set_classifier(1, TestClassifierGranular(0.1))
    lcpl.fit(data, labels, placeholder_label=placeholder)
    result = lcpl.predict(data, 0.5)
    assert np.array_equal(result, expected_result)


def test_fit_only_one_class(data):
    labels = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])
    lcpl = LocalClassifierPerLevel(local_classifier=LogisticRegression())
    lcpl.fit(data, labels)
    result = lcpl.predict(data)
    assert np.array_equal(result, [[1, 2]] * 4)


@pytest.mark.parametrize("input_labels", [np.array([["1", "2"]]), np.array([[1, 2]])])
def test_replace_classifiers(input_labels):
    lcpl = LocalClassifierPerLevel(local_classifier=TestClassifier(0.6))
    lcpl.fit(np.array([[1]]), input_labels, replace_classifiers=True)
    assert lcpl.get_classifier(0).__class__ == ConstantClassifier


def test_predict_strings_cut_off():
    data = np.array([[1], [2]])
    labels = np.array([["t_1", "t_2", "t_3"], ["t_1", "t_2", "t_3"]])
    lcpl = LocalClassifierPerLevel(local_classifier=TestClassifier(0.6))
    lcpl.fit(data, labels)
    result = lcpl.predict(data)
    assert np.array_equal(result, [["t_1", "t_2", "t_3"], ["t_1", "t_2", "t_3"]])


@pytest.mark.parametrize("input_labels", [np.array([["1", "2"]]), np.array([[1, 2]])])
def test_set_classifier(input_labels):
    graph = nx.from_edgelist(input_labels, create_using=nx.DiGraph)
    lcpl = LocalClassifierPerLevel(
        local_classifier=TestClassifier(0.6), hierarchy=graph
    )
    lcpl.set_classifier(0, RandomForestClassifier())
    assert lcpl.get_classifier(0).__class__ == RandomForestClassifier


def test_predict_proba_default(data):
    labels = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 2, 8]])
    lcpl = LocalClassifierPerLevel(
        local_classifier=RandomForestClassifier(random_state=0)
    )
    lcpl.fit(data, labels, replace_classifiers=False)
    result = lcpl.predict_proba(data)
    ground_truth = [
        np.array([[1.0], [1.0], [1.0], [1.0]]),
        np.array(
            [
                [0.64, 0.32, 0.04],
                [0.27, 0.69, 0.04],
                [0.07, 0.3, 0.63],
                [0.75, 0.08, 0.17],
            ]
        ),
        np.array(
            [
                [0.64, 0.0, 0.32, 0.04],
                [0.27, 0.0, 0.69, 0.04],
                [0.01, 0.06, 0.3, 0.63],
                [0.01, 0.74, 0.08, 0.17],
            ]
        ),
    ]
    for i in range(max(len(result), len(ground_truth))):
        assert np.array_equal(result[i], ground_truth[i])


def test_algorithm_not_found(data):
    labels = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 2, 8]])
    lcpl = LocalClassifierPerLevel(local_classifier=RandomForestClassifier())
    lcpl.fit(data, labels, replace_classifiers=False)
    with pytest.raises(KeyError):
        lcpl.predict_proba(data, "unknown")


def test_multiple_roots():
    X = np.array([[1], [2], [3], [4], [5], [6], [7]])
    Y = np.array(
        [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["g", "h"],
            ["i", "j"],
            ["k", "l"],
            ["m", "n"],
        ]
    )
    lcpp = LocalClassifierPerLevel(local_classifier=RandomForestClassifier())
    lcpp.fit(X, Y)
    prediction = lcpp.predict(X)
    assert np.array_equal(prediction, Y)


def test_repeated_taxonomy():
    X = np.array([[1], [20]])
    Y = np.array([["a", "b", "c"], ["d", "b", "e"]])
    lcpp = LocalClassifierPerLevel(
        local_classifier=RandomForestClassifier(), unique_taxonomy=False
    )
    lcpp.fit(X, Y)
    prediction = lcpp.predict(X)
    assert np.array_equal(prediction, Y)

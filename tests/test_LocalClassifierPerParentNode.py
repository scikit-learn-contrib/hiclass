import logging
import tempfile

import networkx as nx
import numpy as np
import pytest
from bert_sklearn import BertClassifier
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from hiclass import LocalClassifierPerParentNode
from hiclass._calibration.Calibrator import _Calibrator
from hiclass.HierarchicalClassifier import make_leveled


@parametrize_with_checks([LocalClassifierPerParentNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerParentNode(
        local_classifier=LogisticRegression(), calibration_method="ivap"
    )
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "f")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"], ["b", "d"], ["c", "f"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.y_cal = np.array([["a", "b"], ["a", "c"], ["b", "d"], ["c", "f"]])
    digraph.X_cal = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.sample_weight_ = None
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node in ["a", "b", "c"]:
            assert isinstance(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"],
                LogisticRegression,
            )
        else:
            with pytest.raises(KeyError):
                isinstance(
                    digraph_logistic_regression.hierarchy_.nodes[node]["classifier"],
                    LogisticRegression,
                )


def test_initialize_local_calibrators(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    digraph_logistic_regression._initialize_local_calibrators()

    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node in ["a", "b", "c"]:
            assert isinstance(
                digraph_logistic_regression.hierarchy_.nodes[node]["calibrator"],
                _Calibrator,
            )
        else:
            with pytest.raises(KeyError):
                isinstance(
                    digraph_logistic_regression.hierarchy_.nodes[node]["classifier"],
                    _Calibrator,
                )


def test_fit_digraph(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["d", "f"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


def test_calibrate_digraph(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)

    calibrators = {
        "a": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"]
            )
        },
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, calibrators)
    digraph_logistic_regression._calibrate_digraph(local_mode=True)

    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["calibrator"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c", "d", "f"]:
        with pytest.raises((TypeError, KeyError)):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["calibrator"]
            )
    assert 1


def test_fit_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["d", "f"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


def test_calibrate_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)

    calibrators = {
        "a": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"]
            )
        },
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, calibrators)
    digraph_logistic_regression._calibrate_digraph(local_mode=True, use_joblib=True)

    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["calibrator"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c", "d", "f"]:
        with pytest.raises((TypeError, KeyError)):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["calibrator"]
            )
    assert 1


@pytest.fixture
def digraph_2d():
    classifier = LocalClassifierPerParentNode()
    classifier.y_ = np.array([["a", "b", "c"], ["d", "e", "f"]])
    classifier.hierarchy_ = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")])
    classifier.logger_ = logging.getLogger("HC")
    classifier.edge_list = tempfile.TemporaryFile()
    classifier.separator_ = "::HiClass::Separator::"
    return classifier


def test_get_parents(digraph_2d):
    ground_truth = np.array(["a", "b", "d", "e"])
    nodes = digraph_2d._get_parents()
    assert_array_equal(ground_truth, nodes)


@pytest.fixture
def x_and_y_arrays():
    graph = LocalClassifierPerParentNode()
    graph.X_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    graph.y_ = np.array([["a", "b", "c"], ["a", "e", "f"], ["d", "g", "h"]])
    graph.hierarchy_ = nx.DiGraph(
        [("a", "b"), ("b", "c"), ("a", "e"), ("e", "f"), ("d", "g"), ("g", "h")]
    )
    graph.root_ = "r"
    graph.sample_weight_ = None
    return graph


def test_get_successors(x_and_y_arrays):
    x, y, weights = x_and_y_arrays._get_successors("a")
    assert_array_equal(x_and_y_arrays.X_[0:2], x)
    assert_array_equal(["b", "e"], y)
    assert weights is None
    x, y, weights = x_and_y_arrays._get_successors("d")
    assert_array_equal([x_and_y_arrays.X_[-1]], x)
    assert_array_equal(["g"], y)
    assert weights is None
    x, y, weights = x_and_y_arrays._get_successors("b")
    assert_array_equal([x_and_y_arrays.X_[0]], x)
    assert_array_equal(["c"], y)
    assert weights is None


@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerParentNode(
        local_classifier=LogisticRegression(),
        calibration_method="ivap",
        probability_combiner="geometric",
        return_all_probabilities=True,
    )

    digraph.separator_ = "::HiClass::Separator::"
    digraph.hierarchy_ = nx.DiGraph(
        [
            ("r", "1"),
            ("r", "2"),
            ("1", "1" + digraph.separator_ + "1.1"),
            ("1", "1" + digraph.separator_ + "1.2"),
            ("2", "2" + digraph.separator_ + "2.1"),
            ("2", "2" + digraph.separator_ + "2.2"),
        ]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.max_levels_ = 2
    digraph.max_level_dimensions_ = np.array([2, 4])
    digraph.dtype_ = "<U30"
    digraph.root_ = "r"

    # for predict_proba
    tmp_labels = digraph._disambiguate(make_leveled(digraph.y_))
    # digraph.y_ = tmp_labels
    digraph.global_classes_ = [
        np.unique(tmp_labels[:, level]).astype("str")
        for level in range(tmp_labels.shape[1])
    ]
    digraph.global_class_to_index_mapping_ = [
        {
            digraph.global_classes_[level][index]: index
            for index in range(len(digraph.global_classes_[level]))
        }
        for level in range(tmp_labels.shape[1])
    ]

    classes_ = [digraph.global_classes_[0]]
    for level in range(1, digraph.max_levels_):
        classes_.append(
            np.sort(
                np.unique(
                    [
                        label.split(digraph.separator_)[level]
                        for label in digraph.global_classes_[level]
                    ]
                )
            )
        )
    digraph.classes_ = classes_
    digraph.class_to_index_mapping_ = [
        {local_labels[index]: index for index in range(len(local_labels))}
        for local_labels in classes_
    ]

    classifiers = {
        "r": {"classifier": LogisticRegression()},
        "1": {"classifier": LogisticRegression()},
        "2": {"classifier": LogisticRegression()},
    }
    classifiers["r"]["classifier"].fit(digraph.X_, ["1", "1", "2", "2"])
    classifiers["1"]["classifier"].fit(
        digraph.X_[:2],
        ["1" + digraph.separator_ + "1.1", "1" + digraph.separator_ + "1.2"],
    )
    classifiers["2"]["classifier"].fit(
        digraph.X_[2:],
        ["2" + digraph.separator_ + "2.1", "2" + digraph.separator_ + "2.2"],
    )
    nx.set_node_attributes(digraph.hierarchy_, classifiers)
    return digraph


def test_predict(fitted_logistic_regression):
    ground_truth = np.array([["2", "2.2"], ["2", "2.1"], ["1", "1.2"], ["1", "1.1"]])
    prediction = fitted_logistic_regression.predict([[7, 8], [5, 6], [3, 4], [1, 2]])
    assert_array_equal(ground_truth, prediction)


def test_predict_sparse(fitted_logistic_regression):
    ground_truth = np.array([["2", "2.2"], ["2", "2.1"], ["1", "1.2"], ["1", "1.1"]])
    prediction = fitted_logistic_regression.predict(
        csr_matrix([[7, 8], [5, 6], [3, 4], [1, 2]])
    )
    assert_array_equal(ground_truth, prediction)


def test_predict_proba(fitted_logistic_regression):
    proba = fitted_logistic_regression.predict_proba([[7, 8], [5, 6], [3, 4], [1, 2]])
    assert len(proba) == 2
    assert proba[0].shape == (4, 2)
    assert proba[1].shape == (4, 4)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )


def test_predict_proba_sparse(fitted_logistic_regression):
    proba = fitted_logistic_regression.predict_proba(
        csr_matrix([[7, 8], [5, 6], [3, 4], [1, 2]])
    )
    assert len(proba) == 2
    assert proba[0].shape == (4, 2)
    assert proba[1].shape == (4, 4)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )


def test_fit_predict():
    lcppn = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcppn.fit(x, y)
    predictions = lcppn.predict(x)
    assert_array_equal(y, predictions)


def test_fit_calibrate_predict_proba():
    lcppn = LocalClassifierPerParentNode(
        local_classifier=LogisticRegression(),
        calibration_method="ivap",
        return_all_probabilities=True,
    )
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcppn.fit(x, y)
    predictions = lcppn.predict(x)
    proba = lcppn.predict_proba(x)
    assert_array_equal(y, predictions)

    assert len(proba) == 2
    assert proba[0].shape == (2, 2)
    assert proba[1].shape == (2, 2)
    assert_array_almost_equal(
        np.sum(proba[0], axis=1), np.ones(len(proba[0])), decimal=10
    )
    assert_array_almost_equal(
        np.sum(proba[1], axis=1), np.ones(len(proba[1])), decimal=10
    )


def test_fit_calibrate_predict_predict_proba_bert():
    classifier = LocalClassifierPerParentNode(
        local_classifier=LogisticRegression(),
        return_all_probabilities=True,
        calibration_method="ivap",
        probability_combiner="geometric",
    )

    classifier.logger_ = logging.getLogger("HC")
    classifier.bert = True
    x = [[0, 1], [2, 3]]
    y = [["a", "b"], ["c", "d"]]
    sample_weight = None
    classifier.fit(x, y, sample_weight)
    classifier.calibrate(x, y)
    classifier.predict(x)
    classifier.predict_proba(x)


# Note: bert only works with the local classifier per parent node
# It does not have the attribute classes_, which are necessary
# for the local classifiers per level and per node
def test_fit_bert():
    bert = BertClassifier()
    clf = LocalClassifierPerParentNode(
        local_classifier=bert,
        bert=True,
    )
    x = ["Batman", "rorschach"]
    y = [
        ["Action", "The Dark Night"],
        ["Action", "Watchmen"],
    ]
    clf.fit(x, y)
    check_is_fitted(clf)
    predictions = clf.predict(x)
    assert_array_equal(y, predictions)


def test_bert_unleveled():
    clf = LocalClassifierPerParentNode(
        local_classifier=BertClassifier(),
        bert=True,
    )
    x = ["Batman", "Jaws"]
    y = [["Action", "The Dark Night"], ["Thriller"]]
    ground_truth = [["Action", "The Dark Night"], ["Action", "The Dark Night"]]
    clf.fit(x, y)
    check_is_fitted(clf)
    predictions = clf.predict(x)
    assert_array_equal(ground_truth, predictions)

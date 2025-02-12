import logging

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted
from hiclass._calibration.Calibrator import _Calibrator
from hiclass.HierarchicalClassifier import make_leveled

from hiclass import LocalClassifierPerNode

from hiclass.BinaryPolicy import ExclusivePolicy


@parametrize_with_checks([LocalClassifierPerNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_with_policy():
    digraph = LocalClassifierPerNode(binary_policy="exclusive")
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.X_ = np.array([[1, 2]])
    digraph.y_ = np.array([["a", "b"]])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.sample_weight_ = None
    return digraph


def test_initialize_binary_policy(digraph_with_policy):
    digraph_with_policy.binary_policy_ = digraph_with_policy._initialize_binary_policy()
    assert isinstance(digraph_with_policy.binary_policy_, ExclusivePolicy)


@pytest.fixture
def digraph_with_unknown_policy():
    digraph = LocalClassifierPerNode(binary_policy="unknown")
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
    return digraph


def test_initialize_unknown_binary_policy(digraph_with_unknown_policy):
    with pytest.raises(KeyError):
        digraph_with_unknown_policy.binary_policy_ = (
            digraph_with_unknown_policy._initialize_binary_policy()
        )
        digraph_with_unknown_policy._initialize_binary_policy()


@pytest.fixture
def digraph_with_object_policy():
    digraph = LocalClassifierPerNode(binary_policy=ExclusivePolicy)
    digraph.hierarchy_ = nx.DiGraph([("a", "b")])
    digraph.y_ = np.array(["a", "b"])
    digraph.logger_ = logging.getLogger("LCPN")
    return digraph


def test_initialize_object_binary_policy(digraph_with_object_policy):
    with pytest.raises(ValueError):
        digraph_with_object_policy._initialize_binary_policy()


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerNode(
        local_classifier=LogisticRegression(), calibration_method="ivap"
    )
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.y_cal = np.array([["a", "b"], ["a", "c"]])
    digraph.X_cal = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.binary_policy_ = ExclusivePolicy(digraph.hierarchy_, digraph.X_, digraph.y_)
    digraph.sample_weight_ = None
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node != digraph_logistic_regression.root_:
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
        if node != digraph_logistic_regression.root_:
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
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)
    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_calibrate_digraph(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)

    calibrators = {
        "b": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["b"]["classifier"]
            )
        },
        "c": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["c"]["classifier"]
            )
        },
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, calibrators)
    digraph_logistic_regression._calibrate_digraph(local_mode=True)

    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["calibrator"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["calibrator"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_fit_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)
    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_calibrate_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "b": {"classifier": LogisticRegression()},
        "c": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)

    calibrators = {
        "b": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["b"]["classifier"]
            )
        },
        "c": {
            "calibrator": _Calibrator(
                digraph_logistic_regression.hierarchy_.nodes["c"]["classifier"]
            )
        },
    }
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, calibrators)
    digraph_logistic_regression._calibrate_digraph(local_mode=True, use_joblib=True)

    with pytest.raises(KeyError):
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["calibrator"])
    for node in ["b", "c"]:
        try:
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["calibrator"]
            )
        except NotFittedError as e:
            pytest.fail(repr(e))
    assert 1


def test_clean_up(digraph_logistic_regression):
    digraph_logistic_regression._clean_up()
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.X_ is None
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.y_ is None
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.binary_policy_ is None
    with pytest.raises(AttributeError):
        assert digraph_logistic_regression.cal_binary_policy_ is None


@pytest.fixture
def fitted_logistic_regression():
    digraph = LocalClassifierPerNode(
        local_classifier=LogisticRegression(),
        return_all_probabilities=True,
        calibration_method="ivap",
        probability_combiner="geometric",
    )

    digraph.separator_ = "::HiClass::Separator::"
    # digraph.hierarchy_ = nx.DiGraph(
    #    [("r", "1"), ("r", "2"), ("1", "1.1"), ("1", "1.2"), ("2", "2.1"), ("2", "2.2")]
    # )

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
    digraph.logger_ = logging.getLogger("LCPN")
    digraph.max_levels_ = 2
    digraph.dtype_ = "<U30"
    digraph.root_ = "r"

    # for predict_proba
    tmp_labels = digraph._disambiguate(make_leveled(digraph.y_))
    # digraph.y_ = tmp_labels
    digraph.max_level_dimensions_ = np.array(
        [len(np.unique(tmp_labels[:, level])) for level in range(tmp_labels.shape[1])]
    )
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
        key: {"classifier": LogisticRegression()}
        for key in np.hstack([digraph.global_classes_[0], digraph.global_classes_[1]])
    }

    classifiers["1"]["classifier"].fit(digraph.X_, [1, 1, 0, 0])
    classifiers["1" + digraph.separator_ + "1.1"]["classifier"].fit(
        digraph.X_, [1, 0, 0, 0]
    )
    classifiers["1" + digraph.separator_ + "1.2"]["classifier"].fit(
        digraph.X_, [0, 1, 0, 0]
    )
    classifiers["2"]["classifier"].fit(digraph.X_, [0, 0, 1, 1])
    classifiers["2" + digraph.separator_ + "2.1"]["classifier"].fit(
        digraph.X_, [0, 0, 1, 0]
    )
    classifiers["2" + digraph.separator_ + "2.2"]["classifier"].fit(
        digraph.X_, [0, 0, 0, 1]
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
    lcpn = LocalClassifierPerNode(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcpn.fit(x, y)
    predictions = lcpn.predict(x)
    assert_array_equal(y, predictions)


def test_fit_calibrate_predict_proba():
    lcpn = LocalClassifierPerNode(
        local_classifier=LogisticRegression(),
        calibration_method="ivap",
        return_all_probabilities=True,
    )
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcpn.fit(x, y)
    lcpn.calibrate(x, y)
    predictions = lcpn.predict(x)
    proba = lcpn.predict_proba(x)
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

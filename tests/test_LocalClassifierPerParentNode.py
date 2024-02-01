import logging
import tempfile

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from hiclass.Explainer import Explainer
from hiclass import LocalClassifierPerParentNode

try:
    import shap
except ImportError:
    shap_installed = False
else:
    shap_installed = True


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@parametrize_with_checks([LocalClassifierPerParentNode()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def digraph_logistic_regression():
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph([("a", "b"), ("a", "c")])
    digraph.y_ = np.array([["a", "b"], ["a", "c"]])
    digraph.X_ = np.array([[1, 2], [3, 4]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.root_ = "a"
    digraph.separator_ = "::HiClass::Separator::"
    digraph.sample_weight_ = None
    return digraph


def test_initialize_local_classifiers(digraph_logistic_regression):
    digraph_logistic_regression._initialize_local_classifiers()
    for node in digraph_logistic_regression.hierarchy_.nodes:
        if node == digraph_logistic_regression.root_:
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


def test_fit_digraph(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True)
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
            )
    assert 1


def test_fit_digraph_joblib_multiprocessing(digraph_logistic_regression):
    classifiers = {
        "a": {"classifier": LogisticRegression()},
    }
    digraph_logistic_regression.n_jobs = 2
    nx.set_node_attributes(digraph_logistic_regression.hierarchy_, classifiers)
    digraph_logistic_regression._fit_digraph(local_mode=True, use_joblib=True)
    try:
        check_is_fitted(digraph_logistic_regression.hierarchy_.nodes["a"]["classifier"])
    except NotFittedError as e:
        pytest.fail(repr(e))
    for node in ["b", "c"]:
        with pytest.raises(KeyError):
            check_is_fitted(
                digraph_logistic_regression.hierarchy_.nodes[node]["classifier"]
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
    digraph = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    digraph.hierarchy_ = nx.DiGraph(
        [
            ("r", "1"),
            ("r", "2"),
            ("1", "1.1"),
            ("1", "1.2"),
            ("2", "2.1"),
            ("2", "2.2"),
        ]
    )
    digraph.y_ = np.array([["1", "1.1"], ["1", "1.2"], ["2", "2.1"], ["2", "2.2"]])
    digraph.X_ = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    digraph.logger_ = logging.getLogger("LCPPN")
    digraph.max_levels_ = 2
    digraph.dtype_ = "<U3"
    digraph.root_ = "r"
    digraph.separator_ = "::HiClass::Separator::"
    classifiers = {
        "r": {"classifier": LogisticRegression()},
        "1": {"classifier": LogisticRegression()},
        "2": {"classifier": LogisticRegression()},
    }
    classifiers["r"]["classifier"].fit(digraph.X_, ["1", "1", "2", "2"])
    classifiers["1"]["classifier"].fit(digraph.X_[:2], ["1.1", "1.2"])
    classifiers["2"]["classifier"].fit(digraph.X_[2:], ["2.1", "2.2"])
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


def test_fit_predict():
    lcppn = LocalClassifierPerParentNode(local_classifier=LogisticRegression())
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    lcppn.fit(x, y)
    predictions = lcppn.predict(x)
    assert_array_equal(y, predictions)


@pytest.fixture
def explainer_data():
    x_train = np.random.randn(4, 3)
    y_train = np.array(
        [["a", "b", "d"], ["a", "b", "e"], ["a", "c", "f"], ["a", "c", "g"]]
    )
    x_test = np.random.randn(1, 3)

    return x_train, x_test, y_train


def test_explainer_tree(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain_with_dict(x_test, traverse_prediction=False)

    for key, val in shap_dict.items():
        # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
        model = lcppn.hierarchy_.nodes[key]["classifier"]
        assert shap_dict[key].shape == (
            len(model.classes_),
            x_test.shape[0],
            x_test.shape[1],
        )


def test_explainer_tree_traversal(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)
    print(lcppn.get_predict_proba(x_test))

    # explainer = Explainer(lcppn, data=x_train, mode="tree")
    # shap_dict = explainer.explain_traversed_nodes(x_test)
    # print(shap_dict)
    #
    # for key, val in shap_dict.items():
    #     # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
    #     model = lcppn.hierarchy_.nodes[key]["classifier"]
    #     assert shap_dict[key].shape == (
    #         len(model.classes_),
    #         x_test.shape[0],
    #         x_test.shape[1],
    #     )


# TODO: Add new test cases with hierarchies without root nodes
def test_explainer_linear(explainer_data):
    logreg = LogisticRegression()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=logreg,
    )

    x_train, x_test, y_train = explainer_data
    lcppn.fit(x_train, y_train)

    lcppn.predict(x_test)
    explainer = Explainer(lcppn, data=x_train, mode="linear")
    shap_dict = explainer.explain_with_dict(x_test, traverse_prediction=False)

    for key, val in shap_dict.items():
        # Assert on shapes of shap values, must match (num_samples, num_features) Note: Logistic regression is based
        # on sigmoid and not softmax, hence there are no separate predictions for each target class
        assert shap_dict[key].shape == x_test.shape


@pytest.fixture
def explainer_data_no_root():
    x_train = np.random.randn(6, 3)
    y_train = np.array(
        [
            ["a", "b", "c"],
            ["x", "y", "z"],
            ["a", "b", "c"],
            ["x", "y", "z"],
            ["a", "b", "c"],
            ["x", "y", "z"],
        ]
    )
    x_test = np.random.randn(1, 3)
    return x_train, x_test, y_train


def test_explainer_tree_no_root(explainer_data_no_root):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data_no_root

    lcppn.fit(x_train, y_train)

    lcppn.predict(x_test)
    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain_with_dict(x_test, traverse_prediction=True)

    for key, val in shap_dict.items():
        # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
        model = lcppn.hierarchy_.nodes[key]["classifier"]
        assert shap_dict[key].shape == (
            len(model.classes_),
            x_test.shape[0],
            x_test.shape[1],
        )


def test_predict_proba(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)

    pred_proba_dict = lcppn.get_predict_proba(x_test)
    assert pred_proba_dict is not None

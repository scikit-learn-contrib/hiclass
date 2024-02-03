import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
    Explainer,
)

try:
    import shap
except ImportError:
    shap_installed = False
else:
    shap_installed = True

classifiers = [
    LocalClassifierPerLevel,
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
]

# TODO : Add parametrized tests


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.fixture
def explainer_data():
    np.random.seed(42)
    x_train = np.random.randn(4, 3)
    y_train = np.array(
        [["a", "b", "d"], ["a", "b", "e"], ["a", "c", "f"], ["a", "c", "g"]]
    )
    x_test = np.random.randn(3, 3)

    return x_train, x_test, y_train


def test_explainer_tree(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain(x_test, traverse_prediction=False)
    print(shap_dict)

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

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain_traversed_nodes(x_test)
    print(shap_dict)
    #
    for key, val in shap_dict.items():
        # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
        model = lcppn.hierarchy_.nodes[key]["classifier"]
        assert shap_dict[key].shape == (
            len(model.classes_),
            x_test.shape[0],
            x_test.shape[1],
        )


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

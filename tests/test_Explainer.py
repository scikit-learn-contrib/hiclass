import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
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


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
def test_explainer_tree(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain(x_test)
    assert shap_dict is not None

    # for key, val in shap_dict.items():
    #     # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
    #     model = lcppn.hierarchy_.nodes[key]["classifier"]
    #     assert shap_dict[key].shape == (
    #         len(model.classes_),
    #         x_test.shape[0],
    #         x_test.shape[1],
    #     )


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
def test_explainer_tree_traversal(explainer_data):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain(x_test)
    assert shap_dict is not None
    #
    # for key, val in shap_dict.items():
    #     # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
    #     model = lcppn.hierarchy_.nodes[key]["classifier"]
    #     assert shap_dict[key].shape == (
    #         len(model.classes_),
    #         x_test.shape[0],
    #         x_test.shape[1],
    #     )


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


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
def test_explainer_tree_no_root(explainer_data_no_root):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = explainer_data_no_root

    lcppn.fit(x_train, y_train)

    lcppn.predict(x_test)
    explainer = Explainer(lcppn, data=x_train, mode="tree")
    shap_dict = explainer.explain(x_test)
    assert shap_dict is not None

    # for key, val in shap_dict.items():
    #     # Assert on shapes of shap values, must match (target_classes, num_samples, num_features)
    #     model = lcppn.hierarchy_.nodes[key]["classifier"]
    #     assert shap_dict[key].shape == (
    #         len(model.classes_),
    #         x_test.shape[0],
    #         x_test.shape[1],
    #     )


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_traversal_path(data, request):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    lcppn.fit(x_train, y_train)
    explainer = Explainer(lcppn, data=x_train, mode="tree")
    traversals = explainer._get_traversed_nodes(x_test)
    preds = lcppn.predict(x_test)
    assert len(preds) == len(traversals)
    for i in range(len(x_test)):
        for j in range(len(traversals[i])):
            if traversals[i][j] == lcppn.root_:
                continue
            label = traversals[i][j].split("::HiClass::Separator::")[-1]
            assert label == preds[i][j - 1]

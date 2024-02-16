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


# TODO : Parametrize tests for remaining classifiers


@pytest.fixture
def explainer_data():
    np.random.seed(42)
    x_train = np.random.randn(4, 3)
    y_train = np.array(
        [["a", "b", "d"], ["a", "b", "e"], ["a", "c", "f"], ["a", "c", "g"]]
    )
    x_test = np.random.randn(5, 3)

    return x_train, x_test, y_train


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
    x_test = np.random.randn(5, 3)
    return x_train, x_test, y_train


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_explainer_tree_lcpn(data, request):
    rfc = RandomForestClassifier()
    lcpn = LocalClassifierPerNode(local_classifier=rfc, replace_classifiers=False)

    x_train, x_test, y_train = request.getfixturevalue(data)

    lcpn.fit(x_train, y_train)

    explainer = Explainer(lcpn, data=x_train, mode="tree")
    explanations = explainer.explain(x_test)
    assert explanations is not None
    y_preds = lcpn.predict(x_test)
    for i in range(len(x_test)):
        y_pred = y_preds[i]
        explanation = explanations[i]
        print(explanation)
        for j in range(len(y_pred)):
            assert str(explanation[j]["node"].data) == y_pred[j]


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_traversal_path_lcppn(data, request):
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
            label = traversals[i][j].split(lcppn.separator_)[-1]
            assert label == preds[i][j - 1]


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_explainer_tree_lcppn(data, request):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = request.getfixturevalue(data)

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    explanations = explainer.explain(x_test)
    assert explanations is not None
    y_preds = lcppn.predict(x_test)
    for i in range(len(x_test)):
        y_pred = y_preds[i]
        explanation = explanations[i]
        for j in range(len(y_pred)):
            assert (
                explanation[j]["predicted_class"].data[0].split(lcppn.separator_)[-1]
                == y_pred[j]
            )


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_traversal_path_lcpn(data, request):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    lcpn = LocalClassifierPerNode(local_classifier=rfc, replace_classifiers=False)

    lcpn.fit(x_train, y_train)
    explainer = Explainer(lcpn, data=x_train, mode="tree")
    traversals = explainer._get_traversed_nodes(x_test)
    preds = lcpn.predict(x_test)
    assert len(preds) == len(traversals)
    for i in range(len(x_test)):
        for j in range(len(traversals[i])):
            label = traversals[i][j].split(lcpn.separator_)[-1]
            assert label == preds[i][j]

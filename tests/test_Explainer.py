import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from hiclass import (
    LocalClassifierPerLevel,
    Explainer,
)

try:
    import shap
except ImportError:
    shap_installed = False
else:
    shap_installed = True

try:
    import xarray
except ImportError:
    xarray_installed = False
else:
    xarray_installed = True


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
def test_traversal_path_lcpl(data, request):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    lcpl = LocalClassifierPerLevel(local_classifier=rfc, replace_classifiers=False)

    lcpl.fit(x_train, y_train)
    explainer = Explainer(lcpl, data=x_train, mode="tree")
    traversals = explainer._get_traversed_nodes(x_test)
    preds = lcpl.predict(x_test)
    assert len(preds) == len(traversals)
    for i in range(len(x_test)):
        for j in range(len(traversals[i])):
            label = traversals[i][j].split(lcpl.separator_)[-1]
            assert label == preds[i][j]


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_explainer_tree_lcpl(data, request):
    rfc = RandomForestClassifier()
    lcpl = LocalClassifierPerLevel(local_classifier=rfc, replace_classifiers=False)

    x_train, x_test, y_train = request.getfixturevalue(data)

    lcpl.fit(x_train, y_train)

    explainer = Explainer(lcpl, data=x_train, mode="tree")
    explanations = explainer.explain(x_test)
    assert explanations is not None
    y_preds = lcpl.predict(x_test)
    for i in range(len(x_test)):
        y_pred = y_preds[i]
        for j in range(len(y_pred)):
            assert str(explanations["node"][i].data[j]) == y_pred[j]

@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.skipif(not xarray_installed, reason="xarray not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
@pytest.mark.parametrize("classifier", [LocalClassifierPerLevel])
def test_explain_with_xr(data, request, classifier):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    clf = classifier(local_classifier=rfc, replace_classifiers=False)

    clf.fit(x_train, y_train)
    explainer = Explainer(clf, data=x_train, mode="tree")
    explanations = explainer._explain_with_xr(x_test)

    # Assert if explainer returns an xarray.Dataset object
    assert isinstance(explanations, xarray.Dataset)

@pytest.mark.parametrize("classifier", [LocalClassifierPerLevel])
@pytest.mark.parametrize("data", ["explainer_data"])
@pytest.mark.parametrize("mode", ["linear", "gradient", "deep", "tree", ""])
def test_explainers(data, request, classifier, mode):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    clf = classifier(local_classifier=rfc, replace_classifiers=False)

    clf.fit(x_train, y_train)
    explainer = Explainer(clf, data=x_train, mode=mode)
    mode_mapping = {
        "linear": shap.LinearExplainer,
        "gradient": shap.GradientExplainer,
        "deep": shap.DeepExplainer,
        "tree": shap.TreeExplainer,
        "": shap.Explainer,
    }
    assert explainer.explainer == mode_mapping[mode]
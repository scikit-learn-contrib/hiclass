import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, Explainer

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
def test_explainer_tree_lcppn(data, request):
    rfc = RandomForestClassifier()
    lcppn = LocalClassifierPerParentNode(
        local_classifier=rfc, replace_classifiers=False
    )

    x_train, x_test, y_train = request.getfixturevalue(data)

    lcppn.fit(x_train, y_train)

    explainer = Explainer(lcppn, data=x_train, mode="tree")
    explanations = explainer.explain(x_test)

    # Assert if explainer returns an xarray.Dataset object
    assert isinstance(explanations, xarray.Dataset)

    # Assert if predictions made are consistent with the explanation object
    y_preds = lcppn.predict(x_test)
    for i in range(len(x_test)):
        y_pred = y_preds[i]
        explanation = explanations["predicted_class"][i]
        for j in range(len(y_pred)):
            assert explanation.data[j].split(lcppn.separator_)[-1] == y_pred[j]


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.skipif(not xarray_installed, reason="xarray not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
def test_explainer_tree_lcpn(data, request):
    rfc = RandomForestClassifier()
    lcpn = LocalClassifierPerNode(local_classifier=rfc, replace_classifiers=False)

    x_train, x_test, y_train = request.getfixturevalue(data)

    lcpn.fit(x_train, y_train)

    explainer = Explainer(lcpn, data=x_train, mode="tree")
    explanations = explainer.explain(x_test)

    # Assert if explainer returns an xarray.Dataset object
    assert isinstance(explanations, xarray.Dataset)
    y_preds = lcpn.predict(x_test)

    # Assert if predictions made are consistent with the explanation object
    for i in range(len(x_test)):
        y_pred = y_preds[i]
        for j in range(len(y_pred)):
            assert str(explanations["node"][i].data[j]) == y_pred[j]


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
    traversals = explainer._get_traversed_nodes_lcppn(x_test)
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
def test_traversal_path_lcpn(data, request):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    lcpn = LocalClassifierPerNode(local_classifier=rfc, replace_classifiers=False)

    lcpn.fit(x_train, y_train)
    explainer = Explainer(lcpn, data=x_train, mode="tree")
    traversals = explainer._get_traversed_nodes_lcpn(x_test)
    preds = lcpn.predict(x_test)

    # Assert if predictions and traversals are of same length
    assert len(preds) == len(traversals)

    # Assert if traversal path in predictions is same as the computed traversal path
    for i in range(len(x_test)):
        for j in range(len(traversals[i])):
            label = traversals[i][j].split(lcpn.separator_)[-1]
            assert label == preds[i][j]


@pytest.mark.skipif(not shap_installed, reason="shap not installed")
@pytest.mark.skipif(not xarray_installed, reason="xarray not installed")
@pytest.mark.parametrize("data", ["explainer_data", "explainer_data_no_root"])
@pytest.mark.parametrize(
    "classifier", [LocalClassifierPerParentNode, LocalClassifierPerNode]
)
def test_explain_with_xr(data, request, classifier):
    x_train, x_test, y_train = request.getfixturevalue(data)
    rfc = RandomForestClassifier()
    clf = classifier(local_classifier=rfc, replace_classifiers=False)

    clf.fit(x_train, y_train)
    explainer = Explainer(clf, data=x_train, mode="tree")
    explanations = explainer._explain_with_xr(x_test)

    # Assert if explainer returns an xarray.Dataset object
    assert isinstance(explanations, xarray.Dataset)


@pytest.mark.parametrize(
    "classifier", [LocalClassifierPerParentNode, LocalClassifierPerNode]
)
def test_imports(classifier):
    x_train = [[76, 12, 49], [88, 63, 31], [5, 42, 24], [17, 90, 55]]
    y_train = [["a", "b", "d"], ["a", "b", "e"], ["a", "c", "f"], ["a", "c", "g"]]

    rfc = RandomForestClassifier()
    clf = classifier(local_classifier=rfc, replace_classifiers=False)
    clf.fit(x_train, y_train)

    explainer = Explainer(clf, data=x_train, mode="tree")
    assert isinstance(explainer.data, np.ndarray)


@pytest.mark.parametrize(
    "classifier", [LocalClassifierPerParentNode, LocalClassifierPerNode]
)
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

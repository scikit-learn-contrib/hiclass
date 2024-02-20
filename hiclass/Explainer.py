"""Explainer API for explaining predictions using shapley values."""

from copy import deepcopy

import numpy as np

from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
)

try:
    import xarray as xr
except ImportError:
    xarray_installed = False
else:
    xarray_installed = True

try:
    import shap
except ImportError:
    shap_installed = False
else:
    shap_installed = True


def _check_imports():
    if not shap_installed:
        raise ImportError(
            "Shap is not installed. Please install it using `pip install shap` first."
        )
    elif not xarray_installed:
        raise ImportError(
            "xarray is not installed. Please install it using `pip install xarray` first."
        )


class Explainer:
    """Explainer class for returning shap values for each of the three hierarchical classifiers."""

    def __init__(self, hierarchical_model, data=None, algorithm="auto", mode=""):
        """
        Initialize the SHAP explainer for a hierarchical model.

        Parameters
        ----------
        hierarchical_model : HierarchicalClassifier
            The hierarchical classification model to explain.
        data : array-like or None, default=None
            The dataset used for creating the SHAP explainer.
        algorithm : str, default="auto"
            The algorithm to use for SHAP explainer.
        mode : str, default=""
            The mode of the SHAP explainer. Can be 'tree', 'gradient', 'deep', 'linear', or '' for default SHAP explainer.
        """
        self.hierarchical_model = hierarchical_model
        self.algorithm = algorithm
        self.mode = mode
        self.data = data

        _check_imports()

        if mode == "tree":
            self.explainer = shap.TreeExplainer
        elif mode == "gradient":
            self.explainer = shap.GradientExplainer
        elif mode == "deep":
            self.explainer = shap.DeepExplainer
        elif mode == "linear":
            self.explainer = shap.LinearExplainer
        else:
            self.explainer = shap.Explainer

    def explain(self, X):
        """
        Generate SHAP values for each node in the hierarchy for the given data.

        Parameters
        ----------
        X : array-like
            Training data to fit the SHAP explainer.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node.
        """
        _check_imports()
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            return self._explain_with_xr(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            return self._explain_with_xr(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            return self._explain_with_xr(X)
        else:
            raise ValueError(f"Invalid model: {self.hierarchical_model}.")

    def _explain_with_xr(self, X):
        """
        Generate SHAP values for each node / level using Local Classifier Per * strategy.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node/level.
        """
        explanations = []
        for sample in X:
            explanation = self._calculate_shap_values(sample.reshape(1, -1))
            explanations.append(explanation)
        return explanations

    def _get_traversed_nodes(self, samples):
        # Helper function to return traversed nodes
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            traversals = []
            start_node = self.hierarchical_model.root_
            for x in samples:
                current = start_node
                traversal_order = []
                while self.hierarchical_model.hierarchy_.neighbors(current):
                    if (
                        "classifier"
                        not in self.hierarchical_model.hierarchy_.nodes[current]
                    ):
                        break  # Break if reached leaf node
                    traversal_order.append(current)
                    successor = self.hierarchical_model.hierarchy_.nodes[current][
                        "classifier"
                    ].predict(x.reshape(1, -1))[0]
                    current = successor
                traversals.append(traversal_order)
            return traversals
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            traversals = []
            predictions = self.hierarchical_model.predict(samples)
            for pred in predictions:
                traversal_order = []
                for i in range(len(pred)):
                    if i == 0:
                        traversal_order.append(pred[i])
                    else:
                        node = (
                            str(traversal_order[i - 1])
                            + self.hierarchical_model.separator_
                            + str(pred[i])
                        )
                        if node in self.hierarchical_model.hierarchy_.nodes:
                            traversal_order.append(node)
                traversals.append(traversal_order)
            return traversals

        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            traversals = []
            for x in samples:
                traversal_order = []
                for current_level in range(
                    len(self.hierarchical_model.local_classifiers_)
                ):
                    classifier = self.hierarchical_model.local_classifiers_[
                        current_level
                    ]
                    predicted_class = classifier.predict(x.reshape(1, -1))[0]

                    if traversal_order:
                        node = (
                            str(traversal_order[-1])
                            + self.hierarchical_model.separator_
                            + str(predicted_class)
                        )
                    else:
                        node = str(predicted_class)
                    if node in self.hierarchical_model.hierarchy_.nodes:
                        traversal_order.append(node)
                traversals.append(traversal_order)
            return traversals

    def _calculate_shap_values(self, X):
        if not xarray_installed:
            raise ImportError(
                "xarray is not installed. Please install it using `pip install xarray` before using "
                "this method."
            )
        traversed_nodes = self._get_traversed_nodes(X)[0]
        datasets = []
        if isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            for node in traversed_nodes:
                # Define the level of the node in hierarchy
                level = len(node.split(self.hierarchical_model.separator_)) - 1

                local_classifier = self.hierarchical_model.local_classifiers_[level]

                # Create a SHAP explainer for the local classifier
                local_explainer = deepcopy(self.explainer)(local_classifier, self.data)

                # Calculate SHAP values for the given sample X
                shap_values = np.array(
                    local_explainer.shap_values(X, check_additivity=False)
                )
                if len(shap_values.shape) < 3:
                    shap_values = shap_values.reshape(
                        1, shap_values.shape[0], shap_values.shape[1]
                    )

                predict_proba = xr.DataArray(
                    local_classifier.predict_proba(X)[0],
                    dims=["class"],
                    coords={"class": local_classifier.classes_},
                )
                classes = xr.DataArray(
                    local_classifier.classes_,
                    dims=["class"],
                    coords={"class": local_classifier.classes_},
                )

                shap_val_local = xr.DataArray(
                    shap_values,
                    dims=["class", "sample", "feature"],
                )
                local_dataset = xr.Dataset(
                    {
                        "node": node.split(self.hierarchical_model.separator_)[-1],
                        "level": level,
                        "predicted_class": local_classifier.predict(X),
                        "predict_proba": predict_proba,
                        "classes": classes,
                        "shap_values": shap_val_local,
                    }
                )
                datasets.append(local_dataset)
            else:
                for node in traversed_nodes:
                    local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                        "classifier"
                    ]

                    # Create a SHAP explainer for the local classifier
                    local_explainer = deepcopy(self.explainer)(
                        local_classifier, self.data
                    )

                    # Calculate SHAP values for the given sample X
                    shap_values = np.array(
                        local_explainer.shap_values(X, check_additivity=False)
                    )
                    if len(shap_values.shape) < 3:
                        shap_values = shap_values.reshape(
                            1, shap_values.shape[0], shap_values.shape[1]
                        )

                    predict_proba = xr.DataArray(
                        local_classifier.predict_proba(X)[0],
                        dims=["class"],
                        coords={
                            "class": local_classifier.classes_,
                        },
                    )
                    classes = xr.DataArray(
                        local_classifier.classes_,
                        dims=["class"],
                        coords={"class": local_classifier.classes_},
                    )

                    shap_val_local = xr.DataArray(
                        shap_values,
                        dims=["class", "sample", "feature"],
                    )

                    local_dataset = xr.Dataset(
                        {
                            "node": node.split(self.hierarchical_model.separator_)[-1],
                            "predicted_class": local_classifier.predict(X),
                            "predict_proba": predict_proba,
                            "classes": classes,
                            "shap_values": shap_val_local,
                        }
                    )
                    datasets.append(local_dataset)
            return datasets

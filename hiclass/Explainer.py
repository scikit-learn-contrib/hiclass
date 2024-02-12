"""Explainer API for explaining predictions using shapley values."""

import shap
import numpy as np
from copy import deepcopy
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
    ConstantClassifier,
)

try:
    import xarray as xr
except ImportError:
    xarray_installed = False
else:
    xarray_installed = True


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

    def explain(self, X, traverse_prediction=False):
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
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            return self.explain_with_xr(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            return self._explain_lcpl(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            return self._explain_lcpn(X)
        else:
            raise ValueError(f"Invalid model: {self.hierarchical_model}.")

    def explain_with_dict(self, X, traverse_prediction):
        """
        Generate SHAP values for each node using Local Classifier Per Parent Node (LCPPN) strategy.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        traverse_prediction : True or False
            If True, restricts calculation of shap values to only traversed hierarchy as predicted by hiclass model.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node.
        """
        if traverse_prediction:
            return self.explain_traversed_nodes(X)
        shap_values_dict = {}
        parent_nodes = self.hierarchical_model._get_parents()
        for parent_node in parent_nodes:
            # Ignore the root node if redundant, do NOT ignore in case of disjoint subtrees
            if (
                parent_node == self.hierarchical_model.root_
                and len(
                    self.hierarchical_model.hierarchy_.nodes[
                        self.hierarchical_model.root_
                    ]["classifier"].classes_
                )
                < 2
            ):
                continue

            # Get the local classifier for the parent node
            local_classifier = self.hierarchical_model.hierarchy_.nodes[parent_node][
                "classifier"
            ]

            # Create a SHAP explainer for the local classifier
            local_explainer = deepcopy(self.explainer)(local_classifier, self.data)

            # TODO: Tree models do not sometimes work if check_additivity is enabled
            # Calculate SHAP values for the given sample X
            if isinstance(local_explainer, shap.TreeExplainer):
                shap_values = np.array(
                    local_explainer.shap_values(X, check_additivity=False)
                )
            else:
                shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[parent_node] = shap_values
        return shap_values_dict

    def explain_traversed_nodes(self, X):
        """
        Generate SHAP values for each node while only calculating shap values for the traversed nodes. Unvisited nodes return `nan` as the shap values.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node.
        """
        shap_values_dict = {}
        traversed_nodes = self._get_traversed_nodes(X)
        for node in traversed_nodes:
            local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                "classifier"
            ]

            # Create explainer with train data
            local_explainer = deepcopy(self.explainer)(local_classifier, self.data)
            shap_values = np.array(local_explainer.shap_values(X))

            if len(shap_values.shape) < 3:
                shap_values = shap_values.reshape(
                    1, shap_values.shape[0], shap_values.shape[1]
                )

            shap_values_dict[node] = shap_values

        for node in self.hierarchical_model.hierarchy_.nodes:
            if node not in traversed_nodes:
                local_classifier = self.hierarchical_model.hierarchy_.nodes[node]
                if len(local_classifier) != 0:
                    shap_val = np.full(
                        (
                            len(local_classifier["classifier"].classes_),
                            X.shape[0],
                            X.shape[1],
                        ),
                        np.nan,
                    )
                    shap_values_dict[node] = shap_val
        return shap_values_dict

    def explain_with_xr(self, X):
        """
        Generate SHAP values for each node using Local Classifier Per Parent Node (LCPPN) strategy.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node.
        """
        explanations = []
        for sample in X:
            explanation = self._calculate_shap_values(sample.reshape(1, -1))
            explanations.append(explanation)
        return explanations

    def _explain_lcpn(self, X):
        shap_values_dict = {}
        for node in self.hierarchical_model.hierarchy_.nodes:
            if node == self.hierarchical_model.root_:
                continue

            if isinstance(
                self.hierarchical_model.hierarchy_.nodes[node]["classifier"],
                ConstantClassifier.ConstantClassifier,
            ):
                continue

            local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                "classifier"
            ]

            local_explainer = deepcopy(self.explainer)(local_classifier, self.data)

            # Calculate SHAP values for the given sample X
            shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[node] = shap_values

        return shap_values_dict

    def _explain_lcpl(self, X):
        """
        Generate SHAP values for each node using Local Classifier Per Level (LCPL) strategy.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        Returns
        -------
        shap_values_dict : dict
            A dictionary of SHAP values for each node.
        """
        shap_values_dict = {}
        start_level = 1
        if len(self.hierarchical_model.local_classifiers_[start_level]) == 1:
            start_level = 2

        for level in range(
            start_level, len(self.hierarchical_model.local_classifiers_)
        ):
            local_classifier = self.hierarchical_model.local_classifiers_[level]
            local_explainer = deepcopy(self.explainer)(local_classifier, self.data)

            # Calculate SHAP values for the given sample X
            shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[level] = shap_values

        return shap_values_dict

    def _get_traversed_nodes(self, samples):
        # Helper function to return traversed nodes
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            traversals = []
            start_node = self.hierarchical_model.root_
            for x in samples:
                current = start_node
                traversal_order = []
                while self.hierarchical_model.hierarchy_.neighbors(current):
                    print(f"Currently traversing: {current}")
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
            pass
        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            pass

    def _calculate_shap_values(self, X):
        traversed_nodes = self._get_traversed_nodes(X)[0]
        datasets = []
        for node in traversed_nodes:
            local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                "classifier"
            ]

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

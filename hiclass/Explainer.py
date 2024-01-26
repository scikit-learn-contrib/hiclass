"""Explainer API for explaining predictions using shapley values."""

import shap
import numpy as np
from copy import deepcopy
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
)

try:
    import xarray as xar
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
        self.explainers = {}  # To store a SHAP explainer for each node
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
            return self._explain_lcppn(X, traverse_prediction)
        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            return self._explain_lcpl(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            return self._explain_lcpl(X)
        else:
            raise ValueError(f"Invalid model: {self.hierarchical_model}.")

    def _explain_lcppn(self, X, traverse_prediction=False):
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
        shap_values_dict = {}

        if traverse_prediction:
            y_pred = self.hierarchical_model.predict(X)
            print(f"y_pred: {y_pred}")
            traversal_path = str(y_pred[0][0])
            for pred in y_pred[0][1:]:
                traversal_path = (
                    traversal_path + self.hierarchical_model.separator_ + pred
                )
            for i in range(self.hierarchical_model.max_levels_)[:-1]:
                node = self.hierarchical_model.separator_.join(
                    traversal_path.split(self.hierarchical_model.separator_)[: i + 1]
                )

                local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                    "classifier"
                ]

                if node not in self.explainers:
                    # Create explainer with train data
                    local_explainer = deepcopy(self.explainer)(
                        local_classifier, self.data
                    )
                    self.explainers[node] = local_explainer
                else:
                    local_explainer = self.explainers[node]

                shap_values = np.array(local_explainer.shap_values(X))
                print(shap_values)
                shap_values_dict[node] = shap_values
            for node in self.hierarchical_model.hierarchy_.nodes:
                if node not in self.explainers:
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

            y_pred_local = local_classifier.predict(X)
            print(f"y_pred_local: {y_pred_local}")

            # Create a SHAP explainer for the local classifier
            if parent_node not in self.explainers:
                # Create explainer with train data
                local_explainer = deepcopy(self.explainer)(local_classifier, self.data)
                self.explainers[parent_node] = local_explainer
            else:
                local_explainer = self.explainers[parent_node]

            # Calculate SHAP values for the given sample X
            shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[parent_node] = shap_values

        return shap_values_dict

    def _explain_lcpn(self, X):
        pass

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

            if level not in self.explainers:
                # Create explainer with train data
                local_explainer = deepcopy(self.explainer)(local_classifier, self.data)
                self.explainers[level] = local_explainer
            else:
                local_explainer = self.explainers[level]

            # Calculate SHAP values for the given sample X
            shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[level] = shap_values

        return shap_values_dict

    def _filter_shap(self, sample, level):
        pass

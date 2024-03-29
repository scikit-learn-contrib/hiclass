"""Explainer API for explaining predictions using shapley values."""

from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
    HierarchicalClassifier,
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


class Explainer:
    """Explainer class for returning shap values for each of the three hierarchical classifiers."""

    def __init__(
        self,
        hierarchical_model: HierarchicalClassifier.HierarchicalClassifier,
        data: None,
        n_jobs: int = 1,
        algorithm: str = "auto",
        mode: str = "",
    ):
        """
        Initialize the SHAP explainer for a hierarchical model.

        Parameters
        ----------
        hierarchical_model : HierarchicalClassifier
            The hierarchical classification model to explain.
        data : array-like or None, default=None
            The dataset used for creating the SHAP explainer.
        n_jobs : int, default=1
            The number of jobs to run in parallel.
        algorithm : str, default="auto"
            The algorithm to use for SHAP explainer. Possible values are 'linear', 'tree', 'auto', 'permutation', or 'partition'
        mode : str, default=""
            The mode of the SHAP explainer. Can be 'tree', 'gradient', 'deep', 'linear', or '' for default SHAP explainer.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>> from hiclass import LocalClassifierPerParentNode, Explainer
        >>> rfc = RandomForestClassifier()
        >>> lcppn = LocalClassifierPerParentNode(local_classifier=rfc, replace_classifiers=False)
        >>> x_train = np.array([[1, 3], [2, 5]])
        >>> y_train = np.array([[1, 2], [3, 4]])
        >>> x_test = np.array([[4, 6]])
        >>> lcppn.fit(x_train, y_train)
        >>> explainer = Explainer(lcppn, data=x_train, mode="tree")
        >>> explanations = explainer.explain(x_test)
        <xarray.Dataset>
        Dimensions:          (class: 3, sample: 1, level: 2, feature: 2)
        Coordinates:
          * class            (class) <U1 '1' '3' '4'
          * level            (level) int64 0 1
        Dimensions without coordinates: sample, feature
        Data variables:
            node             (sample, level) <U13 'hiclass::root' '3'
            predicted_class  (sample, level) <U1 '3' '4'
            predict_proba    (sample, level, class) float64 0.29 0.71 nan nan nan 1.0
            classes          (sample, level, class) object '1' '3' nan nan nan '4'
            shap_values      (level, class, sample, feature) float64 -0.135 ... 0.0
        """
        self.hierarchical_model = hierarchical_model
        self.algorithm = algorithm
        self.mode = mode
        self.data = np.array(data)
        self.n_jobs = n_jobs

        # Check if hierarchical model is fitted
        check_is_fitted(self.hierarchical_model)

        if mode == "linear":
            self.explainer = shap.LinearExplainer
        elif mode == "gradient":
            self.explainer = shap.GradientExplainer
        elif mode == "deep":
            self.explainer = shap.DeepExplainer
        elif mode == "tree":
            self.explainer = shap.TreeExplainer
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
        explanation : xarray.Dataset
            An xarray.Dataset object representing the explanations for each sample passed.
        """
        # Check if sample data is valid
        check_array(X)

        if (
            isinstance(self.hierarchical_model, LocalClassifierPerParentNode)
            or isinstance(self.hierarchical_model, LocalClassifierPerLevel)
            or isinstance(self.hierarchical_model, LocalClassifierPerNode)
        ):
            return self._explain_with_xr(X)
        else:
            raise ValueError(f"Invalid model: {self.hierarchical_model}.")

    def _explain_with_xr(self, X):
        """
        Generate SHAP values for each node using the SHAP package.

        Parameters
        ----------
        X : array-like
            Sample data for which to generate SHAP values.

        Returns
        -------
        explanation : xarray.Dataset
            An xarray Dataset consisting of SHAP values for each sample.
        """
        explanations = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._calculate_shap_values)(sample.reshape(1, -1)) for sample in X
        )

        dataset = xr.concat(explanations, dim="sample")
        return dataset

    def _get_traversed_nodes_lcppn(self, samples):
        """
        Return a list of all traversed nodes as per the provided LocalClassifierPerParentNode model.

        Parameters
        ----------
        samples : array-like
            Sample data for which to generate traversed nodes.

        Returns
        -------
        traversals : list
            A list of all traversed nodes as per LocalClassifierPerParentNode (LCPPN) strategy.
        """
        # Initialize array with shape (#samples, #levels)
        traversals = np.empty(
            (samples.shape[0], self.hierarchical_model.max_levels_),
            dtype=self.hierarchical_model.dtype_,
        )

        # Initialize first element as root node
        traversals[:, 0] = self.hierarchical_model.root_

        # For subsequent nodes, calculate mask and find predictions
        for level in range(1, traversals.shape[1]):
            predecessors = set(traversals[:, level - 1])
            predecessors.discard("")
            for predecessor in predecessors:
                mask = np.isin(traversals[:, level - 1], predecessor)
                predecessor_x = samples[mask]
                if predecessor_x.shape[0] > 0:
                    successors = list(
                        self.hierarchical_model.hierarchy_.successors(predecessor)
                    )
                    if len(successors) > 0:
                        classifier = self.hierarchical_model.hierarchy_.nodes[
                            predecessor
                        ]["classifier"]
                        traversals[mask, level] = classifier.predict(
                            predecessor_x
                        ).flatten()
        return traversals

    def _get_traversed_nodes_lcpn(self, samples):
        """
        Return a list of all traversed nodes as per the provided LocalClassifierPerNode model.

        Parameters
        ----------
        samples : array-like
            Sample data for which to generate traversed nodes.

        Returns
        -------
        traversals : list
            A list of all traversed nodes as per LocalClassifierPerNode (LCPN) strategy.
        """
        traversals = np.empty(
            (samples.shape[0], self.hierarchical_model.max_levels_),
            dtype=self.hierarchical_model.dtype_,
        )

        predictions = self.hierarchical_model.predict(samples)

        traversals[:, 0] = predictions[:, 0]
        separator = np.full(
            (samples.shape[0], 3),
            self.hierarchical_model.separator_,
            dtype=self.hierarchical_model.dtype_,
        )

        for level in range(1, traversals.shape[1]):
            traversals[:, level] = np.core.defchararray.add(
                traversals[:, level - 1],
                np.core.defchararray.add(separator[:, 0], predictions[:, level]),
            )

        # For inconsistent hierarchies, levels with empty nodes should be ignored
        mask = predictions == ""
        traversals[mask] = ""

        return traversals

    def _calculate_shap_values(self, X):
        """
        Return an xarray.Dataset object for a single sample provided. This dataset is aligned on the `level` attribute.

        Parameters
        ----------
        X : array-like
            Data for single sample for which to generate SHAP values.

        Returns
        -------
        explanation : xarray.Dataset
            A single explanation for the prediction of given sample.
        """
        traversed_nodes = []
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            traversed_nodes = self._get_traversed_nodes_lcppn(X)[0]
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            traversed_nodes = self._get_traversed_nodes_lcpn(X)[0]
        datasets = []
        level = 0
        for node in traversed_nodes:
            # Skip if classifier is not found, can happen in case of imbalanced hierarchies
            if "classifier" not in self.hierarchical_model.hierarchy_.nodes[node]:
                continue

            local_classifier = self.hierarchical_model.hierarchy_.nodes[node][
                "classifier"
            ]

            # Create a SHAP explainer for the local classifier
            local_explainer = deepcopy(self.explainer)(local_classifier, self.data)

            current_node = node.split(self.hierarchical_model.separator_)[-1]

            # Calculate SHAP values for the given sample X
            shap_values = np.array(
                local_explainer.shap_values(X, check_additivity=False)
            )

            if len(shap_values.shape) < 3:
                shap_values = shap_values.reshape(
                    1, shap_values.shape[0], shap_values.shape[1]
                )

            if isinstance(self.hierarchical_model, LocalClassifierPerNode):
                simplified_labels = [
                    f"{current_node}_{int(label)}"
                    for label in local_classifier.classes_
                ]
                predicted_class = current_node
            else:
                simplified_labels = [
                    label.split(self.hierarchical_model.separator_)[-1]
                    for label in local_classifier.classes_
                ]
                predicted_class = (
                    local_classifier.predict(X)
                    .flatten()[0]
                    .split(self.hierarchical_model.separator_)[-1]
                )

            classes = xr.DataArray(
                simplified_labels,
                dims=["class"],
                coords={"class": simplified_labels},
            )

            shap_val_local = xr.DataArray(
                shap_values,
                dims=["class", "sample", "feature"],
                coords={"class": simplified_labels},
            )

            prediction_probability = local_classifier.predict_proba(X)[0]

            predict_proba = xr.DataArray(
                prediction_probability,
                dims=["class"],
                coords={
                    "class": simplified_labels,
                },
            )

            local_dataset = xr.Dataset(
                {
                    "node": current_node,
                    "predicted_class": predicted_class,
                    "predict_proba": predict_proba,
                    "classes": classes,
                    "shap_values": shap_val_local,
                    "level": level,
                }
            )
            level = level + 1
            datasets.append(local_dataset)
        sample_explanation = xr.concat(datasets, dim="level")
        return sample_explanation

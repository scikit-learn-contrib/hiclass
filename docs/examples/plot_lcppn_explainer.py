# -*- coding: utf-8 -*-
"""
============================================
Explaining Local Classifier Per Parent Node
============================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPPN model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability`.
SHAP values are calculated based on a synthetic platypus diseases dataset that can be downloaded `here <https://gist.githubusercontent.com/ashishpatel16/9306f8ed3ed101e7ddcb519776bcbd80/raw/3f225c3f80dd8cbb1b6252f6c372a054ec968705/platypus_diseases.csv>`_.
"""
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerParentNode, Explainer
import shap
from hiclass.datasets import load_platypus

# Load train and test splits
X_train, X_test, Y_train, Y_test = load_platypus()

# Use random forest classifiers for every node
rfc = RandomForestClassifier()
classifier = LocalClassifierPerParentNode(
    local_classifier=rfc, replace_classifiers=False
)

# Train local classifier per parent node
classifier.fit(X_train, Y_train)

# Get predictions
predictions = classifier.predict(X_test)

# Define Explainer
explainer = Explainer(classifier, data=X_train.values, mode="tree")
explanations = explainer.explain(X_test.values)
print(explanations)

# Use .sel() method to apply the filter and obtain filtered results
shap_val_respiratory = explainer.filter_by_class(
    explanations,
    class_name="Respiratory",
    sample_indices=explainer.get_sample_indices(predictions, "Respiratory"),
)


# Plot feature importance on test set
shap.plots.violin(
    shap_val_respiratory,
    feature_names=X_train.columns.values,
    plot_size=(13, 8),
)

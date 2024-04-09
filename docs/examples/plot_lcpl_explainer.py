# -*- coding: utf-8 -*-
"""
=========================================
Explaining Local Classifier Per Level
=========================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPL model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability`.
SHAP values are calculated based on a synthetic platypus diseases dataset that can be downloaded here.
"""
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerLevel, Explainer
import shap
from hiclass.datasets import load_platypus

# Load train and test splits
X_train, X_test, Y_train, Y_test = load_platypus()

# Use random forest classifiers for every level
rfc = RandomForestClassifier()
classifier = LocalClassifierPerLevel(local_classifier=rfc, replace_classifiers=False)

# Train local classifiers per level
classifier.fit(X_train, Y_train)

# Define Explainer
explainer = Explainer(classifier, data=X_train, mode="tree")
explanations = explainer.explain(X_test.values)
print(explanations)

# Example of filtering for levels and classes using the .sel() method
# Here we select explanations for "Respiratory" at the first level
respiratory_idx = classifier.predict(X_test)[:, 0] == "Respiratory"
shap_filter = {"level": 0, "class_": "Respiratory", "sample": respiratory_idx}
shap_val_respiratory = explanations.sel(**shap_filter)

# Plot feature importance for the "Respiratory" class
shap.plots.violin(
    shap_val_respiratory.shap_values,
    feature_names=X_train.columns.values,
    plot_size=(13, 8),
)

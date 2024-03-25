# -*- coding: utf-8 -*-
"""
=========================================
Explaining Local Classifier Per Node
=========================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPPN model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability`.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerParentNode, Explainer

# Define data
X_train = np.array(
    [
        [40.7, 1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 5.0, 34.3],
        [39.2, 0.0, 2.0, 4.0, 1.0, 3.0, 1.0, 2.0, 34.1],
        [40.6, 0.0, 3.0, 1.0, 4.0, 5.0, 0.0, 6.0, 27.7],
        [36.5, 0.0, 3.0, 1.0, 2.0, 2.0, 0.0, 2.0, 39.9],
    ]
)
X_test = np.array([[35.5, 0.0, 1.0, 1.0, 3.0, 3.0, 0.0, 2.0, 37.5]])
Y_train = np.array(
    [
        ["Gastrointestinal", "Norovirus", ""],
        ["Respiratory", "Covid", ""],
        ["Allergy", "External", "Bee Allergy"],
        ["Respiratory", "Cold", ""],
    ]
)

# Use random forest classifiers for every node
rfc = RandomForestClassifier()
classifier = LocalClassifierPerParentNode(
    local_classifier=rfc, replace_classifiers=False
)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Define Explainer
explainer = Explainer(classifier, data=X_train, mode="tree")
explanations = explainer.explain(X_test)
print(explanations)

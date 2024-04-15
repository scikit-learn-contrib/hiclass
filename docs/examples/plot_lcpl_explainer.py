# -*- coding: utf-8 -*-
"""
=========================================
Explaining Local Classifier Per Level
=========================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPL model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability`.
SHAP values are calculated based on a synthetic platypus diseases dataset that can be downloaded `here <https://gist.githubusercontent.com/ashishpatel16/9306f8ed3ed101e7ddcb519776bcbd80/raw/3f225c3f80dd8cbb1b6252f6c372a054ec968705/platypus_diseases.csv>`_.
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

# Predict
predictions = classifier.predict(X_test)

# Since Covid is a kind of Respiratory diseases, let's filter explanations for these classes

# Let's get sample indices where 'Covid' is predicted what can be done with .get_sample_indices() method
sample_idx = explainer.get_sample_indices(predictions, "Covid")

# Shapley values filtering by classes with .filter_by_class() method
shap_values_covid = explainer.filter_by_class(
    explanations, class_name="Covid", sample_indices=sample_idx
)
shap_values_resp = explainer.filter_by_class(
    explanations, class_name="Respiratory", sample_indices=sample_idx
)


# This code snippet demonstrates how to visually compare the mean absolute SHAP values for 'Covid' vs. 'Respiratory' diseases.

# Feature names for the X-axis
feature_names = X_train.columns.values

shap.summary_plot(
    [shap_values_covid, shap_values_resp],
    features=X_test.iloc[sample_idx],
    feature_names=X_train.columns.values,
    plot_type="bar",
    class_names=["Covid", "Respiratory"],
)
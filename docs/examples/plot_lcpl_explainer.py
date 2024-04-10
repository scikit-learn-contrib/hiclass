# -*- coding: utf-8 -*-
"""
=========================================
Explaining Local Classifier Per Level
=========================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPL model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability.
SHAP values are calculated based on a synthetic platypus diseases dataset that can be downloaded here.
"""
import matplotlib.pyplot as plt
import numpy as np
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

# Let's filter the Shapley values corresponding to the Covid (level 1)
# and 'Respiratory' (level 0)

covid_idx = classifier.predict(X_test)[:, 1] == "Covid"

shap_filter_covid = {"level": 1, "class": "Covid", "sample": covid_idx}
shap_filter_resp = {"level": 0, "class": "Respiratory", "sample": covid_idx}
shap_val_covid = explanations.sel(**shap_filter_covid)
shap_val_resp = explanations.sel(**shap_filter_resp)


# This code snippet demonstrates how to visually compare the mean absolute SHAP values for 'Covid' vs. 'Respiratory' diseases.

# Feature names for the X-axis
feature_names = X_train.columns.values

# Calculating the mean absolute SHAP values for each feature
mean_abs_shap_covid = np.mean(np.abs(shap_val_covid["shap_values"]), axis=0)
mean_abs_shap_resp = np.mean(np.abs(shap_val_resp["shap_values"]), axis=0)

# Creating the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Setting up the positions for bars on the chart
bar_width = 0.35
index = np.arange(len(mean_abs_shap_covid))

# Plotting bars for 'Covid'
bars1 = ax.bar(index, mean_abs_shap_covid, bar_width, label="Covid")

# Plotting bars for 'Respiratory'
bars2 = ax.bar(index + bar_width, mean_abs_shap_resp, bar_width, label="Respiratory")

# Adding labels, title, and legend
ax.set_xlabel("Features")
ax.set_ylabel("Mean Absolute SHAP Value")
ax.set_title("Mean Absolute SHAP Values for Covid vs Respiratory")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

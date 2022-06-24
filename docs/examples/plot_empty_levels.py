# -*- coding: utf-8 -*-
"""
==========================
Different Number of Levels
==========================

HiClass supports different number of levels in the hierarchy.
For this example, we will train a local classifier per node
with the following hierarchy:

.. figure:: ../algorithms/local_classifier_per_node.svg
   :align: center
"""
from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerNode

# Define data
X_train = [[1], [2], [3], [4]]
X_test = [[4], [3], [2], [1]]
Y_train = [
    ["Reptile", "Snake"],
    ["Reptile", "Lizard"],
    ["Mammal", "Cat"],
    ["Mammal", "Wolf", "Dog"],
]

# Use random forest classifiers for every node
rf = LogisticRegression()
classifier = LocalClassifierPerNode(local_classifier=rf)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

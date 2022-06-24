# -*- coding: utf-8 -*-
"""
==========================
Different Number of Levels
==========================

HiClass supports different number of levels in the hierarchy.
For this example, we will train a local classifier per parent node
with a hierarchy similar to the following image:

.. figure:: ../algorithms/local_classifier_per_parent_node.svg
   :align: center
"""
from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerParentNode

# Define data
X_train = [[1], [2], [3], [4], [5]]
X_test = [[5], [4], [3], [2], [1]]
Y_train = [
    ["Bird"],
    ["Reptile", "Snake"],
    ["Reptile", "Lizard"],
    ["Mammal", "Cat"],
    ["Mammal", "Wolf", "Dog"],
]

# Use random forest classifiers for every node
rf = LogisticRegression()
classifier = LocalClassifierPerParentNode(local_classifier=rf)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

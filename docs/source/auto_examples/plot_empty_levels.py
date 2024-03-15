# -*- coding: utf-8 -*-
"""
==========================
Different Number of Levels
==========================

HiClass supports different number of levels in the hierarchy.
For this example, we will train a local classifier per node
with a hierarchy similar to the following image:

.. figure:: ../algorithms/local_classifier_per_node.svg
   :align: center
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerNode

# Define data
X_train = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
X_test = [[9, 10], [7, 8], [5, 6], [3, 4], [1, 2]]
Y_train = np.array(
    [
        ["Bird"],
        ["Reptile", "Snake"],
        ["Reptile", "Lizard"],
        ["Mammal", "Cat"],
        ["Mammal", "Wolf", "Dog"],
    ],
    dtype=object,
)

# Use random forest classifiers for every node
rf = LogisticRegression()
classifier = LocalClassifierPerNode(local_classifier=rf)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

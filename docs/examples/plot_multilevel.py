# -*- coding: utf-8 -*-
"""
==========================
Multi-Label Classification
==========================

HiClass supports assigning multiple labels to a single sample.
In this example, we will train a local classifier per node
with a hierarchy similar to the following image:

.. figure:: ../algorithms/local_classifier_per_node_multilevel.png
   :align: center

Without multi-label classification, a sample can only belong
to a single class. However, many real-world problems require
a sample to belong to multiple classes.
For example, a duck has 2 wings and 2 legs.
An Human has no wings, but 2 legs.
A dragonfly has 4 wings and 6 legs.

We can encode this information in our Y_train data as follows:
"""
from sklearn.tree import DecisionTreeClassifier

from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode

# Define data
X_train = [[1, 2], [3, 4], [5, 6]]
X_test = [[1, 2], [3, 4], [5, 6]]
Y_train = [
    [["has_wings", "2"], ["has_legs", "2"]],  # Duck
    [["has_legs", "2"]],  # Human
    [["has_wings", "4"], ["has_legs", "6"]],  # Dragonfly
]

# Use decision tree classifiers for every node
tree = DecisionTreeClassifier()
classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

# -*- coding: utf-8 -*-
"""
==========================
Multi-Label Classification
==========================

HiClass supports assigning multiple labels to a single sample.
In this example, we will train a local classifier per node
with a hierarchy depicted as the following image:

.. figure:: ../algorithms/multilabel_hierarchy.svg
   :align: center

Without multi-label classification, a sample can only belong
to a single class. However, in some cases a sample may belong
to multiple classes, especially when the classes are not mutually exclusive.

For example, in the given hierarchy if you want to encode that a sample
is a mermaid, e.g. half human and half fish, you can assign the labels for
that data point as follows.

Important to note here is that you need to specify the whole path
from the root to the most specific nodes for each multi-label, as
can be seen in the 2nd training example for a minotaur.

As a result, the predictions will be an array of fixed dimensions, with
empty levels inserted as fillers, as can be seen in the output for the 3rd
sample.
"""

from sklearn.tree import DecisionTreeClassifier

from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode

# Define data
X_train = [[1, 2], [3, 4], [5, 6]]
X_test = [[1, 2], [3, 4], [5, 6]]
Y_train = [
    [["Mammal", "Human"], ["Fish"]],  # Mermaid
    [["Mammal", "Human"], ["Mammal", "Bovine"]],  # Minotaur
    [["Mammal", "Human"]],  # just a Human
]

# Use decision tree classifiers for every node
tree = DecisionTreeClassifier()
classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

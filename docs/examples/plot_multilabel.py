# -*- coding: utf-8 -*-
"""
==============================================
Using Hierarchical Multi-Label Classification
==============================================

A simple example to show how to use multi-label classification in HiClass.
Please have a look at Algorithms Overview Section for :ref:`Multi-Label Classification` for the motivation and background behind the implementation.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode

# Define data
X_train = [[1, 2], [3, 4], [5, 6]]
X_test = [[1, 2], [3, 4], [5, 6]]

# Define labels
Y_train = np.array(
    [
        [["Mammal", "Human"], ["Fish"]],  # Mermaid
        [["Mammal", "Human"], ["Mammal", "Bovine"]],  # Minotaur
        [["Mammal", "Human"]],  # just a Human
    ],
    dtype=object,
)

# Use decision tree classifiers for every node
tree = DecisionTreeClassifier()
classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

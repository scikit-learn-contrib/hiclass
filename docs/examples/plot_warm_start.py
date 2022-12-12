# -*- coding: utf-8 -*-
"""
===========================
Fitting additional nodes
===========================

The local classifiers per node and per parent node support `warm_start=True`,
which allows to add more estimators to an already fitted model.

Nodes that were trained previously are skipped, and only new nodes are fitted.

.. tabs::

    .. code-tab:: python
        :caption: LocalClassifierPerNode

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, warm_start=True)
        classifier.fit(X, y)


    .. code-tab:: python
        :caption: LocalClassifierPerParentNode

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerParentNode(local_classifier=rf, warm_start=True)
        classifier.fit(X, y)

In the code below, there is a working example with the local classifier per parent node.
However, the code can be easily updated by replacing lines 21-22 with the example shown in the tabs above.

"""
from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerParentNode

# Define data
X_1 = [[1], [2]]
X_2 = [[3], [4], [5]]
Y_1 = [
    ["Animal", "Mammal", "Sheep"],
    ["Animal", "Mammal", "Cow"],
]
Y_2 = [
    ["Animal", "Reptile", "Snake"],
    ["Animal", "Reptile", "Lizard"],
    ["Animal", "Mammal", "Cow"],
]
X_test = [[5], [4], [3], [2], [1]]

# Use logistic regression classifiers for every parent node
# And warm_start=True to allow training with more data in the future.
lr = LogisticRegression()
classifier = LocalClassifierPerParentNode(local_classifier=lr, warm_start=True)

# Train local classifier per parent node
classifier.fit(X_1, Y_1)

# Fit additional data
classifier.fit(X_2, Y_2)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

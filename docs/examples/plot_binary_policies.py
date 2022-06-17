# -*- coding: utf-8 -*-
"""
===========================
Binary Training Policies
===========================

The siblings policy is used by default on the local classifier per node, but the remaining ones can be selected with the parameter :literal:`binary_policy`, for example:

.. tabs::

    .. code-tab:: python
        :caption: Exclusive

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="exclusive")

    .. code-tab:: python
        :caption: Less exclusive

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="less_exclusive")

    .. code-tab:: python
        :caption: Less inclusive

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="less_inclusive")

    .. code-tab:: python
        :caption: Inclusive

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="inclusive")

    .. code-tab:: python
        :caption: Siblings

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="siblings")

    .. code-tab:: python
        :caption: Exclusive siblings

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="exclusive_siblings")

In the code below, the inclusive policy is selected.
However, the code can be easily updated by replacing lines 20-21 with the examples shown in the tabs above.

.. seealso::

   Mathematical definition on the different policies is given at :ref:`Training Policies`.
"""
from sklearn.ensemble import RandomForestClassifier

from hiclass import LocalClassifierPerNode

# Define data
X_train = [[1], [2], [3], [4]]
X_test = [[4], [3], [2], [1]]
Y_train = [
    ["Animal", "Mammal", "Sheep"],
    ["Animal", "Mammal", "Cow"],
    ["Animal", "Reptile", "Snake"],
    ["Animal", "Reptile", "Lizard"],
]

# Use random forest classifiers for every node
# And exclusive siblings policy to select training examples for binary classifiers.
rf = RandomForestClassifier()
classifier = LocalClassifierPerNode(local_classifier=rf, binary_policy="inclusive")

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

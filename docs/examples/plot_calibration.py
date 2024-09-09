# -*- coding: utf-8 -*-
"""
=====================
Calibrating a Classifier
=====================

A minimalist example showing how to calibrate a Hiclass LCN model. The calibration method can be selected with the :literal:`calibration_method` parameter, for example:

.. tabs::

    .. code-tab:: python
        :caption: Isotonic Regression

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic'
        )

    .. code-tab:: python
        :caption: Platt scaling

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='platt'
        )

    .. code-tab:: python
        :caption: Beta scaling

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='beta'
        )

    .. code-tab:: python
        :caption: IVAP

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='ivap'
        )

    .. code-tab:: python
        :caption: CVAP

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='cvap'
        )

Furthermore, probabilites of multiple levels can be aggregated by defining a probability combiner:

.. tabs::

    .. code-tab:: python
        :caption: Multiply (Default)

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic',
            probability_combiner='multiply'
        )

    .. code-tab:: python
        :caption: Geometric Mean

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic',
            probability_combiner='geometric'
        )

    .. code-tab:: python
        :caption: Arithmetic Mean

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic',
            probability_combiner='arithmetic'
        )

    .. code-tab:: python
        :caption: No Aggregation

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic',
            probability_combiner=None
        )


A hierarchical classifier can be calibrated by calling calibrate on the model or by using a Pipeline:

.. tabs::

    .. code-tab:: python
        :caption: Default

        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic'
        )
        
        classifier.fit(X_train, Y_train)
        classifier.calibrate(X_cal, Y_cal)
        classifier.predict_proba(X_test)

    .. code-tab:: python
        :caption: Pipeline

        from hiclass import Pipeline
        
        rf = RandomForestClassifier()
        classifier = LocalClassifierPerNode(
            local_classifier=rf,
            calibration_method='isotonic'
        )

        pipeline = Pipeline([
            ('classifier', classifier),
        ])

        pipeline.fit(X_train, Y_train)
        pipeline.calibrate(X_cal, Y_cal)
        pipeline.predict_proba(X_test)

In the code below, isotonic regression is used to calibrate the model.

"""
from sklearn.ensemble import RandomForestClassifier

from hiclass import LocalClassifierPerNode

# Define data
X_train = [[1], [2], [3], [4]]
X_test = [[4], [3], [2], [1]]
X_cal = [[5], [6], [7], [8]]
Y_train = [
    ["Animal", "Mammal", "Sheep"],
    ["Animal", "Mammal", "Cow"],
    ["Animal", "Reptile", "Snake"],
    ["Animal", "Reptile", "Lizard"],
]

Y_cal = [
    ["Animal", "Mammal", "Cow"],
    ["Animal", "Mammal", "Sheep"],
    ["Animal", "Reptile", "Lizard"],
    ["Animal", "Reptile", "Snake"],
]

# Use random forest classifiers for every node
rf = RandomForestClassifier()

# Use local classifier per node with isotonic regression as calibration method
classifier = LocalClassifierPerNode(
    local_classifier=rf,
    calibration_method='isotonic',
    probability_combiner='multiply'
)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Calibrate local classifier per node
classifier.calibrate(X_cal, Y_cal)

# Predict probabilities
probabilities = classifier.predict_proba(X_test)

# Print probabilities and labels for the last level
print(classifier.classes_[2])
print(probabilities)

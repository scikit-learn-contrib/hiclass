Local Hierarchical Classifier
=============================

A :literal:`local hierarchical classifier` is a supervised machine learning model, where the output of the classification algorithm is defined over a pre-established hierarchical class taxonomy. In HiClass, there are 3 main approaches for local hierarchical classification, i.e., the most common design patterns for local hierarchical classification identified in the literature [1]_, which are the `LocalClassifierPerNode <TODO>`_, `LocalClassifierPerParentNode <TODO>`_ and `LocalClassifierPerLevel <TODO>`_. Similar to classifiers in scikit-learn, a hierarchical classifier from HiClass can be used as a building block of a machine learning pipeline.

In this example, the :literal:`LocalClassifierPerNode` is imported along with the :literal:`RandomForestClassifier` from scikit-learn:

.. code-block:: python

    from hiclass import LocalClassifierPerNode
    from sklearn.ensemble import RandomForestClassifier

We will be using a :literal:`RandomForestClassifier` for each node in the :literal:`LocalClassifierPerNode`, except for the root node. This :literal:`LocalClassifierPerNode` model will have the same structure pre-defined in the hierarchical data used to train the model. This is how to create both objects:

.. code-block:: python

    rf = RandomForestClassifier()
    classifier = LocalClassifierPerNode(local_classifier=rf)

If you wish to use the :literal:`LocalClassifierPerParentNode` or :literal:`LocalClassifierPerLevel` instead, you can just replace the first import line with one of these alternatives:

.. code-block:: python

    from hiclass import LocalClassifierPerParentNode
    from hiclass import LocalClassifierPerLevel

Then replace the constructor with one of the following options:

.. code-block:: python

    classifier = LocalClassifierPerParentNode(local_classifier=rf)
    classifier = LocalClassifierPerLevel(local_classifier=rf)

Note that the :literal:`LocalClassifierPerParentNode` will have a :literal:`RandomForestClassifier` for each parent node existing in the hierarchy, while the :literal:`LocalClassifierPerLevel` will possess a :literal:`RandomForestClassifier` for each level in the training labels.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

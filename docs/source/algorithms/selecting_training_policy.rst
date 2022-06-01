Selecting a training policy
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

.. _local-classifier-per-node-overview:

Local Classifier Per Node
=========================

One of the most popular approaches in the literature, the local classifier per node consists of training one binary classifier for each node of the class taxonomy, except for the root node. A visual representation of the local classifier per node is shown in the image below.

.. figure:: local_classifier_per_node.svg
   :align: center

   Visual representation of the local classifier per node approach, adapted from [1]_.

.. toctree::
    :hidden:

    training_policies
    selecting_training_policy

Each binary classifier is trained in parallel using the library `Ray <https://www.ray.io/>`_.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

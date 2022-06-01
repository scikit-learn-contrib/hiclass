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

Each binary classifier is trained in parallel using the library `Ray <https://www.ray.io/>`_. In order to avoid inconsistencies, prediction is performed in a top-down manner. For example, given a hypothetical test example, the local classifier per node firstly queries the binary classifiers at nodes 1 and 2. Let's suppose that in this case the probability of the test example belonging to class 1 is 0.8, while the probability of belonging to class 2 is 0.5, then class 1 is picked. At the next level, only the classifiers at nodes 1.1 and 1.2 are queried, and again the one with the highest probability is selected.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

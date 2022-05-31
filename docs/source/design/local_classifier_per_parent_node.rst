.. _local-classifier-per-parent-node-overview:

Local Classifier Per Parent Node
================================

The local classifier per parent node approach consists of training a multi-class classifier for each parent node existing in the hierarchy, as shown in the image below.

.. figure:: local_classifier_per_parent_node.svg
   :align: center

Each local classifier is trained in parallel using the library `Ray <https://www.ray.io/>`_.

   Visual representation of the local classifier per parent node approach, adapted from [1]_.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

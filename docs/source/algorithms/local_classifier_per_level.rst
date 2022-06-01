.. _local-classifier-per-level-overview:

Local Classifier Per Level
==========================

The local classifier per level approach consists of training a multi-class classifier for each level of the class taxonomy. An example is displayed in the figure below.

.. figure:: local_classifier_per_level.svg
   :align: center

   Visual representation of the local classifier per level approach, adapted from [1]_.

Similar to the other hierarchical classifiers, the local classifier per level can also be trained in parallel and prediction is performed in a top-down style to avoid inconsistencies. For example, supposing that for a given test example the classifier at the first level returns the probabilities 0.91 and 0.7 for classes 1 and 2, respectively, then the one with the highest probability is considered as the correct prediction, which in this case is class 1. For the second level, only the probabilities for classes 1.1 and 1.2 are considered and the one with the highest probability is the final prediction.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

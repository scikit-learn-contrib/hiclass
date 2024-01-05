# -*- coding: utf-8 -*-
"""
==========================
Multi-Label Classification
==========================

HiClass supports multi-label classification.
This means that a sample can belong to multiple classes at the same level of the hierarchy.

++++++++++++++++++++++++++
Motivation
++++++++++++++++++++++++++
In many hierarchical classification problems, a sample can belong to multiple classes at the same level of the hierarchy.
This is the case, for problems in which the classes are not mutually exclusive.
For example, consider a dog breed classification problem, in which we detect the breed of a dog from available data, such as images.
We know that a dog can be a mix of multiple breeds, such as a dog that is half dachshund and half Golden Retriever.
In this case, we would like to assign both the dachshund and Golden Retriever labels to the sample.

Another example, is that of Greek mythology creatures.

.. figure:: ../algorithms/multilabel_hierarchy.svg
   :align: center


++++++++++++++++++++++++++++++++++++++++
Background - Classification Terminology
++++++++++++++++++++++++++++++++++++++++
To explain what we mean with multi-label hierarchical classification, we first need to define some terminology.
Since HiClass is a sklearn-compatible library, we use the same terminology as sklearn.
In a multi-class classification problem, a sample can belong to only one of multiple classes.
In a multi-label classification problem, a sample can belong to multiple classes.
Hierarchical classification problem is a multi-label classification problem, in which the classes are organized in a hierarchy that is represented as graph, such as a tree or directed acyclic graph (DAG), in which the nodes correspond to the classes to be predicted.
In a hierarchical classification problem, the classes are organized in a hierarchy that is represented as graph, such as a tree or directed acyclic graph (DAG), in which the nodes correspond to the classes to be predicted.
A sample belonging to a single class at each level of the hierarchy, in which the level is defined as the all nodes, i.e., classes, that are the same distance from the root node.
A consequence of this definition is that a sample can only be classified by a single path through the hierarchy.
In multi-label hierarchical classification, a sample can belong to multiple classes at the same level of the hierarchy, i.e., a sample can be classified by multiple paths through the hierarchy.
In the next sections we outline how HiClass implements multi-label hierarchical classification.

++++++++++++++++++++++++++
Design - Target Format
++++++++++++++++++++++++++
Since there is no sklearn specific multi-label hierarchical format, HiClass implements its own format.
A major design goal was to come up with a format, that keeps the sklearn-compatible format in the non-multi-label hierarchical classification case and extends naturally to the multi-label hierarchical classification case.
The non-multi-label hierarchical classification format, is the specification of a single path through the hierarchy, given by a list of labels/classes from root to most specific class.

.. code-block:: python

   y = [
      ["Mammal", "Human"],
      ["Fish"]
   ]


In the multi-label hierarchical classification case, we extend this format by adding a new dimension to the 2-dimensional array which captures the different paths through the hierarchy.
This results in a 3-dimensional array in which the first dimension corresponds to the rows/samples, the second dimension corresponds to the different paths through the hierarchy and the third dimension corresponds to the concrete labels/classes in the path.
The example data therefore would we encoded as:

.. code-block:: python

   y = [
      ["Mammal", "Human"], ["Mammal", "Bovine"]
   ]

Important to note here is that we specify the whole list of nodes from the root to the most specific nodes for each path.
Even in cases in which only the leave nodes are different, we still need to specify the whole path.
This is a consequence of requiring a fixed dimension for the target array, which is required by sklearn.
Furthermore, this makes for a readable and easy to comprehend format that also works well for hierarchies that are not trees, but DAGs.


++++++++++++++++++++++++++
Fitting the Classifiers
++++++++++++++++++++++++++
In this section we outline how the fitting of the local classifiers is implemented in HiClass for multi-label hierarchical classification.
.. For background information on the classifier strategies, please refer to the :ref:`hiclass_classifier_strategies` section.
Here we only focus on the multi-label hierarchical classification case for the :class:`hiclass.MultiLabelLocalClassifierPerNode` and :class:`hiclass.MultiLabelLocalClassifierPerNode` classifiers.

Local Classifier Per Node
-------------------------
The :class:`hiclass.MultiLabelLocalClassifierPerNode` classifier is a multi-label classifier that fits a binary local classifier for each node in the hierarchy.
:class:`hiclass.BinaryPolicy` defines which samples belong to the positive and which ones to the negative class for a given local classifier.
For multi-label hierarchical classification, if a sample belongs to multiple classes at the same level of the hierarchy, it is considered to belong to the positive class for all of these classes.
Since positive and negative samples are mutually exclusive, a sample can only belong to the positive or negative class of a single local classifier.
For example, for the Retriever classifier the example image is seen as a positive sample, since it belongs to the Golden Retriever class, even though it could also be seen as a negative example, since it belongs to the Hound class.
For the Hound classifier, the example image is also seen as a positive sample, since it does not belong to the Dachshund class, which is a subclass of Hound.

Local Classifier Per Node
-------------------------
The :class:`hiclass.MultiLabelLocalClassifierPerParentNode` trains a true multi-class classifier for each non-leaf node, i.e., a node that has children in the hierarchy.
The classifier is trained on samples that have the label of one of the nodes children.
For the multi-label case this means, that a sample can belong to multiple children of a node.
Internally, this is implemented by duplicating the sample and assigning each duplicate to one of the children of the node.
Thereby the classifier itself does not need to support the sklearn multi-label format and can be a standard sklearn classifier.

++++++++++++++++++++++++++
Prediction
++++++++++++++++++++++++++
So far we have only discussed the fitting of the classifiers, in this section we outline how the prediction is implemented in HiClass for multi-label hierarchical classification.
HiClass follows a top-down prediction strategy in which a data sample is classified by nodes in the hierarchy starting from the root node.
In the single path case, the data sample is assigned the label with the highest probability.
For example, in the :class:`hiclass.MultiLabelLocalClassifierPerNode` case, the data sample is assigned the label of the binary classifier that outputs the highest probability.
This leads to only a single path through the hierarchy for each data sample.
In contrast, when we want to allow for multiple paths through the hierarchy, we need to consider other ways of assigning labels to data samples.
HiClass implements two strategies for this: Threshold and Tolerance.

Theshold
-------------------------
The Threshold strategy assigns a label to a data sample if the probability of the label is above a given threshold.
The threshold :math:`\lambda \in [0, 1]` is a parameter that is passed to the predict function, and specifies an absolute probability value.

.. math::
   Predictions(Node) = \{c \in Children(Node): \mathbb{P}(c) > \lambda\}

While this strategy is simple to implement and understand, it has the disadvantage that it is not possible to specify a different threshold for each node in the hierarchy, requiring a global threshold for all nodes.
Furthermore, with the top down-prediction strategy, if the predicted probability is below the threshold for a node, the prediction stops regardless of the probabilities of the nodes further down the hierarchy.

Tolerance
-------------------------
The Tolerance strategy assigns a label to a data sample if the probability is within a given tolerance of the highest probability.
The tolerance :math:`\gamma \in [0, 1]` is a parameter that is passed to the predict function, and specifies a relative probability value.

.. math::
   Predictions(Node) = \{ c \in Children(Node):  \mathbb{P}(c) ≥ max( \mathbb{P}(children) ) - \gamma \}

This strategy has the advantage of always predicting at least one class at each level since the tolerance is relative to the highest probability.

++++++++++++++++++++++++++
Metrics
++++++++++++++++++++++++++
To evaluate the performance of the multi-label hierarchical classifiers, we extend the hierarchical precision, recall and F-Score metrics to the multi-label case.
The hierarchical precision, recall and F-Score are defined as follows are defined in .


++++++++++++++++++++++++++
Example
++++++++++++++++++++++++++
"""

from sklearn.tree import DecisionTreeClassifier
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode

# Define data
X_train = [[1, 2], [3, 4], [5, 6]]
X_test = [[1, 2], [3, 4], [5, 6]]

# Define labels
Y_train = [
    [["Mammal", "Human"], ["Fish"]],  # Mermaid
    [["Mammal", "Human"], ["Mammal", "Bovine"]],  # Minotaur
    [["Mammal", "Human"]],  # just a Human
]

# Use decision tree classifiers for every node
tree = DecisionTreeClassifier()
classifier = MultiLabelLocalClassifierPerNode(local_classifier=tree)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

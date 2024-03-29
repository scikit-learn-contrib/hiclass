# -*- coding: utf-8 -*-
"""
=====================
Parallel Training
=====================

Larger datasets require more time for training.
While by default the models in HiClass are trained using a single core,
it is possible to train each local classifier in parallel by leveraging the library Ray [1]_.
If Ray is not installed, the parallelism defaults to Joblib.
In this example, we demonstrate how to train a hierarchical classifier in parallel by
setting the parameter :literal:`n_jobs` to use all the cores available. Training
is performed on a mock dataset from Kaggle [2]_.

.. [1] https://www.ray.io/
.. [2] https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification
"""
import sys
from os import cpu_count
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hiclass import LocalClassifierPerParentNode
from hiclass.datasets import load_hierarchical_text_classification

# Load train and test splits
X_train, X_test, Y_train, Y_test = load_hierarchical_text_classification()

# We will use logistic regression classifiers for every parent node
lr = LogisticRegression(max_iter=1000)

pipeline = Pipeline(
    [
        ("count", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        (
            "lcppn",
            LocalClassifierPerParentNode(local_classifier=lr, n_jobs=cpu_count()),
        ),
    ]
)

# Fixes bug AttributeError: '_LoggingTee' object has no attribute 'fileno'
# This only happens when building the documentation
# Hence, you don't actually need it for your code to work
sys.stdout.fileno = lambda: False

# Now, let's train the local classifier per parent node
pipeline.fit(X_train, Y_train)

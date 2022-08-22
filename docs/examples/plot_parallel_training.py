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

import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hiclass import LocalClassifierPerParentNode


# Download training data
url = "https://zenodo.org/record/6657410/files/train_40k.csv?download=1"
path = "train_40k.csv"
response = requests.get(url)
with open(path, "wb") as file:
    file.write(response.content)

# Load training data into pandas dataframe
training_data = pd.read_csv(path).fillna(" ")

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

# Select training data
X_train = training_data["Title"]
Y_train = training_data[["Cat1", "Cat2", "Cat3"]]

# Fixes bug AttributeError: '_LoggingTee' object has no attribute 'fileno'
# This only happens when building the documentation
# Hence, you don't actually need it for your code to work
sys.stdout.fileno = lambda: False

# Now, let's train the local classifier per parent node
pipeline.fit(X_train, Y_train)

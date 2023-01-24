# -*- coding: utf-8 -*-
"""
=====================
BERT sklearn
=====================

In order to use `bert-sklearn <https://github.com/charles9n/bert-sklearn>`_ with HiClass, some of scikit-learns checks need to be disabled.
The reason is that BERT expects text as input for the features, but scikit-learn expects numerical features.
Hence, the checks will fail.
To disable scikit-learn's checks, we can simply use the parameter :literal:`bert=True` in the constructor of the local hierarchical classifier.
"""
from bert_sklearn import BertClassifier
from hiclass import LocalClassifierPerParentNode

# Define data
X_train = X_test = [
    "Batman",
    "Rorschach",
]
Y_train = [
    ["Action", "The Dark Night"],
    ["Action", "Watchmen"],
]

# Use BERT for every node
bert = BertClassifier()
classifier = LocalClassifierPerParentNode(
    local_classifier=bert,
    bert=True,
)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
print(predictions)

# -*- coding: utf-8 -*-
"""
=====================
Model Persistence
=====================

HiClass is fully compatible with Pickle.
Pickle can be used to easily store machine learning models on disk.
In this example, we demonstrate how to use pickle to store and load trained classifiers.
"""

# Author: Fabio Malcher Miranda
# License: BSD 3 clause

import pickle

from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerLevel

# Define data
X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
X_test = [[7, 8], [5, 6], [3, 4], [1, 2]]
Y_train = [
    ['Animal', 'Mammal', 'Sheep'],
    ['Animal', 'Mammal', 'Cow'],
    ['Animal', 'Reptile', 'Snake'],
    ['Animal', 'Reptile', 'Lizard'],
]

# Use Logistic Regression classifiers for every level in the hierarchy
lr = LogisticRegression()
classifier = LocalClassifierPerLevel(local_classifier=lr)

# Train local classifier per level
classifier.fit(X_train, Y_train)

# Save the model to disk
filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Some time in the future...

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Predict
predictions = loaded_model.predict(X_test)
print(predictions)

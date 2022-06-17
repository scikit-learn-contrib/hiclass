# -*- coding: utf-8 -*-
"""
=====================
Building Pipelines
=====================

HiClass can be adopted in scikit-learn pipelines, and fully supports sparse matrices as input.
This example desmonstrates the use of both of these features.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hiclass import LocalClassifierPerParentNode

# Define data
X_train = [
    'Struggling to repay loan',
    'Unable to get annual report',
]
X_test = [
    'Unable to get annual report',
    'Struggling to repay loan',
]
Y_train = [
    ['Loan', 'Student loan'],
    ['Credit reporting', 'Reports']
]

# We will use logistic regression classifiers for every parent node
lr = LogisticRegression()

# Let's build a pipeline using CountVectorizer and TfidfTransformer
# to extract features as sparse matrices
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lcppn', LocalClassifierPerParentNode(local_classifier=lr)),
])

# Now, let's train local classifier per parent node
pipeline.fit(X_train, Y_train)

# Finally, let's predict using the pipeline
predictions = pipeline.predict(X_test)
print(predictions)

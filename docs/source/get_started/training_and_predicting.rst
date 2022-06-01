Training and Predicting
=======================

HiClass adheres to the same API from the popular machine learning library scikit-learn. Hence, training is as easy as calling the :literal:`fit` method on the training data:

.. code-block:: python

    classifier.fit(X_train, Y_train)

Prediction is performed by calling the :literal:`predict` method on the test features:

.. code-block:: python

    predictions = classifier.predict(X_test)

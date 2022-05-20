Hello Hiclass!
==============

It is now time to stitch the code together. Here is the full example:

.. code-block:: python

    """Contents of hello_hiclass.py"""
    from hiclass import LocalClassifierPerNode
    from sklearn.ensemble import RandomForestClassifier

    # define data
    X_train = [[1], [2], [3], [4]]
    X_test = [[4], [3], [2], [1]]
    Y_train = [
        ['Animal', 'Mammal', 'Sheep'],
        ['Animal', 'Mammal', 'Cow'],
        ['Animal', 'Reptile', 'Snake'],
        ['Animal', 'Reptile', 'Lizard'],
    ]

    # Use random forest classifiers for every node
    rf = RandomForestClassifier()
    classifier = LocalClassifierPerNode(local_classifier=rf)

    # Train local classifier per node
    classifier.fit(X_train, Y_train)

    # Predict
    predictions = classifier.predict(X_test)
    print(predictions)

Save the code above in a file called :literal:`hello_hiclass.py`, then open a terminal and run the following command:

.. code-block:: bash

    python hello_hiclass.py

The array below should be printed on the terminal:

.. code-block:: python

    [['Animal' 'Reptile' 'Lizard']
     ['Animal' 'Reptile' 'Snake']
     ['Animal' 'Mammal' 'Cow']
     ['Animal' 'Mammal' 'Sheep']]

There is more to HiClass than what is shown in this "Hello World" example, such as training with missing data points, storing trained models and computation of hierarchical metrics. These concepts are covered in the next tutorial.
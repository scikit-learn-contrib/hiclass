.. _calibration-overview:

===========================
Classifier Calibration
===========================
HiClass provides support for probability calibration using various post-hoc calibration methods. 

++++++++++++++++++++++++++
Motivation
++++++++++++++++++++++++++
While many machine learning models can output uncertainty scores, these scores are known to be often poorly calibrated [1]_ [2]_. Model calibration aims to improve the quality of probabilistic forecasts by learning a transformation of the scores, using a separate dataset.

++++++++++++++++++++++++++
Methods
++++++++++++++++++++++++++

HiClass supports the following calibration methods:

* Isotonic Regression [3]_

* Platt Scaling [4]_

* Beta Calibration [5]_

* Inductive Venn-Abers Calibration [6]_

* Cross Venn-Abers Calibration [6]_

++++++++++++++++++++++++++
Probability Aggregation
++++++++++++++++++++++++++

Combining probabilities over multiple levels is another method to improve probabilistic forecasts. The following methods are supported:

Conditional Probability Aggregation (Multiply Aggregation)
--------------
Given a node hierarchy with :math:`n` levels, the probability of a node :math:`A_i`, where :math:`i` denotes the level, is calculated as:

:math:`\displaystyle{\mathbb{P}(A_1 \cap A_2 \cap \ldots \cap A_i) = \mathbb{P}(A_1) \cdot \mathbb{P}(A_2 \mid A_1) \cdot \mathbb{P}(A_3 \mid A_1 \cap A_2) \cdot \ldots}`
:math:`\displaystyle{\cdot \mathbb{P}(A_i \mid A_1 \cap A_2 \cap \ldots \cap A_{i-1})}`

Arithmetic Mean Aggregation
--------------
:math:`\displaystyle{\mathbb{P}(A_i) = \frac{1}{i} \sum_{j=1}^{i} \mathbb{P}(A_{j})}`

Geometric Mean Aggregation
--------------
:math:`\displaystyle{\mathbb{P}(A_i) = \exp{\left(\frac{1}{i} \sum_{j=1}^{i} \ln \mathbb{P}(A_{j})\right)}}`

++++++++++++++++++++++++++
Code sample
++++++++++++++++++++++++++

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier

    from hiclass import LocalClassifierPerNode

    # Define data
    X_train = [[1], [2], [3], [4]]
    X_test = [[4], [3], [2], [1]]
    X_cal = [[5], [6], [7], [8]]
    Y_train = [
        ["Animal", "Mammal", "Sheep"],
        ["Animal", "Mammal", "Cow"],
        ["Animal", "Reptile", "Snake"],
        ["Animal", "Reptile", "Lizard"],
    ]

    Y_cal = [
        ["Animal", "Mammal", "Cow"],
        ["Animal", "Mammal", "Sheep"],
        ["Animal", "Reptile", "Lizard"],
        ["Animal", "Reptile", "Snake"],
    ]

    # Use random forest classifiers for every node
    rf = RandomForestClassifier()

    # Use local classifier per node with isotonic regression as calibration method
    classifier = LocalClassifierPerNode(
        local_classifier=rf, calibration_method="isotonic", probability_combiner="multiply"
    )

    # Train local classifier per node
    classifier.fit(X_train, Y_train)

    # Calibrate local classifier per node
    classifier.calibrate(X_cal, Y_cal)

    # Predict probabilities
    probabilities = classifier.predict_proba(X_test)

    # Print probabilities and labels for the last level
    print(classifier.classes_[2])
    print(probabilities)

.. [1] Niculescu-Mizil, Alexandru; Caruana, Rich (2005): Predicting good probabilities with supervised learning. In: Saso Dzeroski (Hg.): Proceedings of the 22nd international conference on Machine learning - ICML '05. the 22nd international conference. Bonn, Germany, 07.08.2005 - 11.08.2005. New York, New York, USA: ACM Press, S. 625-632.

.. [2] Chuan Guo; Geoff Pleiss; Yu Sun; Kilian Q. Weinberger (2017): On Calibration of Modern Neural Networks. In: Doina Precup und Yee Whye Teh (Hg.): Proceedings of the 34th International Conference on Machine Learning, Bd. 70: PMLR (Proceedings of Machine Learning Research), S. 1321-1330.

.. [3] Zadrozny, Bianca; Elkan, Charles (2002): Transforming classifier scores into accurate multiclass probability estimates. In: Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: Association for Computing Machinery (KDD ’02), S. 694-699.

.. [4] Platt, John (2000): Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In: Adv. Large Margin Classif. 10.

.. [5] Kull, Meelis; Filho, Telmo Silva; Flach, Peter (2017): Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. In: Aarti Singh und Jerry Zhu (Hg.): Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, Bd. 54: PMLR (Proceedings of Machine Learning Research), S. 623-631.

.. [6] Vovk, Vladimir; Petej, Ivan; Fedorova, Valentina (2015): Large-scale probabilistic predictors with and without guarantees of validity. In: C. Cortes, N. Lawrence, D. Lee, M. Sugiyama und R. Garnett (Hg.): Advances in Neural Information Processing Systems, Bd. 28: Curran Associates, Inc. 


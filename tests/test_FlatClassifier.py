import numpy as np
from numpy.testing import assert_array_equal

from hiclass import FlatClassifier


def test_fit_predict():
    flat = FlatClassifier()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([["a", "b"], ["b", "c"]])
    flat.fit(x, y)
    predictions = flat.predict(x)
    assert_array_equal(y, predictions)

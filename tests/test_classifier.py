import numpy as np
import pytest

from hiclass.Classifier import (
    DuplicateFilter,
    Classifier,
    ConstantClassifier,
    NodeClassifier,
)
from sklearn.linear_model import LogisticRegression


class Message(object):
    def __init__(self, msg):
        self.msg = msg


class EstimatorWithoutClone:
    pass


def test_duplicate_filter():
    duplicate_filter = DuplicateFilter()
    assert duplicate_filter.filter(Message("Test message"))
    assert not duplicate_filter.filter(Message("Test message"))
    assert not duplicate_filter.filter(Message("Test message"))
    assert duplicate_filter.filter(Message("New message"))


def test_not_implemented_methods():
    classifier = Classifier()
    dummy_data = np.array([[1]])
    classifier.fit(dummy_data, dummy_data)
    with pytest.raises(NotImplementedError):
        classifier.max_depth = 0
        classifier.placeholder_label = -1
        classifier.predict(dummy_data)
    with pytest.raises(NotImplementedError):
        classifier.get_classifier(1)
    with pytest.raises(NotImplementedError):
        classifier.set_classifier(1, LogisticRegression())
    with pytest.raises(NotImplementedError):
        classifier._get_nodes_to_predict(0)
    with pytest.raises(NotImplementedError):
        classifier._choose_required_prediction_args(0, [0], dummy_data)
    with pytest.raises(NotImplementedError):
        classifier._stopping_criterion(0)
    with pytest.raises(NotImplementedError):
        classifier._create_prediction(dummy_data)
    with pytest.raises(NotImplementedError):
        classifier._create_prediction(
            [0], dummy_data, dummy_data, dummy_data, 1, dummy_data, 1, dummy_data
        )
    # trigger node classifier
    node_classifier = NodeClassifier()
    with pytest.raises(NotImplementedError):
        node_classifier._get_nodes_with_classifier()


def test_correct_non_sklean_clone():
    classifier = Classifier(local_classifier=EstimatorWithoutClone())
    with pytest.raises(AttributeError):
        classifier._copy_classifier()
    dummy_clone = "Estimator"
    classifier.local_classifier.clone = lambda: dummy_clone
    assert classifier._copy_classifier() == dummy_clone


def test_correct_max_depth_init():
    classifier = Classifier()
    classifier._initialize_classifiers = lambda: None
    dummy_data = np.array([[1], [2]])
    dummy_labels = np.array([[2, 4], [3, 5]])

    classifier.fit(dummy_data, dummy_labels)
    assert classifier.max_depth == 3


@pytest.mark.parametrize("current_class", [0, 1, 2, 3])
def test_ConstantClassifier(current_class):
    x = ConstantClassifier(current_class, 4)
    result = x.predict_proba([[1], [2]])
    output = [0, 0, 0, 0]
    output[current_class] = 1
    assert np.array_equal(result, [output] * 2)

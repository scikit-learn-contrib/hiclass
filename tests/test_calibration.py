import pytest
import numpy as np
from unittest.mock import Mock
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from hiclass._calibration.VennAbersCalibrator import _InductiveVennAbersCalibrator, _CrossVennAbersCalibrator
from hiclass._calibration.PlattScaling import _PlattScaling
from hiclass._calibration.IsotonicRegression import _IsotonicRegression
from hiclass._calibration.Calibrator import _Calibrator

@pytest.fixture
def binary_calibration_data():
    prob = np.array([[0.37, 0.63],
                     [0.39, 0.61],
                     [0.42, 0.58],
                     [0.51, 0.49],
                     [0.51, 0.49],
                     [0.45, 0.55],
                     [0.48, 0.52],
                     [0.60, 0.40],
                     [0.54, 0.46],
                     [0.57, 0.43],
                     [0.57, 0.43],
                     [0.62, 0.38]]) 
    
    assert_array_equal(np.sum(prob, axis=1), np.ones(len(prob)))

    ground_truth_labels = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0])
    return prob, ground_truth_labels

@pytest.fixture
def binary_test_scores():
    X = np.array([[0.22, 0.78],
                  [0.48, 0.52],
                  [0.66, 0.34],
                  [0.01, 0.99],
                  [0.77, 0.23],
                  [0.50, 0.50],
                  [1.00, 0.00]])
    
    assert_array_equal(np.sum(X, axis=1), np.ones(len(X)))
    return X

@pytest.fixture
def binary_cal_X():
    X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8], 
                  [1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8], 
                  [1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8]])
    return X

@pytest.fixture
def binary_mock_estimator(binary_calibration_data, binary_test_scores):
    # return calibration scores or test scores depending on input size
    side_effect = lambda X: binary_calibration_data[0] if len(X) == len(binary_calibration_data[0]) else binary_test_scores

    lr = LogisticRegression()
    binary_estimator = Mock(spec=lr)
    binary_estimator.predict_proba.side_effect = side_effect
    binary_estimator.classes_ = np.array([0, 1])
    binary_estimator.get_params.return_value = lr.get_params()
    return binary_estimator

@pytest.fixture
def multiclass_calibration_data():
    prob = np.array([[0.12, 0.24, 0.64],
                     [0.44, 0.22, 0.34],
                     [0.30, 0.40, 0.30],
                     [0.11, 0.33, 0.56],
                     [0.44, 0.22, 0.34],
                     [0.66, 0.33, 0.01],
                     [0.14, 0.66, 0.20],
                     [0.44, 0.12, 0.44],
                     [0.64, 0.24, 0.12],
                     [0.20, 0.77, 0.03]
    ])

    assert_array_equal(np.sum(prob, axis=1), np.ones(len(prob)))

    ground_truth_labels = np.array([2, 0, 1, 2, 0, 0, 1, 0, 0, 1])
    return prob, ground_truth_labels

@pytest.fixture
def multiclass_test_scores():
    X = np.array([[0.23, 0.47, 0.30],
                  [0.44, 0.21, 0.35],
                  [0.22, 0.22, 0.56],
                  [0.25, 0.50, 0.25],
                  [0.01, 0.72, 0.27],
                  [0.12, 0.35, 0.53],
                  [0.30, 0.23, 0.47]
    ])
    
    assert_array_equal(np.sum(X, axis=1), np.ones(len(X)))
    return X

@pytest.fixture
def multiclass_mock_estimator(multiclass_calibration_data, multiclass_test_scores):
    # return calibration scores or test scores depending on input size
    side_effect = lambda X: multiclass_calibration_data[0] if len(X) == len(multiclass_calibration_data[0]) else multiclass_test_scores

    multiclass_estimator = Mock(spec=LogisticRegression)
    multiclass_estimator.predict_proba.side_effect = side_effect
    multiclass_estimator.classes_ = np.array([0, 1, 2])
    return multiclass_estimator

def test_inductive_venn_abers_calibrator(binary_calibration_data, binary_test_scores):
    scores, ground_truth_labels = binary_calibration_data
    test_scores = binary_test_scores
    calibrator = _InductiveVennAbersCalibrator()
    calibrator.fit(scores=scores[:, 1], y=ground_truth_labels)

    intervalls = calibrator.predict_intervall(test_scores[:, 1])
    proba = calibrator.predict_proba(test_scores[:, 1])

    assert_array_almost_equal(calibrator.F1, np.array([0, 0.33333333, 0.375, 0.375, 0.4, 0.5, 0.5, 0.66666666, 0.66666666, 1.0, 1.0]))
    assert_array_almost_equal(calibrator.F0, np.array([0, 0, 0.2, 0.2, 0.2, 0.25, 0.25, 0.33333333, 0.33333333, 0.5, 0.66666666]))
    assert_array_almost_equal(intervalls, np.array([[0.66666666, 1.0], [0.25, 0.66666666], [0, 0.33333333], [0.66666666, 1.0], [0, 0.33333333], [0.25, 0.5], [0,0.33333333]]))
    assert_array_almost_equal(proba, np.array([0.74999999, 0.47058823, 0.24999999, 0.74999999, 0.24999999, 0.4, 0.24999999]))

def test_cross_venn_abers_calibrator(binary_calibration_data, binary_test_scores, binary_cal_X):
    cal_scores, y_cal = binary_calibration_data
    test_scores = binary_test_scores

    calibrator = _CrossVennAbersCalibrator(LogisticRegression(), n_folds=5)
    calibrator.fit(y_cal, cal_scores, X=binary_cal_X)
    proba = calibrator.predict_proba(test_scores[:, 1])
    expected = np.array([0.602499, 0.51438, 0.397501, 0.602499, 0.346239, 0.51438, 0.328119])
    assert len(calibrator.ivaps) == 5
    assert_array_almost_equal(proba, expected)

def test_platt_scaling(binary_calibration_data, binary_test_scores):
    cal_scores, cal_labels = binary_calibration_data
    calibrator = _PlattScaling()
    calibrator.fit(cal_labels, cal_scores[:, 1])
    proba = calibrator.predict_proba(binary_test_scores[:, 1])

    assert proba.shape == (len(binary_test_scores),)
    assert_array_almost_equal(proba, np.array([0.8827398, 0.46130191, 0.15976229, 0.97756289, 0.07045925, 0.42011212, 0.01095897]))

def test_isotonic_regression(binary_calibration_data, binary_test_scores):
    cal_scores, cal_labels = binary_calibration_data
    calibrator = _IsotonicRegression()
    calibrator.fit(cal_labels, cal_scores[:, 1])
    proba = calibrator.predict_proba(binary_test_scores[:, 1])
    
    assert proba.shape == (len(binary_test_scores),)
    assert_array_almost_equal(proba, np.array([1.0, 0.33333333, 0.0, 1.0, 0.0, 0.33333333, 0.0]))

def test_illegal_calibration_method_raises_error(binary_mock_estimator):
    with pytest.raises(ValueError, match="abc is not a valid calibration method."):
        _Calibrator(binary_mock_estimator, method="abc")

def test_not_fitted_calibrator_throws_error(binary_test_scores, binary_mock_estimator):
    for calibrator in [_PlattScaling(), 
                       _IsotonicRegression(), 
                       _InductiveVennAbersCalibrator(), 
                       _CrossVennAbersCalibrator(binary_mock_estimator)]:
            with pytest.raises(NotFittedError):
                calibrator.predict_proba(binary_test_scores)

def test_valid_calibration(binary_calibration_data, binary_test_scores, binary_cal_X, binary_mock_estimator):
    _, y_cal = binary_calibration_data

    for method in _Calibrator.available_methods:
        lr = LogisticRegression()
        lr.fit(binary_cal_X, y_cal)

        calibrator = _Calibrator(lr, method)
        calibrator.fit(X=binary_cal_X, y=y_cal)
        proba = calibrator.predict_proba(binary_test_scores)

        assert proba.shape == (len(binary_test_scores),2)

def test_multiclass_calibration(multiclass_calibration_data, multiclass_test_scores, multiclass_mock_estimator):
    scores, y_cal = multiclass_calibration_data

    calibrator = _Calibrator(multiclass_mock_estimator, method="ivap")
    calibrator.fit(X=scores, y=y_cal)
    assert len(calibrator.calibrators) == scores.shape[1]

    proba = calibrator.predict_proba(multiclass_test_scores)
    assert proba.shape == multiclass_test_scores.shape
    assert_array_almost_equal(np.sum(proba, axis=1), np.ones(len(proba)))

    expected = np.array([[0.27777778, 0.55555556, 0.16666667],
                         [0.52173913, 0.13043478, 0.34782609],
                         [0.33333333, 0.16666667, 0.5       ],
                         [0.28571429, 0.57142857, 0.14285714],
                         [0.13483146, 0.70786517, 0.15730337],
                         [0.16666667, 0.41666667, 0.41666667],
                         [0.42857143, 0.14285714, 0.42857143]])
    
    assert_array_almost_equal(proba, expected)

def test_multiclass_probability_merge(multiclass_mock_estimator, multiclass_calibration_data, multiclass_test_scores):
    calibrator = _Calibrator(estimator=multiclass_mock_estimator, method="ivap")
    X_cal, y_cal = multiclass_calibration_data
    calibrator.fit(X_cal, y_cal)

    calibrator_1 = Mock(spec=_PlattScaling)
    calibrator_1.predict_proba.return_value = np.array([0.40, 0.60, 0.20, 0.80, 0.75, 0.25, 0.85])
    
    calibrator_2 = Mock(spec=_PlattScaling)
    calibrator_2.predict_proba.return_value = np.array([0.30, 0.70, 0.35, 0.65, 0.30, 0.70, 0.20])

    calibrator_3 = Mock(spec=_PlattScaling)
    calibrator_3.predict_proba.return_value = np.array([0.55, 0.45, 0.10, 0.90, 0.45, 0.55, 0.40])

    calibrator.calibrators = [calibrator_1, calibrator_2, calibrator_3]
    proba = calibrator.predict_proba(multiclass_test_scores)

    expected = np.array([[0.32, 0.24, 0.44],
                         [0.34, 0.40, 0.26],
                         [0.31, 0.54, 0.15],
                         [0.34, 0.28, 0.38],
                         [0.5,  0.20, 0.30],
                         [0.17, 0.46, 0.37],
                         [0.59, 0.14, 0.27]])
    
    assert_array_equal(np.sum(expected, axis=1), np.ones(len(expected)))
    assert_array_almost_equal(proba, expected, decimal=2)

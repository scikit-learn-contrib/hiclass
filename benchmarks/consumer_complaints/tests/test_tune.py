import resource

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from pyfakefs.fake_filesystem_unittest import Patcher
from scripts.data import flatten_labels
from scripts.tune import (
    configure_pipeline,
    compute_md5,
    save_trial,
    load_trial,
    limit_memory,
    cross_validate,
    delete_non_hyperparameters,
)
from sklearn.pipeline import Pipeline


@pytest.fixture
def random_forest_config():
    cfg = DictConfig(
        {
            "model": "flat",
            "classifier": "random_forest",
            "n_jobs": 1,
            "n_estimators": 100,
            "criterion": "entropy",
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "min_weight_fraction_leaf": 0.5,
            "min_impurity_decrease": 0.4,
            "bootstrap": True,
            "oob_score": False,
            "class_weight": "balanced",
            "ccp_alpha": 0.3,
            "output_dir": ".",
            "n_splits": 2,
        }
    )
    return cfg


def test_configure_pipeline_1(random_forest_config):
    classifier = configure_pipeline(random_forest_config)
    assert classifier is not None
    assert isinstance(classifier, Pipeline)


def test_compute_md5_1(random_forest_config):
    hyperparameters = delete_non_hyperparameters(random_forest_config)
    md5 = compute_md5(hyperparameters)
    expected = "14a11881777bc9589583efba8af9b752"
    assert expected == md5


def test_save_and_load_trial_1(random_forest_config):
    with Patcher():
        print(random_forest_config)
        print(type(random_forest_config))
        save_trial(random_forest_config, [1, 2, 3])
        scores = load_trial(random_forest_config)
        assert [1, 2, 3] == scores


@pytest.fixture
def logistic_regression_config():
    cfg = DictConfig(
        {
            "model": "local_classifier_per_node",
            "classifier": "logistic_regression",
            "n_jobs": 1,
            "penalty": "l2",
            "dual": False,
            "tol": 1e-6,
            "C": 0.1,
            "fit_intercept": True,
            "intercept_scaling": 5,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 100,
            "multi_class": "auto",
            "output_dir": ".",
            "n_splits": 2,
        }
    )
    return cfg


def test_configure_pipeline_2(logistic_regression_config):
    classifier = configure_pipeline(logistic_regression_config)
    assert classifier is not None
    assert isinstance(classifier, Pipeline)


def test_compute_md5_2(logistic_regression_config):
    hyperparameters = delete_non_hyperparameters(logistic_regression_config)
    md5 = compute_md5(hyperparameters)
    expected = "f393e84d6faa5dc4c62aacbf70ef5865"
    assert expected == md5


def test_save_and_load_trial_2(logistic_regression_config):
    logistic_regression_config.output_dir = "."
    with Patcher():
        save_trial(logistic_regression_config, [4, 5, 6])
        scores = load_trial(logistic_regression_config)
        assert [4, 5, 6] == scores


def test_load_trial(logistic_regression_config):
    logistic_regression_config.output_dir = "."
    with Patcher():
        scores = load_trial(logistic_regression_config)
        assert [] == scores


def test_limit_memory():
    limit_memory(2)
    expected = 2 * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert expected == soft
    assert expected == hard


@pytest.fixture
def X():
    X = pd.Series(
        [
            "I am a complaint",
            "I am another complaint",
            "I am a third complaint",
            "I am a fourth complaint",
        ]
    )
    return X


@pytest.fixture
def y():
    y = pd.DataFrame(
        {
            "Product": ["a", "a", "a", "a"],
            "Sub-product": ["b", "b", "b", "b"],
        }
    )
    return y


def test_cross_validate_1(logistic_regression_config, X, y):
    with Patcher():
        scores = cross_validate(logistic_regression_config, X, y)
        assert [1.0, 1.0] == scores


def test_cross_validate_2(random_forest_config, X, y):
    with Patcher():
        y = flatten_labels(y)
        scores = cross_validate(random_forest_config, X, y)
        assert [1.0, 1.0] == scores


def test_cross_validate_3(logistic_regression_config, X, y):
    with Patcher():
        save_trial(logistic_regression_config, [0.5, 0.5])
        scores = cross_validate(logistic_regression_config, X, y)
        assert [0.5, 0.5] == scores


def test_cross_validate_4(logistic_regression_config, X, y):
    with Patcher():
        save_trial(logistic_regression_config, [0.5])
        scores = cross_validate(logistic_regression_config, X, y)
        assert [0.5, 1.0] == scores
        scores = load_trial(logistic_regression_config)
        assert [0.5, 1.0] == scores


@pytest.fixture
def hyperparameters():
    hp = OmegaConf.create(
        {
            "model": "flat",
            "classifier": "lightgbm",
            "n_jobs": 1,
            "x_train": "x_train.csv",
            "y_train": "y_train.csv",
            "output_dir": ".",
            "mem_gb": 1,
            "n_splits": 2,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }
    )
    return hp


def test_delete_non_hyperparameters(hyperparameters):
    hyperparameters = delete_non_hyperparameters(hyperparameters)
    expected = {
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
    }
    assert expected == hyperparameters

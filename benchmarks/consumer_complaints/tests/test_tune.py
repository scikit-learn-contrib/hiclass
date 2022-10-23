import pytest
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from scripts.tune import (
    configure_lightgbm,
    configure_logistic_regression,
    configure_random_forest,
    configure_pipeline,
    compute_md5,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_configure_lightgbm():
    cfg = DictConfig(
        {
            "n_jobs": 1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample_for_bin": 200000,
            "class_weight": None,
            "min_split_gain": 0.0,
            "min_child_weight": 0.001,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        }
    )
    classifier = configure_lightgbm(cfg)
    assert classifier is not None
    assert isinstance(classifier, LGBMClassifier)


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
        }
    )
    return cfg


def test_configure_random_forest(random_forest_config):
    classifier = configure_random_forest(random_forest_config)
    assert classifier is not None
    assert isinstance(classifier, RandomForestClassifier)


def test_configure_pipeline_1(random_forest_config):
    classifier = configure_pipeline(random_forest_config)
    assert classifier is not None
    assert isinstance(classifier, Pipeline)


def test_compute_md5_1(random_forest_config):
    md5 = compute_md5(random_forest_config)
    expected = "bce939a7765a18f23cb1133ae9eb2cac"
    assert expected == md5


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
        }
    )
    return cfg


def test_configure_logistic_regression(logistic_regression_config):
    classifier = configure_logistic_regression(logistic_regression_config)
    assert classifier is not None
    assert isinstance(classifier, LogisticRegression)


def test_configure_pipeline_2(logistic_regression_config):
    classifier = configure_pipeline(logistic_regression_config)
    assert classifier is not None
    assert isinstance(classifier, Pipeline)


def test_compute_md5_2(logistic_regression_config):
    md5 = compute_md5(logistic_regression_config)
    expected = "ec072c7a92187ba58487d55ac6633332"
    assert expected == md5

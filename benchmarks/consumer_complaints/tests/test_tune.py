from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from scripts.tune import configure_lightgbm, configure_logistic_regression
from sklearn.linear_model import LogisticRegression


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


def test_configure_logistic_regression():
    cfg = DictConfig(
        {
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
    classifier = configure_logistic_regression(cfg)
    assert classifier is not None
    assert isinstance(classifier, LogisticRegression)

import pandas as pd
from pandas.testing import assert_series_equal
from scripts.train import join, get_flat_classifier, get_hierarchical_classifier
from scripts.train import parse_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)


def test_parser():
    parser = parse_args(
        [
            "--n-jobs",
            "8",
            "--x-train",
            "x_train.csv.zip",
            "--y-train",
            "y_train.csv.zip",
            "--trained-model",
            "model.sav",
            "--classifier",
            "xgboost",
            "--random-state",
            "0",
            "--model",
            "flat",
        ]
    )
    assert parser.n_jobs is not None
    assert 8 == parser.n_jobs
    assert parser.x_train is not None
    assert "x_train.csv.zip" == parser.x_train
    assert parser.y_train is not None
    assert "y_train.csv.zip" == parser.y_train
    assert parser.trained_model is not None
    assert "model.sav" == parser.trained_model
    assert parser.classifier is not None
    assert "xgboost" == parser.classifier
    assert parser.random_state is not None
    assert 0 == parser.random_state
    assert parser.model is not None
    assert "flat" == parser.model


def test_join_1():
    y = pd.DataFrame(
        {
            "Product": ["Debt collection", "Checking or savings account"],
            "Sub-product": ["I do not know", "Checking account"],
        }
    )
    flat_y = join(y)
    ground_truth = pd.Series(
        [
            "Debt collection:sep:I do not know",
            "Checking or savings account:sep:Checking account",
        ]
    )
    assert_series_equal(ground_truth, flat_y)


def test_join_2():
    y = pd.DataFrame(
        {
            "Product": ["Debt collection", "Checking or savings account"],
            "Sub-product": ["I do not know", "Checking account"],
        }
    )
    separator = ","
    flat_y = join(y, separator)
    ground_truth = pd.Series(
        [
            "Debt collection,I do not know",
            "Checking or savings account,Checking account",
        ]
    )
    assert_series_equal(ground_truth, flat_y)


def test_get_flat_classifier():
    n_jobs = 1
    base_classifier = "logistic_regression"
    random_state = 0
    flat = get_flat_classifier(n_jobs, base_classifier, random_state)
    assert flat is not None
    assert isinstance(flat, LogisticRegression)
    base_classifier = "random_forest"
    flat = get_flat_classifier(n_jobs, base_classifier, random_state)
    assert flat is not None
    assert isinstance(flat, RandomForestClassifier)
    base_classifier = "xgboost"
    flat = get_flat_classifier(n_jobs, base_classifier, random_state)
    assert flat is not None
    assert isinstance(flat, XGBClassifier)


def test_get_hierarchical_classifier():
    n_jobs = 1
    local_classifier = "logistic_regression"
    hierarchical_classifier = "local_classifier_per_node"
    random_state = 0
    model = get_hierarchical_classifier(
        n_jobs, local_classifier, hierarchical_classifier, random_state
    )
    assert model is not None
    assert isinstance(model, LocalClassifierPerNode)
    local_classifier = "random_forest"
    hierarchical_classifier = "local_classifier_per_parent_node"
    model = get_hierarchical_classifier(
        n_jobs, local_classifier, hierarchical_classifier, random_state
    )
    assert model is not None
    assert isinstance(model, LocalClassifierPerParentNode)
    local_classifier = "xgboost"
    hierarchical_classifier = "local_classifier_per_level"
    model = get_hierarchical_classifier(
        n_jobs, local_classifier, hierarchical_classifier, random_state
    )
    assert model is not None
    assert isinstance(model, LocalClassifierPerLevel)

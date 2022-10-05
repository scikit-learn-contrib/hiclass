from io import StringIO

import pytest
from omegaconf import DictConfig
from pyfakefs.fake_filesystem_unittest import Patcher
from scripts.train import parse_args, load_parameters


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
            "lightgbm",
            "--random-state",
            "0",
            "--model",
            "flat",
            "--best-parameters",
            "best_parameters.yml",
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
    assert "lightgbm" == parser.classifier
    assert parser.random_state is not None
    assert 0 == parser.random_state
    assert parser.model is not None
    assert "flat" == parser.model
    assert parser.best_parameters is not None
    assert "best_parameters.yml" == parser.best_parameters


@pytest.fixture
def tuned_parameters():
    cfg = StringIO()
    cfg.write("name: optuna\n")
    cfg.write("best_params:\n")
    cfg.write("  C: 0.001\n")
    cfg.write("  class_weight: balanced\n")
    cfg.write("  dual: false\n")
    cfg.write("  fit_intercept: false\n")
    cfg.write("  intercept_scaling: 3\n")
    cfg.write("  max_iter: 100\n")
    cfg.write("  multi_class: auto\n")
    cfg.write("  penalty: l2\n")
    cfg.write("  solver: liblinear\n")
    cfg.write("  tol: 1.0e-06\n")
    cfg.write("best_value: 0.9387345438252911\n")
    cfg.seek(0)
    return cfg


def test_load_parameters(tuned_parameters):
    expected = DictConfig(
        {
            "C": 0.001,
            "class_weight": "balanced",
            "dual": False,
            "fit_intercept": False,
            "intercept_scaling": 3,
            "max_iter": 100,
            "multi_class": "auto",
            "penalty": "l2",
            "solver": "liblinear",
            "tol": 1.0e-06,
        }
    )
    with Patcher() as patcher:
        patcher.fs.create_file("best_parameters.yml", contents=tuned_parameters.read())
        parameters = load_parameters("best_parameters.yml")
    assert expected == parameters

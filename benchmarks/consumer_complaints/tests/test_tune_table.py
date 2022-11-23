import pytest
from omegaconf import OmegaConf
from scripts.tune_table import parse_args, delete_non_hyperparameters


def test_parser():
    parser = parse_args(
        [
            "--folder",
            "folder",
            "--model",
            "flat",
            "--classifier",
            "lightgbm",
            "--output",
            "output.md",
        ]
    )
    assert parser.folder is not None
    assert "folder" == parser.folder
    assert parser.model is not None
    assert "flat" == parser.model
    assert parser.classifier is not None
    assert "lightgbm" == parser.classifier
    assert parser.output is not None
    assert "output.md" == parser.output


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

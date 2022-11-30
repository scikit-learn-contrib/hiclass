from argparse import Namespace
from io import StringIO

import pytest
from omegaconf import OmegaConf
from pyfakefs.fake_filesystem_unittest import Patcher
from scripts.tune_table import (
    parse_args,
    compute,
    create_table,
)

from scripts.tune import save_trial


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
def lightgbm_config():
    cfg = OmegaConf.create(
        {
            "model": "flat",
            "classifier": "lightgbm",
            "n_jobs": 12,
            "x_train": "x_train.csv",
            "y_train": "y_train.csv",
            "output_dir": ".",
            "mem_gb": 1,
            "n_splits": 2,
            "num_leaves": 100,
            "n_estimators": 200,
            "min_child_samples": 6,
        }
    )
    return cfg


def test_compute(lightgbm_config):
    expected_hyperparameters = {
        "num_leaves": 100,
        "n_estimators": 200,
        "min_child_samples": 6,
    }
    with Patcher():
        save_trial(lightgbm_config, [1, 2, 3])
        hyperparameters, scores, avg, std = compute(".")
        assert [expected_hyperparameters] == hyperparameters
        assert [[1, 2, 3]] == scores
        assert [2] == avg
        assert [0.816496580927726] == std


@pytest.fixture
def args():
    return Namespace(
        output="output.md",
        model="flat",
        classifier="lightgbm",
        folder=".",
    )


@pytest.fixture
def expected_content():
    content = StringIO()
    content.write("# Model: flat\n")
    content.write("## Base classifier: lightgbm\n")
    content.write("|Parameters|Scores|Average|Standard deviation|\n")
    content.write("|----------|------|-------|------------------|\n")
    content.write(
        "|{'num_leaves': 100, 'n_estimators': 200, 'min_child_samples': 6}|[1, 2, 3]|2.000|0.816|\n"
    )
    return content.getvalue()


def test_create_table(lightgbm_config, args, expected_content):
    with Patcher() as patcher:
        save_trial(lightgbm_config, [1, 2, 3])
        create_table(args)
        assert patcher.fs.exists("output.md")
        with open("output.md", "r") as f:
            content = f.read()
            print(expected_content)
            print(content)
            assert expected_content == content

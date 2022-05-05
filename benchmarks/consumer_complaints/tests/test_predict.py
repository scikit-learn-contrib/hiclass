import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.predict import separate, parse_args


def test_parser():
    parser = parse_args(
        [
            "--trained-model",
            "model.sav",
            "--x-test",
            "x_test.csv.zip",
            "--predictions",
            "predictions.csv.zip",
            "--classifier",
            "hist_gradient",
        ]
    )
    assert parser.trained_model is not None
    assert "model.sav" == parser.trained_model
    assert parser.x_test is not None
    assert "x_test.csv.zip" == parser.x_test
    assert parser.predictions is not None
    assert "predictions.csv.zip" == parser.predictions
    assert parser.classifier is not None
    assert "hist_gradient" == parser.classifier


def test_separate_1():
    y = np.array(
        [
            "Debt collection:sep:I do not know",
            "Checking or savings account:sep:Checking account",
        ]
    )
    y = separate(y)
    ground_truth = pd.DataFrame(
        [
            ["Debt collection", "I do not know"],
            ["Checking or savings account", "Checking account"],
        ]
    )
    assert_frame_equal(ground_truth, y)


def test_separate_2():
    y = np.array(
        [
            "Debt collection/I do not know",
            "Checking or savings account/Checking account",
        ]
    )
    separator = "/"
    y = separate(y, separator)
    ground_truth = pd.DataFrame(
        [
            ["Debt collection", "I do not know"],
            ["Checking or savings account", "Checking account"],
        ]
    )
    assert_frame_equal(ground_truth, y)

from datetime import date

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from pyfakefs.fake_filesystem_unittest import Patcher
from scripts.statistics import parse_args

from scripts.statistics import get_file_modification, create_dataframe, save_statistics


def test_parser():
    parser = parse_args(
        [
            "--data",
            "complaints.csv.zip",
            "--x-train",
            "x_train.csv.zip",
            "--y-train",
            "y_train.csv.zip",
            "--x-test",
            "x_test.csv.zip",
            "--y-test",
            "y_test.csv.zip",
            "--statistics",
            "statistics.csv",
        ]
    )
    assert parser.data is not None
    assert "complaints.csv.zip" == parser.data
    assert parser.x_train is not None
    assert "x_train.csv.zip" == parser.x_train
    assert parser.y_train is not None
    assert "y_train.csv.zip" == parser.y_train
    assert parser.x_test is not None
    assert "x_test.csv.zip" == parser.x_test
    assert parser.y_test is not None
    assert "y_test.csv.zip" == parser.y_test
    assert parser.statistics is not None
    assert "statistics.csv" == parser.statistics


def test_get_file_modification():
    with Patcher() as patcher:
        patcher.fs.create_file("complaints.csv.zip")
        assert date.today().strftime("%d/%m/%Y") == get_file_modification(
            "complaints.csv.zip"
        )


def test_create_dataframe():
    statistics = create_dataframe("2021-01-01", 70, 30)
    assert (1, 3) == statistics.shape
    assert "2021-01-01" == statistics["Snapshot"].values[0]
    assert 70 == statistics["Training set size"].values[0]
    assert 30 == statistics["Test set size"].values[0]


@pytest.fixture
def statistics():
    statistics = pd.DataFrame(
        {
            "Snapshot": ["02/11/2020"],
            "Training set size": [70],
            "Test set size": [30],
        }
    )
    return statistics


def test_save_statistics(statistics):
    with Patcher() as patcher:
        save_statistics(statistics, "statistics.csv")
        assert patcher.fs.exists("statistics.csv")
        result = pd.read_csv("statistics.csv")
        assert_frame_equal(statistics, result)

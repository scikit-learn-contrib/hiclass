from datetime import date

from pyfakefs.fake_filesystem_unittest import Patcher
from scripts.statistics import parse_args

from scripts.statistics import get_file_modification


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

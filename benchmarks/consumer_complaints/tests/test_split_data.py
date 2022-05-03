import zipfile
from io import BytesIO

import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from scripts.split_data import (
    parse_args,
    get_nrows,
    load_data,
    split_data,
)


def test_parser():
    parser = parse_args(
        [
            "--data",
            "complaints.csv.zip",
            "--x-train",
            "x_train.csv.zip",
            "--x-test",
            "x_test.csv.zip",
            "--y-train",
            "y_train.csv.zip",
            "--y-test",
            "y_test.csv.zip",
            "--random-state",
            "0",
            "--nrows",
            "1000",
        ]
    )
    assert parser.data is not None
    assert "complaints.csv.zip" == parser.data
    assert parser.x_train is not None
    assert "x_train.csv.zip" == parser.x_train
    assert parser.x_test is not None
    assert "x_test.csv.zip" == parser.x_test
    assert parser.y_train is not None
    assert "y_train.csv.zip" == parser.y_train
    assert parser.y_test is not None
    assert "y_test.csv.zip" == parser.y_test
    assert parser.random_state is not None
    assert 0 == parser.random_state
    assert parser.nrows is not None
    assert "1000" == parser.nrows


def test_get_nrows():
    assert 1000 == get_nrows("1000")
    assert 1 == get_nrows("1")
    assert get_nrows("None") is None
    assert get_nrows("asdf") is None


def test_load_data():
    data = BytesIO()
    content = "Consumer complaint narrative,Product,Sub-product\n"
    content += ",Student loan,Private student loan\n"
    content += "Incorrect information on your report,,Private student loan\n"
    content += "Incorrect information on your report,Student loan,\n"
    content += (
        "Incorrect information on your report,Student loan,Private student loan\n"
    )
    with zipfile.ZipFile(data, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("complaints.csv", content)
    x, y = load_data(data)
    ground_truth = pd.DataFrame(
        {
            "Consumer complaint narrative": ["Incorrect information on your report"],
            "Product": ["Student loan"],
            "Sub-product": ["Private student loan"],
        }
    )
    assert_series_equal(ground_truth["Consumer complaint narrative"], x)
    assert_frame_equal(ground_truth[["Product", "Sub-product"]], y)


def test_split_data():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    random_state = 42
    x_train, x_test, y_train, y_test = split_data(x, y, random_state)
    assert [1, 8, 3, 10, 5, 4, 7] == x_train
    assert [9, 2, 6] == x_test
    assert ["a", "h", "c", "j", "e", "d", "g"] == y_train
    assert ["i", "b", "f"] == y_test

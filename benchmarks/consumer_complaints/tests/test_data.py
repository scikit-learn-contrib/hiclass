from io import StringIO

import pandas as pd
from pandas._testing import assert_series_equal
from pandas.testing import assert_frame_equal

from scripts.data import (
    load_dataframe,
    save_dataframe,
    join,
)


def test_load_dataframe():
    data = StringIO()
    data.write("a,b\n")
    data.write("1,2\n")
    data.write("3,4\n")
    data.seek(0)
    ground_truth = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    metadata = load_dataframe(data)
    assert_frame_equal(ground_truth, metadata)


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


def test_save_dataframe():
    ground_truth = '"Narrative","Product","Sub-product"\n'
    ground_truth += '"Incorrect information","Student loan","Private student loan"\n'
    data = pd.DataFrame(
        {
            "Narrative": ["Incorrect information"],
            "Product": ["Student loan"],
            "Sub-product": ["Private student loan"],
        }
    )
    output = StringIO()
    save_dataframe(data, output)
    output.seek(0)
    assert ground_truth == output.read()

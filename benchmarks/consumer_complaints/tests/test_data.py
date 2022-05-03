from io import StringIO

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.data import (
    load_dataframe,
    save_dataframe,
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

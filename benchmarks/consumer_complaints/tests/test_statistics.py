from scripts.statistics import parse_args


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

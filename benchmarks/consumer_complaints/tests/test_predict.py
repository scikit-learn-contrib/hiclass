from scripts.predict import parse_args


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

from scripts.tune_table import parse_args


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

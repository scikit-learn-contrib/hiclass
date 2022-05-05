from io import StringIO

import numpy as np

from scripts.metrics import parse_args, compute_f1


def test_parser():
    parser = parse_args(
        [
            "--predictions",
            "predictions.tsv",
            "--ground-truth",
            "ground_truth.tsv",
            "--metrics",
            "metrics.tsv",
        ]
    )
    assert parser.predictions is not None
    assert "predictions.tsv" == parser.predictions
    assert parser.ground_truth is not None
    assert "ground_truth.tsv" == parser.ground_truth
    assert parser.metrics is not None
    assert "metrics.tsv" == parser.metrics


def test_compute_f1():
    ground_truth = "f1_hierarchical\n"
    ground_truth += "1.0\n"
    output = StringIO()
    y = np.array(
        [["Reports", "Credit"], ["Debt", "Mortgage"], ["Loan", "Student loan"]]
    )
    compute_f1(y, y, output)
    output.seek(0)
    assert ground_truth == output.read()

#!/usr/bin/env python3
"""Script to compute metrics."""
import argparse
import sys
from argparse import Namespace
from typing import TextIO

import numpy as np
from hiclass.metrics import f1

from data import load_dataframe


def parse_args(args: list) -> Namespace:
    """
    Parse a list of arguments.

    Parameters
    ----------
    args : list
        Arguments to parse.

    Returns
    -------
    _ : Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute metrics")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Input TSV file with predictions",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Input TSV file with ground truth",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Output TSV file with computed metrics",
    )
    return parser.parse_args(args)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, output: TextIO) -> None:
    """
    Compute hierarchical f1 score.

    Parameters
    ----------
    y_true : np.ndarray
        Expected output.
    y_pred : np.ndarray
        Predicted output.
    output : TextIO
        File where output will be written.
    """
    output.write("f1_hierarchical\n")
    f1_hierarchical = f1(y_true, y_pred)
    output.write(f"{f1_hierarchical}\n")


def main():  # pragma: no cover
    """Compute traditional and hierarchical metrics."""
    args = parse_args(sys.argv[1:])
    predictions = load_dataframe(args.predictions)
    ground_truth = load_dataframe(args.ground_truth)
    with open(args.metrics, "w") as output:
        compute_f1(ground_truth, predictions, output)


if __name__ == "__main__":
    main()  # pragma: no cover

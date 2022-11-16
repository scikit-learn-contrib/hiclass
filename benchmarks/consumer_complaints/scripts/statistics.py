#!/usr/bin/env python3
"""Script to store basic statistical information."""
import argparse
import os
import sys
from argparse import Namespace
from datetime import datetime

import pandas as pd

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
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Full dataset for timestamp extraction",
    )
    parser.add_argument(
        "--x-train",
        type=str,
        required=True,
        help="Input CSV file with training features",
    )
    parser.add_argument(
        "--y-train",
        type=str,
        required=True,
        help="Input CSV file with training labels",
    )
    parser.add_argument(
        "--x-test",
        type=str,
        required=True,
        help="Input CSV file with testing features",
    )
    parser.add_argument(
        "--y-test",
        type=str,
        required=True,
        help="Input CSV file with testing labels",
    )
    parser.add_argument(
        "--statistics",
        type=str,
        required=True,
        help="Path to store statistics in CSV format",
    )
    return parser.parse_args(args)


def get_file_modification(file_path: str) -> str:
    """
    Get the modification date of a file.

    Parameters
    ----------
    file_path : str
        Path to file.

    Returns
    -------
    _ : str
        Modification date.
    """
    date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%d/%m/%Y")
    return date


def create_dataframe(snapshot: str, x_train: int, x_test: int) -> pd.DataFrame:
    """
    Create dataframe with statistics.

    Parameters
    ----------
    snapshot : str
        Snapshot date.
    x_train : int
        Number of training examples.
    x_test : int
        Number of testing examples.

    Returns
    -------
    statistics : pd.DataFrame
        Basic statistics.
    """
    statistics = pd.DataFrame(
        {
            "Snapshot": [snapshot],
            "Training set size": [x_train],
            "Test set size": [x_test],
        }
    )
    return statistics


def save_statistics(stats: pd.DataFrame, output_path: str) -> None:
    """
    Save statistics to CSV file.

    Parameters
    ----------
    stats : pd.DataFrame
        Basic statistics.
    output_path : str
        Path to store statistics.
    """
    stats.to_csv(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover
    args = parse_args(sys.argv[1:])
    snapshot = get_file_modification(args.data)
    x_train = load_dataframe(args.x_train).squeeze().shape[0]
    y_train = load_dataframe(args.y_train).shape[0]
    x_test = load_dataframe(args.x_test).squeeze().shape[0]
    y_test = load_dataframe(args.y_test).shape[0]
    assert x_train == y_train
    assert x_test == y_test
    statistics = create_dataframe(snapshot, x_train, x_test)
    save_statistics(statistics, args.statistics)

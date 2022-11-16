#!/usr/bin/env python3
"""Script to store basic statistical information."""
import argparse
import os
from argparse import Namespace
from datetime import datetime


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

#!/usr/bin/env python3
"""Script to split train and test data."""
import argparse
import sys
from argparse import Namespace
from io import BytesIO
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from data import save_dataframe


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
    parser = argparse.ArgumentParser(description="Split data into train and test sets.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input CSV file containing consumer complaints",
    )
    parser.add_argument(
        "--x-train",
        type=str,
        required=True,
        help="Output CSV file to write training features",
    )
    parser.add_argument(
        "--x-test",
        type=str,
        required=True,
        help="Output CSV file to write test features",
    )
    parser.add_argument(
        "--y-train",
        type=str,
        required=True,
        help="Output CSV file to write training labels",
    )
    parser.add_argument(
        "--y-test",
        type=str,
        required=True,
        help="Output CSV file to write test labels",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        required=True,
        help="Random state to enable reproducibility",
    )
    parser.add_argument(
        "--nrows",
        type=str,
        required=True,
        help="Number of rows to read from CSV file",
    )
    return parser.parse_args(args)


def get_nrows(nrows: str):
    """
    Convert a nrows string either to integer or None.

    Parameters
    ----------
    nrows : str
        String with number of rows or 'None'.

    Returns
    -------
    nrows : Union[int, None]
        Number of rows as int or None if conversion fails.
    """
    try:
        return int(nrows)
    except ValueError:
        return None


def load_data(file_path: Union[str, BytesIO], nrows: int = None) -> tuple:
    """
    Load data for training and test.

    Parameters
    ----------
    file_path : Union[str, BytesIO]
        Path for zipped CSV file with consumer complaints.
    nrows : int, default=None
        Number of rows to read from CSV file.

    Returns
    -------
    x, y : tuple
        Consumer complaint narrative and hierarchical labels.
    """
    data = pd.read_csv(
        file_path,
        compression="zip",
        header=0,
        sep=",",
        low_memory=False,
        usecols=["Consumer complaint narrative", "Product", "Sub-product"],
        nrows=nrows,
    )
    # Remove rows with NaN in any column
    data.dropna(
        subset=["Consumer complaint narrative", "Product", "Sub-product"], inplace=True
    )
    # Rebuild index
    data.reset_index(drop=True, inplace=True)
    x = data["Consumer complaint narrative"]
    y = data[["Product", "Sub-product"]]
    # Alternative y can be built with columns "Issue" and "Sub-issue"
    return x, y


def split_data(x: pd.Series, y: pd.DataFrame, random_state: int) -> tuple:
    """
    Split data in train and test subsets.

    Parameters
    ----------
    x : pd.Series
        Consumer complaint narrative.
    y : pd.DataFrame
        hierarchical labels.
    random_state : int
        Random state to enable reproducibility.

    Returns
    -------
    x_train, x_test, y_train, y_test : tuple
        Train and test split.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def main():  # pragma: no cover
    """Split train and test data."""
    args = parse_args(sys.argv[1:])
    x, y = load_data(args.data, get_nrows(args.nrows))
    x_train, x_test, y_train, y_test = split_data(x, y, args.random_state)
    save_dataframe(x_train, args.x_train)
    save_dataframe(x_test, args.x_test)
    save_dataframe(y_train, args.y_train)
    save_dataframe(y_test, args.y_test)


if __name__ == "__main__":
    main()  # pragma: no cover

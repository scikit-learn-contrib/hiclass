#!/usr/bin/env python3
"""Script to share common functions for data manipulation."""
import csv
from typing import TextIO

import numpy as np
import pandas as pd


def load_dataframe(path: TextIO) -> pd.DataFrame:
    """
    Load a dataframe from a CSV file.

    Parameters
    ----------
    path : TextIO
        Path to CSV file.

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe.
    """
    return pd.read_csv(path, compression="infer", header=0, sep=",", low_memory=False)


def flatten_labels(y, separator: str = ":sep:"):
    """
    Flatten hierarchical labels into a single column.

    Parameters
    ----------
    y : pd.DataFrame
        hierarchical labels.
    separator : str, default=":sep:"
        Separator used to differentiate between columns.

    Returns
    -------
    y : pd.Series
        Joined labels.
    """
    y = y[y.columns].apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)
    return y


def unflatten_labels(y: np.ndarray, separator: str = ":sep:") -> pd.DataFrame:
    """
    Separate flat labels back into hierarchical labels.

    Parameters
    ----------
    y : np.ndarray
        Flat labels.
    separator : str, default=":sep:"
        Separator used to differentiate between columns.

    Returns
    -------
    y : pd.DataFrame
        Hierarchical labels.
    """
    y = pd.Series(y)
    y = y.str.split(
        separator,
        expand=True,
    )
    return y


def save_dataframe(dataframe: pd.DataFrame, file_path: TextIO) -> None:
    """
    Save dataframe to CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to save.
    file_path : TextIO
        Path to save dataframe.
    """
    dataframe.to_csv(
        file_path,
        index=False,
        header=True,
        sep=",",
        compression="infer",
        quoting=csv.QUOTE_ALL,
    )

#!/usr/bin/env python3
"""Script to share common functions for data manipulation."""
import csv
from typing import TextIO

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted


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


class LabelConcatenator(TransformerMixin, BaseEstimator):
    """Concatenate target labels for flat classification."""

    def fit(self, y):
        """
        Fit label concatenator.

        Parameters
        ----------
        y : pd.DataFrame
            Hierarchical labels.

        Returns
        -------
        self : LabelConcatenator
            Fitted label concatenator.
        """
        return self

    def transform(self, y, separator: str = ":sep:"):
        """
        Join hierarchical labels into a single column.

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

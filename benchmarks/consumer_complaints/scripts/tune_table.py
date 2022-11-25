#!/usr/bin/env python3
"""Script to create table with tuning results for flat and hierarchical approaches."""
import argparse
import glob
import pickle
import sys
from argparse import Namespace
from typing import Tuple, List

import numpy as np


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
    parser = argparse.ArgumentParser(
        description="Create table with hyper-parameter tuning results"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder where the tuning results are stored",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model used for tuning",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Classifier used for tuning",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output to write the table in markdown format (.md)",
    )
    return parser.parse_args(args)


def compute(
    folder: str,
) -> Tuple[List[dict], List[list], List[np.ndarray], List[np.ndarray]]:
    """
    Compute average and standard deviation of the tuning results.

    Parameters
    ----------
    folder : str
        Folder where the tuning results are stored.

    Returns
    -------
    hyperparameters : List[dict]
        Hyperparameters tested for tuning.
    scores : List[list]
        Scores for each hyperparameter combination tested.
    avg : List[np.ndarray]
        Averages of k-fold cross-validation.
    std : List[np.ndarray]
        Standard deviations of k-fold cross-validation.
    """
    results = glob.glob(f"{folder}/*.sav")
    if "{}/trained_model.sav".format(folder) in results:
        results.remove(f"{folder}/trained_model.sav")
    hyperparameters = []
    scores = []
    avg = []
    std = []
    for result in results:
        parameters, s = pickle.load(open(result, "rb"))
        hyperparameters.append(parameters)
        scores.append([round(i, 3) for i in s])
        avg.append(np.mean(s))
        std.append(np.std(s))
    return hyperparameters, scores, avg, std


def create_table(args):
    """Create table with tuning results for flat and hierarchical approaches."""
    with open(args.output, "w") as fout:
        fout.write(f"# Model: {args.model}\n")
        fout.write(f"## Base classifier: {args.classifier}\n")
        fout.write("|Parameters|Scores|Average|Standard deviation|\n")
        fout.write("|----------|------|-------|------------------|\n")
        hyperparameters, scores, avg, std = compute(args.folder)
        for hp, sc, av, st in zip(hyperparameters, scores, avg, std):
            fout.write(f"|{hp}|{sc}|{av:.3f}|{st:.3f}|\n")


if __name__ == "__main__":  # pragma: no cover
    args = parse_args(sys.argv[1:])
    create_table(args)

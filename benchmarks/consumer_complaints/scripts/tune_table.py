#!/usr/bin/env python3
"""Script to create table with tuning results for flat and hierarchical approaches."""
import argparse
import glob
import pickle
import sys
from argparse import Namespace
from typing import Tuple, List

import numpy as np
from omegaconf import OmegaConf


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


def delete_non_hyperparameters(hyperparameters: OmegaConf) -> dict:
    """
    Delete non-hyperparameters from the dictionary.

    Parameters
    ----------
    hyperparameters : OmegaConf
        Hyperparameters to delete non-hyperparameters from.

    Returns
    -------
    hyperparameters : dict
        Hyperparameters without non-hyperparameters.

    """
    hyperparameters = OmegaConf.to_container(hyperparameters)
    del hyperparameters["model"]
    del hyperparameters["classifier"]
    del hyperparameters["n_jobs"]
    del hyperparameters["x_train"]
    del hyperparameters["y_train"]
    del hyperparameters["output_dir"]
    del hyperparameters["mem_gb"]
    del hyperparameters["n_splits"]
    return hyperparameters


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
    results = glob.glob(f"{folder}/[!trained_model]*.sav")
    hyperparameters = []
    scores = []
    avg = []
    std = []
    for result in results:
        parameters, s = pickle.load(open(result, "rb"))
        parameters = delete_non_hyperparameters(parameters)
        hyperparameters.append(parameters)
        scores.append([round(i, 3) for i in s])
        avg.append(np.mean(s))
        std.append(np.std(s))
    return hyperparameters, scores, avg, std


def create_table():  # pragma: no cover
    """Create table with tuning results for flat and hierarchical approaches."""
    args = parse_args(sys.argv[1:])
    with open(args.output, "w") as fout:
        fout.write(f"# Model: {args.model}\n")
        fout.write(f"## Base classifier: {args.classifier}\n")
        fout.write("|Parameters|Scores|Average|Standard deviation|\n")
        fout.write("|----------|------|-------|------------------|\n")
        hyperparameters, scores, avg, std = compute(args.folder)
        for hp, sc, av, st in zip(hyperparameters, scores, avg, std):
            fout.write(f"|{hp}|{sc}|{av:.3f}|{st:.3f}|\n")


if __name__ == "__main__":  # pragma: no cover
    create_table()

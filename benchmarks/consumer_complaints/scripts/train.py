#!/usr/bin/env python3
"""Script to train with flat or hierarchical approaches."""
import argparse
import pickle
import sys
from argparse import Namespace

from joblib import parallel_backend
from omegaconf import DictConfig, OmegaConf

from data import load_dataframe, join
from tune import configure_pipeline


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
        "--n-jobs",
        type=int,
        required=True,
        help="Number of jobs to run training in parallel",
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
        "--trained-model",
        type=str,
        required=True,
        help="Path to store trained model",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Algorithm used for fitting, e.g., logistic_regression or random_forest",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        required=True,
        help="Random state to enable reproducibility",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model used for training, e.g., flat, lcpl, lcpn or lcppn",
    )
    parser.add_argument(
        "--best-parameters",
        type=str,
        required=True,
        help="Path to optuna's tuned parameters",
    )
    return parser.parse_args(args)


def load_parameters(yml: str) -> DictConfig:
    """
    Load parameters from a YAML file.

    Parameters
    ----------
    yml : str
        Path to YAML file containing tuned parameters.

    Returns
    -------
    cfg : DictConfig
        Dictionary containing all configuration information.
    """
    cfg = OmegaConf.load(yml)
    return cfg["best_params"]


def train() -> None:  # pragma: no cover
    """Train with flat or hierarchical approaches."""
    args = parse_args(sys.argv[1:])
    x_train = load_dataframe(args.x_train).squeeze()
    y_train = load_dataframe(args.y_train)
    if args.model == "flat":
        y_train = join(y_train)
    best_params = load_parameters(args.best_parameters)
    best_params.model = args.model
    best_params.classifier = args.classifier
    best_params.n_jobs = args.n_jobs
    pipeline = configure_pipeline(best_params)
    with parallel_backend("threading", n_jobs=args.n_jobs):
        pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(args.trained_model, "wb"))


if __name__ == "__main__":
    train()  # pragma: no cover

#!/usr/bin/env python3
"""Script to train with flat or hierarchical approaches."""
import argparse
import pickle
import sys
from argparse import Namespace

import pandas as pd
from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data import load_dataframe

# Base classifiers used for building models
classifiers = {
    "logistic_regression": LogisticRegression(
        max_iter=10000,
        n_jobs=1,
    ),
    "random_forest": RandomForestClassifier(
        n_jobs=1,
    ),
    "hist_gradient": HistGradientBoostingClassifier(),
}


# Hierarchical classifiers used for training
hierarchical_classifiers = {
    "lcpn": LocalClassifierPerNode(),
    "lcppn": LocalClassifierPerParentNode(),
    "lcpl": LocalClassifierPerLevel(),
}


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
    return parser.parse_args(args)


def join(y: pd.DataFrame, separator: str = ":sep:") -> pd.Series:
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


def get_flat_classifier(
    n_jobs: int,
    base_classifier: str,
    random_state: int,
) -> BaseEstimator:
    """
    Build flat classifier for pipeline.

    Parameters
    ----------
    n_jobs : int
        Number of threads to fit in parallel.
    base_classifier : str
        Classifier used for fitting.
    random_state : int
        Random state to enable reproducibility.

    Returns
    -------
    classifier : BaseEstimator
        Flat classifier.
    """
    model = classifiers[base_classifier]
    if isinstance(model, HistGradientBoostingClassifier):
        model = model.set_params(random_state=random_state)
    else:
        model.set_params(n_jobs=n_jobs, random_state=random_state)
    return model


def get_hierarchical_classifier(
    n_jobs: int,
    local_classifier: str,
    hierarchical_classifier: str,
    random_state: int,
) -> BaseEstimator:
    """
    Build hierarchical classifier for pipeline.

    Parameters
    ----------
    n_jobs : int
        Number of threads to fit in parallel.
    local_classifier : str
        Classifier used for fitting.
    hierarchical_classifier : str
        Classifier used for hierarchical classification.
    random_state : int
        Random state to enable reproducibility.

    Returns
    -------
    classifier: BaseEstimator
        Hierarchical classifier.
    """
    local_classifier = classifiers[local_classifier]
    local_classifier.set_params(random_state=random_state)
    model = hierarchical_classifiers[hierarchical_classifier]
    model.set_params(local_classifier=local_classifier, n_jobs=n_jobs)
    return model


def main():  # pragma: no cover
    """Train with flat or hierarchical approaches."""
    args = parse_args(sys.argv[1:])
    x_train = load_dataframe(args.x_train).squeeze()
    y_train = load_dataframe(args.y_train)
    if args.model == "flat":
        y_train = join(y_train)
        classifier = get_flat_classifier(
            args.n_jobs, args.classifier, args.random_state
        )
    else:
        classifier = get_hierarchical_classifier(
            args.n_jobs,
            args.classifier,
            args.model,
            args.random_state,
        )
    pipeline = Pipeline(
        [
            ("count", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(args.trained_model, "wb"))


if __name__ == "__main__":
    main()  # pragma: no cover

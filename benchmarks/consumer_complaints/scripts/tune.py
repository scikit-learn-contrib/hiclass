from os import chdir
from typing import TextIO

import hydra
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from hiclass.metrics import f1


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


def configure_lightgbm(cfg: DictConfig) -> BaseEstimator:
    """
    Setup LightGBM with parameters passed as argument.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    classifier : BaseEstimator
        Estimator with hyper-parameters configured.
    """
    classifier = LGBMClassifier(
        n_jobs=cfg.n_jobs,
        num_leaves=cfg.num_leaves,
        min_data_in_leaf=cfg.min_data_in_leaf,
    )
    return classifier


def configure_logistic_regression(cfg: DictConfig) -> BaseEstimator:
    """
    Setup LogisticRegression with parameters passed as argument.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    classifier : BaseEstimator
        Estimator with hyper-parameters configured.
    """
    classifier = LogisticRegression(
        n_jobs=cfg.n_jobs,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
    )
    return classifier


def configure_random_forest(cfg: DictConfig) -> BaseEstimator:
    """
    Setup RandomForest with parameters passed as argument.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    classifier : BaseEstimator
        Estimator with hyper-parameters configured.
    """
    classifier = RandomForestClassifier(
        n_jobs=cfg.n_jobs,
        n_estimators=cfg.n_estimators,
        criterion=cfg.criterion,
    )
    return classifier


@hydra.main(config_path="../configs", config_name="logistic_regression", version_base="1.2")
def optimize(cfg: DictConfig) -> float:
    configure = {
        "lightgbm": configure_lightgbm,
        "logistic_regression": configure_logistic_regression,
        "random_forest": configure_random_forest,
    }
    classifier = configure[cfg.classifier](cfg)
    x_train = load_dataframe(cfg.x_train).squeeze()
    y_train = join(load_dataframe(cfg.y_train))
    pipeline = Pipeline(
        [
            ("count", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", classifier),
        ]
    )
    score = cross_val_score(pipeline, x_train, y_train, scoring=make_scorer(f1))
    return np.mean(score)


if __name__ == "__main__":
    optimize()

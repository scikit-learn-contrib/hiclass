#!/usr/bin/env python3
"""Script to perform hyper-parameter tuning for flat and hierarchical approaches."""
import hashlib
import json
import logging
import os
import pickle
import resource
from typing import List, Union

import hydra
import numpy as np
import pandas as pd
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from numpy.core._exceptions import _ArrayMemoryError
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from data import load_dataframe, flatten_labels, unflatten_labels
from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)
from hiclass.metrics import f1

log = logging.getLogger("TUNE")


configure_flat = {
    "lightgbm": LGBMClassifier(),
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
}


configure_hierarchical = {
    "local_classifier_per_node": LocalClassifierPerNode(),
    "local_classifier_per_parent_node": LocalClassifierPerParentNode(),
    "local_classifier_per_level": LocalClassifierPerLevel(),
}


non_hyperparameters = [
    "model",
    "classifier",
    "n_jobs",
    "x_train",
    "y_train",
    "output_dir",
    "mem_gb",
    "n_splits",
]


def delete_non_hyperparameters(cfg: OmegaConf) -> dict:
    """
    Delete non-hyperparameters from the dictionary.

    Parameters
    ----------
    cfg : OmegaConf
        Dictionary to delete non-hyperparameters from.

    Returns
    -------
    hyperparameters : dict
        Dictionary containing only hyperparameters.

    """
    hyperparameters = OmegaConf.to_container(cfg)
    for key in non_hyperparameters:
        if key in hyperparameters:
            del hyperparameters[key]
    return hyperparameters


def configure_pipeline(cfg: DictConfig) -> Pipeline:
    """
    Configure pipeline with parameters passed as argument.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    pipeline : Pipeline
        Pipeline with hyper-parameters configured.
    """
    if cfg.model == "flat":
        classifier = configure_flat[cfg.classifier]
        classifier.set_params(**delete_non_hyperparameters(cfg))
    else:
        local_classifier = configure_flat[cfg.classifier]
        local_classifier.set_params(**delete_non_hyperparameters(cfg))
        local_classifier.set_params(n_jobs=1)
        classifier = configure_hierarchical[cfg.model]
        classifier.set_params(
            local_classifier=local_classifier, n_jobs=cfg.n_jobs, verbose=30
        )
    pipeline = Pipeline(
        [
            ("count", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def compute_md5(cfg: dict) -> str:
    """
    Compute MD5 hash of configuration.

    Parameters
    ----------
    cfg : dict
        Dictionary containing hyperparameters.

    Returns
    -------
    md5 : str
        MD5 hash of configuration.

    """
    md5 = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    return md5


def save_trial(cfg: DictConfig, scores: List[float]) -> None:
    """
    Save trial information.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.
    scores : List[float]
        List of scores for each fold.
    """
    hyperparameters = delete_non_hyperparameters(cfg)
    md5 = compute_md5(hyperparameters)
    filename = f"{cfg.output_dir}/{md5}.sav"
    with open(filename, "wb") as file:
        pickle.dump((hyperparameters, scores), file)


def load_trial(cfg: DictConfig) -> List[float]:
    """
    Load trial information.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    scores : List[float]
        The cross-validation scores or empty list if file does not exist.
    """
    hyperparameters = delete_non_hyperparameters(cfg)
    md5 = compute_md5(hyperparameters)
    filename = f"{cfg.output_dir}/{md5}.sav"
    if os.path.exists(filename):
        (_, scores) = pickle.load(open(filename, "rb"))
        log.info(f"Loaded trial with F-scores {scores}")
        return scores
    else:
        return []


def limit_memory(mem_gb: int) -> None:
    """
    Limit memory usage to avoid job being killed by slurm.

    Parameters
    ----------
    mem_gb : int
        Memory limit in GB.
    """
    mem_bytes = mem_gb * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def cross_validate(cfg: DictConfig, X: pd.DataFrame, y: pd.DataFrame) -> List[float]:
    """
    Cross validate pipeline, skipping folds that have already been computed.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.
    X : pd.DataFrame
        Dataframe containing the features.
    y : pd.DataFrame
        Dataframe containing the labels.

    Returns
    -------
    scores : List[float]
        List of scores for each fold.
    """
    pipeline = configure_pipeline(cfg)
    scores = load_trial(cfg)
    with parallel_backend("threading", n_jobs=cfg.n_jobs):
        kf = KFold(n_splits=cfg.n_splits)
        fold = 0
        for train_index, test_index in kf.split(X):
            if fold >= len(scores):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                if cfg.model == "flat":
                    y_pred = unflatten_labels(y_pred)
                    y_test = unflatten_labels(y_test)
                scores.append(f1(y_test, y_pred))
                save_trial(cfg, scores)
                log.info(f"Fold #{fold} obtained F-score: {scores[fold]}")
            fold += 1
    return scores


@hydra.main(
    config_path="../configs", config_name="logistic_regression", version_base="1.2"
)
def optimize(cfg: DictConfig) -> Union[np.ndarray, float]:  # pragma: no cover
    """
    Perform hyper-parameter tuning.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    score : Union[np.ndarray, float]
        The mean cross-validation score.
    """
    try:
        limit_memory(cfg.mem_gb)
        X = load_dataframe(cfg.x_train).squeeze()
        y = load_dataframe(cfg.y_train)
        if cfg.model == "flat":
            y = flatten_labels(y)
        scores = cross_validate(cfg, X, y)
        return np.mean(scores)
    except (ValueError, MemoryError, _ArrayMemoryError):
        return 0


if __name__ == "__main__":  # pragma: no cover
    optimize()

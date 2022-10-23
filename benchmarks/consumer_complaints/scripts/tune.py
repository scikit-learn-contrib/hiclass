#!/usr/bin/env python3
"""Script to perform hyper-parameter tuning for flat and hierarchical approaches."""
import hashlib
import json
import os
import pickle
import resource
from typing import List

import hydra
import numpy as np
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data import load_dataframe, join
from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)
from hiclass.metrics import f1


def configure_lightgbm(cfg: DictConfig) -> BaseEstimator:
    """
    Configure LightGBM with parameters passed as argument.

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
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
    )
    return classifier


def configure_logistic_regression(cfg: DictConfig) -> BaseEstimator:
    """
    Configure LogisticRegression with parameters passed as argument.

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
    Configure RandomForest with parameters passed as argument.

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


configure_flat = {
    "lightgbm": configure_lightgbm,
    "logistic_regression": configure_logistic_regression,
    "random_forest": configure_random_forest,
}


configure_hierarchical = {
    "local_classifier_per_node": LocalClassifierPerNode(),
    "local_classifier_per_parent_node": LocalClassifierPerParentNode(),
    "local_classifier_per_level": LocalClassifierPerLevel(),
}


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
        classifier = configure_flat[cfg.classifier](cfg)
    else:
        local_classifier = configure_flat[cfg.classifier](cfg)
        local_classifier.set_params(n_jobs=1)
        classifier = configure_hierarchical[cfg.model]
        classifier.set_params(local_classifier=local_classifier, n_jobs=cfg.n_jobs)
    pipeline = Pipeline(
        [
            ("count", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def compute_md5(cfg: DictConfig) -> str:
    """
    Compute MD5 hash of configuration.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    md5 : str
        MD5 hash of configuration.

    """
    dictionary = OmegaConf.to_object(cfg)
    md5 = hashlib.md5(
        json.dumps(dictionary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return md5


def save_trial(cfg: DictConfig, score: List[float]) -> None:
    """
    Save trial information.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.
    score : List[float]
        List of scores for each fold.
    """
    md5 = compute_md5(cfg)
    filename = f"{cfg.output_dir}/{md5}.sav"
    pickle.dump((cfg, score), open(filename, "wb"))


def limit_memory(mem_gb: int) -> None:
    """
    Limit memory usage.

    Parameters
    ----------
    mem_gb : int
        Memory limit in GB.
    """
    mem_bytes = mem_gb * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


@hydra.main(
    config_path="../configs", config_name="logistic_regression", version_base="1.2"
)
def optimize(cfg: DictConfig) -> np.ndarray:  # pragma: no cover
    """
    Perform hyper-parameter tuning.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary containing all configuration information.

    Returns
    -------
    score : np.ndarray
        Array containing the mean cross-validation score.
    """
    md5 = compute_md5(cfg)
    filename = f"{cfg.output_dir}/{md5}.sav"
    if os.path.exists(filename):
        (_, score) = pickle.load(open(filename, "rb"))
        return np.mean(score)
    try:
        limit_memory(cfg.mem_gb)
        x_train = load_dataframe(cfg.x_train).squeeze()
        y_train = load_dataframe(cfg.y_train)
        if cfg.model == "flat":
            y_train = join(y_train)
        pipeline = configure_pipeline(cfg)
        with parallel_backend("threading", n_jobs=cfg.n_jobs):
            score = cross_val_score(
                pipeline, x_train, y_train, scoring=make_scorer(f1), n_jobs=1
            )
        save_trial(cfg, score)
        return np.mean(score)
    except MemoryError:
        raise MemoryError("Memory limit exceeded.")
        # return 0


if __name__ == "__main__":  # pragma: no cover
    optimize()

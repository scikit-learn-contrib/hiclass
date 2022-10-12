#!/usr/bin/env python3
"""Script to perform hyper-parameter tuning for flat and hierarchical approaches."""

import hydra
import numpy as np
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
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
        boosting_type=cfg.boosting_type,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample_for_bin=cfg.subsample_for_bin,
        class_weight=cfg.class_weight,
        min_split_gain=cfg.min_split_gain,
        min_child_weight=cfg.min_child_weight,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
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
    x_train = load_dataframe(cfg.x_train).squeeze()
    y_train = load_dataframe(cfg.y_train)
    if cfg.model == "flat":
        y_train = join(y_train)
    pipeline = configure_pipeline(cfg)
    score = cross_val_score(pipeline, x_train, y_train, scoring=make_scorer(f1), n_jobs=1)
    return np.mean(score)


if __name__ == "__main__":  # pragma: no cover
    optimize()

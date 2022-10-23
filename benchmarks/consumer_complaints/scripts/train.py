#!/usr/bin/env python3
"""Script to train with flat or hierarchical approaches."""
import argparse
import pickle
import sys
from argparse import Namespace

import hydra
from joblib import parallel_backend
from omegaconf import DictConfig, OmegaConf

from data import load_dataframe, join
from tune import configure_pipeline


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


@hydra.main(version_base="1.2", config_path="../configs", config_name="submitit")
def train(cfg: DictConfig) -> None:  # pragma: no cover
    """Train with flat or hierarchical approaches."""
    x_train = load_dataframe(cfg.x_train).squeeze()
    y_train = load_dataframe(cfg.y_train)
    if cfg.model == "flat":
        y_train = join(y_train)
    best_params = load_parameters(cfg.best_parameters)
    best_params.model = cfg.model
    best_params.classifier = cfg.classifier
    best_params.n_jobs = cfg.n_jobs
    pipeline = configure_pipeline(best_params)
    with parallel_backend("threading", n_jobs=cfg.n_jobs):
        pipeline.fit(x_train, y_train)
    pickle.dump(pipeline, open(cfg.trained_model, "wb"))


if __name__ == "__main__":
    train()  # pragma: no cover

from os import getcwd, chdir
from typing import TextIO

import hydra
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

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


def optimize_lightgbm(cfg: DictConfig) -> float:
    local_classifier = LGBMClassifier(
        n_jobs=cfg.n_jobs,
        num_leaves=cfg.num_leaves,
        min_data_in_leaf=cfg.min_data_in_leaf,
    )
    chdir("../../../..")
    # raise NotImplementedError(getcwd())
    x_train = load_dataframe(cfg.x_train).squeeze()
    y_train = load_dataframe(cfg.y_train)
    score = cross_val_score(local_classifier, x_train, y_train, scoring=make_scorer(f1))
    return score


def optimize_logistic_regression(cfg: DictConfig) -> float:
    raise NotImplementedError(cfg)


@hydra.main(config_path="../configs", config_name="logistic_regression")
def optimize(cfg: DictConfig) -> float:
    classifier: str = cfg.classifier
    optimizers = {
        "lightgbm": optimize_lightgbm,
        "logistic_regression": optimize_logistic_regression,
    }
    return optimizers[classifier](cfg)
    return 1


if __name__ == "__main__":
    optimize()

import numpy as np


def _normalize_probabilities(proba):
    if isinstance(proba, np.ndarray):
        return np.nan_to_num(proba / proba.sum(axis=1, keepdims=True))
    return [
        np.nan_to_num(
            level_probabilities / level_probabilities.sum(axis=1, keepdims=True)
        )
        for level_probabilities in proba
    ]

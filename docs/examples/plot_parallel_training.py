# -*- coding: utf-8 -*-
"""
=====================
Parallel Training
=====================

Larger datasets require more time for training.
While by default the models in HiClass are trained using a single core,
it is possible to train each local classifier in parallel by leveraging the library Ray [1]_.
In this example, we demonstrate how to train a hierarchical classifier in parallel using a mock dataset from Kaggle [2]_.

.. [1] https://www.ray.io/
.. [2] https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification
"""

import requests


def download(url: str, path: str) -> None:
    """
    Download a file from the internet.

    Parameters
    ----------
    url : str
        The address of the file to be downloaded.
    path : str
        The path to store the downloaded file.
    """
    response = requests.get(url)
    with open(path, "wb") as file:
        file.write(response.content)


training_data_url = "https://zenodo.org/record/6657410/files/train_40k.csv?download=1"
training_data_path = "train_40k.csv"

download(training_data_url, training_data_path)

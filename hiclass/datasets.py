"""Datasets util for downloading and maintaining sample datasets."""

import csv
import logging
import os
import tempfile

import numpy as np
import requests
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use temp directory to store cached datasets
CACHE_DIR = tempfile.gettempdir()

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset urls
HIERARCHICAL_TEXT_CLASSIFICATION_URL = (
    "https://zenodo.org/record/6657410/files/train_40k.csv?download=1"
)


def _download_file(url, destination):
    """Download file from given URL to specified destination."""
    try:
        response = requests.get(url)
        # Raise HTTPError if response code is not OK
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download file from {url}: {str(e)}")


def load_hierarchical_text_classification(test_size=0.3, random_state=42):
    """
    Load hierarchical text classification dataset.

    Parameters
    ----------
    test_size : float, default=0.3
        The proportion of the dataset to include in the test split.
    random_state : int or None, default=42
        Controls the randomness of the dataset. Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    list
        List containing train-test split of inputs.

    Raises
    ------
    RuntimeError
        If failed to access or process the dataset.
    Examples
    --------
    >>> from hiclass.datasets import load_hierarchical_text_classification
    >>> X_train, X_test, Y_train, Y_test = load_hierarchical_text_classification()
    >>> X_train[:3]
    38015                                Nature's Way Selenium
    2281         Music In Motion Developmental Mobile W Remote
    36629    Twinings Ceylon Orange Pekoe Tea, Tea Bags, 20...
    Name: Title, dtype: object
    >>> X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
    (28000,) (12000,) (28000, 3) (12000, 3)
    """
    dataset_name = "hierarchical_text_classification.csv"
    cached_file_path = os.path.join(CACHE_DIR, dataset_name)

    # Check if the file exists in the cache
    if not os.path.exists(cached_file_path):
        try:
            logger.info("Downloading hierarchical text classification dataset..")
            _download_file(HIERARCHICAL_TEXT_CLASSIFICATION_URL, cached_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to access or download dataset: {str(e)}")

    data = [row for row in csv.reader(open(cached_file_path))]
    data.pop(0)
    data = np.array(data)
    X = data[:, 1]
    y = data[:, 7:]

    # Return tuple (X_train, X_test, y_train, y_test)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

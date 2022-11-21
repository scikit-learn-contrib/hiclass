"""Shared code for all tests."""
import hashlib
import os
from typing import Union

import gdown


def md5(file_path: str) -> str:
    """
    Compute the MD5 hash of a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    md5sum : str
        MD5 hash of the file.
    """
    with open(file_path, "r") as file:
        md5sum = hashlib.md5(file.read().encode("utf-8")).hexdigest()
        return md5sum


def download(dataset: dict, fuzzy: bool) -> None:
    """
    Download a dataset if the file does not exist yet.

    Parameters
    ----------
    dataset : dict
        Dictionary containing the URL, path and MD5 hash of the dataset.
    fuzzy : bool
        Whether to use fuzzy matching to find the file name.
    """
    if dataset:
        gdown.cached_download(
            dataset["url"],
            dataset["path"],
            quiet=False,
            fuzzy=fuzzy,
            md5=dataset["md5"],
        )


def get_dataset(prefix: str) -> Union[dict, None]:
    """
    Get the dataset information.

    Parameters
    ----------
    prefix : str
        Prefix of the environment variables.

    Returns
    -------
    dataset : Union[dict, None]
        Dictionary containing the URL, path and MD5 hash of the dataset
        or None if environment variables are not set.
    """
    try:
        uppercase = prefix.upper()
        lowercase = prefix.lower()
        dataset = {
            "url": os.environ["{}_URL".format(uppercase)],
            "path": "tests/fixtures/{}.csv".format(lowercase),
            "md5": os.environ["{}_MD5".format(uppercase)],
        }
    except KeyError:
        return None
    else:
        return dataset


def download_fungi_dataset() -> None:
    """Download the fungi dataset only if the environment variables are set."""
    train = get_dataset("FUNGI_TRAIN")
    download(train, fuzzy=False)
    test = get_dataset("FUNGI_TEST")
    download(test, fuzzy=False)


def download_complaints_dataset() -> None:
    """Download the complaints dataset only if the environment variables are set."""
    x_train = get_dataset("COMPLAINTS_X_TRAIN")
    download(x_train, fuzzy=True)
    y_train = get_dataset("COMPLAINTS_Y_TRAIN")
    download(y_train, fuzzy=True)
    x_test = get_dataset("COMPLAINTS_X_TEST")
    download(x_test, fuzzy=True)
    y_test = get_dataset("COMPLAINTS_Y_TEST")
    download(y_test, fuzzy=True)


def pytest_sessionstart(session):
    """
    Download the datasets before the tests start.

    Parameters
    ----------
    session : pytest.Session
        The pytest session object.
    """
    download_fungi_dataset()
    download_complaints_dataset()

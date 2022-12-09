"""Shared code for all tests."""
import hashlib
import os
from typing import Union, List, TextIO

import numpy as np
import pandas as pd

try:
    import gdown
except ImportError:
    gdown_installed = False
else:
    gdown_installed = True


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
    if gdown_installed and dataset:
        gdown.cached_download(
            dataset["url"],
            dataset["path"],
            quiet=True,
            fuzzy=fuzzy,
            md5=dataset["md5"],
        )


def get_dataset(prefix: str, suffix: str) -> Union[dict, None]:
    """
    Get the dataset information.

    Parameters
    ----------
    prefix : str
        Prefix of the environment variables.
    suffix : str
        Suffix of the output file.

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
            "path": "tests/fixtures/{prefix}.{suffix}".format(
                prefix=lowercase, suffix=suffix
            ),
            "md5": os.environ["{}_MD5".format(uppercase)],
        }
    except KeyError:
        return None
    else:
        return dataset


def download_fungi_dataset() -> None:
    """Download the fungi dataset only if the environment variables are set."""
    train = get_dataset("FUNGI_TRAIN", "fasta")
    download(train, fuzzy=False)
    test = get_dataset("FUNGI_TEST", "fasta")
    download(test, fuzzy=False)


def download_complaints_dataset() -> None:
    """Download the complaints dataset only if the environment variables are set."""
    x_train = get_dataset("COMPLAINTS_X_TRAIN", "csv.zip")
    download(x_train, fuzzy=True)
    y_train = get_dataset("COMPLAINTS_Y_TRAIN", "csv.zip")
    download(y_train, fuzzy=True)
    x_test = get_dataset("COMPLAINTS_X_TEST", "csv.zip")
    download(x_test, fuzzy=True)
    y_test = get_dataset("COMPLAINTS_Y_TEST", "csv.zip")
    download(y_test, fuzzy=True)


def get_ranks(taxxi: str) -> List[str]:
    """
    Get the taxonomy ranks from a taxxi record.

    Parameters
    ----------
    taxxi : str

    Returns
    -------
    ranks : List[str]
        List of taxonomic ranks.
    """
    split = taxxi.split(",")
    kingdom = split[0]
    kingdom = kingdom[kingdom.find("tax=") + 4 :]
    phylum = split[1]
    classs = split[2]
    order = split[3]
    family = split[4]
    genus = split[5]
    if len(split) == 6:
        return [kingdom, phylum, classs, order, family, genus]
    elif len(split) == 7:
        species = split[6][:-1]
        return [kingdom, phylum, classs, order, family, genus, species]


# Returns taxonomy ranks from training dataset
def get_taxonomy(taxxi: List[str]) -> np.ndarray:
    """
    Get the taxonomy ranks from a FASTA IDs.

    Parameters
    ----------
    taxxi : List[str]
        List of FASTA IDs in TAXXI format.

    Returns
    -------
    taxonomy : np.ndarray
        Array of taxonomic ranks.
    """
    taxonomy = np.array([get_ranks(record) for record in taxxi])
    return taxonomy


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

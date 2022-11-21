import hashlib
import os
import urllib.request


def md5(file_path):
    with open(file_path, "r") as file:
        return hashlib.md5(file.read().encode("utf-8")).hexdigest()


def download(dataset):
    if not os.path.exists(dataset["path"]) or md5(dataset["path"]) != dataset["md5"]:
        print(f"Downloading file {dataset['path']}")
        urllib.request.urlretrieve(dataset["url"], dataset["path"])
        assert md5(dataset["path"]) == dataset["md5"]


def download_fungi_dataset():
    # Download the fungi dataset if not already present
    # only if the environment variables are set
    if "FUNGI_TRAIN_URL" in os.environ and "FUNGI_TRAIN_MD5" in os.environ:
        train = {
            "url": os.environ["FUNGI_TRAIN_URL"],
            "path": "tests/fixtures/fungi_train.fasta",
            "md5": os.environ["FUNGI_TRAIN_MD5"],
        }
        download(train)
    if "FUNGI_TEST_URL" in os.environ and "FUNGI_TEST_MD5" in os.environ:
        test = {
            "url": os.environ["FUNGI_TEST_URL"],
            "path": "tests/fixtures/fungi_test.fasta",
            "md5": os.environ["FUNGI_TEST_MD5"],
        }
        download(test)


def pytest_sessionstart(session):
    download_fungi_dataset()

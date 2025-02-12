import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import hiclass.datasets
from hiclass.datasets import load_hierarchical_text_classification


def test_load_hierarchical_text_classification_shape():
    X_train, X_test, y_train, y_test = load_hierarchical_text_classification(
        test_size=0.2, random_state=42
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_load_hierarchical_text_classification_random_state():
    X_train_1, X_test_1, y_train_1, y_test_1 = load_hierarchical_text_classification(
        test_size=0.2, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = load_hierarchical_text_classification(
        test_size=0.2, random_state=42
    )
    assert_array_equal(X_train_1, X_train_2)
    assert_array_equal(X_test_1, X_test_2)
    assert_array_equal(y_train_1, y_train_2)
    assert_array_equal(y_test_1, y_test_2)


def test_load_hierarchical_text_classification_file_exists():
    dataset_name = "hierarchical_text_classification.csv"
    cached_file_path = os.path.join(tempfile.gettempdir(), dataset_name)

    if os.path.exists(cached_file_path):
        os.remove(cached_file_path)

    if not os.path.exists(cached_file_path):
        load_hierarchical_text_classification()
        assert os.path.exists(cached_file_path)


def test_download_dataset():
    dataset_name = "hierarchical_text_classification.csv"
    url = hiclass.datasets.HIERARCHICAL_TEXT_CLASSIFICATION_URL
    cached_file_path = os.path.join(tempfile.gettempdir(), dataset_name)

    if os.path.exists(cached_file_path):
        os.remove(cached_file_path)

    if not os.path.exists(cached_file_path):
        hiclass.datasets._download_file(url, cached_file_path)
        assert os.path.exists(cached_file_path)


def test_download_error_load_hierarchical_text():
    dataset_name = "hierarchical_text_classification.csv"
    backup_url = hiclass.datasets.HIERARCHICAL_TEXT_CLASSIFICATION_URL
    hiclass.datasets.HIERARCHICAL_TEXT_CLASSIFICATION_URL = ""
    cached_file_path = os.path.join(tempfile.gettempdir(), dataset_name)

    if os.path.exists(cached_file_path):
        os.remove(cached_file_path)

    if not os.path.exists(cached_file_path):
        with pytest.raises(RuntimeError):
            load_hierarchical_text_classification()

    hiclass.datasets.HIERARCHICAL_TEXT_CLASSIFICATION_URL = backup_url


def test_url_links():
    assert hiclass.datasets.HIERARCHICAL_TEXT_CLASSIFICATION_URL != ""

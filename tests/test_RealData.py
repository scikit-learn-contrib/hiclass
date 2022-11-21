import os
from os.path import exists

import pytest

from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)

try:
    import skbio
except ImportError:
    skbio_installed = False
else:
    skbio_installed = True

try:
    from hitac._utils import compute_possible_kmers
except ImportError:
    hitac_installed = False
else:
    hitac_installed = True


@pytest.mark.skipif(
    not exists("tests/fixtures/fungi_train.csv")
    or not exists("tests/fixtures/fungi_test.csv"),
    reason="dataset not available",
)
@pytest.mark.skipif(
    "FUNGI_TRAIN_URL" not in os.environ
    or "FUNGI_TRAIN_MD5" not in os.environ
    or "FUNGI_TEST_URL" not in os.environ
    or "FUNGI_TEST_MD5" not in os.environ,
    reason="environment variables not set",
)
@pytest.mark.skipif(not skbio_installed, reason="scikit-bio not installed")
@pytest.mark.skipif(not hitac_installed, reason="hitac not installed")
@pytest.mark.parametrize(
    "model",
    [
        LocalClassifierPerNode(),
        LocalClassifierPerParentNode(),
        LocalClassifierPerLevel(),
    ],
)
def test_fungi(model):
    assert False

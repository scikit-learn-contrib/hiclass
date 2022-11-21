import os
from multiprocessing import cpu_count
from os.path import exists

import pytest
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression

from hiclass import (
    LocalClassifierPerNode,
)
from hiclass.metrics import f1
from tests.conftest import get_taxonomy

try:
    import skbio
except ImportError:
    skbio_installed = False
else:
    skbio_installed = True

try:
    from hitac._utils import (
        compute_possible_kmers,
        _extract_reads,
        compute_frequencies,
        extract_qiime2_taxonomy,
    )
except ImportError:
    hitac_installed = False
else:
    hitac_installed = True


try:
    from q2_types.feature_data import DNAIterator
except ImportError:
    qiime2_installed = False
else:
    qiime2_installed = True


@pytest.mark.skipif(
    not exists("tests/fixtures/fungi_train.fasta")
    or not exists("tests/fixtures/fungi_test.fasta"),
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
@pytest.mark.skipif(not qiime2_installed, reason="qiime2 not installed")
@pytest.mark.parametrize(
    "model",
    [
        LocalClassifierPerNode(),
        # LocalClassifierPerParentNode(),
        # LocalClassifierPerLevel(),
    ],
)
def test_fungi(model):
    # Variables
    train = "tests/fixtures/fungi_train.fasta"
    test = "tests/fixtures/fungi_test.fasta"
    kmer_size = 6
    alphabet = "ACGT"
    threads = min(cpu_count(), 12)
    logistic_regression_parameters = {
        "solver": "liblinear",
        "multi_class": "auto",
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 10000,
        "verbose": 0,
        "n_jobs": 1,
    }

    # Training
    kmers = compute_possible_kmers(kmer_size=kmer_size, alphabet=alphabet)
    train = DNAIterator(skbio.read(str(train), format="fasta", constructor=skbio.DNA))
    training_ids, training_sequences = _extract_reads(train)
    x_train = compute_frequencies(training_sequences, kmers, threads=threads)
    y_train = get_taxonomy(training_ids)
    lr = LogisticRegression(**logistic_regression_parameters)
    model = model.set_params(local_classifier=lr, n_jobs=threads)
    with parallel_backend("threading", n_jobs=threads):
        model.fit(x_train, y_train)

    # Testing
    test = DNAIterator(skbio.read(str(test), format="fasta", constructor=skbio.DNA))
    test_ids, test_sequences = _extract_reads(test)
    x_test = compute_frequencies(test_sequences, kmers, threads)
    y_test = get_taxonomy(test_ids)
    predictions = model.predict(x_test)
    assert f1(y_true=y_test, y_pred=predictions) == 1.0

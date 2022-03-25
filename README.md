# HiClass

HiClass is an open-source Python library for hierarchical classification compatible with scikit-learn

[![Deploy PyPI](https://github.com/mirand863/hiclass/actions/workflows/deploy-pypi.yml/badge.svg?event=push)](https://github.com/mirand863/hiclass/actions/workflows/deploy-pypi.yml) [![Documentation Status](https://readthedocs.org/projects/hiclass/badge/?version=latest)](https://hiclass.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/mirand863/hiclass/branch/main/graph/badge.svg?token=PR8VLBMMNR)](https://codecov.io/gh/mirand863/hiclass) [![Downloads Conda](https://img.shields.io/conda/dn/conda-forge/hiclass?label=conda)](https://anaconda.org/conda-forge/hiclass) [![Downloads pypi](https://img.shields.io/pypi/dm/hiclass?label=pypi)](https://pypi.org/project/hiclass/)  [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

✨ Here are a couple of **demos** that show HiClass in action on hierarchical datasets:

- Classify a consumer complaints dataset from the consumer financial protection bureau: [consumer-complaints]()
- Classify a 16S rRNA dataset from the TAXXI benchmark: [16s-rrna]()

## Quick Links

- [Features](#features)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [Who is using HiClass](#who-is-using-this)
- [Install](#install)
- [Quick start](#quick-start)
- [Step-by-step- walk-through](#step-by-step-walk-through)
- [API documentation](#api-documentation)
- [FAQ](#faq)
- [Support](#support)
- [Contributing](#contributing)
- [Getting latest updates](#getting-latest-updates)
- [Citation](#citation)

## Features

- **Python lists and NumPy arrays:** Handles Python lists and NumPy arrays elegantly, out-of-the-box.
- **Pandas Series and DataFrames:** If you prefer to work with pandas, that is not an issue as HiClass also works with Pandas.
- **Sparse matrices:** HiClass also supports features (X_train and X_test) built with sparse matrices, both for training and predicting.
- **Parallel training:** Training can be performed in parallel on the hierarchical classifiers, which allows parallelization regardless of the implementations available on scikit-learn.
- **Build pipelines and perform hyper-parameter tuning:** Since the hierarchical classifiers inherit from the BaseEstimator of scikit-learn, pipelines can be built and grid search executed to find out the best parameters.

**Don't see a feature on this list?** Search our [issue tracker](https://github.com/mirand863/hiclass/issues) if someone has already requested it and add a comment to it explaining your use-case, or open a new issue if not. We prioritize our roadmap based on user feedback, so we'd love to hear from you.

## Install

### Option 1: Conda

HiClass and its dependencies can be easily installed with conda:

```shell
conda install -c conda-forge hiclass
```

### Option 2: Pip

Alternatively, HiClass and its dependencies can also be installed with pip:

```shell
pip install hiclass
```

## Quick start

Here's a quick example showcasing how you can train and predict using a local classifier per node.

```python
from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier

# define data
X_train = [[1], [2], [3], [4]]
X_test = [[4], [3], [2], [1]]
Y_train = [
    ['Animal', 'Mammal', 'Sheep'],
    ['Animal', 'Mammal', 'Cow'],
    ['Animal', 'Reptile', 'Snake'],
    ['Animal', 'Reptile', 'Lizard'],
]

# Use random forest classifiers for every node
rf = RandomForestClassifier()
lcpn = LocalClassifierPerNode(local_classifier=rf)

# Train local classifier per node
lcpn.fit(X_train, Y_train)

# Predict
predictions = lcpn.predict(X_test)
```

## Step-by-step walk-through

A step-by-step walk-through is available on our interactive notebook hosted on [Google Colab](https://colab.research.google.com/drive/1Idzht9dNoB85pjc9gOL24t9ksrXZEA-9?usp=sharing).

This will guide you through the process of installing hiclass with conda, training and predicting a small dataset.

## API Documentation

Here's our official API documentation, available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/).

If you notice any issues with the documentation or walk-through, please let us know by opening an issue here: [https://github.com/mirand863/hiclass/issues](https://github.com/mirand863/hiclass/issues).

## Contributing

We are a small team on a mission to democratize hierarchical classification, and we'll take all the help we can get! If you'd like to get involved, here's information on where we could use your help: [Contributing.md](https://github.com/mirand863/hiclass/blob/master/CONTRIBUTING.md)

## Getting Latest Updates

If you'd like to get updates when we release new versions, please click on the "Watch" button on the top and select "Releases only". Github will then send you notifications along with a changelog with each new release.

## Citation

If you use HiClass, please cite:

> Miranda, Fábio M., Niklas Köehnecke, and Bernhard Y. Renard. "HiClass: a Python library for local hierarchical classification compatible with scikit-learn." arXiv preprint arXiv:2112.06560 (2021).

```latex
@article{miranda2021hiclass,
  title={HiClass: a Python library for local hierarchical classification compatible with scikit-learn},
  author={Miranda, F{\'a}bio M and K{\"o}ehnecke, Niklas and Renard, Bernhard Y},
  journal={arXiv preprint arXiv:2112.06560},
  year={2021}
}
```

In addition, we would like to list publications that use our software on our repository. Please email the reference, the name of your lab, department and institution to fabio.malchermiranda@hpi.de

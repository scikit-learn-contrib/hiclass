# HiClass

HiClass is an open-source python library for hierarchical classification compatible with scikit-learn

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

## Install

### Option 1: Conda

HiClass and its dependencies can be easily installed with conda:

```
conda install -c conda-forge hiclass
```

### Option 2: Pip

Alternatively, HiClass and its dependencies can also be installed with pip:

```
pip install hiclass
```

## API Documentation

Here's our official API documentation, available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/).

If you notice any issues with the documentation or walk-through, please let us know by opening an issue here: [https://github.com/mirand863/hiclass/issues](https://github.com/mirand863/hiclass/issues).

## TODO: rewrite quick start

~~An example usage can be found below. For a more thorough example, see [our interactive notebook](https://colab.research.google.com/drive/1Idzht9dNoB85pjc9gOL24t9ksrXZEA-9?usp=sharing). The full API documentation is available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/).~~

```python
from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier

# define data
X_train, X_test = get_some_train_data()  # (n, num_features)
Y_train = get_some_labels()  # (n, num_largest_hierarchy)
# Use random forest classifiers for every node and run a classification
rf = RandomForestClassifier()
lcpn = LocalClassifierPerNode(local_classifier=rf)
lcpn.fit(X_train, Y_train)
predictions = lcpn.predict(X_test)
```

## Contributing

We are a small team on a mission to democratize hierarchical classification, and we'll take all the help we can get! If you'd like to get involved, here's information on where we could use your help: [Contributing.md](https://github.com/mirand863/hiclass/blob/master/CONTRIBUTING.md)

## Getting Latest Updates

If you'd like to get updates when we release new versions, please click on the "Watch" button on the top and select "Releases only". Github will then send you notifications along with a changelog with each new release.

## Citation

If you use HiClass, please cite:

> Miranda, Fábio M., Niklas Köehnecke, and Bernhard Y. Renard. "HiClass: a Python library for local hierarchical classification compatible with scikit-learn." arXiv preprint arXiv:2112.06560 (2021).

```
@article{miranda2021hiclass,
  title={HiClass: a Python library for local hierarchical classification compatible with scikit-learn},
  author={Miranda, F{\'a}bio M and K{\"o}ehnecke, Niklas and Renard, Bernhard Y},
  journal={arXiv preprint arXiv:2112.06560},
  year={2021}
}
```

In addition, we would like to list publications that use our software on our repository. Please email the reference, the name of your lab, department and institution to fabio.malchermiranda@hpi.de

# Hiclass - Hierarchical Classification Library

[![pipeline status](https://gitlab.com/dacs-hpi/hiclass/badges/master/pipeline.svg)](https://gitlab.com/dacs-hpi/hiclass/-/commits/master) [![coverage report](https://gitlab.com/dacs-hpi/hiclass/badges/master/coverage.svg)](https://gitlab.com/dacs-hpi/hiclass/-/commits/master) [![Documentation Status](https://readthedocs.org/projects/hiclass/badge/?version=latest)](https://hiclass.readthedocs.io/en/latest/?badge=latest)

This library implements the three local classifier approaches described in [[1]](#1).

## Installation

[![Install with conda](https://anaconda.org/conda-forge/hiclass/badges/installer/conda.svg)](https://anaconda.org/conda-forge/hiclass)

HiClass and its dependencies can be easily installed with conda:

```shell
conda install -c conda-forge hiclass
```

[![Install with pip](https://badge.fury.io/py/hiclass.svg)](https://pypi.org/project/hiclass/)

Alternatively, HiClass and its dependencies can also be installed with pip:

```shell
pip install hiclass
```

Lastly, `pipenv` can also be used to install HiClass and its dependencies. In order to use this, first install it via:
```shell
pip install pipenv
```
Afterwards, you can create an environment and install the dependencies via (for dev dependencies, add `--dev`)
```shell
pipenv install
```
To activate the environment, run:
```shell
pipenv shell
```
For more information, take a look at the [pipenv documentation](https://pipenv.pypa.io/en/latest/).

If you do not wish to use pipenv, you can find the requirements in `Pipfile` under `packages` and `dev-packages`.

## Usage

An example usage can be found below. For a more thorough example, see [our interactive notebook](https://colab.research.google.com/drive/1Idzht9dNoB85pjc9gOL24t9ksrXZEA-9?usp=sharing). The full API documentation is available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/).

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

## References
<a id="1">[1]</a> 
Silla, C.N. and Freitas, A.A. (2011).
A survey of hierarchical classification across different application domains.
Data Mining and Knowledge Discovery, 22(1), pp.31-72.

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

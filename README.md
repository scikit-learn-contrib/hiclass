# HiClass

HiClass is an open-source Python library for hierarchical classification compatible with scikit-learn.

[![Deploy PyPI](https://github.com/mirand863/hiclass/actions/workflows/deploy-pypi.yml/badge.svg?event=push)](https://github.com/mirand863/hiclass/actions/workflows/deploy-pypi.yml) [![Documentation Status](https://readthedocs.org/projects/hiclass/badge/?version=latest)](https://hiclass.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/mirand863/hiclass/branch/main/graph/badge.svg?token=PR8VLBMMNR)](https://codecov.io/gh/mirand863/hiclass) [![Downloads Conda](https://img.shields.io/conda/dn/conda-forge/hiclass?label=conda)](https://anaconda.org/conda-forge/hiclass) [![Downloads pypi](https://img.shields.io/pypi/dm/hiclass?label=pypi)](https://pypi.org/project/hiclass/)  [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

✨ Here are a couple of **demos** that show HiClass in action on hierarchical datasets:

- Classify a consumer complaints dataset from the consumer financial protection bureau: [consumer-complaints](https://colab.research.google.com/drive/1rQTDxWcck-PH4saKzrofQ7Sg9W23lYZv?usp=sharing)
- Classify a 16S rRNA dataset from the TAXXI benchmark: [16s-rrna]()

## Quick links

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
- [Getting the latest updates](#getting-the-latest-updates)
- [Citation](#citation)

## Features

- **Python lists and NumPy arrays:** Handles Python lists and NumPy arrays elegantly, out-of-the-box.
- **Pandas Series and DataFrames:** If you prefer to use pandas, that is not an issue as HiClass also works with Pandas.
- **Sparse matrices:** HiClass also supports features (X_train and X_test) built with sparse matrices, both for training and predicting, which can save you heaps of memory.
- **Parallel training:** Training can be performed in parallel on the hierarchical classifiers, which allows parallelization regardless of the implementations available on scikit-learn.
- **Build pipelines:** Since the hierarchical classifiers inherit from the BaseEstimator of scikit-learn, pipelines can be built to automate machine learning workflows.
- **Hierarchical metrics:** HiClass supports the computation of hierarchical precision, recall and f-score, which are more appropriate for hierarchical data than traditional metrics.
- **Compatible with pickle:** Easily store trained models on disk for future use.

**Don't see a feature on this list?** Search our [issue tracker](https://github.com/mirand863/hiclass/issues) if someone has already requested it and add a comment to it explaining your use-case, or open a new issue if not. We prioritize our roadmap based on user feedback, so we'd love to hear from you.

## Benchmarks

### Consumer complaints dataset with ~600K training examples

This first benchmark was executed on Google Colab with only 1 core, using Logistic Regression as the base classifier.

|Classifier|Training Time (hh::mm:ss)|Memory Usage (GB)|Disk Usage (MB)|F-score|
|----------|:-----------------------:|:---------------:|:-------------:|:-----:|
|[Local Classifier per Parent Node](https://colab.research.google.com/drive/1yZlQ9UnBEGdkIpnJ3pBwvbZ-U0SXL-UG?usp=sharing)|01:00:01|5.21|118|**0.7630**|
|[Local Classifier per Node](https://colab.research.google.com/drive/1rQTDxWcck-PH4saKzrofQ7Sg9W23lYZv?usp=sharing)|**00:21:14**|**4.70**|120|0.7587|
|[Local Classifier per Level](https://colab.research.google.com/drive/1b_Qb2d6RhSO7ICYTIsxH6ZqCVgeKWmll?usp=sharing)|03:11:42|9.69|120|0.7626|
|[Flat Classifier](https://colab.research.google.com/drive/10jgzA65WaoTc7tFfrlKlhlwPBs3PFy9m?usp=sharing)|03:09:35|8.98|**104**|0.7565|

This second benchmark is similar to the last one, except that it was executed on a cluster node running GNU/Linux with 512 GB physical memory and 128
cores provided by two AMD EPYC™ 7742 processors, and each model had 12 cores available for training.

|Classifier|Training Time (hh::mm:ss)|Memory Usage (GB)|Disk Usage (MB)|F-score|
|----------|:-----------------------:|:---------------:|:-------------:|:-----:|
|Local Classifier per Parent Node|00:21:54|3.87|116|**0.7606**|
|Local Classifier per Node|**00:05:33**|**3.76**|118|0.7563|
|Local Classifier per Level|01:52:42|3.87|118|**0.7606**|
|Flat Classifier|01:48:54|7.23|**103**|0.7553|

This third benchmark was also executed on the same cluster node as the previous benchmark and 12 cores were provided for each model, however, the base classifier was LightGBM instead.

|Classifier|Training Time (hh::mm:ss)|Memory Usage (GB)|Disk Usage (MB)|F-score|
|----------|:-----------------------:|:---------------:|:-------------:|:-----:|
|Local Classifier per Parent Node|**00:24:42**|3.87|77|0.7127|
|Local Classifier per Node|00:30:50|4.87|311|**0.7503**|
|Local Classifier per Level|01:45:57|**3.81**|29|0.5732|
|Flat Classifier|00:28:07|4.34|**20**|0.1260|

Lastly, this fourth benchmark was also executed on the same cluster node as the previous benchmarks and 12 cores were provided for each model, however, the base classifier was random forest instead.

|Classifier|Training Time (hh::mm:ss)|Memory Usage (GB)|Disk Usage (GB)|F-score|
|----------|:-----------------------:|:---------------:|:-------------:|:-----:|
|Local Classifier per Parent Node|03:04:23|**34.98**|**11**|0.7133|
|Local Classifier per Node|02:21:05|39.16|12|**0.7450**|
|Local Classifier per Level|03:58:59|136.50|43|0.7093|
|Flat Classifier|**00:31:02**|77.32|37|0.6405|

For reproducibility, a Snakemake pipeline was created. Instructions on how to run it and source code are available at [https://github.com/mirand863/hiclass/tree/main/benchmarks/consumer_complaints](https://github.com/mirand863/hiclass/tree/main/benchmarks/consumer_complaints).

We would love to benchmark with larger datasets, if we can find large ones in the public domain. If you have any suggestions for hierarchical datasets that are open, please let us know by opening an issue. We would also be delighted if you are able to share benchmarks from your own large datasets. Please send us a PR!

## Roadmap

Here is our public roadmap: https://github.com/mirand863/hiclass/projects/1.

We do Just-In-Time planning, and we tend to reprioritize based on your feedback. Hence, items you see on this roadmap are subject to change. We prioritize features based on the number of people asking for it, features/fixes that are small enough and can be addressed while we work on other related features, features/fixes that help improve stability & relevance and features that address interesting use cases that excite us! If you'd like to have a request prioritized, we ask that you add a detailed use-case for it, either as a comment on an existing issue (besides a thumbs-up) or in a new issue. The detailed context helps.

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

Here's a quick example showcasing how you can train and predict using a local classifier per node, with a `RandomForestClassifier` for each node:

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
classifier = LocalClassifierPerNode(local_classifier=rf)

# Train local classifier per node
classifier.fit(X_train, Y_train)

# Predict
predictions = classifier.predict(X_test)
```

HiClass can also be adopted in scikit-learn pipelines, and fully supports sparse matrices as input. In order to demonstrate the use of both of these features, we will use the following example:

```python
from hiclass import LocalClassifierPerParentNode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# define data
X_train = [
    'Struggling to repay loan',
    'Unable to get annual report',
]
X_test = [
    'Unable to get annual report',
    'Struggling to repay loan',
]
Y_train = [
    ['Loan', 'Student loan'],
    ['Credit reporting', 'Reports']
]
```

Now, let's build a pipeline that will use `CountVectorizer` and `TfidfTransformer` to extract features as sparse matrices:

```python
# Use logistic regression classifiers for every parent node
lr = LogisticRegression()
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lcppn', LocalClassifierPerParentNode(local_classifier=lr)),
])
```

Finally, let's train and predict with the pipeline we just created:

```python
# Train local classifier per parent node
pipeline.fit(X_train, Y_train)

# Predict
predictions = pipeline.predict(X_test)
```

## Step-by-step walk-through

A step-by-step walk-through is available on our interactive notebook hosted on [Google Colab](https://colab.research.google.com/drive/1Idzht9dNoB85pjc9gOL24t9ksrXZEA-9?usp=sharing).

This will guide you through the process of installing hiclass with conda, training and predicting a small dataset.

## API documentation

Here's our official API documentation, available on [Read the Docs](https://hiclass.readthedocs.io/en/latest/).

If you notice any issues with the documentation or walk-through, please let us know by opening an issue here: [https://github.com/mirand863/hiclass/issues](https://github.com/mirand863/hiclass/issues).

## Support

If you run into any problems or issues, please create a [Github issue](https://github.com/mirand863/hiclass/issues) and we'll try our best to help.

We strive to provide good support through our issue tracker on Github. However, if you'd like to receive private support with:

- Phone / video calls to discuss your specific use case and get recommendations
- Private discussions over Slack or Mattermost

Please reach out to us at fabio.malchermiranda@hpi.de.

## Contributing

We are a small team on a mission to democratize hierarchical classification, and we'll take all the help we can get! If you'd like to get involved, here's information on contribution guidelines and how to test the code locally: [CONTRIBUTING.md](https://github.com/mirand863/hiclass/blob/main/CONTRIBUTING.md)

## Getting the latest updates

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

In addition, we would like to list publications that use HiClass to solve hierarchical problems. If you would like your manuscript to be added to this list, please email the reference, the name of your lab, department and institution to fabio.malchermiranda@hpi.de

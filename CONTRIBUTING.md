## General guidelines

To contribute, fork the repository and send a pull request.

When submitting code, please make every effort to follow existing conventions and style in order to keep the code as readable as possible.

Where appropriate, please provide unit tests or integration tests. Unit tests should be pytest based tests and be added to <project>/tests.

Please make sure all tests pass before submitting a pull request. It is also good if you squash your commits and add the tags #major or #minor to the pull request title if need be, otherwise your pull request will be considered a patch bump. Please check [https://semver.org/](https://semver.org/) for more information about versioning.

## Testing the code locally

To test the code locally you need to install the dependencies for the library in the current environment. Additionally, you need to install the dependencies for testing. All of those dependencies can be installed with:

```
pip install -e ".[dev]"
```

To run the tests simply execute:

```
pytest -v --flake8 --pydocstyle --cov=hiclass --cov-fail-under=90 --cov-report html
```

Lastly, you can set up the git hooks scripts to fix formatting errors locally during commits:

```
pre-commit install
```

If black is not executed locally and there are formatting errors, the CI/CD pipeline will fail.

## Building the documentation locally

To build the documentation locally, you need to install another set of dependencies that are specific for the documentation. It is easier to create a separate conda environment and run the following command:

```
pip install -r docs/requirements.txt
```

To build the documentation you need to change to run the following commands:

```
cd docs
make html
```

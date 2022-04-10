from sklearn.utils.estimator_checks import parametrize_with_checks

from hiclass import LocalClassifierPerLevel


@parametrize_with_checks([LocalClassifierPerLevel()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

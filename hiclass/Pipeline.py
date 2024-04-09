"""Custom Pipeline class that supports the `calibrate` method."""
from sklearn.pipeline import Pipeline as skPipeline


class Pipeline(skPipeline):
    """Custom Pipeline class that supports the `calibrate` method."""

    def __init__(self, steps, **kwargs):
        """Create Pipeline object."""
        super().__init__(steps, **kwargs)

    def calibrate(self, X, y, **params):
        """Transform the data and apply `calibrate` with the final estimator."""
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].calibrate(Xt, y)

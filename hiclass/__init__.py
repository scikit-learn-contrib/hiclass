"""Init module for the library."""

import os
from ._version import get_versions
from .LocalClassifierPerLevel import LocalClassifierPerLevel
from .LocalClassifierPerNode import LocalClassifierPerNode
from .LocalClassifierPerParentNode import LocalClassifierPerParentNode
from .LocalClassifierPerLevel import LocalClassifierPerLevel
from .Pipeline import Pipeline
from .FlatClassifier import FlatClassifier
from .MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
from .MultiLabelLocalClassifierPerParentNode import (
    MultiLabelLocalClassifierPerParentNode,
)
from .Explainer import Explainer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
    "Pipeline",
    "FlatClassifier",
    "Explainer",
    "MultiLabelLocalClassifierPerNode",
    "MultiLabelLocalClassifierPerParentNode",
    "datasets",
]

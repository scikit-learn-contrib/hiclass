"""Init module for the library."""
import os
from ._version import get_versions
from .LocalClassifierPerNode import LocalClassifierPerNode
from .LocalClassifierPerParentNode import LocalClassifierPerParentNode
from .LocalClassifierPerLevel import LocalClassifierPerLevel
from .Pipeline import Pipeline

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
    "Pipeline",
]

"""Init module for the library."""
from .LocalClassifierPerLevel import LocalClassifierPerLevel
from .LocalClassifierPerNode import LocalClassifierPerNode
from .LocalClassifierPerParentNode import LocalClassifierPerParentNode
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
]

"""Init module for the library."""

from .LocalClassifierPerLevel import LocalClassifierPerLevel
from .LocalClassifierPerNode import LocalClassifierPerNode
from .LocalClassifierPerParentNode import LocalClassifierPerParentNode
from .MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
from .MultiLabelLocalClassifierPerParentNode import (
    MultiLabelLocalClassifierPerParentNode,
)
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
    "MultiLabelLocalClassifierPerNode",
    "MultiLabelLocalClassifierPerParentNode",
]

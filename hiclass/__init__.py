"""Init module for the library."""

import os
from ._version import get_versions
from .LocalClassifierPerNode import LocalClassifierPerNode
from .LocalClassifierPerParentNode import LocalClassifierPerParentNode
from .LocalClassifierPerLevel import LocalClassifierPerLevel

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
]

# Try to import the Explainer module
try:
    from .explainer import Explainer
    # If successful, add 'Explainer' to __all__
    __all__.append('Explainer')
except ImportError:
    # If the import fails, Explainer is not available
    pass

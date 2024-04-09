"""Init the probability combiner module."""

from .MultiplyCombiner import MultiplyCombiner
from .ArithmeticMeanCombiner import ArithmeticMeanCombiner
from .GeometricMeanCombiner import GeometricMeanCombiner

__all__ = [
    "MultiplyCombiner",
    "ArithmeticMeanCombiner",
    "GeometricMeanCombiner",
]

init_strings = [
    "multiply",
    "geometric",
    "arithmetic",
]
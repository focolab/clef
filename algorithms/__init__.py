"""
CLEF Algorithm Package

This package contains the algorithm factory and various closed-loop trigger
algorithms for microscopy experiments.
"""

from .algorithm_factory import (
    create_algorithm,
    list_available_algorithms,
    register_algorithm,
)

__all__ = [
    'create_algorithm',
    'list_available_algorithms',
    'register_algorithm',
]
# veco_ai/__init__.py
"""
veco-ai package initializer.

- Avoids circular imports.
- Exposes the public API (Vectorize).
"""

__version__ = "0.1.0"

from .veco_ai import Vectorize

__all__ = ["Vectorize", "__version__"]

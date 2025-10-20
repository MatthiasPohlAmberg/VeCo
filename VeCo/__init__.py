# VeCo/__init__.py
"""
VeCo package initializer.

- Avoids circular imports.
- Exposes the public API (Vectorize).
"""

__version__ = "0.1.0"

from .veco import Vectorize

__all__ = ["Vectorize", "__version__"]

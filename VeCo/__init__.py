# VeCo/__init__.py
"""
VeCo package initializer.

- Vermeidet Zirkularimporte.
- Exportiert die öffentliche API (Vectorize).
"""

from .veco import Vectorize  # ✅ relativer Import, kein Kreis
__all__ = ["Vectorize"]

"""FLUED research package.

The stable v2 modules remain available at their original import paths.  The
v3.3 public prototype is exposed under ``flued.v33`` so ablation scripts can
import the architecture without reaching into private files.
"""

from .v33 import FLUEDV33, FLUEDV33Config, FLUEDV33Output

__all__ = ["FLUEDV33", "FLUEDV33Config", "FLUEDV33Output"]

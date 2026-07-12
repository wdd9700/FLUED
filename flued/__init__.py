"""FLUED research package.

The stable v2 modules remain available at their original import paths. Versioned
v3.3 and v3.4 prototypes are exposed without requiring callers to reach into
private implementation files.
"""

from .v33 import FLUEDV33, FLUEDV33Config, FLUEDV33Output
from .v34 import FLUEDV34, FLUEDV34Config, FLUEDV34Probe, FLUEDV34ProbeConfig, V34ProbeOutput

__all__ = [
    "FLUEDV33",
    "FLUEDV33Config",
    "FLUEDV33Output",
    "FLUEDV34",
    "FLUEDV34Config",
    "FLUEDV34Probe",
    "FLUEDV34ProbeConfig",
    "V34ProbeOutput",
]

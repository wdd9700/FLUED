"""FLUED research package.

The stable v2 modules remain available at their original import paths. Versioned
v3.3+ implementations are exposed without requiring callers to reach into
private implementation files; v3.6 (KDA generation) is the current mainline.
"""

from .v33 import FLUEDV33, FLUEDV33Config, FLUEDV33Output
from .v34 import FLUEDV34, FLUEDV34Config, FLUEDV34Probe, FLUEDV34ProbeConfig, V34ProbeOutput
from .v36 import FLUEDV36, V36Config, V36Output
from .hnet_repro import HNetRepro, HNetReproConfig

__all__ = [
    "FLUEDV33",
    "FLUEDV33Config",
    "FLUEDV33Output",
    "FLUEDV34",
    "FLUEDV34Config",
    "FLUEDV34Probe",
    "FLUEDV34ProbeConfig",
    "V34ProbeOutput",
    "FLUEDV36",
    "V36Config",
    "V36Output",
    "HNetRepro",
    "HNetReproConfig",
]

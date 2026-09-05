"""Utilities for adapting the 2D acoustic FNO notebook to pino.hdf5."""

from .data import PinoHDF5Dataset, create_splits
from .model import AcousticFNO3D
from .model_ais_mqfno import AISMQFNO
from .model_factorized import FactorizedAcousticFNO

__all__ = [
    "AcousticFNO3D",
    "AISMQFNO",
    "FactorizedAcousticFNO",
    "PinoHDF5Dataset",
    "create_splits",
]

# These architectures live only on branches that include their implementation
# modules.  Keep the shared package importable when those optional files are not
# present, while preserving the public exports whenever they are available.
try:
    from .model_deeponet import FactorizedTemporalBranchEncoder, SDRDeepONet
except ModuleNotFoundError as error:
    if error.name != f"{__name__}.model_deeponet":
        raise
else:
    __all__.extend(("FactorizedTemporalBranchEncoder", "SDRDeepONet"))

try:
    from .model_sdr import SDRPINO
except ModuleNotFoundError as error:
    if error.name != f"{__name__}.model_sdr":
        raise
else:
    __all__.append("SDRPINO")

"""Top-level package for OPICS."""

from SiEPIC.opics import libraries
from SiEPIC.opics.network import Network
from SiEPIC.opics.utils import netlistParser
#from SiEPIC.opics.globals import C, F
from SiEPIC.opics.globals import C

__author__ = "Jaspreet Jhoja"
__email__ = "jj@alumni.ubc.ca"
__version__ = "0.3.2"

# initialize OPICS package

name = "opics"

__all__ = [
    "Network",
    "libraries",
    "globals",
    "netlistParser",
    "C",
    "F",
]

print(
    r"""
   ____  ____  _______________
  / __ \/ __ \/  _/ ____/ ___/
 / / / / /_/ // // /    \__ \
/ /_/ / ____// // /___ ___/ /
\____/_/   /___/\____//____/
"""
)
print("OPICS version", __version__)

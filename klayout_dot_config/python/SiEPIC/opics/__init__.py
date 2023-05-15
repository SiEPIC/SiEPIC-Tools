"""Top-level package for OPICS.

Frequency-domain Circuit simulator using S-Parameters

Component library is pulled from each Technology PDK, from the sub-folders "CML/*"


"""

# Required packages:
from SiEPIC.install import install
import pya
if not install('numpy'):
    pya.MessageBox.warning(
    "Missing package", "The OPICS circuit simulator does not function without the package 'numpy'.",  pya.MessageBox.Ok)    
if not install('scipy'):
    pya.MessageBox.warning(
    "Missing package", "The OPICS circuit simulator does not function without the package 'scipy'.",  pya.MessageBox.Ok)    
if not install('yaml'):
    pya.MessageBox.warning(
    "Missing package", "The OPICS circuit simulator does not function without the package 'yaml'.",  pya.MessageBox.Ok)    
if not install('defusedxml'):
    pya.MessageBox.warning(
    "Missing package", "The OPICS circuit simulator does not function without the package 'defusedxml'.",  pya.MessageBox.Ok)    


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

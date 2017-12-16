import pya

# Netlist extraction will merge straight+bend sections into waveguide (1), 
# or extract each bend, straight section, etc. (0)
#WAVEGUIDE_extract_simple = 1
SIMPLIFY_NETLIST_EXTRACTION = True

#Create GUI's
from .core import WaveguideGUI, MonteCarloGUI, CalibreGUI, Net
WG_GUI = WaveguideGUI()
MC_GUI = MonteCarloGUI()
DRC_GUI = CalibreGUI()
#Define global Net object that implements netlists and pin searching/connecting
NET = Net()

#Define enumeration for pins
from .utils import enum
PIN_TYPES = enum('I/O', 'OPTICAL', 'ELECTRICAL')

try:
  import numpy
except ImportError:
  MODULE_NUMPY = False
  
#ACTIONS = []
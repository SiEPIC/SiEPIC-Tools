import pya

# Netlist extraction will merge straight+bend sections into waveguide (1), 
# or extract each bend, straight section, etc. (0)
#WAVEGUIDE_extract_simple = 1
SIMPLIFY_NETLIST_EXTRACTION = True

#Create GUI's
from .core import WaveguideGUI, MonteCarloGUI, Net, Component
WG_GUI = WaveguideGUI()
MC_GUI = MonteCarloGUI()

# ******** lukasc
# don't use a global one.. based on cells
#Define global Net object that implements netlists and pin searching/connecting
# NET = Net()

NET_DISCONNECTED = Net()

# don't use a global one.. based on cells
#Define global Component object
#COMPONENT = Component()

#Define enumeration for pins
from .utils import enum
PIN_TYPES = enum('OPTICALIO', 'OPTICAL', 'ELECTRICAL')
PIN_LENGTH = 100  # 0.1 micron

try:
  MODULE_NUMPY = True
  import numpy
except ImportError:
  MODULE_NUMPY = False
  
#ACTIONS = []

KLAYOUT_VERSION = int(pya.Application.instance().version().split('.')[1])

# Waveguide DevRec: space between the waveguide and the DevRec polygon
WG_DEVREC_SPACE = 1

# Path to Waveguide, path snapping to nearest pin. Search for pin with this distance:
PATH_SNAP_PIN_MAXDIST = 20


# Load INTC element library details KLayout application data path
import os 
path = os.path.join(pya.Application.instance().application_data_path(), 'Lumerical_CMLs')
path = os.path.join(path,"Lumerical_INTC_CMLs.txt")
INTC_ELEMENTS = ''
if os.path.exists(path):
  print('loading Lumerical_INTC_CMLs.txt')
  fh = open(path, "r")
  INTC_ELEMENTS = fh.read()
  fh.close()
  
try:
  INTC
except:
  INTC = None  
  print('resetting Lumerical INTERCONNECT Python integration')

try:
  FDTD
except:
  FDTD = None  
  print('resetting Lumerical FDTD Python integration')

try:
  LUMAPI
except:
  LUMAPI = None  
  print('resetting Lumerical Python integration')


try:
  TEMP_FOLDER
except:
  import tempfile
  TEMP_FOLDER = tempfile.mkdtemp()

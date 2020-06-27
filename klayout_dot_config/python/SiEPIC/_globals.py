import pya

# Netlist extraction will merge straight+bend sections into waveguide (1),
# or extract each bend, straight section, etc. (0)
#WAVEGUIDE_extract_simple = 1
SIMPLIFY_NETLIST_EXTRACTION = True

# Commenting this out. by @tlima
# Only usage of WG_GUI found in :
#  - SiEPIC-Tools/klayout_dot_config/python/SiEPIC/scripts.py
# Only usage of WG_GUI found in :
#  - SiEPIC-Tools/klayout_dot_config/python/SiEPIC/lumerical/interconnect.py
# # Create GUI's
# from .core import WaveguideGUI, MonteCarloGUI
# WG_GUI = WaveguideGUI()
# MC_GUI = MonteCarloGUI()

# ******** lukasc
# don't use a global one.. based on cells
# Define global Net object that implements netlists and pin searching/connecting
# NET = Net()

from .core import Net
NET_DISCONNECTED = Net()

# don't use a global one.. based on cells
# Define global Component object
# from .core import Component
# COMPONENT = Component()


# Define an Enumeration type for Python
# TODO: maybe move to standard enum for python3
# https://docs.python.org/3/library/enum.html
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define enumeration for pins
PIN_TYPES = enum('OPTICALIO', 'OPTICAL', 'ELECTRICAL')
PIN_LENGTH = 100  # 0.1 micron


MODULE_NUMPY = False
try:
    import numpy
    MODULE_NUMPY = True
except ImportError:
    from .install import install_numpy
    try:
        MODULE_NUMPY = install_numpy()
    except Exception as e:
        print("Could not install numpy with pip. ERROR:", e)

#ACTIONS = []
try: # KOM - testing functionality for cases without GUI
    KLAYOUT_VERSION = int(pya.Application.instance().version().split('.')[1])
except: # KOM - testing functionality for cases without GUI
    KLAYOUT_VERSION = int(26) # KOM - testing functionality for cases without GUI

# Waveguide DevRec: space between the waveguide and the DevRec polygon
WG_DEVREC_SPACE = 1

# Path to Waveguide, path snapping to nearest pin. Search for pin with this distance:
PATH_SNAP_PIN_MAXDIST = 20


# Load INTC element library details KLayout application data path
try: # KOM - testing functionality for cases without GUI
    import os
    path = os.path.join(pya.Application.instance().application_data_path(), 'Lumerical_CMLs')
    path = os.path.join(path, "Lumerical_INTC_CMLs.txt")
    INTC_ELEMENTS = ''
    if os.path.exists(path):
        print('loading Lumerical_INTC_CMLs.txt')
        fh = open(path, "r")
        INTC_ELEMENTS = fh.read()
        fh.close()
except:  # KOM - testing functionality for cases without GUI
    pass

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
    MODE
except:
    MODE = None
    print('resetting Lumerical MODE Python integration')

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


import pya
Python_Env = "" # tag which defines whether we are loading library in script or GUI env
if 'Application' in dir(pya):
    try:
        # import pya, which is available when running within KLayout
        if pya.Application.instance().main_window():
            Python_Env = "KLayout_GUI"
            print('Python Environment: KLayout GUI')
        else:
            Python_Env = "KLayout_batch"
            print('Python Environment: KLayout batch mode')
    except:
        Python_Env = "Script"
else:
    Python_Env = "Script"

# Netlist extraction will merge straight+bend sections into waveguide (1),
# or extract each bend, straight section, etc. (0)
#WAVEGUIDE_extract_simple = 1
SIMPLIFY_NETLIST_EXTRACTION = True
    
if Python_Env == "KLayout_GUI":
    # Create GUI's
    from .core import WaveguideGUI, MonteCarloGUI
    WG_GUI = WaveguideGUI()
    MC_GUI = MonteCarloGUI()
    
# ******** lukasc
# don't use a global one.. based on cells
# Define global Net object that implements netlists and pin searching/connecting
# NET = Net()

from .core import Net, Component
NET_DISCONNECTED = Net()

# don't use a global one.. based on cells
# Define global Component object
#COMPONENT = Component()


# Define an Enumeration type for Python
# TODO: maybe move to standard enum for python3
# https://docs.python.org/3/library/enum.html
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define enumeration for pins
PIN_TYPES = enum('OPTICALIO', 'OPTICAL', 'ELECTRICAL')
PIN_LENGTH = 20  # 10 nm on each side. Previous was 2x50 nm, but shorter works well for Waveguide DRC checking


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

KLAYOUT_VERSION = int(pya.__version__.split('.')[1])
KLAYOUT_VERSION_3 = int(pya.__version__.split('.')[2])

# Waveguide DevRec: space between the waveguide and the DevRec polygon
WG_DEVREC_SPACE = 1

# Path to Waveguide, path snapping to nearest pin. Search for pin with this distance:
PATH_SNAP_PIN_MAXDIST = 20

    
INTC_ELEMENTS = ''
if Python_Env == "KLayout_GUI":
    # Load INTC element library details KLayout application data path
    import os
    path = os.path.join(pya.Application.instance().application_data_path(), 'Lumerical_CMLs')
    path = os.path.join(path, "Lumerical_INTC_CMLs.txt")
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
        MODE
    except:
        MODE = None
        print('resetting Lumerical MODE Python integration')
    
    try:
        LUMAPI
    except:
        LUMAPI = None
        print('resetting Lumerical Python integration')
    
if 'TEMP_FOLDER' not in locals(): 
    import tempfile
    TEMP_FOLDER = tempfile.mkdtemp()

'''
SiEPIC-Tools package for KLayout
'''

__version__ = "0.5.20"

print("KLayout SiEPIC-Tools version %s" %__version__)

import pya
if '__version__' in dir(pya):
    # pya.__version__ was introduced in KLayout version 0.28.6
    KLAYOUT_VERSION = int(pya.__version__.split('.')[1])
else:
    KLAYOUT_VERSION = int(pya.Application.instance().version().split('.')[1])
    KLAYOUT_VERSION_3 = int(pya.Application.instance().version().split('.')[2])
    
if KLAYOUT_VERSION < 28:  
    # pya.Technologies was introduced in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
    # SiEPIC-Tools is being updated to use this functionality, hence will no longer be supported for KLayout 0.26
    raise Exception('\nSiEPIC-Tools is no longer compatible with older versions (0.26) of KLayout.\nPlease download an install the latest version from www.klayout.de')
else:
    from . import _globals
    
    if _globals.Python_Env == "KLayout_GUI":
        from . import extend, _globals, core, examples, github, scripts, utils, setup, install, verification
    else:
        from . import _globals, core, utils, extend, verification, scripts
        


try:
    # Start timer
    import time
    start_time = time.time()
    from .scripts import version_check

    version_check()

    execution_time = time.time() - start_time
    print(f"Version check, time: {execution_time} seconds")
except:
    pass

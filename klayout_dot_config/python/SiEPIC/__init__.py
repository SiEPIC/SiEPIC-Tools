'''
SiEPIC-Tools package for KLayout
'''

__version__ = '0.5.2'

print("KLayout SiEPIC-Tools version %s" %__version__)

import pya
KLAYOUT_VERSION = int(pya.__version__.split('.')[1])

if KLAYOUT_VERSION < 28:  
    # pya.Technologies was introduced in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
    # SiEPIC-Tools is being updated to use this functionality, hence will no longer be supported for KLayout 0.26
    raise Exception('\nSiEPIC-Tools is no longer compatible with older versions (0.26) of KLayout.\nPlease download an install the latest version from www.klayout.de')
else:
    from . import _globals
    
    if _globals.Python_Env == "KLayout_GUI":
        from . import extend, _globals, core, examples, github, scripts, utils, setup, install, verification
    else:
        from . import extend, _globals, verification
    


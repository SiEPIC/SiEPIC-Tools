'''
SiEPIC-Tools package for KLayout
'''

__version__ = '0.4.7'

print("KLayout SiEPIC-Tools version %s" %__version__)

from . import _globals

if _globals.Python_Env == "KLayout_GUI":
    from . import extend, _globals, core, examples, github, scripts, utils, setup, install, verification
else:
    from . import extend, _globals
    


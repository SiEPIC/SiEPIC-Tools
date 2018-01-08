'''
################################################################################
#
#  SiEPIC-Tools
#  
################################################################################

Component simulations using Lumerical FDTD, to generate Compact Models

- component_simulation: single component simulation

usage:

import SiEPIC.lumerical.fdtd


################################################################################
'''

import pya


def run_FDTD(verbose=False):
  import lumapi_fdtd as lumapi
  from .. import _globals
  _globals.FDTD  # Python Lumerical FDTD integration handle
  
  if not _globals.FDTD: # Not running, start a new session
    _globals.FDTD = lumapi.open('fdtd')
    print(_globals.FDTD)
  else: # found open FDTD session
    try:
      lumapi.evalScript(_globals.FDTD, "?'KLayout integration test.';")
    except: # but can't communicate with INTC; perhaps it was closed by the user
      INTC = lumapi.open('fdtd') # run again.
  try: # check again
    lumapi.evalScript(_globals.FDTD, "?'KLayout integration test.';")
  except:
    raise Exception ("Can't run Lumerical FDTD. Unknown error.")










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
  if verbose:
    print(_globals.FDTD)  # Python Lumerical FDTD integration handle
  
  if not _globals.FDTD: # Not running, start a new session
    _globals.FDTD = lumapi.open('fdtd')
    if verbose:
      print(_globals.FDTD)  # Python Lumerical FDTD integration handle
  else: # found open FDTD session
    try:
      lumapi.evalScript(_globals.FDTD, "?'KLayout integration test.';")
    except: # but can't communicate with FDTD; perhaps it was closed by the user
      _globals.FDTD = lumapi.open('fdtd') # run again.
      if verbose:
        print(_globals.FDTD)  # Python Lumerical FDTD integration handle
  try: # check again
    lumapi.evalScript(_globals.FDTD, "?'KLayout integration test.';")
  except:
    raise Exception ("Can't run Lumerical FDTD. Unknown error.")


def generate_component_sparam(verbose = False):
  if verbose:
    print('SiEPIC.lumerical.fdtd: generate_component_sparam()')
    
  import lumapi_fdtd as lumapi
  from .. import _globals
  if verbose:
    print(_globals.FDTD)  # Python Lumerical INTERCONNECT integration handle

  run_FDTD()
  lumapi.evalScript(_globals.FDTD, "b=0:0.01:10; plot(b,sin(b),'Congratulations, Lumerical is now available from KLayout','','Congratulations, Lumerical is now available from KLayout');")

  lumapi.evalScript(_globals.FDTD, "out=c;")
  c=lumapi.getVar(_globals.FDTD, "out")









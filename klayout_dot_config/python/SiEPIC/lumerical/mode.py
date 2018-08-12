# $autorun
'''
################################################################################
#
#  SiEPIC-Tools
#   Mustafa Hammood    Mustafa@ece.ubc.ca
#   August 2018
################################################################################

Waveguide simulations using Lumerical MODE, to generate effective index models

- find_neff: single waveguide system neff model
- find_neff_supermode: supermode calculation for a 2 waveguide system

usage:

import SiEPIC.lumerical.mode


################################################################################
'''

import sys
if 'pya' in sys.modules: # check if in KLayout
  import pya

try:
    import pyparsing
except:
    try:
        import pip
        import pya
        install = pya.MessageBox.warning(
            "Install package?", "Install package 'pyparsing' using pip? [required for Lumerical MODE]",  pya.MessageBox.Yes + pya.MessageBox.No)
        if install == pya.MessageBox.Yes:
            # try installing using pip
            from SiEPIC.install import get_pip_main
            main = get_pip_main()
            main(['install', 'pyparsing'])
    except ImportError:
        pass
        
        
def run_MODE(verbose=False):
  from . import load_lumapi
  from .. import _globals
  lumapi = _globals.LUMAPI
  if not lumapi:
    print("SiEPIC.lumerical.mode.run_MODE: lumapi not loaded; reloading load_lumapi.")
    import sys
    if sys.version_info[0] == 3:
        if sys.version_info[1] < 4:
            from imp import reload
        else:
            from importlib import reload
    elif sys.version_info[0] == 2:
        from imp import reload
    reload(load_lumapi)

  if not lumapi:
    print("SiEPIC.lumerical.mode.run_MODE: lumapi not loaded")
    pya.MessageBox.warning("Cannot load Lumerical Python integration.", "Some SiEPIC-Tools Lumerical functionality will not be available.", pya.MessageBox.Cancel)
    return

  if verbose:
    print(_globals.MODE)  # Python Lumerical MODE integration handle

  if not _globals.MODE: # Not running, start a new session
    _globals.MODE = lumapi.open('mode')
    if verbose:
      print(_globals.MODE)  # Python Lumerical MODE integration handle
  else: # found open MODE session
    try:
      lumapi.evalScript(_globals.MODE, "?'KLayout integration test.\n';\n")
    except: # but can't communicate with MODE; perhaps it was closed by the user
      _globals.MODE = lumapi.open('mode') # run again.
      if verbose:
        print(_globals.MODE)  # Python Lumerical MODE integration handle
  try: # check again
    lumapi.evalScript(_globals.MODE, "?'KLayout integration test.\n';\n")
  except:
    raise Exception ("Can't run Lumerical MODE via Python integration.")
    
    
def find_neff_supermode(w_1 = 500e-9, w_2 = 500e-9, gap = 200e-9, pol = 'TE', verbose = False):

 # search for 2 waveguide system script file in technology
  from ..utils import get_technology
  TECHNOLOGY = get_technology()
  tech_name = TECHNOLOGY['technology_name']

  import os, fnmatch
  dir_path = pya.Application.instance().application_data_path()
  search_str = 'blank.lms'
  matches = []
  for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
      for filename in fnmatch.filter(filenames, search_str):
        if tech_name in root:
          matches.append(os.path.join(root, filename))
  
  from .. import _globals
  lumapi = _globals.LUMAPI
  if not lumapi:
    print('SiEPIC.lumerical.mode.find_neff_supermode: lumapi not loaded')
    return
    
  filename = matches[0]
  run_MODE()
  lumapi.evalScript(_globals.MODE,"load('%s'); gap = %s; width_1 = %s; width_2 = %s;" % (filename, gap, w_1, w_2))  
  lumapi.evalScript(_globals.MODE,"WG_supermode;") 
  lumapi.evalScript(_globals.MODE,"switchtolayout;")  
  n_eff1_fit = lumapi.getVar(_globals.MODE, "n_eff1_fit")
  n_eff2_fit = lumapi.getVar(_globals.MODE, "n_eff2_fit")
  print(n_eff1_fit+' and '+n_eff2_fit)
  return [n_eff1_fit, n_eff2_fit]
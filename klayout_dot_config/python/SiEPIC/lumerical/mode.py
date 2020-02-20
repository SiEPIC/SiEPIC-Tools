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

  #lumapi.evalScript(_globals.MODE,"switchtolayout;")  
  
  # Below commands DO NOT WORK, see issue at kx
  
  #n_eff1_fit = lumapi.getVar(_globals.MODE, "n_eff1_fit")
  #n_eff2_fit = lumapi.getVar(_globals.MODE, "n_eff2_fit")
  
  # Temporary cave-man parsing solution
  import os, fnmatch
  dir_path = pya.Application.instance().application_data_path()
  search_str = 'n_eff1_fit'
  matches = []
  for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
      for filename in fnmatch.filter(filenames, search_str):
        if tech_name in root:
          matches.append(os.path.join(root, filename))
          
  os.chdir(dir_path+'/tech/EBeam/mode')

  n_eff1_fit = open("n_eff1_fit","r")
  theStr = n_eff1_fit.read()
  theStr = (theStr.replace("\n",",")).split(",")
  n_eff1_fit_0 = float(theStr[0])
  n_eff1_fit_1 = float(theStr[1])
  #n_eff1_fit_2 = float(theStr[2])
  
  n_eff2_fit = open("n_eff2_fit","r")
  theStr = n_eff2_fit.read()
  theStr = (theStr.replace("\n",",")).split(",")
  n_eff2_fit_0 = float(theStr[0])
  n_eff2_fit_1 = float(theStr[1])
  #n_eff2_fit_2 = float(theStr[2])
  
  
  n_g1_fit = open("n_g1_fit","r")
  theStr = n_g1_fit.read()
  theStr = (theStr.replace("\n",",")).split(",")
  n_g1_fit_0 = float(theStr[0])
  n_g1_fit_1 = float(theStr[1])
  #n_g1_fit_2 = float(theStr[2])
  
  n_g2_fit = open("n_g2_fit","r")
  theStr = n_g2_fit.read()
  theStr = (theStr.replace("\n",",")).split(",")
  n_g2_fit_0 = float(theStr[0])
  n_g2_fit_1 = float(theStr[1])
  #n_g2_fit_2 = float(theStr[2])
  
  n_eff1_fit = [n_eff1_fit_0, n_eff1_fit_1]
  n_eff2_fit = [n_eff2_fit_0, n_eff2_fit_1]
  n_g1_fit = [n_g1_fit_0, n_g1_fit_1]
  n_g2_fit = [n_g2_fit_0, n_g2_fit_1]
  
  
  print('n_eff1='+str(n_eff1_fit)+' and n_eff2='+str(n_eff2_fit)+' and n_g1='+str(n_g1_fit)+' and n_g2='+str(n_g2_fit))
  
  return [n_eff1_fit, n_eff2_fit, n_g1_fit, n_g2_fit]


# run Lumerical MODE for dispersion analysis 
# of two-waveguide contra-directional coupler (unperturbed waveguides)
# By Mustafa Hammood, https://github.com/mustafacc/SiEPIC_Photonics_Package
def dispersion_2WG(params, verbose = False):

 # search for 2 waveguide system script file in technology
  from ..utils import get_technology
  TECHNOLOGY = get_technology()
  tech_name = TECHNOLOGY['technology_name']

  import os, fnmatch
  dir_path = pya.Application.instance().application_data_path()
  search_str = 'dispersion_2WG.lsf'
  matches = []
  for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
      for filename in fnmatch.filter(filenames, search_str):
        if tech_name in root:
          matches.append(os.path.join(root, filename))
  
  run_MODE()
  from .. import _globals
  lumapi = _globals.LUMAPI
  if not lumapi:
    print('SiEPIC.lumerical.mode.dispersion_2WG: lumapi not loaded')
    return params
    
  filename = matches[0]
  lumapi.evalScript(_globals.MODE,"cd('%s');" % os.path.dirname(filename))

  # feed parameters into model
  a = params["index"]

  lumapi.evalScript(_globals.MODE, "gap = %s; width_1 = %s; width_2 = %s; thick_Si = %s; period = %s;" 
                      % (params["gap"]*1e-6, params["wg1_width"]*1e-6, params["wg2_width"]*1e-6, params['thickness_Si'], params["grating_period"]*1e-6))

  lumapi.evalScript(_globals.MODE, "wavelength = %s; wavelength_stop = %s; number_of_plot_points = %s; pol = %s;" 
                      % (params['wavelength_start'], params['wavelength_stop'], params['wavelength_points'], params['pol']) )

    
  lumapi.evalScript(_globals.MODE,"slab = 0;")
  lumapi.evalScript(_globals.MODE,"thick_Slab = 0;")

  # run dispersion analysis script
  lumapi.evalScript(_globals.MODE,"dispersion_2WG;") 
    
  # contra-directional coupling
  params['neff_data'] = lumapi.getVar(_globals.MODE, 'n_eff_data')
  params['lambda_fit'] = lumapi.getVar(_globals.MODE, 'lambda_fit')
  params['ng_contra'] = lumapi.getVar(_globals.MODE, 'ng0')
  params['contra_lambda'] = lumapi.getVar(_globals.MODE, 'contra_lambda')
    
  # self-coupling
  params['ng1'] = lumapi.getVar(_globals.MODE, 'ng0_self1')
  params['ng2'] = lumapi.getVar(_globals.MODE, 'ng0_self2')
  params['lambda_self1'] = lumapi.getVar(_globals.MODE, 'self1_lambda')
  params['lambda_self2'] = lumapi.getVar(_globals.MODE, 'self2_lambda')

    
  return params


# run Lumerical MODE EME for CDC
# By Mustafa Hammood, https://github.com/mustafacc/SiEPIC_Photonics_Package
def cdc_EME(params, verbose = False):

  from ..utils import get_technology
  TECHNOLOGY = get_technology()
  tech_name = TECHNOLOGY['technology_name']

  import os, fnmatch
  dir_path = pya.Application.instance().application_data_path()
  search_str = 'ContraDC_EME.lms'
  matches = []
  for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
      for filename in fnmatch.filter(filenames, search_str):
        if tech_name in root:
          matches.append(os.path.join(root, filename))
  
  run_MODE()
  from .. import _globals
  lumapi = _globals.LUMAPI
  if not lumapi:
    print('SiEPIC.lumerical.mode.cdc_EME: lumapi not loaded')
    return params
    
  filename = matches[0]
  lumapi.evalScript(_globals.MODE,"cd('%s');" % os.path.dirname(filename))

  projectname = 'ContraDC_EME'
  scriptname = 'ContraDC_EMEscript'

  lumapi.evalScript(_globals.MODE, "gap = %s; Wa = %s; Wb = %s; thick_Si = %s; grating_period = %s;" 
                      % (params["gap"]*1e-6, params["wg1_width"]*1e-6, params["wg2_width"]*1e-6, params['thickness_Si'], params["grating_period"]*1e-6))

  lumapi.evalScript(_globals.MODE, "dWa = %s; dWb = %s; number_of_periods=%s" 
                      % (params["corrugation_width1"]*1e-6, params["corrugation_width2"]*1e-6, params["number_of_periods"]) )

  lumapi.evalScript(_globals.MODE, "min_wavelength = %s; max_wavelength = %s; pol = %s;" 
                      % (params['wavelength_start'], params['wavelength_stop'], params['pol']) )

  lumapi.evalScript(_globals.MODE, "wl_ref = %s;" 
                      % (params['contra_lambda']) )

   
  lumapi.evalScript(_globals.MODE,"load('%s'); %s; ?'delta_lambda = '+num2str(delta_lambda)" % (projectname, scriptname) )
        
    
  params['delta_lambda_contra'] = lumapi.getVar(_globals.MODE,"delta_lambda")
  params['lambda_contra'] = lumapi.getVar(_globals.MODE,"lambda0")
  params['delta_lambda_self1'] = lumapi.getVar(_globals.MODE,"delta_lambda_self1")
  params['delta_lambda_self2'] = lumapi.getVar(_globals.MODE,"delta_lambda_self2")

  return params

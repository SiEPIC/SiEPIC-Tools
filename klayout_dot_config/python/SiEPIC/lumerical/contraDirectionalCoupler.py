# SiEPIC Tools Contra-directional coupler simulation module
# Mustafa Hammood     Mustafa@ece.ubc.ca 
# August 2018
# KLayout-INTERCONNECT CML integration added by Lukas Chrostowski
#
# Using CMT and TMM to determine the transfer function of a contra-DC device
# Based on MATLAB script from J.St.Yves and W.Shi of uLaval, 2016




#%% linear algebra numpy manipulation functions
# Takes a 4*4 matrix and switch the first 2 inputs with first 2 outputs
def switchTop( P ):
  import numpy as np
  P_FF = np.asarray([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
  P_FG = np.asarray([[P[0][2],P[0][3]],[P[1][2],P[1][3]]])
  P_GF = np.asarray([[P[2][0],P[2][1]],[P[3][0],P[3][1]]])
  P_GG = np.asarray([[P[2][2],P[2][3]],[P[3][2],P[3][3]]])
    
  H1 = P_FF-np.matmul(np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1)),P_GF)
  H2 = np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1))
  H3 = np.matmul(-np.linalg.matrix_power(P_GG,-1),P_GF)
  H4 = np.linalg.matrix_power(P_GG,-1)
  H = np.vstack((np.hstack((H1,H2)),np.hstack((H3,H4))))
    
  return H

# Swap columns of a given array
def swap_cols(arr, frm, to):
  arr[:,[frm, to]] = arr[:,[to, frm]]
  return arr

# Swap rows of a given array
def swap_rows(arr, frm, to):
  arr[[frm, to],:] = arr[[to, frm],:]
  return arr
    
#%% the bread and butter
def contraDC_model(params, verbose = True):

  import cmath, math
  import sys, os, time
  import numpy as np
  import scipy.linalg    
  
  #%% System constants Constants
  c = 299792458           #[m/s]
  dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
  j = cmath.sqrt(-1)      # imaginary
    
  # ...

#%% phase matching analysis        
def phaseMatch_analysis(params, verbose = False):

  import numpy as np

  neff_data, lambda_fit, ng_contra, ng1, ng2, lambda_self1, lambda_self2, grating_period = params['neff_data'], params['lambda_fit'], params['ng_contra'], params['ng1'], params['ng2'], params['lambda_self1'], params['lambda_self2'], params['grating_period']

  neff1_fit = np.polyfit(lambda_fit[:,0], neff_data[:,0],1)
  neff2_fit = np.polyfit(lambda_fit[:,0], neff_data[:,1],1)

  neff1 = np.polyval(neff1_fit, lambda_fit)
  neff2 = np.polyval(neff2_fit, lambda_fit)
  neff_avg = (neff1+neff2)/2
  phaseMatch = lambda_fit/(2*grating_period)*1e6

  # find contra-directional coupling wavelength
  tol = 1e-3
  phaseMatchindex = np.where(abs(phaseMatch-neff_avg)<tol); phaseMatchindex=phaseMatchindex[0]
  phaseMatchindex = phaseMatchindex[int(phaseMatchindex.size/2)]
  lambda_phaseMatch = lambda_fit[phaseMatchindex]
  print('lambda_phaseMatch: %s' % lambda_phaseMatch)

  # average effective indices at phase match
  params['neff1_phaseMatch'] = neff1[phaseMatchindex][0]
  params['neff1_dispersion'] = neff1_fit[0]
  params['neff2_phaseMatch'] = neff2[phaseMatchindex][0]
  params['neff2_dispersion'] = neff2_fit[0]

  return params
  

def cdc_simulation(verbose=True, FDTD_settings = None):

  import pya
  # Get technology and layout details
  from ..utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
  dbum = TECHNOLOGY['dbu']*1e-6 # dbu to m conversion
  
  # get selected instances; only one
  from ..utils import select_instances
  from .. import _globals
  
  # print error message if no or more than one component selected
  selected_instances = select_instances()
  error = pya.QMessageBox()
  error.setStandardButtons(pya.QMessageBox.Ok )
  if len(selected_instances) != 1:
    error.setText("Error: Need to have one component selected.")
    response = error.exec_()
    return
    
  for obj in selected_instances:
    c = cell.find_components(cell_selected=[obj.inst().cell],verbose=True)

  # check if selected PCell is a contra DC
  if c[0].cell.basic_name() != "Contra-Directional Coupler":
    error.setText("Error: selected component must be a contra-DC PCell.")
    response = error.exec_()
    return

  # parse PCell parameters into params array
  if c[0].cell.is_pcell_variant():
    params = c[0].cell.pcell_parameters_by_name()
  else:
    error.setText("Error: selected component must be a contra-DC PCell.")
    response = error.exec_()
    return
  print(params)
    
  # parse into individual variables in meters
  a = params["index"]
  
#  simulation_lambda_start
#  simulation_lambda_end
#  simulation_number_points
      
  if params["sinusoidal"] == False:
    sinusoidal = 0;
  else:
    sinusoidal = 1;
    error.setText("Error: The simulation model currently does not support sinusoidal gratings.")
    response = error.exec_()
    return

  if params["AR"] == False:
    error.setText("Error: The simulation model currently only supports anti-reflection gratings.")
    response = error.exec_()
    return

  # Simulation parameters
  # get FDTD settings from XML file
  if not FDTD_settings:
    from SiEPIC.utils import load_FDTD_settings
    FDTD_settings=load_FDTD_settings()
    if FDTD_settings:
      if verbose:
        print(FDTD_settings)

  mode_selection = FDTD_settings['mode_selection']
  if 'fundamental TM mode' in mode_selection or '2' in mode_selection:
    params['pol']=2  # TM
  else:
    params['pol']=1  # TE

  # wavelength
  params['wavelength_start'] = FDTD_settings['wavelength_start']
  params['wavelength_stop'] =  FDTD_settings['wavelength_stop']
  params['wavelength_points'] = FDTD_settings['frequency_points_monitor']

  params['thickness_Si']=FDTD_settings['thickness_Si']
    
  from SiEPIC import lumerical

  # Calculate the 2 waveguide modes
  params = lumerical.mode.dispersion_2WG(params, verbose)
  
  # Find phase matching condition
  params = phaseMatch_analysis(params, verbose)

  # Simulate an infinite CDC using Lumerical's EME
  params = lumerical.mode.cdc_EME(params, verbose)

  def get_kappa(delta_lambda, lambda0, ng):
    kappa = np.pi*ng*delta_lambda/(lambda0**2)
    return kappa
        
  kappa_contra = get_kappa(params['delta_lambda_contra'], params['lambda_contra'], params['ng_contra'])
  kappa_self1 = get_kappa(params['delta_lambda_self1'], params['lambda_self1'], params['ng1'])/10
  kappa_self2 = get_kappa(params['delta_lambda_self2'], params['lambda_self2'], params['ng2'])/10     

  print('Kappa contra = %s, Kappa self 1 = %s, Kappa self 2 = %2' % (kappa_contra, kappa_self1,kappa_self2) )
  
  
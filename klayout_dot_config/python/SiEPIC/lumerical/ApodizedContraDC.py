# SiEPIC Tools Contra-directional coupler simulation module
# Mustafa Hammood     Mustafa@ece.ubc.ca 
# August 2018
#
# Using CMT and TMM to determine the transfer function of a contra-DC device
# Based on MATLAB script from J.St.Yves and W.Shi of uLaval, 2016

import cmath
import sys, os
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
        

def TOP():
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
    
  # parse PCell parameters into params array
  for obj in selected_instances:
    c = cell.find_components(cell_selected=[obj.inst().cell],verbose=True)
    if c:
      if c[0].cell.is_pcell_variant():
        params = c[0].cell.pcell_parameters_by_name()
        for key in params.keys():
          print ("Parameter: %s, Value: %s" % (key, params[key]) )

  print(params)

            if instance.cell.basic_name() == "Waveguide":
  
  # check if selected PCell is a contra DC
  if "component: ebeam_contra_dc" not in text:
    error.setText("Error: selected component must be a contra-DC PCell.")
    response = error.exec_()
    return
    
  # parse into individual variables in meters
  N = params["number_of_periods"]
  period = params["grating_period"]*1e-6
  dW_1 = params["corrugation_width1"]*1e-6
  dW_2 = params["corrugation_width2"]*1e-6
  W_1 = params["wg1_width"]*1e-6
  W_2 = params["wg2_width"]*1e-6
  gap = params["gap"]*1e-6
  a = params["index"]
  
  if params["sinusoidal"] == False:
    sinusoidal = 0;
  else:
    sinusoidal = 1;
    
  from SiEPIC.lumerical.fdtd import generate_CDC_bandstructure
  from SiEPIC.lumerical.mode import find_neff_supermode
  
  [n_eff1_fit, n_eff2_fit, n_g1_fit, n_g2_fit] = find_neff_supermode(W_1, W_2, gap)
  [bandwidth, lambda_0] = generate_CDC_bandstructure(W_1, W_2, dW_1, dW_2, period, gap, sinusoidal)
  #[bandwidth, lambda_0] = [7e-9, 1550e-9]
  
  os.chdir('C:\\Users\\Mustafa\\Desktop')

  # create data files
  dataFile = open("contraDC_params","w")
  dataFile.write(str(gap)+','+str(W_1)+','+str(W_2)+','+str(dW_1)+','+str(dW_2)+','+str(period)+','+str(a)+','+str(N))
  
  dataFile = open("contraDC_mode","w")
  dataFile.write(str(n_eff1_fit)+','+str(n_eff2_fit)+','+str(n_g1_fit)+','+str(n_g2_fit))
  
  dataFile = open("contraDC_fdtd","w")
  dataFile.write(str(bandwidth)+','+str(lambda_0))
  
  #os.
  #os.system('python matlabenginestart.py')
  import subprocess
  p = subprocess.Popen(["start", "cmd", "/k", "python matlabenginestart.py"], shell = True)
  
  
def ApodizedContraDC(starting_wavelength = 1520, ending_wavelength = 1580, resolution = 500, neffwg1 = 2.5220, neffwg2 = 2.3282, Dneffwg1 = -9.6552e5, Dneffwg2 = -1.2195e6, period = 320):
  if MODULE_NUMPY:
    import numpy as np

  # Constants
  c = 299792458           #[m/s]
  dneffdT = 1.87E-04    #[/K] assuming dneff/dn=1 (very confined)
  j = cmath.sqrt(-1)      # imaginary
  
  # Properties
  name='undefName'
  
  NormalTemp=300                 #[K] Temperature of the mesured chip
  DeviceTemp=300                 #[K] Temperature of the device (for dNeff)
  
  #starting_wavelength=1520   #nanometers
  #ending_wavelength=1580
  #resolution=500                     #number of wavelengths to compute
  
  N_Corrugations=1000             #Number of corrugations along the grating
  period=period*1e-9                     #[m] corrugation period
  
  centralWL_neff=1550e-9
  alpha=10*100                     #Loss of the medium, in dB/m (10dB/cm*100cm/m)
  
  kappaMax=9000                  #contra-directional coupling
  kappaMin=0
  a=0                                    #for gaussian function, set to 0 for custom
  ApoFunc=np.exp(-np.linspace(0,1,num=1000)**2)     #Function used for apodization (window function)
  
  mirror= False                #makes the apodization function symetrical
  N_seg=20                   #Number of flat steps in the coupling profile
  antiRefCoeff=.3          #k11 and k22 relative to k12 (bragg, same-waveguide reflection)
  
  rch=0                        #random chirping, maximal fraction of index randomly changing each segment
  lch=0                         #linear chirp across the length of the device
  kch=0                        #coupling dependant chirp, normalized to the max coupling
  
  import math
  
  Lambda = np.linspace(starting_wavelength, ending_wavelength, resolution)*1e-9
  #print(Lambda)
  alpha_e = alpha/10*math.log(10)
  neff_detuning_factor = 0.9776
  
  # Waveguides models
  neff_a_data=neffwg1+Dneffwg1*(Lambda-centralWL_neff)
  #print(neff_a_data)
  neff_b_data=neffwg2+Dneffwg2*(Lambda-centralWL_neff)
  Lambda_data_left=Lambda
  Lambda_data_right=Lambda
  
  beta_data_left=2*math.pi/Lambda_data_left*neff_a_data
  #print(beta_data_left)
  beta_data_right=2*math.pi/Lambda_data_right*neff_b_data
            
  beta_left=np.interp(Lambda, Lambda_data_left, beta_data_left)
  #print(beta_left)
  beta_right=np.interp(Lambda, Lambda_data_right, beta_data_right)
  
  betaL=beta_left
  betaR=beta_right
  
  # Calculating reflection wavelenghts
  f= 2*math.pi/(beta_left+beta_right) #=grating period at phase match
  minimum = min(abs(f-period)) #index of closest value
  idx = np.where(abs(f-period) == minimum)
  beta12Wav = Lambda.item(idx[0][0])
  
  f= 2*math.pi/(2*beta_left)
  minimum = min(abs(f-period))
  idx = np.where(abs(f-period) == minimum)
  beta1Wav = Lambda.item(idx[0][0])
  
  f= 2*math.pi/(2*beta_right)
  minimum = min(abs(f-period))
  idx = np.where(abs(f-period) == minimum)
  beta2Wav = Lambda.item(idx[0][0])
  
  T=      np.zeros((1, Lambda.size))
  R=      np.zeros((1, Lambda.size))
  T_co=   np.zeros((1, Lambda.size))
  R_co=   np.zeros((1, Lambda.size))
  
  mode_kappa_a1=1
  mode_kappa_a2=0 #no initial cross coupling
  mode_kappa_b2=1
  mode_kappa_b1=0
  
  LeftRightTransferMatrix = np.zeros((4,4,Lambda.size))
  TopDownTransferMatrix = np.zeros((4,4,Lambda.size))
  InOutTransferMatrix = np.zeros((4,4,Lambda.size))
  
  # Apodization & segmenting
  l_seg = N_Corrugations*period/N_seg
  L_seg=l_seg
  n_apodization=np.arange(N_seg)-0.5
  zaxis= (np.arange(N_seg)-1)*l_seg
  
  if a !=0:
    kappa_apodG=np.exp(-a*((n_apodization)-0.5*N_seg)**2/N_seg**2)
    ApoFunc=kappa_apodG
      
  profile= (ApoFunc-min(ApoFunc))/(max(ApoFunc)-(min(ApoFunc))) # normalizes the profile
  
  n_profile = np.linspace(0,N_seg,profile.size)
  profile=np.interp(n_apodization, n_profile, profile)

  kappa_apod=kappaMin+(kappaMax-kappaMin)*profile
  
  lenghtLambda=Lambda.size
  

  kappa_12max= max(kappa_apod)
  n=np.arange(N_seg)
  steps = 20;
  step=0
  
  for ii in range(lenghtLambda):
    P=1
  
    for n in range(N_seg):
      L_0=(n-1)*l_seg

      kappa_12=kappa_apod.item(n)
      #kappa_21=conj(kappa_12); #unused: forward coupling!
      kappa_11=antiRefCoeff*kappa_apod.item(n)
      kappa_22=antiRefCoeff*kappa_apod.item(n)
      
      beta_del_1=beta_left-math.pi/period-j*alpha_e/2
      
      beta_del_2=beta_right-math.pi/period-j*alpha_e/2

      # S1 = Matrix of propagation in each guide & direction
      S_1=[ [j*beta_del_1.item(ii), 0, 0, 0], [0, j*beta_del_2.item(ii), 0, 0],
                [0, 0, -j*beta_del_1.item(ii), 0],[0, 0, 0, -j*beta_del_2.item(ii)]]

      # S2 = transfert matrix
      S_2= [[-j*beta_del_1.item(ii),  0,  -j*kappa_11*np.exp(j*2*beta_del_1.item(ii)*L_0),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0)],
                [0,  -j*beta_del_2.item(ii),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  -j*kappa_22*np.exp(j*2*beta_del_2.item(ii)*L_0)],
                [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1.item(ii)*L_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*beta_del_1.item(ii),  0],
                [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2.item(ii)*L_0),  0,  j*beta_del_2.item(ii)]]
      

      kk=np.expm1(np.asarray(S_1)*l_seg)*np.expm1(np.asarray(S_2)*l_seg)*P
      P = kk
      
  #LeftRightTransferMatrix(:,:,ii) = P
      
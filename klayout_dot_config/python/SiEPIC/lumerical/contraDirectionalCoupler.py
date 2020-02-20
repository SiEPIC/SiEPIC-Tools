# SiEPIC Tools Contra-directional coupler simulation module
# Mustafa Hammood     Mustafa@ece.ubc.ca 
# August 2018
# KLayout-INTERCONNECT CML integration added by Lukas Chrostowski
#
# Using CMT and TMM to determine the transfer function of a contra-DC device
# Based on MATLAB script from J.St.Yves and W.Shi of uLaval, 2016



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

    ApoFunc=np.exp(-np.linspace(0,1,num=1000)**2)     #Function used for apodization (window function)

    mirror = False                #makes the apodization function symetrical
    N_seg = 501                   #Number of flat steps in the coupling profile

    chirp = False
    if chirp == True:
        rch= 0.04                        #random chirping, maximal fraction of index randomly changing each segment
        lch= 0.2                         #linear chirp across the length of the device
        kch= -.1                        #coupling dependant chirp, normalized to the max coupling
    else:
        rch= 0                        #random chirping, maximal fraction of index randomly changing each segment
        lch= 0                         #linear chirp across the length of the device
        kch= 0                        #coupling dependant chirp, normalized to the max coupling


    neff_detuning_factor = 1  ## Mustafa: What is this?
    
    #%% calculate waveguides propagation constants

    alpha_e = 100*params['alpha']/10*math.log(10)
    neffThermal = 0 # dneffdT*(simulation_setup.deviceTemp-simulation_setup.chipTemp)

    # Waveguides models
    Lambda = np.linspace(params['wavelength_start'], params['wavelength_stop'], num=int(params['wavelength_points']))

    neffwg1 = params['neff1_phaseMatch']
    Dneffwg1 = params['neff1_dispersion']
    neffwg2 = params['neff2_phaseMatch']
    Dneffwg2 = params['neff2_dispersion'] 


    neff_a_data = neffwg1+Dneffwg1*(Lambda-params['lambda_contra'])
    neff_a_data = neff_a_data*neff_detuning_factor+neffThermal
    neff_b_data=neffwg2+Dneffwg2*(Lambda-params['lambda_contra'])
    neff_b_data = neff_b_data*neff_detuning_factor+neffThermal
    Lambda_data_left=Lambda
    Lambda_data_right=Lambda

    beta_data_left=2*math.pi/Lambda_data_left*neff_a_data
    beta_data_right=2*math.pi/Lambda_data_right*neff_b_data

    #%% makes sense until HERE


    beta_left=np.interp(Lambda, Lambda_data_left, beta_data_left); betaL=beta_left
    beta_right=np.interp(Lambda, Lambda_data_right, beta_data_right); betaR=beta_right    
  
    # Calculating reflection wavelenghts
    period = params['grating_period']
    
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
    
    T =      np.zeros((1, Lambda.size),dtype=complex)
    R =      np.zeros((1, Lambda.size),dtype=complex)
    T_co =   np.zeros((1, Lambda.size),dtype=complex)
    R_co =   np.zeros((1, Lambda.size),dtype=complex)
    
    E_Thru = np.zeros((1, Lambda.size),dtype=complex)
    E_Drop = np.zeros((1, Lambda.size),dtype=complex)
    
    mode_kappa_a1=1
    mode_kappa_a2=0 #no initial cross coupling
    mode_kappa_b2=1
    mode_kappa_b1=0
  
    LeftRightTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
    TopDownTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
    InOutTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
  
    # Apodization & segmenting
    a = params["index"]
    l_seg = params['number_of_periods']*period/N_seg
    L_seg=l_seg
    n_apodization=np.arange(N_seg)+0.5
    zaxis= (np.arange(N_seg))*l_seg

    if  a!=0:
        kappa_apodG=np.exp(-a*((n_apodization)-0.5*N_seg)**2/N_seg**2)
        ApoFunc=kappa_apodG

        profile= (ApoFunc-min(ApoFunc))/(max(ApoFunc)-(min(ApoFunc))) # normalizes the profile

        n_profile = np.linspace(0,N_seg,profile.size)
        profile=np.interp(n_apodization, n_profile, profile)

        kappaMin = params['kappa_contra']*profile[0]
        kappaMax = params['kappa_contra']
    
        kappa_apod=kappaMin+(kappaMax-kappaMin)*profile
        
    else:
        kappa_apod = params['kappa_contra']*np.ones(N_seg)
        profile = 0

    lenghtLambda=Lambda.size
    
    kappa_12max= max(kappa_apod)
    
    n=np.arange(N_seg)
    chirpWL = np.ones(N_seg)
    for ii in range(lenghtLambda):
        
        P=1
  
        for n in range(N_seg):
            L_0=(n)*l_seg

            kappa_12=kappa_apod.item(n)
            #kappa_21=conj(kappa_12); #unused: forward coupling!
            kappa_11=params['kappa_self1']
            kappa_22=params['kappa_self2']
      
            beta_del_1=beta_left*chirpWL.item(n)-math.pi/period-j*alpha_e/2
            beta_del_2=beta_right*chirpWL.item(n)-math.pi/period-j*alpha_e/2

            # S1 = Matrix of propagation in each guide & direction
            S_1=[  [j*beta_del_1.item(ii), 0, 0, 0], [0, j*beta_del_2.item(ii), 0, 0],
                   [0, 0, -j*beta_del_1.item(ii), 0],[0, 0, 0, -j*beta_del_2.item(ii)]]

            # S2 = transfert matrix
            S_2=  [[-j*beta_del_1.item(ii),  0,  -j*kappa_11*np.exp(j*2*beta_del_1.item(ii)*L_0),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0)],
                   [0,  -j*beta_del_2.item(ii),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  -j*kappa_22*np.exp(j*2*beta_del_2.item(ii)*L_0)],
                   [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1.item(ii)*L_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*beta_del_1.item(ii),  0],
                   [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2.item(ii)*L_0),  0,  j*beta_del_2.item(ii)]]

            P0=np.matmul(scipy.linalg.expm(np.asarray(S_1)*l_seg),scipy.linalg.expm(np.asarray(S_2)*l_seg))
            if n == 0:
                P1 = P0*P
            else:
                P1 = np.matmul(P0,P)
                
            P = P1
            
        LeftRightTransferMatrix[:,:,ii] = P
        
        # Calculating In Out Matrix
        # Matrix Switch, flip inputs 1&2 with outputs 1&2
        H = switchTop( P )
        InOutTransferMatrix[:,:,ii] = H
        
        # Calculate Top Down Matrix
        P2 = P
        # switch the order of the inputs/outputs
        P2=np.vstack((P2[3][:], P2[1][:], P2[2][:], P2[0][:])) # switch rows
        P2=swap_cols(P2,1,2) # switch columns
        # Matrix Switch, flip inputs 1&2 with outputs 1&2
        P3 = switchTop( P2 )
        # switch the order of the inputs/outputs
        P3=np.vstack((P3[3][:], P3[0][:], P3[2][:], P3[1][:])) # switch rows
        P3=swap_cols(P3,2,3) # switch columns
        P3=swap_cols(P3,1,2) # switch columns

        TopDownTransferMatrix[:,:,ii] = P3
        T[0,ii] = H[0,0]*mode_kappa_a1+H[0,1]*mode_kappa_a2
        R[0,ii] = H[3,0]*mode_kappa_a1+H[3,1]*mode_kappa_a2

        T_co[0,ii] = H[1,0]*mode_kappa_a1+H[1,0]*mode_kappa_a2
        R_co[0,ii] = H[2,0]*mode_kappa_a1+H[2,1]*mode_kappa_a2

        E_Thru[0,ii] = mode_kappa_a1*T[0,ii]+mode_kappa_a2*T_co[0,ii]
        E_Drop[0,ii] = mode_kappa_b1*R_co[0,ii] + mode_kappa_b2*R[0,ii]

        #%% return results
        params['E_Thru'] = E_Thru
        params['E_Drop'] = E_Drop
        params['wavelength'] = Lambda
        params['TransferMatrix'] = LeftRightTransferMatrix
        
    return params    
  # ...


#%% generate S-parameters
# Source: J. Frei, X.-D. Cai, and S. Muller. Multiport s-parameter and 
# T-parameter conversion with symmetry extension. IEEE Transactions on 
# Microwave Theory and Techniques, 56(11):2493?2504, 2008.

def gen_sparams( params):
    import numpy as np
    import scipy.io as sio
    import os
    from .. import _globals

    T = params['TransferMatrix']
    lambda0 = params['wavelength']*1e9
    f =  299792458/lambda0
    
    span = lambda0.__len__()
    T11 = T[0][0][:span]; T11 = np.matrix.transpose(T11)
    T12 = T[0][1][:span]; T12 = np.matrix.transpose(T12)
    T13 = T[0][2][:span]; T13 = np.matrix.transpose(T13)
    T14 = T[0][3][:span]; T14 = np.matrix.transpose(T14)
    
    T21 = T[1][0][:span]; T21 = np.matrix.transpose(T21)
    T22 = T[1][1][:span]; T22 = np.matrix.transpose(T22)
    T23 = T[1][2][:span]; T23 = np.matrix.transpose(T23)
    T24 = T[1][3][:span]; T24 = np.matrix.transpose(T24)

    T31 = T[2][0][:span]; T31 = np.matrix.transpose(T31)
    T32 = T[2][1][:span]; T32 = np.matrix.transpose(T32)
    T33 = T[2][2][:span]; T33 = np.matrix.transpose(T33)
    T34 = T[2][3][:span]; T34 = np.matrix.transpose(T34)

    T41 = T[3][0][:span]; T41 = np.matrix.transpose(T41)
    T42 = T[3][1][:span]; T42 = np.matrix.transpose(T42)
    T43 = T[3][2][:span]; T43 = np.matrix.transpose(T43)
    T44 = T[3][3][:span]; T44 = np.matrix.transpose(T44)
    
    S11=(T13*T44-T14*T43)/(T33*T44-T34*T43)
    S21=(T23*T44-T24*T43)/(T33*T44-T34*T43)
    S31=(T44)/(T33*T44-T34*T43)
    S41=(-T43)/(T33*T44-T34*T43)
    
    S12=(T14*T33-T13*T34)/(T33*T44-T34*T43)
    S22=(T24*T33-T23*T34)/(T33*T44-T34*T43)
    S32=(-T34)/(T33*T44-T34*T43)
    S42=(T33)/(T33*T44-T34*T43)
    
    S13=(T11*T33*T44-T11*T34*T43-T13*T44*T31+T13*T34*T41+T14*T43*T31-T14*T33*T41)/(T33*T44-T34*T43)
    S23=(T21*T33*T44-T21*T34*T43-T23*T44*T31+T23*T34*T41+T24*T43*T31-T24*T33*T41)/(T33*T44-T34*T43)
    S33=(T34*T41-T44*T31)/(T33*T44-T34*T43)
    S43=(T43*T31-T33*T41)/(T33*T44-T34*T43)

    S14=(T12*T33*T44-T12*T34*T43-T13*T44*T32+T13*T34*T42+T14*T43*T32-T14*T33*T42)/(T33*T44-T34*T43)
    S24=(T22*T33*T44-T22*T34*T43-T23*T44*T32+T23*T34*T42+T24*T43*T32-T24*T33*T42)/(T33*T44-T34*T43)
    S34=(T34*T42-T44*T32)/(T33*T44-T34*T43)
    S44=(T43*T32-T33*T42)/(T33*T44-T34*T43)
    
    S = {}
    S['f'] = np.matrix.transpose(f)
    S['lambda'] = np.matrix.transpose(lambda0)
    
    S['S11'] = S11
    S['S21'] = S21
    S['S31'] = S31
    S['S41'] = S41

    S['S12'] = S12
    S['S22'] = S22
    S['S32'] = S32
    S['S42'] = S42
    
    S['S13'] = S13
    S['S23'] = S23
    S['S33'] = S33
    S['S43'] = S43

    S['S14'] = S14
    S['S24'] = S24
    S['S34'] = S34
    S['S44'] = np.matrix.transpose(S44)

    file_sparam  = os.path.join(_globals.TEMP_FOLDER, '%s.dat' % "ContraDC_sparams.mat")
    # export S-parameter data to file named xxx.dat to be loaded in INTERCONNECT
    print(" S-Parameter file: %s" % file_sparam)
    sio.savemat(file_sparam, S)
    params['file_sparam']=file_sparam
    

def cdc_simulation(verbose=True, FDTD_settings = None):

  alpha = 10  # dB/cm
  wavelength_points = 1000

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

  params['alpha'] = alpha
  params['wavelength_points'] = wavelength_points
          
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
    import numpy as np
    kappa = np.pi*ng*delta_lambda/(lambda0**2)
    return kappa
        
  params['kappa_contra'] = kappa_contra = get_kappa(params['delta_lambda_contra'], params['lambda_contra'], params['ng_contra'])
  params['kappa_self1'] = kappa_self1 = get_kappa(params['delta_lambda_self1'], params['lambda_self1'], params['ng1'])/10
  params['kappa_self2'] = kappa_self2 = get_kappa(params['delta_lambda_self2'], params['lambda_self2'], params['ng2'])/10     

  print('Kappa contra = %s, Kappa self 1 = %s, Kappa self 2 = %s' % (kappa_contra, kappa_self1,kappa_self2) )
  
  params = contraDC_model(params)
  
  params = gen_sparams(params)
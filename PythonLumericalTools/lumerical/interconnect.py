'''
################################################################################
#
#  SiEPIC-Tools
#  
################################################################################

Circuit simulations using Lumerical INTERCONNECT and a Compact Model Library

- run_INTC: run INTERCONNECT using Python integration
- circuit_simulation:  run simulation using netlist as input
- circuit_simulation_monte_carlo: perform many simulations


################################################################################
'''

if not 'INTC' in globals():
  INTC = None  
  # print('resetting Lumerical INTERCONNECT Python integration')

import sys
if 'pya' in sys.modules: # check if in KLayout
  import pya

# Run Lumerical Interconnect
def run_INTC(verbose=False):

  if verbose:
    print("SiEPIC.lumerical.interconnect.run_INTC()")  # Python Lumerical INTERCONNECT integration handle

  from . import load_lumapi
  lumapi = load_lumapi.LUMAPI
  global INTC

  if not lumapi:
    print("SiEPIC.lumerical.interconnect.run_INTC: lumapi not loaded; reloading load_lumapi.")
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
    print("SiEPIC.lumerical.interconnect.run_INTC: lumapi not loaded")
    pya.MessageBox.warning("Cannot load Lumerical Python integration.", "Cannot load Lumerical Python integration. \nSome SiEPIC-Tools Lumerical functionality will not be available.", pya.MessageBox.Cancel)
      #    warning = pya.QMessageBox()
      #    warning.setStandardButtons(pya.QMessageBox.Cancel)
      #    warning.setText("Cannot load Lumerical Python integration.") 
      #    warning.setInformativeText("Some SiEPIC-Tools Lumerical functionality will not be available.")
      #    pya.QMessageBox_StandardButton(warning.exec_())
    return
    
  
  if verbose:
    print("Checking INTERCONNECT integration handle: %s" % INTC)  # Python Lumerical INTERCONNECT integration handle
  
  if not INTC: # Not running, start a new session
    INTC = lumapi.open('interconnect')
    if verbose:
      print("Started new session. INTERCONNECT integration handle: %s" % INTC)  # Python Lumerical INTERCONNECT integration handle
  else: # found open INTC session
    try:
      lumapi.evalScript(INTC, "?'Python integration test.';")
    except: # but can't communicate with INTC; perhaps it was closed by the user
      INTC = lumapi.open('interconnect') # run again.
      if verbose:
        print("Re-Started new session. INTERCONNECT integration handle: %s" % INTC)  # Python Lumerical INTERCONNECT integration handle
  try: # check again
    lumapi.evalScript(INTC, "?'Python integration test.';")
  except:
    raise Exception ("Can't run Lumerical INTERCONNECT via Python integration.")

# Generate a spice main file, which contains the simulation parameters
def spice_main(circuit_name, folder, num_detectors):

  import os
  filename_main = os.path.join(folder, '%s_main.spi' % circuit_name)
  filename_subckt = os.path.join(folder,  '%s.spi' % circuit_name)

  file_subckt = open(filename_subckt, 'r')
  ports = None
  for line in file_subckt:
    # find instantiation of subckt
    # check if circuit_name is in the line, and the line doesn't start with a space or a period
    if circuit_name in line and not line[0] in " .":
      fields = line.split(' ')
      fields=[c.replace('\n','') for c in fields]
      print ("  - %s" % fields)
      # make sure this line contains the subckt circuit name
      if circuit_name in fields:
        print (":   - %s" % fields)
        while '' in fields:
          fields.remove('')
        topcell = fields.pop(0)
        print (":   - %s" % fields)
        ports = fields[0:fields.index(circuit_name)]
  if not ports:
    print("no ports found; double check spice netlist file.")
    return
  print("ports: %s" % ports)
  if len(ports) != num_detectors+1:
    print("error: incorrect number of ports than expected (num_detectors+1).")
    return

  # Write the spice file
  file = open(filename_main, 'w')
  text =  '* Spice output from SiEPIC-Tools \n'
  text += '* Optical Network Analyzer:\n'
  text += '.ona input_unit=wavelength input_parameter=start_and_stop\n'
  text += '  + minimum_loss=80\n'
  text += '  + analysis_type=scattering_data\n'
  text += '  + multithreading=user_defined number_of_threads=1\n'
  text += '  + orthogonal_identifier=1\n'
  text += '  + start=1500.000e-9\n'
  text += '  + stop=1600.000e-9\n'
  text += '  + number_of_points=3000\n'
  text += '  + output=%s,%s\n' % (topcell, ports[-1])
  for i in range(0,len(ports)-1):
    text += '  + input(%s)=%s,%s\n' % (i+1, topcell, ports[i])
  text += '.INCLUDE "%s"\n' % filename_subckt 

  file.write (text)
  file.close()
  print(' generated spice _main file')


def circuit_simulation(circuit_name, folder, num_detectors, matlab_data_files=[], simulate=True, verbose=False ):
  if verbose:
    print('*** circuit_simulation()')

  #  circuit_name = topcell.name.replace('.','') # remove "."
  #  circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _
  
  import os
  filename_main = os.path.join(folder, '%s_main.spi' % circuit_name)
  print(filename_main)
  filename_subckt = os.path.join(folder,  '%s.spi' % circuit_name)
  if not os.path.exists(filename_subckt):
      print(" %s netlist file not found" %filename_subckt)
      return
  if 1 or not os.path.exists(filename_main):
      # generate the main spice file
      spice_main(circuit_name, folder, num_detectors)

  # Output files
  filename_lsf = os.path.join(folder, '%s.lsf' % circuit_name)
  filename_icp = os.path.join(folder, '%s.icp' % circuit_name)
  
  
  # Write the Lumerical INTERCONNECT start-up script.
  file = open(filename_lsf, 'w')
  text_lsf = 'switchtolayout;\n'
  text_lsf += 'deleteall;\n'
  text_lsf += "importnetlist('%s');\n" % filename_main
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_thickness", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_width", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_grid", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_x", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_y", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_non_uniform", "wafer", "Number");\n'  % circuit_name
  text_lsf += 'select("::Root Element::%s");\n' % circuit_name
  text_lsf += 'set("run setup script",2);\n'
  text_lsf += "save('%s');\n" % filename_icp
  text_lsf += 'run;\n'
  for i in range(1, num_detectors+1):
    if matlab_data_files:
      # convert simulation data into simple datasets:
      wavelenth_scale = 1e9
      text_lsf += 'temp = getresult("ONA_1", "input %s/mode 1/gain");\n' % i
      text_lsf += 't%s = matrixdataset("Simulation");\n' % i
      text_lsf += 't%s.addparameter("wavelength",temp.wavelength*%s);\n' % (i, wavelenth_scale)
      text_lsf += 't%s.addattribute("Simulation, Detector %s",getresultdata("ONA_1", "input %s/mode 1/gain"));\n' % (i,i, i)
    else:
      text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i, i)
      
  # load measurement data files
  m_count=0
  if matlab_data_files:
    for m in matlab_data_files:
      if '.mat' in m:
        m_count += 1
        # INTERCONNECT can't deal with our measurement files... load and save data.
        from scipy.io import loadmat, savemat        # used to load MATLAB data files
        # *** todo, use DFT rules to determine which measurements we should load.
        PORT=2 # Which Fibre array port is the output connected to?
        matData = loadmat(m, squeeze_me=True, struct_as_record=False)
        wavelength = matData['scandata'].wavelength
        power = matData['scandata'].power[:,PORT-1]
        savemat(m, {'wavelength': wavelength, 'power': power})
        
        # INTERCONNECT load data
        head, tail = os.path.split(m)
        tail = tail.split('.mat')[0]
        text_lsf += 'matlabload("%s");\n' % m
        text_lsf += 'm%s = matrixdataset("Measurement");\n' % m_count
        text_lsf += 'm%s.addparameter("wavelength",wavelength*%s);\n'  % (m_count, wavelenth_scale)
        text_lsf += 'm%s.addattribute("Measured: %s",power);\n'  % (m_count, tail)
  
  text_lsf += 'visualize(t1'
  for i in range(2, num_detectors+1):
    text_lsf += ', t%s' % i
  for i in range(1, m_count+1):
    text_lsf += ', m%s' % i
  text_lsf += ');\n'
  
  file.write (text_lsf)
  file.close()
  
  if verbose:
    print(text_lsf)

  if simulate:
    # Run using Python integration:
    try: 
      from . import load_lumapi
      lumapi = load_lumapi.LUMAPI
      global INTC
      # Launch INTERCONNECT:
      run_INTC()
      lumapi.evalScript(INTC, "?'Test';")
    except:
      import sys
      if 'pya' in sys.modules: # check if in KLayout
        from .. import scripts
        scripts.open_folder(tmp_folder)
        INTC_commandline(filename_main)
    try:
      lumapi.evalScript(INTC, "cd ('" + folder + "');")
      lumapi.evalScript(INTC, circuit_name + ";")
    except:
      pass
  else:
    import sys
    if 'pya' in sys.modules: # check if in KLayout
      from .. import scripts
      scripts.open_folder(tmp_folder)
    
  if verbose:
    print('Done Lumerical INTERCONNECT circuit simulation.')

  
 
'''
'''
def circuit_simulation_monte_carlo(circuit_name, folder, num_detectors, params = None, topcell = None, verbose=True, opt_in_selection_text=[], matlab_data_files=[], simulate=True):
  print('*** circuit_simulation_monte_carlo()')

  # Default simulation parameters
  if not params:
    params = { 
      'num_wafers': 1, 
      'num_dies': 3,
      'histograms': {
        'fsr': False,
        'gain': False,
        'wavelength': False
      },
      'waf_var': {
        'width': {
          'std_dev': 5.0,        # [nm], Within wafer Sigma RMS for width
          'corr_len': 4.5e-3     # [m],  wafer correlation length
        },
        'height': {
          'std_dev': 2.0,        # [nm], Within wafer Sigma RMS for thickness
          'corr_len': 4.5e-3     # [m], wafer correlation length  
        }
      },
      'waf_to_waf_var': {
        'width': {
          'std_dev': 5.0  # [nm], wafer Sigma RMS for width
        },
        'thickness': {
          'std_dev': 3.0 # [nm], wafer Sigma RMS for thickness 
        }
      }
    }


  if int(params['num_wafers'])<1 or int(params['num_dies'])<1:
    print("Insufficient number of dies: The number of die per wafer, and number of wafers, for Monte Carlo simulations need to be 1 or more.")
    return

  #  circuit_name = topcell.name.replace('.','') # remove "."
  #  circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _
  
  import os
  filename_main = os.path.join(folder, '%s_main.spi' % circuit_name)
  print(filename_main)
  filename_subckt = os.path.join(folder,  '%s.spi' % circuit_name)
  if not os.path.exists(filename_subckt):
      print(" %s netlist file not found" %filename_subckt)
      return
  if 1 or not os.path.exists(filename_main):
      # generate the main spice file
      spice_main(circuit_name, folder, num_detectors)

  # Output files
  filename_lsf = os.path.join(folder, '%s.lsf' % circuit_name)
  filename_icp = os.path.join(folder, '%s.icp' % circuit_name)

  # Write the Lumerical INTERCONNECT start-up script.
  file = open(filename_lsf, 'w')

  text_lsf = '###DEVELOPER:Zeqin Lu, zqlu@ece.ubc.ca, University of British Columbia \n' 
  text_lsf += 'switchtolayout;\n'
  text_lsf += 'deleteall;\n'
  text_lsf += "importnetlist('%s');\n" % filename_main
  text_lsf += 'addproperty("::Root Element", "wafer_uniformity_thickness", "wafer", "Matrix");\n' 
  text_lsf += 'addproperty("::Root Element", "wafer_uniformity_width", "wafer", "Matrix");\n' 
  text_lsf += 'addproperty("::Root Element", "N", "wafer", "Number");\n'  
  text_lsf += 'addproperty("::Root Element", "selected_die", "wafer", "Number");\n' 
  text_lsf += 'addproperty("::Root Element", "wafer_length", "wafer", "Number");\n'   
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_thickness", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_width", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_grid", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_x", "wafer", "Number");\n'  % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_y", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_non_uniform", "wafer", "Number");\n'  % circuit_name
  text_lsf += 'select("::Root Element::%s");\n'  % circuit_name
  text_lsf += 'set("MC_non_uniform",99);\n'  
  text_lsf += 'n_wafer = %s;  \n'  % params['num_wafers']  #  GUI INPUT: Number of testing wafer
  text_lsf += 'n_die = %s;  \n'  % params['num_dies']  #  GUI INPUT: Number of testing die per wafer
  text_lsf += 'kk = 1;  \n'
  text_lsf += 'select("ONA_1");\n'
  text_lsf += 'num_points = get("number of points");\n'
  
  for i in range(0, num_detectors):
    text_lsf += 'mc%s = matrixdataset("mc%s"); # initialize visualizer data, mc%s \n' % (i+1, i+1, i+1)
    text_lsf += 'Gain_Data_input%s = matrix(num_points,n_wafer*n_die);  \n' % (i+1) 

  ###Define histograms datasets
  if(params['histograms']['fsr']==True):
    text_lsf += 'fsr_dataset = matrix(1,n_wafer*n_die,1);\n'
  if(params['histograms']['wavelength']==True):
    text_lsf += 'freq_dataset = matrix(1,n_wafer*n_die,1);\n'
  if(params['histograms']['gain']==True):
    text_lsf += 'gain_dataset = matrix(1,n_wafer*n_die,1);\n'
  
  text_lsf += '#Run Monte Carlo simulations; \n'
  text_lsf += 'for (jj=1; jj<=n_wafer; jj=jj+1) {   \n'
  ############################## Wafer generation ###########################################
  text_lsf += ' wafer_length = %s;  \n'  % 100e-3 # datadict["wafer_length_x"]  # [m], GUI INPUT: wafer length
  text_lsf += ' wafer_cl_width = %s;  \n' % params['waf_var']['width']['corr_len']  # [m],  GUI INPUT: wafer correlation length
  text_lsf += ' wafer_cl_thickness = %s;  \n' % params['waf_var']['height']['corr_len']  # [m],  GUI INPUT: wafer correlation length  
  text_lsf += ' wafer_clx_width = wafer_cl_width;  \n'  
  text_lsf += ' wafer_cly_width = wafer_cl_width; \n'   
  text_lsf += ' wafer_clx_thickness = wafer_cl_thickness;  \n'  
  text_lsf += ' wafer_cly_thickness = wafer_cl_thickness; \n'  
  text_lsf += ' N = 500;  \n'        
  text_lsf += ' wafer_grid=wafer_length/N; \n'   
  text_lsf += ' wafer_RMS_w = %s;     \n' % params['waf_var']['width']['std_dev'] # [nm], GUI INPUT: Within wafer Sigma RMS for width
  text_lsf += ' wafer_RMS_t = %s;   \n' % params['waf_var']['height']['std_dev']    # [nm], GUI INPUT: Within wafer Sigma RMS for thickness
  text_lsf += ' x = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += ' y = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += ' xx = meshgridx(x,y) ;  \n'
  text_lsf += ' yy = meshgridy(x,y) ;  \n'
  text_lsf += ' wafer_Z_thickness = wafer_RMS_t*randnmatrix(N,N);  \n'
  text_lsf += ' wafer_F_thickness = exp(-(xx^2/(wafer_clx_thickness^2/2)+yy^2/(wafer_cly_thickness^2/2))); \n'  # Gaussian filter
  text_lsf += ' wafer_uniformity_thickness = real( 2/sqrt(pi)*wafer_length/N/sqrt(wafer_clx_thickness)/sqrt(wafer_cly_thickness)*invfft(fft(wafer_Z_thickness,1,0)*fft(wafer_F_thickness,1,0), 1, 0)  );    \n' # wafer created using Gaussian filter   
  text_lsf += ' wafer_Z_width = wafer_RMS_w*randnmatrix(N,N);  \n'
  text_lsf += ' wafer_F_width = exp(-(xx^2/(wafer_clx_width^2/2)+yy^2/(wafer_cly_width^2/2))); \n'  # Gaussian filter
  text_lsf += ' wafer_uniformity_width = real( 2/sqrt(pi)*wafer_length/N/sqrt(wafer_clx_width)/sqrt(wafer_cly_width)*invfft(fft(wafer_Z_width,1,0)*fft(wafer_F_width,1,0), 1, 0)  );    \n' # wafer created using Gaussian filter 
  
  ######################## adjust Wafer mean ###################
  text_lsf += ' mean_RMS_w = %s;     \n' % params['waf_to_waf_var']['width']['std_dev'] # [nm], GUI INPUT:  wafer Sigma RMS for width
  text_lsf += ' mean_RMS_t = %s;   \n' % params['waf_to_waf_var']['thickness']['std_dev']    # [nm], GUI INPUT:  wafer Sigma RMS for thickness
  text_lsf += ' wafer_uniformity_thickness = wafer_uniformity_thickness + randn(0,mean_RMS_t); \n'
  text_lsf += ' wafer_uniformity_width = wafer_uniformity_width + randn(0,mean_RMS_w); \n'
  
  ##################################### pass wafer to Root ###################
  text_lsf += ' #pass wafers to object \n'
  text_lsf += ' select("::Root Element");  \n' 
  text_lsf += ' set("wafer_uniformity_thickness", wafer_uniformity_thickness);  \n'
  text_lsf += ' set("wafer_uniformity_width", wafer_uniformity_width);  \n'
  text_lsf += ' set("N",N);  \n'
  text_lsf += ' set("wafer_length",wafer_length);  \n'
  
  #################################### embed wafer selection script in Root ###################
  text_lsf += ' select("::Root Element");\n'
  text_lsf += ' set("setup script",'+ "'" +  ' \n'
  text_lsf += '  ######################## high resolution interpolation for dies ################# \n'
  text_lsf += '  MC_grid = 5e-6;  \n'   # [m], mesh grid
  text_lsf += '  die_span_x = %s; \n'  % 5e-3 # datadict["die_length_x"]  # [m]    GUI INPUT: die length X
  text_lsf += '  die_span_y = %s; \n'  % 5e-3 # datadict["die_length_y"]  # [m]    GUI INPUT: die length Y
  text_lsf += '  MC_resolution_x = die_span_x/MC_grid;  \n'
  text_lsf += '  MC_resolution_y = die_span_y/MC_grid;  \n'
  text_lsf += '  die_num_x = floor(wafer_length/die_span_x); \n'
  text_lsf += '  die_num_y = floor(wafer_length/die_span_y); \n'
  text_lsf += '  die_num_total = die_num_x*die_num_y; \n'
  text_lsf += '  x = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += '  y = linspace(-wafer_length/2,wafer_length/2,N); \n'
              # pick die for simulation, and do high resolution interpolation 
  text_lsf += '  j=selected_die; \n'
  text_lsf += '  die_min_x = -wafer_length/2+(j-1)*die_span_x -floor((j-1)/die_num_x)*wafer_length; \n'
  text_lsf += '  die_max_x = -wafer_length/2+j*die_span_x -floor((j-1)/die_num_x)*wafer_length; \n'
  text_lsf += '  die_min_y = wafer_length/2-ceil(j/die_num_y)*die_span_y; \n'
  text_lsf += '  die_max_y = wafer_length/2-(ceil(j/die_num_y)-1)*die_span_y; \n'
  text_lsf += '  x_die = linspace(die_min_x, die_max_x, MC_resolution_x); \n'
  text_lsf += '  y_die = linspace(die_min_y, die_max_y, MC_resolution_y); \n'
  text_lsf += '  die_xx = meshgridx(x_die,y_die) ;  \n'
  text_lsf += '  die_yy = meshgridy(x_die,y_die) ;  \n'
  text_lsf += '  MC_uniformity_thickness = interp(wafer_uniformity_thickness, x, y, x_die, y_die); # interpolation \n'
  text_lsf += '  MC_uniformity_width = interp(wafer_uniformity_width, x, y, x_die, y_die); # interpolation \n'
  ######################### pass die to object ####################################
  text_lsf += '  select("::Root Element::%s");  \n' % circuit_name
  text_lsf += '  set("MC_uniformity_thickness",MC_uniformity_thickness);  \n'
  text_lsf += '  set("MC_uniformity_width",MC_uniformity_width);  \n'
  text_lsf += '  set("MC_resolution_x",MC_resolution_x);  \n'
  text_lsf += '  set("MC_resolution_y",MC_resolution_y);  \n'
  text_lsf += '  set("MC_grid",MC_grid);  \n'
  text_lsf += '  set("MC_non_uniform",1);  \n'
  text_lsf += " '"+'); \n'
  
  text_lsf += ' for (ii=1;  ii<=n_die; ii=ii+1) {   \n'
  text_lsf += '  switchtodesign; \n'
  text_lsf += '  setnamed("ONA_1","peak analysis", "single");\n'
  text_lsf += '  select("::Root Element");  \n'
  text_lsf += '  set("selected_die",ii);  \n'
  text_lsf += '  run;\n'
  text_lsf += '  select("ONA_1");\n'
  text_lsf += '  T=getresult("ONA_1","input 1/mode 1/transmission");\n'
  text_lsf += '  wavelength = T.wavelength;\n'   
  
  for i in range(0, num_detectors):
    text_lsf += '  if (kk==1) { mc%s.addparameter("wavelength",wavelength);} \n' % (i+1) 
    text_lsf += '  mc%s.addattribute("run", getattribute( getresult("ONA_1", "input %s/mode 1/gain"), getattribute(getresult("ONA_1", "input %s/mode 1/gain")) ) );\n' % (i+1, i+1, i+1)
    text_lsf += '  Gain_Data_input%s(1:num_points, kk) = getattribute( getresult("ONA_1", "input %s/mode 1/gain"), getattribute(getresult("ONA_1", "input %s/mode 1/gain")) ); \n'  % (i+1, i+1, i+1)
    
  #add simulation data to their corresponding datalists  
  if(params['histograms']['fsr']==True):
      text_lsf += '  fsr_select = getresult("ONA_1", "input 1/mode 1/peak/free spectral range");\n'
      text_lsf += '  fsr_dataset(1,kk) = real(fsr_select.getattribute(getattribute(fsr_select)));\n'

  if(params['histograms']['wavelength']==True):
      text_lsf += '  freq_dataset(1,kk) = getresult("ONA_1", "input 1/mode 1/peak/frequency");\n'

  if(params['histograms']['gain']==True):
      text_lsf += '  gain_select = getresult("ONA_1", "input 1/mode 1/peak/gain");\n'
      text_lsf += '  gain_dataset(1,kk) = real(gain_select.getattribute(getattribute(gain_select)));\n'

  text_lsf += '  switchtodesign; \n'
  text_lsf += '  kk = kk + 1;  \n'
  text_lsf += ' }\n'   # end for wafer iteration
  text_lsf += '}\n'  # end for die iteration
  text_lsf += '?"Spectrum data for each input can be found in the Script Workspace tab:";\n'    
  for i in range(0, num_detectors): 
      text_lsf += '?"Gain_Data_input%s"; \n' %(i+1)
  text_lsf += '?"Plot spectrums using script: plot(wavelength, Gain_Data_input#)";\n'  
  for i in range(0, num_detectors):
    text_lsf += 'visualize(mc%s);\n' % (i+1)
  
  #### Display Histograms for the selected components
  #FSR
  if(params['histograms']['fsr']==True):
      text_lsf += 'dataset = fsr_dataset*1e9;\n'  #select fsr dataset defined above
      text_lsf += 'bin_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n' #define number of bins according to the number of data
      text_lsf += 'histc(dataset, bin_hist, "Free Spectral Range (nm)", "Count", "Histogram - FSR");\n' #generate histogram 
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n' #define plot legends
      
  #wavelength
  if(params['histograms']['wavelength']==True):
      text_lsf += 'dataset = freq_dataset*1e9;\n'
      text_lsf += 'num_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n'
      text_lsf += 'histc(dataset, bin_hist, "Wavelength (nm)", "Count", "Histogram - Peak wavelength");\n'
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n'

  #Gain
  if(params['histograms']['gain']==True):
      text_lsf += 'dataset = gain_dataset;\n'
      text_lsf += 'num_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n'
      text_lsf += 'histc(dataset, bin_hist, "Gain (dB)", "Count", "Histogram - Peak gain");\n'
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n'





  '''

  for i in range(1, num_detectors+1):
    if matlab_data_files:
      # convert simulation data into simple datasets:
      wavelenth_scale = 1e9
      text_lsf += 'temp = getresult("ONA_1", "input %s/mode 1/gain");\n' % i
      text_lsf += 't%s = matrixdataset("Simulation");\n' % i
      text_lsf += 't%s.addparameter("wavelength",temp.wavelength*%s);\n' % (i, wavelenth_scale)
      text_lsf += 't%s.addattribute("Simulation, Detector %s",getresultdata("ONA_1", "input %s/mode 1/gain"));\n' % (i,i, i)
    else:
      text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i, i)
      
  # load measurement data files
  m_count=0
  if matlab_data_files:
    for m in matlab_data_files:
      if '.mat' in m:
        m_count += 1
        # INTERCONNECT can't deal with our measurement files... load and save data.
        from scipy.io import loadmat, savemat        # used to load MATLAB data files
        # *** todo, use DFT rules to determine which measurements we should load.
        PORT=2 # Which Fibre array port is the output connected to?
        matData = loadmat(m, squeeze_me=True, struct_as_record=False)
        wavelength = matData['scandata'].wavelength
        power = matData['scandata'].power[:,PORT-1]
        savemat(m, {'wavelength': wavelength, 'power': power})
        
        # INTERCONNECT load data
        head, tail = os.path.split(m)
        tail = tail.split('.mat')[0]
        text_lsf += 'matlabload("%s");\n' % m
        text_lsf += 'm%s = matrixdataset("Measurement");\n' % m_count
        text_lsf += 'm%s.addparameter("wavelength",wavelength*%s);\n'  % (m_count, wavelenth_scale)
        text_lsf += 'm%s.addattribute("Measured: %s",power);\n'  % (m_count, tail)
  
  text_lsf += 'visualize(t1'
  for i in range(2, num_detectors+1):
    text_lsf += ', t%s' % i
  for i in range(1, m_count+1):
    text_lsf += ', m%s' % i
  text_lsf += ');\n'
  
  '''
  
  file.write (text_lsf)
  file.close()
  
  if verbose:
    print(text_lsf)
  #

  if simulate:
    # Run using Python integration:
    try: 
      from . import load_lumapi
      lumapi = load_lumapi.LUMAPI
      global INTC
      # Launch INTERCONNECT:
      run_INTC()
      lumapi.evalScript(INTC, "?'Test';")
    except:
      import sys
      if 'pya' in sys.modules: # check if in KLayout
        from .. import scripts
        scripts.open_folder(folder)
        INTC_commandline(filename_main)
    try:
      lumapi.evalScript(INTC, "cd ('" + folder + "');")
      lumapi.evalScript(INTC, circuit_name + ";")
    except:
      pass
  else:
    import sys
    if 'pya' in sys.modules: # check if in KLayout
      from .. import scripts
      scripts.open_folder(folder)

  if verbose:
    print('Done Lumerical INTERCONNECT Monte Carlo circuit simulation.')


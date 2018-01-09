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


'''
################################################################################
Lumerical FDTD Solutions simulation for a component
generate S-Parameters

definitions:
  z: thickness direction
  z=0: centre of the waveguides, so we can use symmetric mesh in z
       which will be better for convergence testing as the mesh won't move
################################################################################
'''
def generate_component_sparam(verbose = False):
  if verbose:
    print('SiEPIC.lumerical.fdtd: generate_component_sparam()')

  # Get technology and layout details
  from ..utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()  
  dbum = TECHNOLOGY['dbu']*1e-6 # dbu to m conversion

  # run Lumerical FDTD Solutions  
  import lumapi_fdtd as lumapi
  from .. import _globals
  if verbose:
    print(_globals.FDTD)  # Python Lumerical INTERCONNECT integration handle
  run_FDTD()

  # get FDTD settings from XML file
  from SiEPIC.utils import load_FDTD_settings
  FDTD_settings=load_FDTD_settings()
  if FDTD_settings:
    if verbose:
      print(FDTD_settings)

  # get selected instances; only one
  from ..utils import select_instances
  selected_instances = select_instances()
  error = pya.QMessageBox()
  error.setStandardButtons(pya.QMessageBox.Ok )
  if len(selected_instances) != 1:
    error.setText("Error: Need to have one component selected.")
    return
  
  # get selected component
  if verbose:
    print(" selected component: %s" % selected_instances[0].inst().cell )
  component = cell.find_components(cell_selected=[selected_instances[0].inst().cell])[0]

  # Configure wavelength and polarization
  # GUI for: pol = {'quasi-TE', 'quasi-TM', 'quasi-TE and -TM'}
  mode_selection = FDTD_settings['mode_selection']
  # GUI for: wavelength = {'1450-1650'} 
  wavelength_start = FDTD_settings['wavelength_start']
  wavelength_stop =  FDTD_settings['wavelength_stop']     

  # get DevRec layer
  devrec_box = component.polygon.bbox()
  print("%s, %s, %s, %s"  % (devrec_box.left*dbum, devrec_box.right*dbum, devrec_box.bottom*dbum, devrec_box.top*dbum) )

  # create FDTD simulation region (extra large)
  FDTDzspan=4e-6
  FDTDxmin,FDTDxmax,FDTDymin,FDTDymax = devrec_box.left*dbum-200e-9, devrec_box.right*dbum+200e-9, devrec_box.bottom*dbum-200e-9, devrec_box.top*dbum+200e-9
  sim_time = max(devrec_box.width(),devrec_box.height())*dbum * 4.5; 
  lumapi.evalScript(_globals.FDTD, " \
    newproject; \
    addfdtd; set('x min',%s); set('x max',%s); set('y min',%s); set('y max',%s); set('z span',%s);\
    set('force symmetric z mesh', 1); set('mesh accuracy',1); \
    setglobalsource('wavelength start',%s); setglobalsource('wavelength stop', %s); \
    setglobalmonitor('frequency points',%s); set('simulation time', %s/c+400e-15); \
    ?'FDTD solver added';    " % (FDTDxmin,FDTDxmax,FDTDymin,FDTDymax,FDTDzspan,wavelength_start,wavelength_stop,FDTD_settings['frequency_points_monitor'], sim_time) )

  # add substrate and cladding:
  lumapi.evalScript(_globals.FDTD, " \
    addrect; set('x min',%s); set('x max',%s); set('y min',%s); set('y max',%s); set('z min', %s); set('z max',%s);\
      set('material', '%s'); set('alpha',0.1);    \
    ?'oxide added';    " % (FDTDxmin-1e-6, FDTDxmax+1e-6, FDTDymin-1e-6, FDTDymax+1e-6, \
      -FDTD_settings['thickness_BOX']-FDTD_settings['thickness_Si']/2, -FDTD_settings['thickness_Si']/2, \
      FDTD_settings['material_Clad'] ) )
  lumapi.evalScript(_globals.FDTD, " \
    addrect; set('x min',%s); set('x max',%s); set('y min',%s); set('y max',%s); set('z min', %s); set('z max',%s);\
      set('material', '%s'); set('alpha',0.1);    \
    ?'oxide added';    " % (FDTDxmin-1e-6, FDTDxmax+1e-6, FDTDymin-1e-6, FDTDymax+1e-6, \
      -FDTD_settings['thickness_Si']/2, FDTD_settings['thickness_Clad']-FDTD_settings['thickness_Si']/2, \
      FDTD_settings['material_Clad'] ) )

  # get polygons from component
  polygons = component.get_polygons()
  if verbose:
    print(" polygons: %s" % [p for p in polygons] )
  polygons_vertices = [[[vertex.x*dbum, vertex.y*dbum] for vertex in p.each_point()] for p in [p.to_simple_polygon() for p in polygons] ]
  if verbose:
    print(" polygons' vertices: %s" % len(polygons_vertices) )
  if len(polygons_vertices) < 1:
    error.setText("Error: Component needs to have polygons.")
    return

  # send polygons to FDTD
  for i in range(0,len(polygons_vertices)):
    lumapi.putMatrix(_globals.FDTD, "polygon_vertices", polygons_vertices[i] )
    lumapi.evalScript(_globals.FDTD, " \
      addpoly; set('vertices',polygon_vertices); \
      set('material', '%s'); set('z span', %s);     \
      ?'Polygons added'; " % (FDTD_settings['material_Si'], FDTD_settings['thickness_Si']) )
    
  # get Component pins
  pins = component.find_pins()
  pins = sorted(pins, key=lambda  p: p.pin_name)
  
  # create FDTD ports
  for p in pins:
    if p.rotation in [180.0, 0.0]:
      lumapi.evalScript(_globals.FDTD, " \
        addport; set('injection axis', 'x-axis'); set('x',%s); set('y',%s); set('y span',%s); set('z span',%s); \
        " % (p.center.x*dbum, p.center.y*dbum,2e-6,FDTDzspan)  )
    if p.rotation in [270.0, 90.0, -90.0]:
      lumapi.evalScript(_globals.FDTD, " \
        addport; set('injection axis', 'y-axis'); set('x',%s); set('y',%s); set('x span',%s); set('z span',%s); \
        " % (p.center.x*dbum, p.center.y*dbum,2e-6,FDTDzspan)  )
    if p.rotation in [180.0, 90.0]:
      p.direction = 'Backward'
    else:
      p.direction = 'Forward'
    lumapi.evalScript(_globals.FDTD, " \
      set('name','%s'); set('direction', '%s'); set('frequency points', %s); set('mode selection', '%s'); \
      ?'Added pin: %s'; " % (p.pin_name, p.direction, 1, mode_selection, p.pin_name)  )
      
    
  # Calculate mode sources
  # Get field profiles, to find |E| = 1e-6 points to find spans
  min_z, max_z = 0,0
  for p in [pins[0]]:  # if all pins are the same, only do it once
    lumapi.evalScript(_globals.FDTD, " \
      select('FDTD::ports::%s'); mode_profiles=getresult('FDTD::ports::%s','mode profiles'); E=mode_profiles.E1; x=mode_profiles.x; y=mode_profiles.y; z=mode_profiles.z; \
      ?'Selected pin: %s'; " % (p.pin_name, p.pin_name, p.pin_name)  )
    E=lumapi.getVar(_globals.FDTD, "E")
    x=lumapi.getVar(_globals.FDTD, "x")
    y=lumapi.getVar(_globals.FDTD, "y")
    z=lumapi.getVar(_globals.FDTD, "z")

    import numpy as np
    # remove the wavelength from the array, 
    # leaving two dimensions, and 3 field components
    Efield_xyz = np.array(E[0,:,:,0,:])
    # find the field intensity (|Ex|^2 + |Ey|^2 + |Ez|^2)
    Efield_intensity = np.empty([Efield_xyz.shape[0],Efield_xyz.shape[1]])
    print(Efield_xyz.shape)
    for a in range(0,Efield_xyz.shape[0]):
      for b in range(0,Efield_xyz.shape[1]):
        Efield_intensity[a,b] = abs(Efield_xyz[a,b,0])**2+abs(Efield_xyz[a,b,1])**2+abs(Efield_xyz[a,b,2])**2
    # find the max field for each z slice (b is the z axis)
    Efield_intensity_b = np.empty([Efield_xyz.shape[1]])
    for b in range(0,Efield_xyz.shape[1]):
      Efield_intensity_b[b] = max(Efield_intensity[:,b])
    # find the z thickness where the field has sufficiently decayed
    indexes = np.argwhere ( Efield_intensity_b > 1e-6 )
    min_index, max_index = int(min(indexes)), int(max(indexes))
    if min_z > z[min_index]:
      min_z = z[min_index]
    if max_z < z[max_index]:
      max_z = z[max_index]
    if verbose:
      print(' Port %s field decays at: %s, %s microns' % (p.pin_name, z[max_index], z[min_index]) )

  if FDTDzspan > max_z-min_z:
    FDTDzspan = max_z-min_z
    if verbose:
      print(' Updating FDTD Z-span to: %s microns' % (FDTDzspan) )
 
  # Configure FDTD region, mesh accuracy 1
  # run single simulation
  from .. import _globals
  import os
  filename = os.path.join(_globals.TEMP_FOLDER, '%s.fsp' % component.instance)
  lumapi.evalScript(_globals.FDTD, " \
    select('FDTD'); set('z span',%s);\
    save('%s');\
    ?'FDTD Z-span updated to %s'; " % (FDTDzspan, filename, FDTDzspan) )

  # Calculate, plot, and get the S-Parameters, S21, S31, S41 ...
  # optionally simulate a subset of the S-Parameters
  # assume input on port 1  
  def FDTD_run_Sparam_simple(pins, out_pins = None, plots = False):
    if verbose:
      print(' Run simulation S-Param FDTD')
    lumapi.evalScript(_globals.FDTD, "run; ")
    port_pins = [pins[0]]+out_pins if out_pins else pins
    for p in port_pins:
      lumapi.evalScript(_globals.FDTD, " \
        P=Port_%s=getresult('FDTD::ports::%s','expansion for port monitor'); \
         " % (p.pin_name,p.pin_name) )
    lumapi.evalScript(_globals.FDTD, "wavelengths=c/P.f*1e6;")
    wavelengths = lumapi.getVar(_globals.FDTD, "wavelengths") 
    Sparams = []
    for p in port_pins[1::]:
      lumapi.evalScript(_globals.FDTD, " \
        Sparam=S_%s_%s= Port_%s.%s/Port_%s.%s;  \
         " % (p.pin_name, pins[0].pin_name, \
              p.pin_name, 'b' if p.direction=='Forward' else 'a', \
              pins[0].pin_name, 'a' if pins[0].direction=='Forward' else 'b') )
      Sparams.append(lumapi.getVar(_globals.FDTD, "Sparam"))
      if plots:
        lumapi.evalScript(_globals.FDTD, " \
          plot (wavelengths, 10*log10(abs(Sparam)^2),  'Wavelength (um)', 'Transmission (dB)', 'S_%s_%s'); \
           " % (p.pin_name, pins[0].pin_name) )
    return Sparams
  
    
    
  
  # user verify ok ?

  # Convergence testing on S-Parameters:
  # find the pin that has the highest Sparam (max over wavelength)
  # use this Sparam for convergence testing
  Sparams = FDTD_run_Sparam_simple(pins, plots = True)
  Sparam_pin_max = np.amax(np.absolute(Sparams), axis=1).argmax() +1
  # convergence test on simulation z-span (assume symmetric)
  # loop in Python so we can check if it is good enough
  test_converged = False
  convergence = []
  Sparams_abs_prev = np.array([np.absolute(Sparams)[Sparam_pin_max,:,:]])
  while not test_converged:
    FDTDzspan += 100e-9
    lumapi.evalScript(_globals.FDTD, " \
      switchtolayout; select('FDTD'); set('z span',%s);\
      ?'FDTD Z-span updated to %s'; " % (FDTDzspan, FDTDzspan) )
    Sparams = FDTD_run_Sparam_simple(pins, out_pins = [pins[Sparam_pin_max]], plots = True)
    Sparams_abs = np.array(np.absolute(Sparams))
    rms_error = np.sqrt(np.mean( (Sparams_abs_prev - Sparams_abs)**2 ))
    convergence.append ( [FDTDzspan, rms_error] )
    Sparams_abs_prev = Sparams_abs
    if verbose:
      print (' convergence: span %s, rms error %s' % (FDTDzspan, rms_error) ) 
    if FDTDzspan > 2e-6:
      test_converged=True

  lumapi.putMatrix(_globals.FDTD, 'convergence', convergence)
  lumapi.evalScript(_globals.FDTD, "plot(convergence(:,1), convergence(:,2), 'Simulation span','RMS error between simulation','Convergence testing');")
  
  return Sparams 
  
  
  # Configure FDTD region, mesh accuracy 4, update FDTD ports mode source frequency points
  lumapi.evalScript(_globals.FDTD, " \
    select('FDTD'); set('mesh accuracy',%s);\
    ?'FDTD mesh accuracy updated %s'; " % (FDTD_settings['mesh_accuracy'], FDTD_settings['mesh_accuracy']) )
  for p in pins:
    lumapi.evalScript(_globals.FDTD, " \
      select('FDTD::ports::%s'); set('frequency points', %s); \
      ?'updated pin: %s'; " % (p.pin_name, FDTD_settings['frequency_points_expansion'], p.pin_name)  )



  
  # Run full S-parameters

'''
# add s-parameter sweep task
deletesweep("s-parameter sweep");
addsweep(3);

# perform simulations using the first 2 ports, use symmetry later.
NsimPorts=2;
 
# define index entries for s-matrix mapping table (rows)
Nports=4;
for (port=1:Nports) { # inject light into each port
 for (mode = mode_selection) { # for each mode
  index1 = struct;
  index1.Port = "port "+num2str(port);
  index1.Mode = "mode "+num2str(mode);
  # add index entries to s-matrix mapping table
  addsweepparameter("s-parameter sweep",index1);
 }
}

# run s-parameter sweep
runsweep("s-parameter sweep");
 
# collect results
S_matrix = getsweepresult("s-parameter sweep","S matrix");
S_parameters = getsweepresult("s-parameter sweep","S parameters");
S_diagnostic = getsweepresult("s-parameter sweep","S diagnostic");
 
# visualize results
#visualize(S_matrix);
visualize(S_parameters);
#visualize(S_diagnostic);
 
# export S-parameter data to file named s_params.dat to be loaded in INTERCONNECT
exportsweep("s-parameter sweep",s_filename);
'''
  # Export S-Parameters

  # Create an INTC model
  




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
  sim_time = max(devrec_box.width(),devrec_box.height())*dbum * 4.2 + 5e-6; 
  lumapi.evalScript(_globals.FDTD, " \
    newproject; \
    addfdtd; set('x min',%s); set('x max',%s); set('y min',%s); set('y max',%s); set('z span',%s);\
    set('force symmetric z mesh', 1); set('mesh accuracy',1); \
    setglobalsource('wavelength start',%s); setglobalsource('wavelength stop', %s); \
    setglobalmonitor('frequency points',%s); set('simulation time', %s/c); \
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
    print(" polygons' vertices: %s" % polygons_vertices )
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
      direction = 'Backward'
    else:
      direction = 'Forward'
    lumapi.evalScript(_globals.FDTD, " \
      set('name','%s'); set('direction', '%s'); set('frequency points', %s); set('mode selection', '%s'); \
      ?'Added pin: %s'; " % (p.pin_name, direction, FDTD_settings['frequency_points_expansion'], mode_selection, p.pin_name)  )
      
    
  # Calculate mode sources
  # Get field profiles, to find |E| = 1e-6 points to find spans
  for p in pins:
    lumapi.evalScript(_globals.FDTD, " \
      select('FDTD::ports::%s'); updateportmodes;  \
      ?'Selected pin: %s'; " % (p.pin_name, p.pin_name)  )

  
  # Configure FDTD region, mesh accuracy 1
  
  # run single simulation
  
  # user verify ok
  
  # convergence test on simulation z-span (assume symmetric)
  # loop in Python so we can check if it is good enough
  
  # Configure FDTD region, mesh accuracy 4
  
  # Run full S-parameters

  # Export S-Parameters

  # Create an INTC model
  




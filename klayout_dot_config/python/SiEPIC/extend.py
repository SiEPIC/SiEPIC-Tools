#################################################################################
#                SiEPIC Class Extension of KLayout PYA Library                  #
#################################################################################
'''
This module extends several pya classes that are useful for the library.

pya.Path and pya.DPath Extensions:
  - get_points(), returns list of pya.Points
  - get_dpoints(), returns list of pya.DPoints
  - is_manhattan(), tests to see if the path is manhattan
  - radius_check(radius), tests to see of all path segments are long enough to be
  converted to a waveguide with bends of radius 'radius'
  - remove_colinear_points(), removes all colinear points in place
  - translate_from_center(offset), returns a new path whose points have been offset
  by 'offset' from the center of the original path
  - snap(pins), snaps the path in place to the nearest pin
  
pya.Polygon and pya.DPolygon Extensions:
  - get_points(), returns list of pya.Points
  - get_dpoints(), returns list of pya.DPoints
  
pya.PCellDeclarationHelper Extensions:
  - print_parameter_list, prints parameter list
  
pya.Cell Extensions:
  - print_parameter_values, if this cell is a pcell, prints the parameter values
  - find_pins: find Pin object of either the specified name or all pins in a cell
  - find_pin

pya.Instance Extensions:
  - find_pins: find Pin objects for all pins in a cell instance
'''
#################################################################################

import pya

warning = pya.QMessageBox()
warning.setStandardButtons(pya.QMessageBox.Ok)
warning.setDefaultButton(pya.QMessageBox.Ok)

#################################################################################
#                SiEPIC Class Extension of Path & DPath Class                   #
#################################################################################

# Function Definitions
#################################################################################

def get_points(self):
  return [pya.Point(pt.x, pt.y) for pt in self.each_point()]

def get_dpoints(self):
  return [pya.DPoint(pt.x, pt.y) for pt in self.each_point()]

def is_manhattan(self):
  if self.__class__ == pya.Path:
    pts = self.get_points()
  else:
    pts = self.get_dpoints()
  check = 1 if len(pts) == 2 else 0
  for i, pt in enumerate(pts):
    if (i==1 or pts[i] == pts[-1]):
      if(pts[i].x == pts[i-1].x or pts[i].y == pts[i-1].y): check += 1
  return check==2
  
def radius_check(self, radius):
  def all2(iterable):
    for element in iterable:
        if not element:
            return False
    return True

  points = self.get_points()
  lengths = [ points[i].distance(points[i-1]) for i, pt in enumerate(points) if i > 0]

  # first and last segment must be >= radius
  check1=(lengths[0] >= radius)
  check2=(lengths[-1] >= radius)
  # middle segments must accommodate two bends, hence >= 2 radius
  check3=[length >= 2*radius for length in lengths if length != lengths[0] or length != lengths[-1]]
  return check1 and check2 and all(check3)

def remove_colinear_points(self):
  from .utils import pt_intersects_segment
  if self.__class__ == pya.Path:
    pts = self.get_points()
  else:
    pts = self.get_dpoints()
  self.points = [pts[0]]+[pts[i] for i in range(1, len(pts)-1) if not pt_intersects_segment(pts[i+1], pts[i-1], pts[i])]+[pts[-1]]
  
def translate_from_center(self, offset):
  from math import pi, cos, sin, acos, sqrt
  from .utils import angle_vector
  pts = [pt for pt in self.get_dpoints()]
  tpts = [pt for pt in self.get_dpoints()]
  for i in range(0,len(pts)):
    if i == 0:
      u = pts[i]-pts[i+1]
      v = -u
    elif i == (len(pts) - 1):
      u = pts[i-1]-pts[i]
      v = -u
    else:
      u = pts[i-1]-pts[i]
      v = pts[i+1]-pts[i]

    if offset < 0:
      o1 = pya.DPoint(abs(offset)*cos(angle_vector(u)*pi/180-pi/2), abs(offset)*sin(angle_vector(u)*pi/180-pi/2))
      o2 = pya.DPoint(abs(offset)*cos(angle_vector(v)*pi/180+pi/2), abs(offset)*sin(angle_vector(v)*pi/180+pi/2))
    else:
      o1 = pya.DPoint(abs(offset)*cos(angle_vector(u)*pi/180+pi/2), abs(offset)*sin(angle_vector(u)*pi/180+pi/2))
      o2 = pya.DPoint(abs(offset)*cos(angle_vector(v)*pi/180-pi/2), abs(offset)*sin(angle_vector(v)*pi/180-pi/2))
      
    p1 = u+o1
    p2 = o1
    p3 = v+o2
    p4 = o2
    d = (p1.x-p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x-p4.x)

    if round(d,10) == 0:
      tpts[i] += p2
    else:
      tpts[i] += pya.DPoint(((p1.x*p2.y-p1.y*p2.x)*(p3.x-p4.x)-(p1.x-p2.x)*(p3.x*p4.y-p3.y*p4.x))/d,
                           ((p1.x*p2.y-p1.y*p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x*p4.y-p3.y*p4.x))/d)

  if self.__class__ == pya.Path:
    return pya.Path([pya.Point(pt.x, pt.y) for pt in tpts], self.width)
  elif self.__class__ == pya.DPath:
    return pya.DPath(tpts, self.width)
    
'''
snap - pya.Path extension
This function snaps the two path endpoints to the nearest pins by adjusting the end segments

Input: 
 - self: the Path object
 - pins: an array of Pin objects, which are paths with 2 points, 
         with the vector giving the direction (out of the component)
Output:
 - modifies the original Path

'''
def snap(self, pins):
  # Import functionality from SiEPIC-Tools:
  from .utils import angle_vector, get_technology
  from . import _globals
  TECHNOLOGY = get_technology()
    
  # Search for pins within this distance to the path endpoints, e.g., 10 microns
  d_min = _globals.PATH_SNAP_PIN_MAXDIST/TECHNOLOGY['dbu'];

  if not len(pins): return

  # array of path vertices:
  pts = self.get_points()

  # angles of all segments:
  ang = angle_vector(pts[1]-pts[0])
  print( '%s, %s' % (ang, [pin.rotation for pin in pins] ))
  
  # sort all the pins based on distance to the Path endpoint
  # only consider pins that are facing each other, 180 degrees 
  pins_sorted = sorted([pin for pin in pins if round((ang - pin.rotation)%360) == 180 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[0]))

  if len(pins_sorted):
    # pins_sorted[0] is the closest one
    dpt = pins_sorted[0].center - pts[0]
    # check if the pin is close enough to the path endpoint
    if dpt.abs() <= d_min:
      # snap the endpoint to the pin
      pts[0] = dpt
      # move the first corner
      if(round(ang % 180) == 0):
        pts[1].y += dpt.y
      else:
        pts[1].x += dpt.x
        
  # do the same thing on the other end:  
  ang = angle_vector(pts[-2]-pts[-1])
  print( '%s, %s' % (ang, [pin.rotation for pin in pins] ))
  pins_sorted = sorted([pin for pin in pins if round((ang - pin.rotation)%360) == 180 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[-1]))
  if len(pins_sorted):
    dpt = pins_sorted[0].center - pts[-1]
    if dpt.abs() <= d_min:
      pts[-1] = dpt
      if(round(ang % 180) == 0):
        pts[-2].y += dpt.y
      else:
        pts[-2].x += dpt.x

  # check that the path has non-zero length after the snapping operation
  test_path = pya.Path()
  test_path.points = pts
  if test_path.length() > 0:
    self.points = pts

# Path Extension
#################################################################################

pya.Path.get_points = get_points
pya.Path.get_dpoints = get_dpoints
pya.Path.is_manhattan = is_manhattan
pya.Path.radius_check = radius_check
pya.Path.remove_colinear_points = remove_colinear_points
pya.Path.translate_from_center = translate_from_center
pya.Path.snap = snap;

# DPath Extension
#################################################################################

pya.DPath.get_points = get_points
pya.DPath.get_dpoints = get_dpoints
pya.DPath.is_manhattan = is_manhattan
pya.DPath.radius_check = radius_check
pya.DPath.remove_colinear_points = remove_colinear_points
pya.DPath.translate_from_center = translate_from_center
pya.DPath.snap = snap;

#################################################################################
#            SiEPIC Class Extension of Polygon & DPolygon Class                 #
#################################################################################

# Function Definitions
#################################################################################

def get_points(self):
  return [pya.Point(pt.x, pt.y) for pt in self.each_point_hull()]

def get_dpoints(self):
  return [pya.DPoint(pt.x, pt.y) for pt in self.each_point_hull()]

#################################################################################

pya.Polygon.get_points = get_points;
pya.Polygon.get_dpoints = get_dpoints;

#################################################################################

pya.DPolygon.get_points = get_points;
pya.DPolygon.get_dpoints = get_dpoints;

#################################################################################
#                    SiEPIC Class Extension of PCell Class                      #
#################################################################################

# Function Definitions
#################################################################################

def print_parameter_list(self):
  types = ['TypeBoolean', 'TypeDouble', 'TypeInt', 'TypeLayer', 'TypeList', 'TypeNone', 'TypeShape', 'TypeString']
  for p in self.get_parameters():
    if ~p.readonly:
      print( "Name: %s, %s, unit: %s, default: %s, description: %s%s" % \
        (p.name, types[p.type], p.unit, p.default, p.description, ", hidden" if p.hidden else ".") )

#################################################################################

pya.PCellDeclarationHelper.print_parameter_list = print_parameter_list
  
#################################################################################
#                    SiEPIC Class Extension of Cell Class                       #
#################################################################################

# Function Definitions
#################################################################################

def print_parameter_values(self):
  print(self.pcell_parameters())
  params = self.pcell_parameters_by_name()
  for key in params.keys():
    print("Parameter: %s, Value: %s") % (key, params[key])

'''
Optical Pins have: 
 1) path on layer PinRec, indicating direction (out of component)
 2) text on layer PinRec, inside the path
Electrical Pins have: 
 1) box on layer PinRec, indicating direction (out of component)
 2) text on layer PinRec, inside the path
'''
def find_pins(self):
  from .core import Pin
  from . import _globals
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  pins = []
  LayerPinRecN = self.layout().layer(TECHNOLOGY['PinRec'])
  it = self.begin_shapes_rec(LayerPinRecN)
  while not(it.at_end()):
    idx = len(pins) # pin index value to be assigned to Pin.idx
    if it.shape().is_path():
      pin_path = it.shape().path.transformed(it.itrans())
      # Find text label (pin name) for this pin
      pin_name = None
      subcell = it.cell()  # cell (component) to which this shape belongs
      iter2 = subcell.begin_shapes_rec_touching(LayerPinRecN, it.shape().bbox())
      while not(iter2.at_end()):
        if iter2.shape().is_text():
          pin_name = iter2.shape().text.string
        iter2.next()
      if pin_name == None:
        raise Exception("Invalid pin detected: %s.\nPins must have a pin name." % pin_path)
      pins.append(Pin(path=pin_path, _type=_globals.PIN_TYPES.OPTICAL, pin_name=pin_name, idx=idx))
#      print( "PinRec, name: %s at (%s)" % (pins[-1].pin_name, pins[-1].center) )
    if it.shape().is_box():
      pin_box = it.shape().box.transformed(it.itrans())
      pin_name = None
      subcell = it.cell()  # cell (component) to which this shape belongs
      iter2 = subcell.begin_shapes_rec_touching(LayerPinRecN, it.shape().bbox())
      while not(iter2.at_end()):
        if iter2.shape().is_text():
          pin_name = iter2.shape().text.string
        iter2.next()
      if pin_name == None:
        raise Exception("Invalid pin detected: %s.\nPins must have a pin name." % pin_path)
      pins.append(Pin(box=pin_box, _type=_globals.PIN_TYPES.ELECTRICAL, pin_name=pin_name, idx=idx))
#      print( "PinRec, name: %s at (%s)" % (pins[-1].pin_name, pins[-1].center) )
      
    it.next()
  return pins
  
def find_pin(self, name):
  from . import _globals
  from .core import Pin
  pins = []
  label = None
  it = self.begin_shapes_rec(self.layout().layer(_globals.TECHNOLOGY['PinRec']))
  while not(it.at_end()):
    if it.shape().is_path():
      pins.append(it.shape().path.transformed(it.itrans()))
    if it.shape().is_text() and it.shape().text.string == name:
      label = it.shape().text.transformed(it.itrans())
    it.next()
    
  if label is None: return None
  
  for pin in pins:
    pts = pin.get_points()
    if (pts[0]+pts[1])*0.5 == pya.Point(label.x, label.y):
      return Pin(pin, _globals.PIN_TYPES.OPTICAL)
    
  return None

# find the pins inside a component
def find_pins_component(self, component):
  pins = self.find_pins()
  for p in pins:
    # add component to the pin
    p.component = component
  return pins

'''
Components:
'''
def find_components(self, verbose=False):
  '''
  Function to traverse the cell's hierarchy and find all the components
  returns list of components (class Component)
  Use the DevRec shapes.  Assumption: One DevRec shape per component.
  
  Find all the DevRec shapes; identify the component it belongs; record the info as a Component 
  for each component instance, also find the Pins and Fibre ports.
  
  Find all the pins for the component, save in components and also return pin list.
  Use the pin names on layer PinRec to sort the pins in alphabetical order
  '''
  if verbose:
    print('*** Cell.find_components:')
  
  components = []

  from .core import Component
  from . import _globals
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  dbu = TECHNOLOGY['dbu']

  # Find all the DevRec shapes
  LayerDevRecN = self.layout().layer(TECHNOLOGY['DevRec'])
  iter1 = self.begin_shapes_rec(LayerDevRecN)
  
  while not(iter1.at_end()):
    idx = len(components) # component index value to be assigned to Component.idx
    subcell = iter1.cell()             # cell (component) to which this shape belongs
    component = subcell.basic_name()   # name library component
    instance = subcell.name      
#    subcell.name                # name of the cell; for PCells, different from basic_name

    found_component = False
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
      box= iter1.shape().box.transformed(iter1.itrans())
      if verbose:
        print("%s: DevRec in cell {%s}, box -- %s; %s" % (idx, subcell.basic_name(), box.p1, box.p2) )
      polygon = pya.Polygon(box) # Save the component outline polygon
      found_component = True
    if iter1.shape().is_polygon():
      polygon = iter1.shape().polygon.transformed(iter1.itrans()) # Save the component outline polygon
      if verbose:
        print("%s: DevRec in cell {%s}, polygon -- %s" % (idx, subcell.basic_name(), polygon))
      found_component = True

    # A component was found. record the instance info as an Optical_component 
    if found_component:
      # Find text label for DevRec, to get Library name
      library = None
      iter2 = subcell.begin_shapes_rec(LayerDevRecN)
      spice_params = ""
      while not(iter2.at_end()):
        if iter2.shape().is_text():
          text = iter2.shape().text
          if verbose:
            print("%s: DevRec label: %s" % (idx, text))
          if text.string.find("Lumerical_INTERCONNECT_library=") > -1:
            library = text.string[len("Lumerical_INTERCONNECT_library="):]
          if text.string.find("Lumerical_INTERCONNECT_component=") > -1:
            component = text.string[len("Lumerical_INTERCONNECT_component="):]
          if text.string.find("Spice_param:") > -1:
            spice_params = text.string[len("Spice_param:"):]
        iter2.next()
      if library == None:
        if verbose:
          print("Missing library information for component: %s" % component )

      # Save the component into the components list      
      components.append(Component(idx=idx, \
         component=component, instance=instance, trans=iter1.trans(), library=library, params=spice_params, polygon=polygon) )

      # find the component pins, and Sort by pin text labels
      pins = sorted(subcell.find_pins_component(components[-1]), key=lambda  p: p.pin_name)

      # find_pins returns pin locations within the subcell; transform to the top cell:
      [p.transform(iter1.trans()) for p in pins]

      # store the pins in the component
      components[-1].pins=pins

    iter1.next()
  # end while iter1 
  return components
# end def find_components
  


def identify_nets(self, verbose=False):
  # function to identify all the nets in the cell layout
  # use the data in Optical_pin, Optical_waveguide to find overlaps
  # and save results in components

  from . import _globals
  from .core import Net

  # output: array of Net[]
  nets = []

  # find components and pins in the cell layout
  components = self.find_components()
  pins = self.find_pins()
  
  # Optical Pins:
  optical_pins = [p for p in pins if p.type==_globals.PIN_TYPES.OPTICAL]
  
  # Loop through all pairs components (c1, c2); only look at touching components
  for c1 in components:
    for c2 in components [ c1.idx+1: len(components) ]:
      if verbose:
        print( " - Components: [%s-%s], [%s-%s]"
          % (c1.component, c1.idx, c2.component, c2.idx) )      

      if c1.polygon.bbox().overlaps(c2.polygon.bbox()) or c1.polygon.bbox().touches(c2.polygon.bbox()):
        # Loop through all the pins (p1) in c1
        # - Compare to all other pins, find other overlapping pins (p2) in c2
        for p1 in c1.pins:
          for p2 in c2.pins:
            if 0:
              print( " - Components, pins: [%s-%s, %s, %s, %s], [%s-%s, %s, %s, %s]"
                % (c1.component, c1.idx, p1.pin_name, p1.center, p1.rotation, c2.component, c2.idx, p2.pin_name, p2.center, p2.rotation) )      
      
            # check that pins are facing each other, 180 degree
            check1 = ((p1.rotation - p2.rotation)%360) == 180
      
            # check that the pin centres are perfectly overlapping 
            # (to avoid slight disconnections, and phase errors in simulations)
            check2 = (p1.center == p2.center)
      
            if check1 and check2:  # found connected pins:
              # make a new optical net index
              net_idx = len(nets)
              # optical net connects two pins; keep track of the pins, Pin[] :
              nets.append ( Net ( idx=net_idx, pins=[p1, p2] ) )
              # assign this net number to the pins
              p1.net = nets[-1]
              p2.net = nets[-1]
              
              if verbose:
                print( " - pin-pin, net: %s, component, pin: [%s-%s, %s, %s, %s], [%s-%s, %s, %s, %s]" 
                  % (net_idx, c1.component, c1.idx, p1.pin_name, p1.center, p1.rotation, c2.component, c2.idx, p2.pin_name, p2.center, p2.rotation) )      
      
  return nets, components


def spice_netlist_export(self, verbose = False):
  # list all Optical_component objects from an array
  # input array, optical_components
  # example output:         
  # X_grating_coupler_1 N$7 N$6 grating_coupler library="custom/genericcml" sch_x=-1.42 sch_y=-0.265 sch_r=0 sch_f=false
  import SiEPIC
  from . import _globals
  from time import strftime 
  from .utils import eng_str


  text_main = '* Spice output from KLayout SiEPIC-Tools v%s, %s.\n\n' % (SiEPIC.__version__, strftime("%Y-%m-%d %H:%M:%S") )
  text_subckt = text_main

  nets, components = self.identify_nets ()

  # convert KLayout GDS rotation/flip to Lumerical INTERCONNECT
  # KLayout defines mirror as an x-axis flip, whereas INTERCONNECT does y-axis flip
  # KLayout defines rotation as counter-clockwise, whereas INTERCONNECT does clockwise
  # input is KLayout Rotation,Flip; output is INTERCONNECT:
  KLayoutInterconnectRotFlip = \
      {(0, False):[0, False], \
       (90, False):[270, False], \
       (180, False):[180, False], \
       (270, False):[90, False], \
       (0, True):[180,True], \
       (90, True):[90, True], \
       (180, True):[0,True], \
       (270, True):[270, False]}
  Lumerical_schematic_scaling=1

  for c in components:
    # optical nets
    nets_str = ''
    for p in c.pins:
      nets_str += " N$" + str(p.net.idx)

    trans = KLayoutInterconnectRotFlip[(c.trans.angle, c.trans.is_mirror())]
     
    flip = ' sch_f=true' if trans[1] else ''
    if trans[0] > 0:
      rotate = ' sch_r=%s' % str(trans[0])
    else:
      rotate = ''
      
    text_subckt += ' %s %s %s ' % ( c.component.replace(' ', '_') +"_"+str(c.idx), nets_str, c.component.replace(' ', '_') ) 
    if c.library != None:
      text_subckt += 'library="%s" ' % c.library
    x, y = c.trans.disp.x, c.trans.disp.y
    text_subckt += '%s lay_x=%s lay_y=%s sch_x=%s sch_y=%s %s%s\n' % \
       ( c.params,
         eng_str(x * 1e-6), eng_str(y * 1e-6), \
         eng_str(x * Lumerical_schematic_scaling), eng_str(y * Lumerical_schematic_scaling), \
         rotate, flip)


  detector_nets=[]
  return text_subckt, text_main, len(detector_nets)


#################################################################################

pya.Cell.print_parameter_values = print_parameter_values
pya.Cell.find_pin = find_pin
pya.Cell.find_pins = find_pins
pya.Cell.find_pins_component = find_pins_component
pya.Cell.find_components = find_components
pya.Cell.identify_nets = identify_nets
pya.Cell.spice_netlist_export = spice_netlist_export

#################################################################################
#                    SiEPIC Class Extension of Instance Class                   #
#################################################################################

# Function Definitions
#################################################################################

def find_pins(self):

  return [pin.transform(self.trans) for pin in self.cell.find_pins()]
  
#################################################################################

pya.Instance.find_pins = find_pins

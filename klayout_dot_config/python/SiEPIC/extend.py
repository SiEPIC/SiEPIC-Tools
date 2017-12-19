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
  points = self.get_points()
  lengths = [ points[i].distance(points[i-1]) for i, pt in enumerate(points) if i > 0]
  return (lengths[0] >= radius) and (lengths[-1] >= radius) and all(length >= 2*radius for length in lengths if length != lengths[0] or length != lengths[-1])

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
    
def snap(self, pins):
  from .utils import angle_vector, get_technology
  from . import _globals
  from math import pi
  TECHNOLOGY = get_technology()
    
  d_min = _globals.PATH_SNAP_PIN_MAXDIST/TECHNOLOGY['dbu'];

  if not len(pins): return
  pts = self.get_points()
  ang = angle_vector(pts[1]-pts[0])
  pins_sorted = sorted([pin for pin in pins if ((pin.rotation - ang)%180) == 0 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[0]))
  if len(pins_sorted):
    dpt = pins_sorted[0].center - pts[0]
    if dpt.abs() <= d_min:
      pts[0] += dpt
      if(round(ang % 180) == 0):
        pts[1].y += dpt.y
      else:
        pts[1].x += dpt.x
  
  ang = angle_vector(pts[-1]-pts[-2])
  pins_sorted = sorted([pin for pin in pins if ((pin.rotation - ang)%180) == 0 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[-1]))
  if len(pins_sorted):
    dpt = pins_sorted[0].center - pts[-1]
    if dpt.abs() <= d_min:
      pts[-1] += dpt
      if(round(ang % 180) == 0):
        pts[-2].y += dpt.y
      else:
        pts[-2].x += dpt.x

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
      pins.append(Pin(pin_path, _globals.PIN_TYPES.OPTICAL, pin_name = pin_name))
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
      pins.append(Pin(box=pin_box, _type=_globals.PIN_TYPES.ELECTRICAL, pin_name = pin_name))
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

'''
Components:
'''
def find_components(self):
  '''
  Function to traverse the cell's hierarchy and find all the components
  returns list of components (class Component)
  Use the DevRec shapes.  Assumption: One DevRec shape per component.
  
  Find all the DevRec shapes; identify the component it belongs; record the info as a Component 
  for each component instance, also find the Pins and Fibre ports.
  
  Use the pin names on layer PinRec to sort the pins in alphabetical order
  '''
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
  
  i=0
  while not(iter1.at_end()):
    i+=1
    subcell = iter1.cell()             # cell (component) to which this shape belongs
    component = subcell.basic_name()   # name library component
    instance = subcell.name      
    subcell.name                # name of the cell; for PCells, different from basic_name

    found_component = False
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
      box= iter1.shape().box.transformed(iter1.itrans())
      print("%s: DevRec in cell {%s}, box -- %s; %s" % (i, subcell.basic_name(), box.p1, box.p2) )
      found_component = True
    if iter1.shape().is_polygon():
      polygon = iter1.shape().polygon.transformed(iter1.itrans())
      print("%s: DevRec in cell {%s}, polygon -- %s" % (i, subcell.basic_name(), polygon))
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
          print("%s: DevRec label: %s" % (i, text))
          if text.string.find("Lumerical_INTERCONNECT_library=") > -1:
            library = text.string[len("Lumerical_INTERCONNECT_library="):]
          if text.string.find("Lumerical_INTERCONNECT_component=") > -1:
            component = text.string[len("Lumerical_INTERCONNECT_component="):]
          if text.string.find("Spice_param:") > -1:
            spice_params = text.string[len("Spice_param:"):]
        iter2.next()
      if library == None:
        print("Missing library information for component: %s" % component )
      # get the cell's x,y coordinates
      x = iter1.trans().disp.x*dbu
      y = iter1.trans().disp.y*dbu
#      flip = iter1.trans().is_mirror()
#      rotate = (int(iter1.trans().rot())*90) % 360
      component_idx = len(components)
      
      # find the component pins, and Sort by pin text labels
      pins = sorted(subcell.find_pins(), key=lambda  p: p.pin_name)
#      [p.display() for p in pins]

      components.append(Component(idx=component_idx, \
         component=component, instance=instance, trans=iter1.trans(), library=library, params=spice_params, pins=pins) )


#            optical_pins.append (Optical_pin (pin_idx, points, component_idx, x, y, 1, pin_info2[p1].pin_text) )
#            optical_components[component_idx].npins += 1
#            optical_components[component_idx].pins.append( pin_idx )

      # reserve space for netlist for this component, based on the number of pins.
#      optical_components[component_idx].nets = [-1] * (optical_components[component_idx].npins)



    # end if found_component   
    iter1.next()
  # end while iter1 
  return components
# end def find_components
  





#################################################################################

pya.Cell.print_parameter_values = print_parameter_values
pya.Cell.find_pin = find_pin
pya.Cell.find_pins = find_pins
pya.Cell.find_components = find_components


#################################################################################
#                    SiEPIC Class Extension of Instance Class                   #
#################################################################################

# Function Definitions
#################################################################################

def find_pins(self):

  return [pin.transform(self.trans) for pin in self.cell.find_pins()]
  
#################################################################################

pya.Instance.find_pins = find_pins

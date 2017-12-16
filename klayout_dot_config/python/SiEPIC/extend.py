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
    
  d_min = 10/TECHNOLOGY['dbu'];

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
        
  self.points = pts

# Path Extension
#################################################################################

if hasattr(pya.Path, "get_points"):
  warning.setText("Warning: The function 'to_points' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.get_points = get_points
  
if hasattr(pya.Path, "get_dpoints"):
  warning.setText("Warning: The function 'to_dpoints' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.get_dpoints = get_dpoints

if hasattr(pya.Path, "is_manhattan"):
  warning.setText("Warning: The function 'is_manhattan' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.is_manhattan = is_manhattan
  
if hasattr(pya.Path, "radius_check"):
  warning.setText("Warning: The function 'radius_check' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.radius_check = radius_check

if hasattr(pya.Path, "remove_colinear_points"):
  warning.setText("Warning: The function 'remove_colinear_points' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.remove_colinear_points = remove_colinear_points

if hasattr(pya.Path, "translate_from_center"):
  warning.setText("Warning: The function 'translate_from_center' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.translate_from_center = translate_from_center

if hasattr(pya.Path, "snap"):
  warning.setText("Warning: The function 'snap' in the class 'Path', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Path.snap = snap;

# DPath Extension
#################################################################################
  
if hasattr(pya.DPath, "get_points"):
  warning.setText("Warning: The function 'to_points' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.get_points = get_points
  
if hasattr(pya.DPath, "get_dpoints"):
  warning.setText("Warning: The function 'to_dpoints' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.get_dpoints = get_dpoints

if hasattr(pya.DPath, "is_manhattan"):
  warning.setText("Warning: The function 'is_manhattan' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.is_manhattan = is_manhattan
  
if hasattr(pya.DPath, "radius_check"):
  warning.setText("Warning: The function 'radius_check' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.radius_check = radius_check

if hasattr(pya.DPath, "remove_colinear_points"):
  warning.setText("Warning: The function 'remove_colinear_points' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.remove_colinear_points = remove_colinear_points

if hasattr(pya.DPath, "translate_from_center"):
  warning.setText("Warning: The function 'translate_from_center' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPath.translate_from_center = translate_from_center

if hasattr(pya.DPath, "snap"):
  warning.setText("Warning: The function 'snap' in the class 'DPath', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
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

if hasattr(pya.Polygon, "get_points"):
  warning.setText("Warning: The function 'to_points' in the class 'Polygon', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Polygon.get_points = get_points;
  
if hasattr(pya.Polygon, "get_dpoints"):
  warning.setText("Warning: The function 'to_dpoints' in the class 'Polygon', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Polygon.get_dpoints = get_dpoints;

#################################################################################

if hasattr(pya.DPolygon, "get_points"):
  warning.setText("Warning: The function 'to_points' in the class 'DPolygon', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.DPolygon.get_points = get_points;
  
if hasattr(pya.DPolygon, "get_dpoints"):
  warning.setText("Warning: The function 'to_dpoints' in the class 'DPolygon', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
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

if hasattr(pya.PCellDeclarationHelper, "print_parameter_list"):
  warning.setText("Warning: The function 'print_parameter_list' in the class 'PCellDeclarationHelper', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
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

def find_pins(self):
  from .utils import get_technology
  from .core import Pin
  from . import _globals

  TECHNOLOGY = get_technology()
  pins = []
  it = self.begin_shapes_rec(TECHNOLOGY['PinRec'])
  while not(it.at_end()):
    if it.shape().is_path():
      pins.append(Pin(it.shape().path.transformed(it.itrans()), _globals.PIN_TYPES.OPTICAL))
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

#################################################################################

if hasattr(pya.Cell, "print_parameter_values"):
  warning.setText("Warning: The function 'get_parameter_values' in the class 'Cell', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Cell.print_parameter_values = print_parameter_values
  
if hasattr(pya.Cell, "find_pin"):
  warning.setText("Warning: The function 'find_pin' in the class 'Cell', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Cell.find_pin = find_pin

if hasattr(pya.Cell, "find_pins"):
  warning.setText("Warning: The function 'find_pins' in the class 'Cell', is already implemented in the KLayout Library.\n\
Redefining might cause instability and will not be performed.")
  warning.exec_()
else:
  pya.Cell.find_pins = find_pins

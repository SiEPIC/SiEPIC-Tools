#################################################################################
#                SiEPIC Class Extension of KLayout PYA Library                  #
#################################################################################
'''
This module extends several pya classes that are useful for the library.

pya.Layout:
  - get_technology: get the technology for the specific layout
  - load_Waveguide_types: load the waveguide types from WAVEGUIDES.XML
  - cell_character_replacement: replace forbidden characters
  
dbu float-int extension:
  - to_dbu and to_itype, convert float (microns) to integer (nanometers) using dbu
  - from_dbu and to_dtype, convert integer (nanometers) to float (microns) using dbu

pya.Path and pya.DPath Extensions:
  - get_points(), returns list of pya.Points
  - get_dpoints(), returns list of pya.DPoints
  - is_manhattan(), tests to see if the path is manhattan
  - is_manhattan_endsegments(), tests to see if the path is manhattan (only the 1st and last segments)
  - radius_check(radius), tests to see of all path segments are long enough to be
    converted to a waveguide with bends of radius 'radius'
  - remove_colinear_points(), removes all colinear points in place
  - unique_points(), remove all but one colinear points
  - translate_from_center(offset), returns a new path whose points have been offset
    by 'offset' from the center of the original path
  - snap(pins), snaps the path in place to the nearest pin
  - snap_m, snaps the path to the nearest metal pins [Karl McNulty, RIT Integrated Photonics Group]
  - to_dtype(dbu), for KLayout < 0.25, integer to float using dbu
  - to_itype(dbu), for KLayout < 0.25, float to integer using dbu

pya.Polygon and pya.DPolygon Extensions:
  - get_points(), returns list of pya.Points
  - get_dpoints(), returns list of pya.DPoints
  - to_dtype(dbu), for KLayout < 0.25, integer using dbu to float

pya.PCellDeclarationHelper Extensions:
  - print_parameter_list, prints parameter list

pya.Cell Extensions:
  - print_parameter_values, if this cell is a pcell, prints the parameter values
  - find_pins: find Pin object of either the specified name or all pins in a cell
  - find_pin
  - find_pins_component
  - find_components
  - identify_nets
  - get_LumericalINTERCONNECT_analyzers
  - get_LumericalINTERCONNECT_analyzers_from_opt_in
  - spice_netlist_export
  - check_component_models
  - pinPoint

pya.Instance Extensions:
  - find_pins: find Pin objects for all pins in a cell instance
  - pinPoint

pya.Point Extensions:
  - to_dtype(dbu): for KLayout < 0.25, convert integer Point using dbu to float DPoint
  - angle_vector
  - to_p(): for KLayout < 0.25

'''
#################################################################################


import pya

from SiEPIC._globals import Python_Env
if Python_Env == "KLayout_GUI":
    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Ok)
    warning.setDefaultButton(pya.QMessageBox.Ok)

#################################################################################
#                SiEPIC Class Extension of Layout Class                         #
#################################################################################

# Gets the technology information and stores it with the layout.
# only loads XML files once, so it is fast
def get_technology(self):
    if 'TECHNOLOGY' not in dir(self):
        from .utils import get_technology_by_name
        self.TECHNOLOGY = get_technology_by_name(self.technology().name)
    return self.TECHNOLOGY

pya.Layout.get_technology = get_technology

# Gets the technology waveguide information and stores it with the layout.
# only loads XML files once, so it is fast
def load_Waveguide_types(self):
    TECHNOLOGY = self.get_technology()
    if 'WaveguideTypes' not in dir(self):
        from .utils import load_Waveguides_by_Tech
        self.WaveguideTypes = load_Waveguides_by_Tech(TECHNOLOGY['technology_name'])
    return self.WaveguideTypes

pya.Layout.load_Waveguide_types = load_Waveguide_types


def cell_character_replacement(self, forbidden_cell_characters = '=', replacement_cell_character = '_'):
    # scan through all cells
#    for i in range(0,self.cells()):
#        cell = self.cell(i)
    for cell in self.cells('*'):
        if True in [c in cell.name for c in forbidden_cell_characters]:
            oldname = cell.name
            for ch in forbidden_cell_characters:
                if ch in cell.name:
                    cell.name = cell.name.replace(ch,replacement_cell_character)
            print(" - cell name: %s, replaced with: %s" % (oldname, cell.name)) 

pya.Layout.cell_character_replacement = cell_character_replacement


#################################################################################
#                SiEPIC Class Extension of Path & DPath Class                   #
#################################################################################

# Function Definitions
#################################################################################


def to_dtype(self, dbu):
    Dpath = pya.DPath(self.get_dpoints(), self.width) * dbu
    Dpath.width = self.width * dbu
    return Dpath


def to_itype(self, dbu):
    path = pya.Path([pt.to_itype(dbu) for pt in self.each_point()], round(self.width / dbu))
    path.width = round(self.width / dbu)
    return path


def get_points(self):
    return [pya.Point(pt.x, pt.y) for pt in self.each_point()]


def get_dpoints(self):
    return [pya.DPoint(pt.x, pt.y) for pt in self.each_point()]


def is_manhattan_endsegments(self):
    if self.__class__ == pya.Path:
        pts = self.get_points()
    else:
        pts = self.get_dpoints()
    check = 1 if len(pts) == 2 else 0
    for i, pt in enumerate(pts):
        if (i == 1 or pts[i] == pts[-1]):
            if(pts[i].x == pts[i - 1].x or pts[i].y == pts[i - 1].y):
                check += 1
    return check == 2


def is_manhattan(self):
    if self.__class__ == pya.Path:
        pts = self.get_points()
    else:
        pts = self.get_dpoints()
    if len(pts) == 2:
        return True
        
    # check that each segment is horizontal or vertical:
    for i, pt in enumerate(pts[0:-1]):
        if not (pts[i].x == pts[i + 1].x or pts[i].y == pts[i + 1].y):
            return False

    # check that all corners are 90 degrees (not 180):
    from SiEPIC.utils import inner_angle_b_vectors
    for i in range(1, len(pts)-1):
        if ((inner_angle_b_vectors(pts[i]-pts[i-1],pts[i+1]-pts[i])+90)%360-90) != 90:
            return False
 
    return True


def radius_check(self, radius, dbu=0.001):
    def all2(iterable):
        for element in iterable:
            if not element:
                return False
        return True

    if type(self) == pya.DPath:
        points = [p.to_itype(dbu) for p in self.get_dpoints() ]
        radius = to_itype(radius,dbu)
    else:
        points = self.get_points()
    
    if len(points) > 2:
    
      lengths = [points[i].distance(points[i - 1]) for i, pt in enumerate(points) if i > 0]
  
      # first and last segment must be >= radius
      check1 = (lengths[0] >= radius)
      check2 = (lengths[-1] >= radius)
      # middle segments must accommodate two bends, hence >= 2 radius
      check3 = [length >= 2 * radius for length in lengths[1:-1]]
      if not(check1 and check2 and all(check3)):
        print('radius check failed')
      return check1 and check2 and all(check3)
    else:
      return True

# remove all colinear points (only keep corners)
def remove_colinear_points(self, verbose=False):
    from .utils import pt_intersects_segment, angle_b_vectors
    if self.__class__ == pya.Path:
        pts = self.get_points()
    else:
        pts = self.get_dpoints()

    # only keep unique path points:
    pts2 = []
    for pt in pts:
        if pt not in pts2:
            pts2.append(pt)
    pts = pts2

    for i in range(1,len(pts)-1):
      turn = ((angle_b_vectors(pts[i]-pts[i-1],pts[i+1]-pts[i])+90)%360-90)/90
      angle = angle_vector(pts[i]-pts[i-1])/90
      if verbose:
        print('%s, %s' %(turn, angle))

    # removed all colinear points
    self.points = [pts[0]] + [pts[i]
                              for i in range(1, len(pts) - 1) if not pt_intersects_segment(pts[i + 1], pts[i - 1], pts[i])] + [pts[-1]]
    return self


def unique_points(self):
    '''
    Check the path and return only the unique points,
    namely eliminate any duplicate points.

    Lukas 2023/11: found using cProfile that previous implementation was very slow,
    which caused waveguide generation to be slow.
        previous: 25 ms per call
        now: 1 ms per call
    '''
    
    if self.__class__ == pya.Path:
        pts = self.get_points()
    else:
        pts = self.get_dpoints()

    # only keep unique path points:
    output = [pts[0]]
    for pt in pts[1:]:
        if pt != output[-1]:
            output.append(pt)
    self.points = output
    return self


def translate_from_center(self, offset):
    from math import pi, cos, sin, acos, sqrt
    from .utils import angle_vector
    pts = [pt for pt in self.get_dpoints()]
    tpts = [pt for pt in self.get_dpoints()]
    for i in range(0, len(pts)):
        if i == 0:
            u = pts[i] - pts[i + 1]
            v = -u
        elif i == (len(pts) - 1):
            u = pts[i - 1] - pts[i]
            v = -u
        else:
            u = pts[i - 1] - pts[i]
            v = pts[i + 1] - pts[i]

        if offset < 0:
            o1 = pya.DPoint(abs(offset) * cos(angle_vector(u) * pi / 180 - pi / 2),
                            abs(offset) * sin(angle_vector(u) * pi / 180 - pi / 2))
            o2 = pya.DPoint(abs(offset) * cos(angle_vector(v) * pi / 180 + pi / 2),
                            abs(offset) * sin(angle_vector(v) * pi / 180 + pi / 2))
        else:
            o1 = pya.DPoint(abs(offset) * cos(angle_vector(u) * pi / 180 + pi / 2),
                            abs(offset) * sin(angle_vector(u) * pi / 180 + pi / 2))
            o2 = pya.DPoint(abs(offset) * cos(angle_vector(v) * pi / 180 - pi / 2),
                            abs(offset) * sin(angle_vector(v) * pi / 180 - pi / 2))

        p1 = u + o1
        p2 = o1
        p3 = v + o2
        p4 = o2
        d = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)

        if round(d, 10) == 0:
            tpts[i] += p2
        else:
            tpts[i] += pya.DPoint(((p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x)) / d,
                                  ((p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x)) / d)

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
    d_min = _globals.PATH_SNAP_PIN_MAXDIST / TECHNOLOGY['dbu']

    if not len(pins):
        return

    # array of path vertices:
    pts = self.get_points()
    
    def snap_endpoints(input_pts, pins):  
      # function takes a list of pts of a path, and finds the closest pins to this path
      # provided that the pins are facing the correct way to make a connection.
      # when you have 2 or more segments (3 or more points), then you can move the two end segments.
    
      pts = input_pts

      # find closest pin to the first pts
      # angle of first segment:
      ang = angle_vector(pts[0] - pts[1])
      # sort all the pins based on distance to the Path endpoint
      # only consider pins that are facing each other, 180 degrees
      pins_sorted = sorted([pin for pin in pins if round((ang - pin.rotation) % 360) ==
                            180 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[0]))

      # angle of last segment (do it before moving the pts)
      ang2 = angle_vector(pts[-1] - pts[-2])
  
      if len(pins_sorted):
          # pins_sorted[0] is the closest one
          dpt = pins_sorted[0].center - pts[0]
          # print(dpt)
          # check if the pin is close enough to the path endpoint
          if dpt.abs() <= d_min:
              # snap the endpoint to the pin
              pts[0] += dpt
              # move the first corner
              if(round(ang % 180) == 0):
                  pts[1].y += dpt.y
              else:
                  pts[1].x += dpt.x
              snapped0 = True
          else:
              snapped0 = False
      else:
          snapped0 = False
  
      # find closest pin to the last pts
      # do the same thing on the other end (check that it isn't the same pin as above):
      pins_sorted = sorted([pin for pin in pins if round((ang2 - pin.rotation) % 360) ==
                            180 and pin.type == _globals.PIN_TYPES.OPTICAL], key=lambda x: x.center.distance(pts[-1]))
      if len(pins_sorted):
          if (len(pts)==2) and (pins_sorted[0].center != pts[-1]):
              snapped1 = False
        
          elif pins_sorted[0].center != pts[0]:
              dpt = pins_sorted[0].center - pts[-1]
              if dpt.abs() <= d_min:
                  # snap the endpoint to the pin
                  pts[-1] += dpt
                  # move the last corner
                  if(round(ang2 % 180) == 0):
                      pts[-2].y += dpt.y
                  else:
                      pts[-2].x += dpt.x
              snapped1 = True
          else:
              snapped1 = False
      else:
          snapped1 = False

      # check that the path has non-zero length after the snapping operation
      test_path = pya.Path()
      test_path.points = pts
      if test_path.length() > 0:
          return_pts = pts
      else:
          return_pts = input_pts
          
      # return flag to track whether BOTH endpoints were snapped
      return return_pts, snapped0 & snapped1


    # Perform snapping:
    if len(pts) > 2:
      newpoints, snapped_both = snap_endpoints(pts, pins)
      # for a previously created extra 2 vertices, check if we still need it
      if len(pts) == 4:
        if newpoints[1] == newpoints[2]:
            newpoints = [ newpoints[0],newpoints[3] ]
      self.points = newpoints  
    elif len(pts) == 2:
      # call the snap and check if it worked. case where the two components' pins are already aligned
      newpoints, snapped_both = snap_endpoints(pts, pins)
      if snapped_both:
        self.points = newpoints
        return True
      else:
        # - snapping failed; case where the two components' pins are not aligned
        # - need to add extra vertices; assume we add two more points, that we have scenario of an S-Bend
        #   - add two more points,  
        ang = angle_vector(pts[0] - pts[1])
        if(round(ang % 180) == 0):
            # move the y coordinate
            newpoints = [ pts[0], 
                          pya.Point((pts[0].x+pts[1].x)/2, pts[0].y),
                          pya.Point((pts[0].x+pts[1].x)/2, pts[1].y),
                          pts[1] ]
        else:
            # move the x coordinate
            newpoints = [ pts[0], 
                          pya.Point(pts[0].x, (pts[0].y+pts[1].y)/2),
                          pya.Point(pts[0].x, (pts[0].y+pts[1].y)/2),
                          pts[1] ]

        #  - call the snap and check if it worked.  
        newpoints, snapped_both = snap_endpoints(newpoints, pins)
        if snapped_both:
          self.points = newpoints
          # if we still need the 2 extra
          if newpoints[1] == newpoints[2]:
             newpoints = [ newpoints[0],newpoints[3] ]
          #  - snapping successful:
          self.points = newpoints  
          return True
        else:
          return False  
    else:
      return False
      

'''
snap_m - pya.Path extension
This function snaps the two metal path endpoints to the nearest metal pins by adjusting the end segments
Author: Karl McNulty, RIT Integrated Photonics Group
Input:
 - self: the Path object
 - pins: an array of Pin objects, which are paths with 2 points,
         with the vector giving the direction (out of the component)
Output:
 - modifies the original Path

'''


def snap_m(self, pins):
    # Import functionality from SiEPIC-Tools:
    from .utils import angle_vector, get_technology
    from . import _globals
    TECHNOLOGY = get_technology()

    # Search for pins within this distance to the path endpoints, e.g., 10 microns
    d_min = _globals.PATH_SNAP_PIN_MAXDIST / TECHNOLOGY['dbu']

    if not len(pins):
        return

    # array of path vertices:
    pts = self.get_points()
    
    def snap_endpoints(input_pts, pins):  
      # function takes a list of pts of a path, and finds the closest pins to this path
      # provided that the pins are facing the correct way to make a connection.
      # when you have 2 or more segments (3 or more points), then you can move the two end segments.
    
      pts = input_pts

      # find closest pin to the first pts
      # angles of all segments:
      ang = angle_vector(pts[0] - pts[1])
      # sort all the pins based on distance to the Path endpoint
      # only consider pins that are facing each other, 180 degrees
      pins_sorted = sorted([pin for pin in pins if round((ang - pin.rotation) % 360) ==
                            180 and pin.type == _globals.PIN_TYPES.ELECTRICAL], key=lambda x: x.center.distance(pts[0]))
  
      if len(pins_sorted):
          # pins_sorted[0] is the closest one
          dpt = pins_sorted[0].center - pts[0]
          # check if the pin is close enough to the path endpoint
          if dpt.abs() <= d_min:
              # snap the endpoint to the pin
              pts[0] += dpt
              if len(pts)>2:
                # move the first corner
                if(round(ang % 180) == 0):
                    pts[1].y += dpt.y
                else:
                    pts[1].x += dpt.x
              snapped0 = True
          else:
              snapped0 = False
      else:
          snapped0 = False
  
      # find closest pin to the last pts
      # do the same thing on the other end (check that it isn't the same pin as above):
      ang = angle_vector(pts[-1] - pts[-2])
      pins_sorted = sorted([pin for pin in pins if round((ang - pin.rotation) % 360) ==
                            180 and pin.type == _globals.PIN_TYPES.ELECTRICAL], key=lambda x: x.center.distance(pts[-1]))
      if len(pins_sorted):
          if pins_sorted[0].center != pts[0]:
              dpt = pins_sorted[0].center - pts[-1]
              if dpt.abs() <= d_min:
                  # snap the endpoint to the pin
                  pts[-1] += dpt
                  if len(pts)>2:
                    # move the last corner
                    if(round(ang % 180) == 0):
                        pts[-2].y += dpt.y
                    else:
                        pts[-2].x += dpt.x
              snapped1 = True
          else:
              snapped1 = False
      else:
          snapped1 = False

      # check that the path has non-zero length after the snapping operation
      test_path = pya.Path()
      test_path.points = pts
      if test_path.length() > 0:
          return_pts = pts
      else:
          return_pts = input_pts
          
      # return flag to track whether BOTH endpoints were snapped
      return return_pts, snapped0 & snapped1

    if len(pts) > 2:
      self.points, snapped_both = snap_endpoints(pts, pins)
    elif len(pts) == 2:
      # call the snap and check if it worked. case where the two components' pins are already aligned
      newpoints, snapped_both = snap_endpoints(pts, pins)
      if snapped_both:
        self.points = newpoints
        return True
      else:
        # - snapping failed; case where the two components' pins are not aligned
        # - need to add extra vertices; assume we add two more points, that we have scenario of an S-Bend
        #   - add two more points,  
        ang = angle_vector(pts[0] - pts[1])
        if(round(ang % 180) == 0):
            # move the y coordinate
            newpoints = [ pts[0], 
                          pya.Point((pts[0].x+pts[1].x)/2, pts[0].y),
                          pya.Point((pts[0].x+pts[1].x)/2, pts[1].y),
                          pts[1] ]
        else:
            # move the x coordinate
            newpoints = [ pts[0], 
                          pya.Point((pts[0].x+pts[1].x)/2, pts[0].y),
                          pya.Point((pts[0].x+pts[1].x)/2, pts[1].y),
                          pts[1] ]

        #  - call the snap and check if it worked.  
        newpoints, snapped_both = snap_endpoints(newpoints, pins)
        if snapped_both:
          #  - snapping successful, added 2 points.  
          self.points = newpoints
          return True
        else:
          return False  
    else:
      return False

# Path Extension
#################################################################################

pya.Path.to_dtype = to_dtype
pya.Path.to_itype = to_itype
pya.Path.get_points = get_points
pya.Path.get_dpoints = get_dpoints
pya.Path.is_manhattan_endsegments = is_manhattan_endsegments
pya.Path.is_manhattan = is_manhattan
pya.Path.radius_check = radius_check
pya.Path.remove_colinear_points = remove_colinear_points
pya.Path.unique_points = unique_points
pya.Path.translate_from_center = translate_from_center
pya.Path.snap = snap
pya.Path.snap_m = snap_m # function added by Karl McNulty (RIT) for metal pin functionality

# DPath Extension
#################################################################################

pya.DPath.to_itype = to_itype
pya.DPath.to_dtype = to_dtype
pya.DPath.get_points = get_points
pya.DPath.get_dpoints = get_dpoints
pya.DPath.is_manhattan_endsegments = is_manhattan_endsegments
pya.DPath.is_manhattan = is_manhattan
pya.DPath.radius_check = radius_check
pya.DPath.remove_colinear_points = remove_colinear_points
pya.DPath.unique_points = unique_points
pya.DPath.translate_from_center = translate_from_center
pya.DPath.snap = snap

#################################################################################
#            SiEPIC Class Extension of Polygon & DPolygon Class                 #
#################################################################################

# Function Definitions
#################################################################################


def get_points(self):
    return [pya.Point(pt.x, pt.y) for pt in self.each_point_hull()]


def get_dpoints(self):
    return [pya.DPoint(pt.x, pt.y) for pt in self.each_point_hull()]


def to_dtype(self, dbu):
    pts = self.get_points()
    pts1 = [p.to_dtype(dbu) for p in pts]
    return pya.DPolygon(pts1)

#################################################################################

pya.Polygon.get_points = get_points
pya.Polygon.get_dpoints = get_dpoints
pya.Polygon.to_dtype = to_dtype

#################################################################################

pya.DPolygon.get_points = get_points
pya.DPolygon.get_dpoints = get_dpoints

#################################################################################
#                    SiEPIC Class Extension of PCell Class                      #
#################################################################################

# Function Definitions
#################################################################################


def print_parameter_list(self):
    types = ['TypeBoolean', 'TypeDouble', 'TypeInt', 'TypeLayer',
             'TypeList', 'TypeNone', 'TypeShape', 'TypeString']
    for p in self.get_parameters():
        if ~p.readonly:
            print("Name: %s, %s, unit: %s, default: %s, description: %s%s" %
                  (p.name, types[p.type], p.unit, p.default, p.description, ", hidden" if p.hidden else "."))

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
        print("Parameter: %s, Value: %s" % (key, params[key]) )



def find_pins(self, verbose=False, polygon_devrec=None, GUI=False):
    '''
    Find Pins in a Cell.
    Optical Pins have:
     1) path on layer PinRec, indicating direction (out of component)
     2) text on layer PinRec, inside the path
    Electrical Pins have:
     1) path on layer PinRecM, indicating direction (out of component)
     2) text on layer PinRecM, inside the path
    
    Old version:
    Electrical Pins have:
     1) box on layer PinRec
     2) text on layer PinRec, inside the box
    '''

    if verbose:
        print("SiEPIC.extend(cell).find_pins()")

    from .core import Pin
    from . import _globals
    from .utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name(self.layout().technology().name)

    # array to store Pin objects
    pins = []

    # Pin Recognition layer
    if not 'PinRec' in TECHNOLOGY:
        if GUI:
            pya.MessageBox.warning(
                "Problem with Technology", "Problem with active Technology %s: missing layer PinRec" % (TECHNOLOGY['technology_name']), pya.MessageBox.Ok)
            return
        else:
            raise Exception("Problem with active Technology %s: missing layer PinRec" % (TECHNOLOGY['technology_name']))
            
    LayerPinRecN = self.layout().layer(TECHNOLOGY['PinRec'])

    error_text = ''
    pin_errors = []

    # iterate through all the PinRec shapes in the cell
    it = self.begin_shapes_rec(LayerPinRecN)
    while not(it.at_end()):
        #    if verbose:
        #      print(it.shape().to_s())
        # Assume a PinRec Path is an optical pin
        if it.shape().is_path():
            
            if it.shape().cell.name != self.name:
                if verbose:
                    print('%s not %s' % (it.shape().cell.name, self.name) )
            if verbose:
                print("Path: %s" % it.shape())
            # get the pin path
            pin_path = it.shape().path.transformed(it.itrans())
            # Find text label (pin name) for this pin by searching inside the Path bounding box
            # Text label must be on DevRec layer
            pin_name = None
            subcell = it.cell()  # cell (component) to which this shape belongs
            iter2 = subcell.begin_shapes_rec_touching(LayerPinRecN, it.shape().bbox())
            while not(iter2.at_end()):
                if iter2.shape().is_text():
                    pin_name = iter2.shape().text.string
                iter2.next()
                if pin_name and pin_path.num_points()==2:
                  # Store the pin information in the pins array
                  pins.append(Pin(path=pin_path, _type=_globals.PIN_TYPES.OPTICAL, pin_name=pin_name))
            if pin_name == None or pin_path.num_points()!=2:
                print("Invalid pin Path detected: %s. Cell: %s" % (pin_path, subcell.name))
                error_text += ("Invalid pin Path detected: %s, in Cell: %s, Optical Pins must have a pin name.\n" %
                               (pin_path, subcell.name))
                pin_errors.append ([pin_path, subcell]) 
                #        raise Exception("Invalid pin Path detected: %s, in Cell: %s.\nOptical Pins must have a pin name." % (pin_path, subcell.name))

        # Assume a PinRec Box is an electrical pin
        # similar to optical pin
        if it.shape().is_simple_polygon():
            pin_box = it.shape().bbox().transformed(it.itrans())
        if it.shape().is_box():
            if verbose:
                print("Box: %s" % it.shape())
            pin_box = it.shape().box.transformed(it.itrans())
        if it.shape().is_simple_polygon() or it.shape().is_box():
            pin_name = None
            subcell = it.cell()  # cell (component) to which this shape belongs
            iter2 = subcell.begin_shapes_rec_touching(LayerPinRecN, it.shape().bbox())
            if verbose:
                print("Box: %s" % it.shape().bbox())
            while not(iter2.at_end()):
                if verbose:
                    print("shape touching: %s" % iter2.shape())
                if iter2.shape().is_text():
                    pin_name = iter2.shape().text.string
                iter2.next()
            if pin_name == None:
                error_text += ("Invalid pin Box detected: %s, Cell: %s, Electrical Pins must have a pin name.\n" %
                               (pin_box, subcell.name))
#        raise Exception("Invalid pin Box detected: %s.\nElectrical Pins must have a pin name." % pin_box)
            pins.append(Pin(box=pin_box, _type=_globals.PIN_TYPES.ELECTRICAL, pin_name=pin_name))

        it.next()

    # Optical IO (Fibre) Recognition layer
    if not 'FbrTgt' in TECHNOLOGY:
        if GUI:
            pya.MessageBox.warning(
                "Problem with Technology", "Problem with active Technology %s: missing layer FbrTgt"% (TECHNOLOGY['technology_name']), pya.MessageBox.Ok)
            return
        else:
            raise Exception("Problem with active Technology %s: missing layer FbrTgt" % (TECHNOLOGY['technology_name']))

    LayerFbrTgtN = self.layout().layer(TECHNOLOGY['FbrTgt'])

    # iterate through all the FbrTgt shapes in the cell
    it = self.begin_shapes_rec(LayerFbrTgtN)
    while not(it.at_end()):
        # Assume a FbrTgt Path is an optical pin
        if it.shape().is_polygon():
            pin_name = self.basic_name().replace(' ', '_')  # default name from the cell
            # or find the text label for the optical IO name
            subcell = it.cell()  # cell (component) to which this shape belongs
            iter2 = subcell.begin_shapes_rec_touching(LayerFbrTgtN, it.shape().bbox())
            if verbose:
                print("Box: %s" % it.shape().bbox())
            while not(iter2.at_end()):
                if verbose:
                    print("shape touching: %s" % iter2.shape())
                if iter2.shape().is_text():
                    pin_name = iter2.shape().text.string
                iter2.next()
            # Store the pin information in the pins array
            # check if this one already exists (duplicate polygons)
            if not([p for p in pins if p.type == _globals.PIN_TYPES.OPTICALIO and 
              p.polygon == it.shape().polygon.transformed(it.itrans())]):
                pins.append(Pin(polygon=it.shape().polygon.transformed(it.itrans()),
                            _type=_globals.PIN_TYPES.OPTICALIO,
                            pin_name=pin_name))
        it.next()

    # Metal Pin Recognition layer 
    if not 'PinRecM' in TECHNOLOGY:
        if GUI:
            pya.MessageBox.warning(
                "Problem with Technology", "Problem with active Technology %s: missing layer PinRecM" % (TECHNOLOGY['technology_name']), pya.MessageBox.Ok)
            return
        else:
            raise Exception("Problem with active Technology %s: missing layer PinRecM" % (TECHNOLOGY['technology_name']))

    try:    
        LayerPinRecN = self.layout().layer(TECHNOLOGY['PinRecM'])
    except:
        LayerPinRecN = self.layout().layer(TECHNOLOGY['PinRec'])

    # Metal Pin recognition by Karl McNulty (RIT)
    # iterate through all the PinRec shapes in the cell
    it = self.begin_shapes_rec(LayerPinRecN)
    while not(it.at_end()):
        #    if verbose:
        #      print(it.shape().to_s())
        # Assume a PinRec Path is an optical pin
        if it.shape().is_path():
            if verbose:
                print("Path: %s" % it.shape())
            # get the pin path
            pin_path = it.shape().path.transformed(it.itrans())
            # Find text label (pin name) for this pin by searching inside the Path bounding box
            # Text label must be on DevRec layer
            pin_name = None
            subcell = it.cell()  # cell (component) to which this shape belongs
            iter2 = subcell.begin_shapes_rec_touching(LayerPinRecN, it.shape().bbox())
            while not(iter2.at_end()):
                if iter2.shape().is_text():
                    pin_name = iter2.shape().text.string
                iter2.next()
            if pin_name == None or pin_path.num_points()!=2:
                print("Invalid pin Path detected: %s. Cell: %s" % (pin_path, subcell.name))
                error_text += ("Invalid pin Path detected: %s, in Cell: %s, Optical Pins must have a pin name.\n" %
                               (pin_path, subcell.name))
            # raise Exception("Invalid pin Path detected: %s, in Cell: %s.\nElectrical Pins must have a pin name." % (pin_path, subcell.name))
            else:
              # Store the pin information in the pins array
              pins.append(Pin(path=pin_path, _type=_globals.PIN_TYPES.ELECTRICAL, pin_name=pin_name))
        it.next()

    if error_text:
        if GUI:
            pya.MessageBox.warning("Problem with component pin:\n", error_text, pya.MessageBox.Ok)

    # return the array of pins
    return pins, pin_errors


def find_pin(self, name):
    from . import _globals
    from .core import Pin
    pins = []
    label = None
    from .utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name(self.layout().technology().name)
    it = self.begin_shapes_rec(self.layout().layer(TECHNOLOGY['PinRec']))
    while not(it.at_end()):
        if it.shape().is_path():
            pins.append(it.shape().path.transformed(it.itrans()))
        if it.shape().is_text() and it.shape().text.string == name:
            label = it.shape().text.transformed(it.itrans())
        it.next()

    if label is None:
        return None

    for pin in pins:
        pts = pin.get_points()
        if (pts[0] + pts[1]) * 0.5 == pya.Point(label.x, label.y):
            return Pin(pin, _globals.PIN_TYPES.OPTICAL)

    return None

# find the pins inside a component


def find_pins_component(self, component):
    pins, _ = self.find_pins()
    for p in pins:
        # add component to the pin
        p.component = component
    return pins


'''
Components:
'''
def find_components(self, cell_selected=None, inst=None, verbose=False):
    '''
    Function to traverse the cell's hierarchy and find all the components
    returns list of components (class Component)
    Use the DevRec shapes.  
    Assumption: One DevRec shape per component.  We return a component "Flattened" if more than one DevRec is found

    Find all the DevRec shapes; identify the component it belongs; record the info as a Component
    for each component instance, also find the Pins and Fibre ports.

    Find all the pins for the component, save in components and also return pin list.
    Use the pin names on layer PinRec to sort the pins in alphabetical order

    cell_selected: only find components that match this specific cell.
    
    inst: return only the component that matches the instance inst

    '''
    if verbose:
        print('*** Cell.find_components:')

    if cell_selected != None and type(cell_selected) != type([]):
          cell_selected=[cell_selected]

    components = []

    from .core import Component
    from . import _globals
    from .utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name(self.layout().technology().name)
    dbu = TECHNOLOGY['dbu']

    # Find all the DevRec shapes
    LayerDevRecN = self.layout().layer(TECHNOLOGY['DevRec'])

    # if we are requesting a specific instance, narrow down the search to the specific area
    if inst:
        iter1 = self.begin_shapes_rec_overlapping(LayerDevRecN, inst.bbox())
    # otherwise search for all components in the cell:
    else:
        iter1 = self.begin_shapes_rec(LayerDevRecN)
        
    component_matched = []

    while not(iter1.at_end()):
        idx = len(components)  # component index value to be assigned to Component.idx
        component_ID = idx
        subcell = iter1.cell()  # cell (component) to which this shape belongs
        if cell_selected and not subcell in cell_selected:
            # check if subcell is one of the arguments to this function: cell_selected
            iter1.next()
            continue
        component = subcell.basic_name().replace(' ', '_')   # name library component
        instance = subcell.name

        found_component = False
        # DevRec must be either a Box or a Polygon:
        if iter1.shape().is_box():
            box = iter1.shape().box.transformed(iter1.itrans())
            if verbose:
                print("%s: DevRec in cell {%s}, box -- %s; %s" %
                      (idx, subcell.basic_name(), box.p1, box.p2))
            polygon = pya.Polygon(box)  # Save the component outline polygon
            DevRec_polygon = pya.Polygon(iter1.shape().box)
            found_component = True
        if iter1.shape().is_polygon():
            polygon = iter1.shape().polygon.transformed(iter1.itrans())  # Save the component outline polygon
            DevRec_polygon = iter1.shape().polygon
            if verbose:
                print("%s: DevRec in cell {%s}, polygon -- %s" %
                      (idx, subcell.basic_name(), polygon))
            found_component = True

        # A component was found. record the instance info as a Component
        if found_component:
            # check if the component is flattened, or a hierarchical sub-cell
            iter3 = subcell.begin_shapes_rec_overlapping(LayerDevRecN, subcell.bbox())
            num_devrec = 0
            while not(iter3.at_end()):
                if iter3.shape().is_box() or iter3.shape().is_polygon():
                    num_devrec += 1
                iter3.next()
                
            if num_devrec > 1:
                print(' * Warning: cell (%s) contains multiple (%s) DevRec layers, suggesting that a cell contains subcells with DevRec layers.' % (instance, num_devrec))
#                # Save the flattened component into the components list
#                components.append(Component(component="Flattened", basic_name="Flattened",
#                                            idx=idx, polygon=polygon, trans=iter1.trans()))
            if 1:
                '''
                # the following doesn't work hierarchically... problem:
                print('inst:')
                instances = [i for i in self.each_overlapping_inst(polygon.bbox().enlarged(-200,-200))]
                if len(instances)==1:
                  instance = instances[0]
                  print('  * Component Instance: %s, %s, %s' % (instance, instance.bbox(), polygon.bbox().enlarged(-200,-200)))
                elif len(instances)>1:
                  print('    more than one Instance overlapping')
                  for instance in instances:
                    print('    Instance: %s, %s' % (instance, polygon.bbox().enlarged(-200,-200)))
                '''  
            
                # Find text label for DevRec, to get Library name
                library = None
                # *** use of subcell assumes that the shapes are hierarchical within the component
                # for flat layout... check within the DevRec shape.
                iter2 = subcell.begin_shapes_rec(LayerDevRecN)
                spice_params = ""
                library, cellName = None, None
                waveguide_type = None
                while not(iter2.at_end()):
                    if iter2.shape().is_text():
                        text = iter2.shape().text
                        if verbose:
                            print("%s: DevRec label: %s" % (idx, text))
                        if text.string.find("Lumerical_INTERCONNECT_library=") > -1:
                            library = text.string[len("Lumerical_INTERCONNECT_library="):]
                        if text.string.find("Lumerical_INTERCONNECT_component=") > -1:
                            component = text.string[len("Lumerical_INTERCONNECT_component="):]
                        if text.string.find("Component=") > -1:
                            component = text.string[len("Component="):]
                        if text.string.find("cellName=") > -1:
                            cellName = text.string[len("cellName="):]
                        if text.string.find("Component_ID=") > -1:
                            cID = int(text.string[len("Component_ID="):])
                            if cID > 0:
                                component_ID = cID
                        if text.string.find("Spice_param:") > -1:
                            spice_params = text.string[len("Spice_param:"):]
                        if text.string.find("waveguide_type=") > -1:
                            waveguide_type = text.string[len("waveguide_type="):]
                    iter2.next()
                if library == None:
                    if verbose:
                        print("Missing library information for component: %s" % component)
                if cellName == None:
                    cellName = subcell.basic_name()

                # Save the component into the components list
                components.append(Component(idx=idx,
                       component=component, instance=instance, 
                       trans=iter1.trans(), library=library, 
                       params=spice_params, polygon=polygon, 
                       DevRec_polygon=DevRec_polygon, cell=subcell, 
                       basic_name=subcell.basic_name(), cellName=cellName, 
                       waveguide_type=waveguide_type))

                # find the component pins, and Sort by pin text labels
                pins = sorted(subcell.find_pins_component(
                    components[-1]), key=lambda p:  '' if p.pin_name == None else p.pin_name)

                # find_pins returns pin locations within the subcell; transform to the top cell:
                [p.transform(iter1.trans()) for p in pins]

                # store the pins in the component
                components[-1].pins = pins

        # find the component that matches the requested instance
        if inst and components:
            if components[-1].basic_name == inst.cell.basic_name() and components[-1].trans==inst.trans:
                if verbose:
                    print('Found requested Inst (exact name and origin match): %s' % inst.trans)
                return components[-1]
        
            if components[-1].trans==inst.trans:
                if verbose:
                    print('Found requested Inst (origin match only, could be the right one): %s' % inst.trans)
                component_matched = components[-1]

        iter1.next()
    # end while iter1
    
    if component_matched:
        return component_matched
    
    return components
# end def find_components


def identify_nets(self, verbose=False):
    # function to identify all the nets in the cell layout
    # use the data in Optical_pin, Optical_waveguide to find overlaps
    # and save results in components

    if verbose:
        print("SiEPIC.extend.identify_nets():")

    from . import _globals
    from .core import Net

    # output: array of Net[]
    nets = []

    # find components and pins in the cell layout
    components = self.find_components()
    pins, _ = self.find_pins()

    # Optical Pins:
    optical_pins = [p for p in pins if p.type == _globals.PIN_TYPES.OPTICAL]

    # Loop through all pairs components (c1, c2); only look at touching components
    for c1 in components:
        for c2 in components[c1.idx + 1: len(components)]:
            if verbose:
                print(" - Components: [%s-%s], [%s-%s].  Pins: %s, %s"
                      % (c1.component, c1.idx, c2.component, c2.idx, c1.pins, c2.pins))

            if c1.polygon.bbox().overlaps(c2.polygon.bbox()) or c1.polygon.bbox().touches(c2.polygon.bbox()):
                # Loop through all the pins (p1) in c1
                # - Compare to all other pins, find other overlapping pins (p2) in c2
                for p1 in [p for p in c1.pins if p.type == _globals.PIN_TYPES.OPTICAL]:
                    for p2 in [p for p in c2.pins if p.type == _globals.PIN_TYPES.OPTICAL]:
                        if verbose:
                            print(" - Components, pins: [%s-%s, %s, %s, %s], [%s-%s, %s, %s, %s]; difference: %s"
                                  % (c1.component, c1.idx, p1.pin_name, p1.center, p1.rotation, c2.component, c2.idx, p2.pin_name, p2.center, p2.rotation, p1.center - p2.center))
                        # check that pins are facing each other, 180 degree
                        check1 = ((p1.rotation - p2.rotation) % 360) == 180

                        # check that the pin centres are perfectly overlapping
                        # (to avoid slight disconnections, and phase errors in simulations)
                        check2 = (p1.center == p2.center)

                        if check1 and check2:  # found connected pins:
                            # make a new optical net index
                            net_idx = len(nets)
                            # optical net connects two pins; keep track of the pins, Pin[] :
                            nets.append(
                                Net(idx=net_idx, pins=[p1, p2], _type=_globals.PIN_TYPES.OPTICAL))
                            # assign this net number to the pins
                            p1.net = nets[-1]
                            p2.net = nets[-1]

                            if verbose:
                                print(" - pin-pin, net: %s, component, pin: [%s-%s, %s, %s, %s], [%s-%s, %s, %s, %s]"
                                      % (net_idx, c1.component, c1.idx, p1.pin_name, p1.center, p1.rotation, c2.component, c2.idx, p2.pin_name, p2.center, p2.rotation))

    return nets, components


# data structure used to find the detectors and which optical nets they are connected to.
class Detector_info:

    def __init__(self, detector_net, detector_number):
        self.detector_net = detector_net
        self.detector_number = detector_number


def get_LumericalINTERCONNECT_analyzers(self, components, verbose=None):
    """
    Find - LumericalINTERCONNECT_Laser
         - LumericalINTERCONNECT_Detector
    get their parameters
    determine which OpticalIO they are connected to, and find their nets
    Assume that the detectors and laser are on the topcell (not subcells); don't perform transformations.

    returns: parameters, nets in order

    usage:
    laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, ignoreOpticalIOs = get_LumericalINTERCONNECT_analyzers(topcell, components)
    """

    topcell = self

    from . import _globals
    from .utils import select_paths, get_technology
    from .core import Net
    TECHNOLOGY = get_technology()

    layout = topcell.layout()
    LayerLumericalN = self.layout().layer(TECHNOLOGY['Lumerical'])

    # default is the 1st polarization
    orthogonal_identifier = 1

    # Find the laser and detectors in the layout.
    iter1 = topcell.begin_shapes_rec(LayerLumericalN)
    n_IO = 0
    detectors_info = []
    laser_net = None
    wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs = 0, 0, 0, 0, 0
    while not(iter1.at_end()):
        subcell = iter1.cell()             # cell (component) to which this shape belongs
        if iter1.shape().is_box():
            box = iter1.shape().box.transformed(iter1.itrans())
            if iter1.cell().basic_name() == ("Lumerical_INTERCONNECT_Detector"):
                n_IO += 1
                # *** todo read parameters from Text labels rather than PCell:
                detector_number = subcell.pcell_parameters_by_name()["number"]
                if verbose:
                    print("%s: Detector {%s} %s, box -- %s; %s" %
                          (n_IO, subcell.basic_name(), detector_number, box.p1, box.p2))
                # find components which have an IO pin inside the Lumerical box:
                components_IO = [c for c in components if any(
                    [box.contains(p.center) for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO])]
                if len(components_IO) > 1:
                    raise Exception("Error - more than 1 optical IO connected to the detector.")
                if len(components_IO) == 0:
                    print("Warning - No optical IO connected to the detector.")
#          raise Exception("Error - 0 optical IO connected to the detector.")
                else:
                    p = [p for p in components_IO[0].pins if p.type == _globals.PIN_TYPES.OPTICALIO]
                    p[0].pin_name += '_detector' + str(n_IO)
                    p[0].net = Net(idx=p[0].pin_name, pins=p)
                    detectors_info.append(Detector_info(p[0].net, detector_number))
                    if verbose:
                        print(" - pin_name: %s" % (p[0].pin_name))

            if iter1.cell().basic_name() == ("Lumerical_INTERCONNECT_Laser"):
                n_IO += 1
                # *** todo read parameters from Text labels rather than PCell:
                wavelength_start = subcell.pcell_parameters_by_name()["wavelength_start"]
                wavelength_stop = subcell.pcell_parameters_by_name()["wavelength_stop"]
                wavelength_points = subcell.pcell_parameters_by_name()["npoints"]
                orthogonal_identifier = subcell.pcell_parameters_by_name()["orthogonal_identifier"]
                ignoreOpticalIOs = subcell.pcell_parameters_by_name()["ignoreOpticalIOs"]
                if verbose:
                    print("%s: Laser {%s}, box -- %s; %s" %
                          (n_IO, subcell.basic_name(), box.p1, box.p2))
                # find components which have an IO pin inside the Lumerical box:
                components_IO = [c for c in components if any(
                    [box.contains(p.center) for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO])]
                if len(components_IO) > 1:
                    raise Exception("Error - more than 1 optical IO connected to the laser.")
                if len(components_IO) == 0:
                    print("Warning - No optical IO connected to the laser.")
#          raise Exception("Error - 0 optical IO connected to the laser.")
                else:
                    p = [p for p in components_IO[0].pins if p.type == _globals.PIN_TYPES.OPTICALIO]
                    p[0].pin_name += '_laser' + str(n_IO)
                    laser_net = p[0].net = Net(idx=p[0].pin_name, pins=p)
                    if verbose:
                        print(" - pin_name: %s" % (p[0].pin_name))

        iter1.next()

    # Sort the detectors:
    detectors_info2 = sorted(detectors_info, key=lambda d: d.detector_number)

    # output:
    detector_nets = []
    for d in detectors_info2:
        detector_nets.append(d.detector_net)

    return laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs

'''
Find 1 opt_in label, and return lasers and detectors
'''


def get_LumericalINTERCONNECT_analyzers_from_opt_in(self, components, verbose=None, opt_in_selection_text=[]):
    """
    From the opt_in label, find the trimmed circuit, and assign a laser and detectors

    returns: parameters, nets in order

    usage:
    laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, ignoreOpticalIOs, detector_list = get_LumericalINTERCONNECT_analyzers_from_opt_in(topcell, components)
    """
    from . import _globals
    from .core import Net

    from SiEPIC.utils import load_DFT
    DFT = load_DFT()
    if not DFT:
        if verbose:
            print(' no DFT rules available.')
        return False, False, False, False, False, False, False, False

    from .scripts import user_select_opt_in
    opt_in_selection_text, opt_in_dict = user_select_opt_in(
        verbose=verbose, option_all=False, opt_in_selection_text=opt_in_selection_text)
    if not opt_in_dict:
        if verbose:
            print(' no opt_in selected.')
        return False, False, False, False, False, False, False, False

    # find closest GC to opt_in (pick the 1st one... ignore the others)
    t = opt_in_dict[0]['Text']
    components_sorted = sorted([c for c in components if [p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO]],
                               key=lambda x: x.trans.disp.to_p().distance(pya.Point(t.x, t.y).to_dtype(1)))
    if not(components_sorted):
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText("To run a simulation, you need to have optical IO in the layout." )
        pya.QMessageBox_StandardButton(warning.exec_())
        return False, False, False, False, False, False, False, False
        
    dist_optin_c = components_sorted[0].trans.disp.to_p().distance(pya.Point(t.x, t.y).to_dtype(1))
    if verbose:
        print(" - Found opt_in: %s, nearest GC: %s.  Locations: %s, %s. distance: %s" % (opt_in_dict[0][
              'Text'], components_sorted[0].instance,  components_sorted[0].center, pya.Point(t.x, t.y), dist_optin_c))
    if dist_optin_c > float(DFT['design-for-test']['opt_in']['max-distance-to-grating-coupler']) * 1000:
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText("To run a simulation, you need to have an opt_in label with %s microns from the nearest grating coupler" % int(
            DFT['design-for-test']['opt_in']['max-distance-to-grating-coupler']))
        pya.QMessageBox_StandardButton(warning.exec_())
        return False, False, False, False, False, False, False, False
    # starting with the opt_in label, identify the sub-circuit, then GCs
    detector_GCs = [c for c in components if [p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO] if (
        c.trans.disp - components_sorted[0].trans.disp).to_p() != pya.DPoint(0, 0)]
    if verbose:
        print("   N=%s, detector GCs: %s" %
              (len(detector_GCs), [c.display() for c in detector_GCs]))
    vect_optin_GCs = [(c.trans.disp - components_sorted[0].trans.disp).to_p()
                      for c in detector_GCs]

    # Laser at the opt_in GC:
    p = [p for p in components_sorted[0].pins if p.type == _globals.PIN_TYPES.OPTICALIO]
    p[0].pin_name += '_laser'
    laser_net = p[0].net = Net(idx=p[0].pin_name, pins=p)
    if verbose:
        print(" - pin_name: %s" % (p[0].pin_name))

    tunable_lasers = DFT['design-for-test']['tunable-laser']
    if type(tunable_lasers) == type({}):
        # single tunable laser
        tunable_lasers = [tunable_lasers]
    for i in range(len(tunable_lasers)):
        if tunable_lasers[i]['wavelength'] == opt_in_dict[0]['wavelength'] and tunable_lasers[i]['polarization'] == opt_in_dict[0]['pol']:
            wavelength_start, wavelength_stop, wavelength_points = float(tunable_lasers[i]['wavelength-start']), float(
                tunable_lasers[i]['wavelength-stop']), int(tunable_lasers[i]['wavelength-points'])
    if not('wavelength_start' in locals()):
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText("No laser at %s nm is available. Tunable laser definition is in the technology's DFT.xml file." %
                        opt_in_dict[0]['wavelength'])
        pya.QMessageBox_StandardButton(warning.exec_())
        return False, False, False, False, False, False, False, False

    if opt_in_dict[0]['pol'] == 'TE':
        orthogonal_identifier = 1
    elif opt_in_dict[0]['pol'] == 'TM':
        orthogonal_identifier = 2
    else:
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText("Unknown polarization: %s." % opt_in_dict[0]['pol'])
        pya.QMessageBox_StandardButton(warning.exec_())
        return False, False, False, False, False, False, False, False
    ignoreOpticalIOs = False

    # find the GCs in the circuit and connect detectors based on DFT rules
    detectors_info = []
    detector_number = 0
    detector_lookuptable = {1: 1, -1: 2, -2: 3}
    detector_list = []
    for d in list(range(int(DFT['design-for-test']['grating-couplers']['detectors-above-laser']) + 0, 0, -1)) + list(range(-1, -int(DFT['design-for-test']['grating-couplers']['detectors-below-laser']) - 1, -1)):
        if pya.DPoint(0, d * float(DFT['design-for-test']['grating-couplers']['gc-pitch']) * 1000) in vect_optin_GCs:
            detector_number += 1
            detector_list += [detector_lookuptable[d]]
            index = vect_optin_GCs.index(pya.DPoint(
                0, d * float(DFT['design-for-test']['grating-couplers']['gc-pitch']) * 1000))
            # detector_GCs[index] # component

            p = [p for p in detector_GCs[index].pins if p.type == _globals.PIN_TYPES.OPTICALIO]
            p[0].pin_name += '_detector' + str(detector_number)
            p[0].net = Net(idx=p[0].pin_name, pins=p)
            detectors_info.append(Detector_info(p[0].net, detector_number))
            if verbose:
                print(" - pin_name: %s" % (p[0].pin_name))

    # Sort the detectors:
    detectors_info2 = sorted(detectors_info, key=lambda d: d.detector_number)

    # output:
    detector_nets = []
    for d in detectors_info2:
        detector_nets.append(d.detector_net)

    return laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs, detector_list


# generate spice netlist file
# example output:
# X_grating_coupler_1 N$7 N$6 grating_coupler library="custom/genericcml"
# sch_x=-1.42 sch_y=-0.265 sch_r=0 sch_f=false
def spice_netlist_export(self, verbose=False, opt_in_selection_text=[]):
    import SiEPIC
    from . import _globals
    from time import strftime
    from .utils import eng_str

    from .utils import get_technology
    TECHNOLOGY = get_technology()
    if not TECHNOLOGY['technology_name']:
        v = pya.MessageBox.warning("Errors", "SiEPIC-Tools requires a technology to be chosen.  \n\nThe active technology is displayed on the bottom-left of the KLayout window, next to the T. \n\nChange the technology using KLayout File | Layout Properties, then choose Technology and find the correct one (e.g., EBeam, GSiP).", pya.MessageBox.Ok)
        return '', '', 0, []

    # get the netlist from the entire layout
    nets, components = self.identify_nets()

    if not components:
        v = pya.MessageBox.warning("Errors", "No components found.", pya.MessageBox.Ok)
        return '', '', 0, []

    # Get information about the laser and detectors:
    # this updates the Optical IO Net
    laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs = \
        get_LumericalINTERCONNECT_analyzers(self, components, verbose=verbose)
    detector_list = []

    # if Laser and Detectors are not defined
    if not laser_net or not detector_nets:
        # Use opt_in labels
        laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs, detector_list = \
            get_LumericalINTERCONNECT_analyzers_from_opt_in(
                self, components, verbose=verbose, opt_in_selection_text=opt_in_selection_text)

        if not laser_net or not detector_nets:
            warning = pya.QMessageBox()
            warning.setStandardButtons(pya.QMessageBox.Ok)
            warning.setText(
                "To run a simulation, you need to define a laser and detector(s), or have an opt_in label.")
            pya.QMessageBox_StandardButton(warning.exec_())
            return '', '', 0, []

    # trim the netlist, based on where the laser is connected
    laser_component = [c for c in components if any(
        [p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO and 'laser' in p.pin_name])]

    from .scripts import trim_netlist
    nets, components = trim_netlist(nets, components, laser_component[0])

    if not components:
        pya.MessageBox.warning("Error: netlist extraction",
                               "Error: netlist extraction. No components found connected to opt_in label.", pya.MessageBox.Ok)
        return '', '', 0, []

    if verbose:
        print("* Display list of components:")
        [c.display() for c in components]
        print("* Display list of nets:")
        [n.display() for n in nets]

    text_main = '* Spice output from KLayout SiEPIC-Tools v%s, %s.\n\n' % (
        SiEPIC.__version__, strftime("%Y-%m-%d %H:%M:%S"))
    text_subckt = text_main

    # convert KLayout GDS rotation/flip to Lumerical INTERCONNECT
    # KLayout defines mirror as an x-axis flip, whereas INTERCONNECT does y-axis flip
    # KLayout defines rotation as counter-clockwise, whereas INTERCONNECT does clockwise
    # input is KLayout Rotation,Flip; output is INTERCONNECT:
    KLayoutInterconnectRotFlip = \
        {(0, False): [0, False],
         (90, False): [270, False],
         (180, False): [180, False],
         (270, False): [90, False],
         (0, True): [180, True],
         (90, True): [90, True],
         (180, True): [0, True],
         (270, True): [270, False]}

    # Determine the Layout-to-Schematic (x,y) coordinate scaling
    # Find the distances between all the components, in order to determine scaling
    sch_positions = [o.Dcenter for o in components]
    sch_distances = []
    for j in range(len(sch_positions)):
        for k in range(j + 1, len(sch_positions)):
            dist = (sch_positions[j] - sch_positions[k]).abs()
            sch_distances.append(dist)
    sch_distances.sort()
    if verbose:
        print("Distances between components: %s" % sch_distances)
    # remove any 0 distances:
    while 0.0 in sch_distances:
        sch_distances.remove(0.0)
    # scaling based on nearest neighbour:
    Lumerical_schematic_scaling = 0.6 / min(sch_distances)
    print("Scaling for Lumerical INTERCONNECT schematic: %s" % Lumerical_schematic_scaling)
    # but if the layout is too big, limit the size
    MAX_size = 0.05 * 1e3
    if max(sch_distances) * Lumerical_schematic_scaling > MAX_size:
        Lumerical_schematic_scaling = MAX_size / max(sch_distances)
    print("Scaling for Lumerical INTERCONNECT schematic: %s" % Lumerical_schematic_scaling)

    # find electrical IO pins
    electricalIO_pins = ""
    DCsources = ""  # string to create DC sources for each pin
    Vn = 1
    SINGLE_DC_SOURCE = 2
    # (1) attach all electrical pins to the same DC source
    # (2) or to individual DC sources
    # (3) or choose based on number of DC sources, if > 5, use single DC source

    # create individual sources:
    for c in components:
        for idx,p in enumerate(c.pins):
            if p.type == _globals.PIN_TYPES.ELECTRICAL:
                if idx > 0:  
                    if p.pin_name == c.pins[idx - 1].pin_name: continue  # Skip pins that have exactly the same name (assume they are internally connected in the component)
                NetName = " " + c.cell.name + '_' + str(c.idx) + '_' + p.pin_name
                electricalIO_pins += NetName
                DCsources += "N" + \
                    str(Vn) + NetName + \
                    ' "dc source" amplitude=0 sch_x=%s sch_y=%s\n' % (-2 - Vn / 3., -2 + Vn / 8.)
                Vn += 1
    electricalIO_pins_subckt = electricalIO_pins

    # create 1 source
    if (SINGLE_DC_SOURCE == 1) or ((SINGLE_DC_SOURCE == 3) and (Vn > 5)):
        electricalIO_pins_subckt = ""
        for c in components:
            for p in c.pins:
                if p.type == _globals.PIN_TYPES.ELECTRICAL:
                    NetName = " N$"
                    electricalIO_pins_subckt += NetName
                    DCsources = "N1" + NetName + ' "dc source" amplitude=0 sch_x=-2 sch_y=0\n'

    # find optical IO pins
    opticalIO_pins = ''
    for c in components:
        for p in c.pins:
            if p.type == _globals.PIN_TYPES.OPTICALIO:
                NetName = ' ' + p.pin_name
                print(p.pin_name)
                opticalIO_pins += NetName

    circuit_name = self.name.replace('.', '')  # remove "."
    if '_' in circuit_name[0]:
        circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _

    # create the top subckt:
    text_subckt += '.subckt %s%s%s\n' % (circuit_name, electricalIO_pins, opticalIO_pins)
    # assign MC settings before importing netlist components
    text_subckt += '.param MC_uniformity_width=0 \n'
    text_subckt += '.param MC_uniformity_thickness=0 \n'
    text_subckt += '.param MC_resolution_x=100 \n'
    text_subckt += '.param MC_resolution_y=100 \n'
    text_subckt += '.param MC_grid=10e-6 \n'
    text_subckt += '.param MC_non_uniform=99 \n'

    for c in components:
        # Check pins to see if explicitly ordered numerically - requires first character in pin name to be a number (Stefan Preble, RIT)
        explicit_ordering = False
        for p in c.pins:
            pinname1 = p.pin_name[0]
            if pinname1.isdigit():
                explicit_ordering = True
            else:
                explicit_ordering = False  # all pins must be numbered
                break
               
        nets_str = ''        
        if explicit_ordering:   # Order the pins numerically (Stefan Preble, RIT)
            for idx, p in enumerate(c.pins):
                if idx > 0:  
                    if p.pin_name == c.pins[idx - 1].pin_name: continue  # Skip pins that have exactly the same name (assume they are internally connected in the component)
                if p.type == _globals.PIN_TYPES.ELECTRICAL:
                    nets_str += " " + c.cell.name + '_' + str(c.idx) + '_' + p.pin_name
                if p.type == _globals.PIN_TYPES.OPTICALIO:
                    nets_str += " " + str(p.net.idx)    
                if p.type == _globals.PIN_TYPES.OPTICAL:
                    nets_str += " N$" + str(p.net.idx)
        else:
            # optical nets: must be ordered electrical, optical IO, then optical
            for p in c.pins:
                if p.type == _globals.PIN_TYPES.ELECTRICAL:
                    nets_str += " " + c.cell.name + '_' + str(c.idx) + '_' + p.pin_name
            for p in c.pins:
                if p.type == _globals.PIN_TYPES.OPTICALIO:
                    nets_str += " " + str(p.net.idx)
            for p in c.pins:
                if p.type == _globals.PIN_TYPES.OPTICAL:
                    nets_str += " N$" + str(p.net.idx)

        trans = KLayoutInterconnectRotFlip[(c.trans.angle, c.trans.is_mirror())]

        flip = ' sch_f=true' if trans[1] else ''
        if trans[0] > 0:
            rotate = ' sch_r=%s' % str(trans[0])
        else:
            rotate = ''

        # Check to see if this component is an Optical IO type.
        pinIOtype = any([p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO])

        if ignoreOpticalIOs and pinIOtype:
            # Replace the Grating Coupler or Edge Coupler with a 0-length waveguide.
            component1 = "ebeam_wg_strip_1550"
            params1 = "wg_length=0u wg_width=0.500u"
        else:
            component1 = c.component
            params1 = c.params
            
        # Remove "$N" from component's name for cell instance arrays of the same name     
        if "$" in component1:
            component1 = component1[:component1.find("$")]
        
        if component1.find(' ')>=0 and component1.find('"')==-1:
          text_subckt += ' %s %s "%s" ' % (component1.replace(' ','_') +
                                       "_" + str(c.idx), nets_str, component1)
        else:
          text_subckt += ' %s %s %s ' % (component1.replace('"', '').replace(' ', '_') +
                                       "_" + str(c.idx), nets_str, component1)
        if c.library != None:
            text_subckt += 'library="%s" ' % c.library
        x, y = c.Dcenter.x, c.Dcenter.y
        text_subckt += '%s lay_x=%s lay_y=%s sch_x=%s sch_y=%s %s%s\n' % \
            (params1,
             eng_str(x * 1e-6), eng_str(y * 1e-6),
             eng_str(x * Lumerical_schematic_scaling), eng_str(y * Lumerical_schematic_scaling),
             rotate, flip)

    text_subckt += '.ends %s\n\n' % (circuit_name)

    if laser_net:
        text_main += '* Optical Network Analyzer:\n'
        text_main += '.ona input_unit=wavelength input_parameter=start_and_stop\n  + minimum_loss=80\n  + analysis_type=scattering_data\n  + multithreading=user_defined number_of_threads=1\n'
        text_main += '  + orthogonal_identifier=%s\n' % orthogonal_identifier
        text_main += '  + start=%4.3fe-9\n' % wavelength_start
        text_main += '  + stop=%4.3fe-9\n' % wavelength_stop
        text_main += '  + number_of_points=%s\n' % wavelength_points
        for i in range(0, len(detector_nets)):
            text_main += '  + input(%s)=%s,%s\n' % (i + 1, circuit_name, detector_nets[i].idx)
        text_main += '  + output=%s,%s\n' % (circuit_name, laser_net.idx)

    # main circuit
    text_subckt += '%s %s %s %s sch_x=-1 sch_y=-1 ' % (
        circuit_name, electricalIO_pins_subckt, opticalIO_pins, circuit_name)
    if len(DCsources) > 0:
        text_subckt += 'sch_r=270\n\n'
    else:
        text_subckt += '\n\n'

    text_main += DCsources

    return text_subckt, text_main, len(detector_nets), detector_list


def check_components_models():

    # Check if all the components in the cell have compact models loaded in INTERCONNECT

    # test for Component.has_compactmodel()
    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    print("* find_components()")
    components = cell.find_components()
    print("* Display list of components")

    if not all([c.has_model() for c in components]):
        # missing models, find which one
        components_havemodels = [[c.has_model(), c.component, c.instance] for c in components]
        missing_models = []
        for c in components_havemodels:
            if c[0] == False:
                missing_models.append([c[1], c[2]])
        missing = ("We have %s component(s) missing models, as follows: %s" %
                   (len(missing_models), missing_models))
        v = pya.MessageBox.warning("Errors", missing, pya.MessageBox.Ok)
    else:
        print('check_components_models(): all models are present.')
        v = pya.MessageBox.warning(
            "All ok", "All components have models. Ok to simulate the circuit.", pya.MessageBox.Ok)

# find the Pin's Point, whose name matches the input, for the given Cell
def pinPoint(self, pin_name, verbose=False):
    pins, _ = self.find_pins()
    if pins:
        matched_pins = [p for p in pins if (p.pin_name==pin_name)]
        if not matched_pins:
            raise Exception("Did not find matching pin (%s), in the components list of pins (%s)." %(pin_name, [p.pin_name for p in pins]) )
            return
        return matched_pins[0].center
    else:
        pass


#################################################################################

pya.Cell.print_parameter_values = print_parameter_values
pya.Cell.find_pin = find_pin
pya.Cell.find_pins = find_pins
pya.Cell.find_pins_component = find_pins_component
pya.Cell.find_components = find_components
pya.Cell.identify_nets = identify_nets
pya.Cell.get_LumericalINTERCONNECT_analyzers = get_LumericalINTERCONNECT_analyzers
pya.Cell.get_LumericalINTERCONNECT_analyzers_from_opt_in = get_LumericalINTERCONNECT_analyzers_from_opt_in
pya.Cell.spice_netlist_export = spice_netlist_export
pya.Cell.pinPoint = pinPoint


#################################################################################
#                    SiEPIC Class Extension of Instance Class                   #
#################################################################################

# Function Definitions
#################################################################################

# find the Pins associated with the Instance:
def find_pins(self, verbose=False):
    if verbose:
        print("Instance.find_pins, self: %s" % self)
        print("Instance.find_pins, cplx_trans: %s" % self.cplx_trans)
    found_pins, errors = self.cell.find_pins(verbose)
    return [pin.transform(self.cplx_trans) for pin in self.cell.find_pins(verbose)[0]], errors

# find the Pin's Point, whose name matches the input, for the given Instance
def pinPoint(self, pin_name, verbose=False):
    from pya import Point
    import SiEPIC.core
    pins, _ = self.find_pins(verbose)
    if verbose:
        print("Instance.pinPoint, pins: %s" % pins)
    if type(pins) == SiEPIC.core.Pin:
        pins = [pins]
    if pins:
        p = [ p for p in pins if (p.pin_name==pin_name)]
        if p:
            return p[0].center
        else:
            print("pinPoint, not found: %s, other pins: %s" % (pin_name, [p.pin_name for p in pins]))
            return Point(0,0)
    else:
        return Point(0,0)

def find_pin(self, pin_name, verbose=False):

    pins, _ = self.find_pins()

    if pins:
        p = [ p for p in pins if (p.pin_name==pin_name)]
        if p:
            if len(p)>1:
                raise Exception ('Multiple Pins with name "%s" found in cell "%s"' % (pin_name, self.cell.basic_name()) )
            return p[0]
        else:
            raise Exception ('Pin with name "%s" not found in cell "%s"' % (pin_name, self.cell.basic_name()) )
    else:
        raise Exception ('No Pins found in cell "%s"' % (self.cell.basic_name()) )


def move_up_hierachy(self):
    # move the instance up one hierarcy level
    each_parent_inst = self.parent_cell.each_parent_inst()
    parent_inst = next(each_parent_inst).inst()
    try:
        parent_inst = next(each_parent_inst).inst()
        parent_inst = next(each_parent_inst).inst()
        raise Exception ('move the cell up one hierarchy works only when the parent cell is instantiated only once.')
    except StopIteration:
        pass                
    parent_trans = parent_inst.trans
    t = pya.Trans(parent_trans*self.trans)

    # new parent cell
    each_parent_cell = self.parent_cell.each_parent_cell()
    newparent_cell = next(each_parent_cell)    
    self.parent_cell = self.layout().cell(newparent_cell)
    # apply transformation from the original parent
    self.transform(parent_trans.inverted())


#################################################################################

pya.Instance.find_pins = find_pins
pya.Instance.find_pin = find_pin
pya.Instance.pinPoint = pinPoint
pya.Instance.move_up_hierachy = move_up_hierachy


#################################################################################
#                    SiEPIC Class Extension of Point Class                      #
#################################################################################

# multiply an integer Point by a constant to get a float DPoint
# new DPoint = Point.to_dtype(TECHNOLOGY['dbu'])
def to_dtype(self, dbu):
    # create a new empty list.  Otherwise, this function would modify the original list
    # http://stackoverflow.com/questions/240178/python-list-of-lists-changes-reflected-across-sublists-unexpectedly
    return pya.DPoint(self.x / (1 / dbu), self.y / (1 / dbu))
    # > 15950 * 0.001 = 15.950000000000001
    # > 15950 / (1/ 0.001) = 15.95


def to_itype(self, dbu):
    # create a new empty list.  Otherwise, this function would modify the original list
    # http://stackoverflow.com/questions/240178/python-list-of-lists-changes-reflected-across-sublists-unexpectedly
    return pya.Point(self.x / (dbu), self.y / (dbu))

# *** Required for two Windows computers, but not others. Unknown.
pya.Point.to_dtype = to_dtype

pya.Point.to_itype = to_itype

# in v > 0.24, these are built-in to KLayout
from SiEPIC._globals import KLAYOUT_VERSION
if KLAYOUT_VERSION < 25:
    def to_p(self):
        return self
    pya.Point.to_dtype = to_dtype
    pya.DPoint.to_itype = to_itype
    pya.Point.to_p = to_p
    pya.DPoint.to_p = to_p


# Find the angle of a vector
def angle_vector(u):
    from math import atan2, pi
    return (atan2(u.y, u.x)) / pi * 180


pya.Point.angle_vector = angle_vector
pya.DPoint.angle_vector = angle_vector

##########################################################################################
#                    SiEPIC Class Extension of Point/Vector Classes                      #
##########################################################################################
from ._globals import MODULE_NUMPY

if MODULE_NUMPY:
    import numpy as np
from numbers import Number
from math import sqrt

# Point-like classes
PointLike = (pya.Point, pya.DPoint, pya.DVector, pya.Vector)


def pyaPoint__rmul__(self, factor):
    """ This implements factor * P """
    if isinstance(factor, Number):
        return self.__class__(self.x * factor, self.y * factor)
    elif MODULE_NUMPY and isinstance(factor, np.ndarray):  # ideally this is never called
        return factor.__mul__(self)
    else:
        return NotImplemented


def pyaPoint__mul__(self, factor):
    """ This implements P * factor """
    if isinstance(factor, Number):
        return self.__class__(self.x * factor, self.y * factor)
    elif MODULE_NUMPY and isinstance(factor, np.ndarray):  # Numpy can multiply any object
        return factor.__mul__(self)
    elif isinstance(factor, PointLike):
        return self.x * factor.x + self.y * factor.y
    else:
        return NotImplemented


def pyaPoint__truediv__(self, dividend):
    """ This implements P / dividend """
    return self.__class__(self.x / dividend, self.y / dividend)


def pyaPoint_norm(self):
    """ This implements the L2 norm """
    return sqrt(self.x ** 2 + self.y ** 2)


for klass in PointLike:
    klass.__rmul__ = pyaPoint__rmul__
    klass.__mul__ = pyaPoint__mul__
    klass.__truediv__ = pyaPoint__truediv__
    klass.norm = pyaPoint_norm

import sys
if sys.version_info[0] > 2:
    assert pya.DPoint(1, 2) / 1.0 == pya.DPoint(1, 2)
    assert 0.5 * pya.DPoint(1, 2) == pya.DPoint(0.5, 1)


'''
dbu float-int extension:
  - to_dbu, convert float (microns) to integer (nanometers) using dbu
  - from_dbu, convert integer (nanometers) to float (microns) using dbu
'''


def to_dbu(f, dbu):
    return int(round(f / dbu))


def to_itype(f, dbu):
    if type(f) == str:
        f = float(f)
    return int(round(f / dbu))


def from_dbu(i, dbu):
    return i * dbu


def to_dtype(i, dbu):
    return i * dbu

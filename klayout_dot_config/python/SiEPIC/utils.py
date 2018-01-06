#################################################################################
#                SiEPIC Tools - utils                                           #
#################################################################################
'''
List of functions:
 

advance_iterator
get_technology_by_name
get_technology
load_DFT
get_layout_variables
enum
find_paths
selected_opt_in_text
select_paths
select_waveguides
select_instances
angle_b_vectors
inner_angle_b_vectors
angle_vector
angle_trunc
points_per_circle
arc
arc_wg
arc_wg_xy
arc_bezier
arc_to_waveguide
translate_from_normal
pt_intersects_segment
layout_pgtext
find_automated_measurement_labels
etree_to_dict: XML parser
xml_to_dict
eng_str


'''

import pya



# Python 2 vs 3 issues:  http://python3porting.com/differences.html
# Python 2: iterator.next()
# Python 3: next(iterator)
# Python 2 & 3: advance_iterator(iterator)
try:
  advance_iterator = next
except NameError:
  def advance_iterator(it):
    return it.next()


'''
Get Technology functions:
 - get_technology_by_name(tech_name)
 - get_technology()
 - get_layout_variables(), also returns layout, cell.

return:
TECHNOLOGY['dbu'] is the database unit
TECHNOLOGY['layer name'] is a LayerInfo object. 

'''

'''
Read the layer table for a given technology.
Usage:
import SiEPIC.utils
SiEPIC.utils.get_technology_by_name('EBeam')
'''
def get_technology_by_name(tech_name, verbose=False):
    if verbose:
      print("get_technology_by_name()")
    from ._globals import KLAYOUT_VERSION
    technology = {}
    technology['technology_name']=tech_name
    if KLAYOUT_VERSION > 24:
      technology['dbu'] = pya.Technology.technology_by_name(tech_name).dbu
    else:
      technology['dbu'] = 0.001

    if KLAYOUT_VERSION > 24:
      lyp_file = pya.Technology.technology_by_name(tech_name).eff_layer_properties_file()	
      technology['base_path'] = pya.Technology.technology_by_name(tech_name).base_path()
    else:
      import os, fnmatch
      dir_path = pya.Application.instance().application_data_path()
      search_str = '*' + tech_name + '.lyp'
      matches = []
      for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
          for filename in fnmatch.filter(filenames, search_str):
              matches.append(os.path.join(root, filename))
      if matches:
        lyp_file = matches[0]
      else:
        raise Exception('Cannot find technology layer properties file %s' % search_str )
      # technology['base_path']
      head, tail = os.path.split(lyp_file)
      technology['base_path'] = os.path.split(head)[0]
      # technology['INTC_CML']
      cml_files = [x for x in os.listdir(technology['base_path']) if x.endswith(".cml")]
      if cml_files:
        technology['INTC_CML'] = cml_files[-1]
        technology['INTC_CML_path'] = os.path.join(technology['base_path'],cml_files[-1])
        technology['INTC_CML_version'] = cml_files[-1].replace(tech_name+'_','')
      else:
        technology['INTC_CML'] = None
        technology['INTC_CML_path'] = None
        technology['INTC_CML_version'] = None
    
    # Layers:
    import platform
    if platform.system() == 'Windows' and KLAYOUT_VERSION > 24: 
      # *** XML not working in Windows
      lv = pya.LayoutView()
      lv.load_layer_props(lyp_file)
      itr = lv.begin_layers()
      while True:
        if itr == lv.end_layers():
          break
        else:
          layerInfo = itr.current().source.split('@')[0]
          if layerInfo == '*/*':
            # likely encoutered a layer group, skip it
            pass
          else:
            technology[itr.current().name] = pya.LayerInfo(int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))
          itr.next()

    else:
      print(lyp_file)
      file = open(lyp_file, 'r') 
      layer_dict = xml_to_dict(file.read())['layer-properties']['properties']
      file.close()
       
      for k in layer_dict:
        layerInfo = k['source'].split('@')[0]
        if 'group-members' in k:
          # encoutered a layer group, look inside:
          j = k['group-members']
          if 'name' in j:
            layerInfo_j = j['source'].split('@')[0]
            technology[j['name']] = pya.LayerInfo(int(layerInfo_j.split('/')[0]), int(layerInfo_j.split('/')[1]))
          else:
            for j in k['group-members']:
              layerInfo_j = j['source'].split('@')[0]
              technology[j['name']] = pya.LayerInfo(int(layerInfo_j.split('/')[0]), int(layerInfo_j.split('/')[1]))
          if k['source'] != '*/*@*':
            technology[k['name']] = pya.LayerInfo(int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))
        else:
          technology[k['name']] = pya.LayerInfo(int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))
    return technology
# end of get_technology_by_name(tech_name)
# test example: give it a name of a technology, e.g., GSiP
# print(get_technology_by_name('EBeam'))
# print(get_technology_by_name('GSiP'))

# Get the current Technology
def get_technology(verbose=False):
    if verbose:
      print("get_technology()")
    from ._globals import KLAYOUT_VERSION
    technology = {}

    # defaults:
    technology['DevRec'] = pya.LayerInfo(68, 0)
    technology['Waveguide'] = pya.LayerInfo(1, 0)
    technology['Si'] = pya.LayerInfo(1, 0)
    technology['PinRec'] = pya.LayerInfo(69, 0)
    technology['Lumerical'] = pya.LayerInfo(733, 0)
    technology['Text'] = pya.LayerInfo(10, 0)
    technology_name = 'EBeam'

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
      # no layout open; return a default technology
      print ("No view selected")
      technology['dbu']=0.001
      technology['technology_name']=technology_name
      return technology

    # "lv.active_cellview().technology" crashes in KLayout 0.24.10 when loading a GDS file (technology not defined yet?)
    if KLAYOUT_VERSION > 24:
      technology_name = lv.active_cellview().technology

    technology['technology_name'] = technology_name
    
    if KLAYOUT_VERSION > 24:
      technology['dbu'] = pya.Technology.technology_by_name(technology_name).dbu
    else:
      if pya.Application.instance().main_window().current_view().active_cellview().is_valid():
        technology['dbu'] = pya.Application.instance().main_window().current_view().active_cellview().layout().dbu
      else:
        technology['dbu'] = 0.001
      
    itr = lv.begin_layers()
    while True:
      if itr == lv.end_layers():
        break
      else:
        layerInfo = itr.current().source.split('@')[0]
        if layerInfo == '*/*':
          # likely encoutered a layer group, skip it
          pass
        else:
          technology[itr.current().name] = pya.LayerInfo(int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))
        itr.next()
        
    return technology


'''
Load Design-for-Test (DFT) rules
These are technology specific, and located in the tech folder, named DFT.xml
'''
def load_DFT():
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  tech_name = TECHNOLOGY['technology_name']

  import os, fnmatch
  dir_path = pya.Application.instance().application_data_path()
  search_str = 'DFT.xml'
  matches = []
  for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
      for filename in fnmatch.filter(filenames, search_str):
        if tech_name in root:
          matches.append(os.path.join(root, filename))
  if matches:
    DFT_file = matches[0]
    
     
    import platform
    if platform.system() == 'Windows': 
      # ***, XML not working on Windows
      DFT_hardcoded = {'design-for-test': {'opt_in': {'max-distance-to-grating-coupler': '10'}, 'grating-couplers': {'minimum-gc-pitch-between-separate-circuits': '40', 'detectors-above-laser': '1', 'gc-orientation': {'ebeam_gc_te1550': '0', 'ebeam_gc_tm1550': '180'}, 'gc-array-orientation': '90', 'gc-pitch': '127', 'detectors-below-laser': '2'}, 'tunable-laser': {'wavelength-start': '1500', 'wavelength': '1550', 'wavelength-stop': '1600', 'wavelength-points': '3000'}}}
      return DFT_hardcoded

    else:
      file = open(DFT_file, 'r') 
      DFT = xml_to_dict(file.read())
      file.close()
    
      return DFT
  else:
    return None

def get_layout_variables():
  from .utils import get_technology
  TECHNOLOGY = get_technology()

  # Configure variables to find in the presently selected cell:
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    print("No view selected")
    raise UserWarning("No view selected. Make sure you have an open layout.")
  # Find the currently selected layout.
  ly = pya.Application.instance().main_window().current_view().active_cellview().layout() 
  if ly == None:
    raise UserWarning("No layout. Make sure you have an open layout.")
  # find the currently selected cell:
  cv = pya.Application.instance().main_window().current_view().active_cellview()
  cell = pya.Application.instance().main_window().current_view().active_cellview().cell
  if cell == None:
    raise UserWarning("No cell. Make sure you have an open layout.")


  return TECHNOLOGY, lv, ly, cell
   
  
#Define an Enumeration type for Python
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Find all paths, full hierarachy scan, return polygons on top cell. 
# for Verfication
def find_paths(layer, cell = None):
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  if cell is None:
    ly = lv.active_cellview().layout() 
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
  else:
    ly = cell.layout()
    
  selection = []
  itr = cell.begin_shapes_rec(ly.layer(layer))
  while not(itr.at_end()):
    if itr.shape().is_path():
      selection.append (itr.shape().path.transformed(itr.trans()))
    itr.next()
    
  return selection

#Return all selected opt_in Text labels. 
# example usage: selected_opt_in_text()[0].shape.text.string
def selected_opt_in_text():
  from .utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
  
  selection = lv.object_selection
  selection = [o for o in selection if (not o.is_cell_inst()) and o.shape.is_text() and 'opt_in' in o.shape.text.string]
  return selection
  
  
#Return all selected paths. If nothing is selected, select paths automatically
def select_paths(layer, cell = None):
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")

  if cell is None:
    ly = lv.active_cellview().layout() 
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
  else:
    ly = cell.layout()
    
  selection = lv.object_selection
  if selection == []:
    itr = cell.begin_shapes_rec(ly.layer(layer))
    while not(itr.at_end()):
      if itr.shape().is_path():
        selection.append(pya.ObjectInstPath())
        selection[-1].layer = ly.layer(layer)
        selection[-1].shape = itr.shape()
        selection[-1].top = cell.cell_index()
        selection[-1].cv_index = 0
      itr.next()
    lv.object_selection = selection
  else:
    lv.object_selection = [o for o in selection if (not o.is_cell_inst()) and o.shape.is_path()]
    
  return lv.object_selection
  
#Return all selected waveguides. If nothing is selected, select waveguides automatically
#Returns all cell_inst
def select_waveguides(cell = None):
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")

  if cell is None:
    ly = lv.active_cellview().layout() 
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
  else:
    ly = cell.layout()

  selection = lv.object_selection
  if selection == []:
    for instance in cell.each_inst():
      if instance.cell.basic_name() == "Waveguide":
        selection.append(pya.ObjectInstPath())
        selection[-1].top = cell.cell_index()
        selection[-1].append_path(pya.InstElement.new(instance))
    lv.object_selection = selection
  else:
    lv.object_selection = [o for o in selection if o.is_cell_inst() and o.inst().cell.basic_name() == "Waveguide"]
    
  return lv.object_selection
  
#Return all selected instances. 
#Returns all cell_inst
def select_instances(cell = None):
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  if cell is None:
    ly = lv.active_cellview().layout() 
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
  else:
    ly = cell.layout()

  selection = lv.object_selection
  if selection == []:
    for instance in cell.each_inst():
      selection.append(pya.ObjectInstPath())
      selection[-1].top = cell.cell_index()
      selection[-1].append_path(pya.InstElement.new(instance))
    lv.object_selection = selection
  else:
    lv.object_selection = [o for o in selection if o.is_cell_inst()]
    
  return lv.object_selection
  

#Find the angle between two vectors (not necessarily the smaller angle)
def angle_b_vectors(u, v):
  from math import atan2, pi
  return (atan2(v.y, v.x)-atan2(u.y, u.x))/pi*180

#Find the angle between two vectors (will always be the smaller angle)
def inner_angle_b_vectors(u, v):
  from math import acos, pi
  return acos((u.x*v.x+u.y*v.y)/(u.abs()*v.abs()))/pi*180

#Find the angle of a vector
def angle_vector(u):
  from math import atan2, pi
  return (atan2(u.y,u.x))/pi*180

#Truncate the angle
def angle_trunc(a, trunc):
  return ((a%trunc)+trunc)%trunc


# Calculate the recommended number of points in a circle, based on 
# http://stackoverflow.com/questions/11774038/how-to-render-a-circle-with-as-few-vertices-as-possible
def points_per_circle(radius):
  from math import acos, pi, ceil
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  err = 1e3*TECHNOLOGY['dbu']/2
  return int(ceil(2*pi/acos(2 * (1 - err / radius)**2 - 1))) if radius > 0.1 else 100

def arc(r, theta_start, theta_stop):
  # function to draw an arc of waveguide
  # radius: radius
  # w: waveguide width
  # length units in dbu
  # theta_start, theta_stop: angles for the arc
  # angles in degrees

  from math import pi, cos, sin
  from .utils import points_per_circle
  
  circle_fraction = abs(theta_stop-theta_start) / 360.0
  npoints = int(points_per_circle(r) * circle_fraction)
  if npoints==0:
    npoints = 1
  da = 2 * pi / npoints * circle_fraction # increment, in radians
  pts = []
  th = theta_start / 360.0 * 2 * pi
  for i in range(0, npoints+1):
    pts.append(pya.Point.from_dpoint(pya.DPoint((r*cos(i*da+th))/1, (r*sin(i*da+th))/1)))
  return pts

def arc_xy(x, y, r, theta_start, theta_stop):
  # function to draw an arc of waveguide
  # radius: radius
  # w: waveguide width
  # length units in dbu
  # theta_start, theta_stop: angles for the arc
  # angles in degrees

  from math import pi, cos, sin
  from .utils import points_per_circle
  
  circle_fraction = abs(theta_stop-theta_start) / 360.0
  npoints = int(points_per_circle(r) * circle_fraction)
  if npoints==0:
    npoints = 1
  da = 2 * pi / npoints * circle_fraction # increment, in radians
  pts = []
  th = theta_start / 360.0 * 2 * pi
  for i in range(0, npoints+1):
    pts.append(pya.Point.from_dpoint(pya.DPoint((x+r*cos(i*da+th))/1, (y+r*sin(i*da+th))/1)))
  return pts

def arc_wg(radius, w, theta_start, theta_stop):
  # function to draw an arc of waveguide
  # radius: radius
  # w: waveguide width
  # length units in dbu
  # theta_start, theta_stop: angles for the arc
  # angles in degrees

  from math import pi, cos, sin
  from .utils import points_per_circle
  
  circle_fraction = abs(theta_stop-theta_start) / 360.0
  npoints = int(points_per_circle(radius) * circle_fraction)
  if npoints==0:
    npoints = 1
  da = 2 * pi / npoints * circle_fraction # increment, in radians
  pts = []
  th = theta_start / 360.0 * 2 * pi
  for i in range(0, npoints+1):
    pts.append(pya.Point.from_dpoint(pya.DPoint(((radius+w/2)*cos(i*da+th))/1, ((radius+w/2)*sin(i*da+th))/1)))
  for i in range(npoints, -1, -1):
    pts.append(pya.Point.from_dpoint(pya.DPoint(((radius-w/2)*cos(i*da+th))/1, ((radius-w/2)*sin(i*da+th))/1)))
  return pya.Polygon(pts)

def arc_wg_xy(x, y, r, w, theta_start, theta_stop):
  # function to draw an arc of waveguide
  # x, y: location of the origin
  # r: radius
  # w: waveguide width
  # length units in dbu
  # theta_start, theta_stop: angles for the arc
  # angles in degrees

  from math import pi, cos, sin
  from .utils import points_per_circle
  
  circle_fraction = abs(theta_stop-theta_start) / 360.0
  npoints = int(points_per_circle(r) * circle_fraction)
  if npoints==0:
    npoints = 1
  da = 2 * pi / npoints * circle_fraction # increment, in radians
  pts = []
  th = theta_start / 360.0 * 2 * pi
  for i in range(0, npoints+1):
    pts.append(pya.Point.from_dpoint(pya.DPoint((x+(r+w/2)*cos(i*da+th))/1, (y+(r+w/2)*sin(i*da+th))/1)))
  for i in range(npoints, -1, -1):
    pts.append(pya.Point.from_dpoint(pya.DPoint((x+(r-w/2)*cos(i*da+th))/1, (y+(r-w/2)*sin(i*da+th))/1)))
  return pya.Polygon(pts)


#Create a bezier curve. While there are parameters for start and stop in degrees, this is currently only implemented for 90 degree bends
def arc_bezier(radius, start, stop, bezier):
  from math import sin, cos, pi
  N=100
  L = radius  # effective bend radius / Length of the bend
  diff = 1./(N-1) # convert int to float
  xp=[0, (1-bezier)*L, L, L]
  yp=[0, 0, bezier*L, L]
  xA = xp[3] - 3*xp[2] + 3*xp[1] - xp[0]
  xB = 3*xp[2] - 6*xp[1] + 3*xp[0]
  xC = 3*xp[1] - 3*xp[0]
  xD = xp[0]
  yA = yp[3] - 3*yp[2] + 3*yp[1] - yp[0]
  yB = 3*yp[2] - 6*yp[1] + 3*yp[0]
  yC = 3*yp[1] - 3*yp[0]
  yD = yp[0]
  
  pts = [pya.Point(-L,0) + pya.Point(xD, yD)]
  for i in range(1, N-1):
    t = i*diff
    pts.append(pya.Point(-L,0) + pya.Point(t**3*xA + t**2*xB + t*xC + xD, t**3*yA + t**2*yB + t*yC + yD))
  pts.extend([pya.Point(0, L-1), pya.Point(0,L)])
  return pts

#Take a list of points and create a polygon of width 'width' 
def arc_to_waveguide(pts, width):
  return pya.Polygon(translate_from_normal(pts, -width/2.) + translate_from_normal(pts, width/2.)[::-1])

#Translate each point by its normal a distance 'trans'
def translate_from_normal(pts, trans):
#  pts = [pya.DPoint(pt) for pt in pts]
  pts = [pt.to_dtype(1) for pt in pts]
  from math import cos, sin, pi
  d = 1./(len(pts)-1)
  a = angle_vector(pts[1]-pts[0])*pi/180 + (pi/2 if trans > 0 else -pi/2)
  tpts = [pts[0] + pya.DPoint(abs(trans)*cos(a), abs(trans)*sin(a))]
  
  for i in range(1, len(pts)-1):
    dpt = (pts[i+1]-pts[i-1])*(2/d)
    tpts.append(pts[i] + pya.DPoint(-dpt.y, dpt.x)*(trans/1/dpt.abs()))
    
  a = angle_vector(pts[-1]-pts[-2])*pi/180 + (pi/2 if trans > 0 else -pi/2)
  tpts.append(pts[-1] + pya.DPoint(abs(trans)*cos(a), abs(trans)*sin(a)))
  
  #Make ends manhattan
  if abs(tpts[0].x - pts[0].x) > abs(tpts[0].y - pts[0].y):
    tpts[0].y = pts[0].y
  else:
    tpts[0].x = pts[0].x
  if abs(tpts[-1].x - pts[-1].x) > abs(tpts[-1].y - pts[-1].y):
    tpts[-1].y = pts[-1].y
  else:
    tpts[-1].x = pts[-1].x
#  return [pya.Point(pt) for pt in tpts]
  return [pt.to_itype(1) for pt in tpts]

#Check if point c intersects the segment defined by pts a and b
def pt_intersects_segment(a, b, c):
  """ How can you determine a point is between two other points on a line segment?
  http://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
  by Cyrille Ka.  Check if c is between a and b? """
  cross = abs((c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y))
  if round(cross, 5) != 0 : return False

  dot = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y)
  if dot < 0 : return False
  return False if dot > (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y) else True


#Add bubble to a cell
# Example
# cell = pya.Application.instance().main_window().current_view().active_cellview().cell
# layout_pgtext(cell, LayerInfo(10, 0), 0, 0, "test", 1)
def layout_pgtext(cell, layer, x, y, text, mag, inv = False):
  pcell = cell.layout().create_cell("TEXT", "Basic", {"text": text, 
                                                      "layer": layer, 
                                                      "mag": mag,
                                                      "inverse": inv })
  dbu = cell.layout().dbu
  cell.insert(pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, x/dbu, y/dbu)))


'''
return all opt_in labels:
 text_out: HTML text
 opt_in: a Dictionary
'''
def find_automated_measurement_labels(topcell=None, LayerTextN=None):
  # example usage:
  # topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
  # LayerText = pya.LayerInfo(10, 0)
  # LayerTextN = topcell.layout().layer(LayerText)
  # find_automated_measurement_labels(topcell, LayerTextN)
  import string
  if not LayerTextN:
    from .utils import get_technology, find_paths
    TECHNOLOGY = get_technology()
    dbu=TECHNOLOGY['dbu']
    LayerTextN=TECHNOLOGY['Text']
  if not topcell:
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
      print("No view selected")
      raise UserWarning("No view selected. Make sure you have an open layout.")
    # Find the currently selected layout.
    ly = pya.Application.instance().main_window().current_view().active_cellview().layout() 
    if ly == None:
      raise UserWarning("No layout. Make sure you have an open layout.")
    # find the currently selected cell:
    cv = pya.Application.instance().main_window().current_view().active_cellview()
    topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
    if topcell == None:
      raise UserWarning("No cell. Make sure you have an open layout.")
    
  text_out = '% X-coord, Y-coord, Polarization, wavelength, type, deviceID, params <br>'
  dbu = topcell.layout().dbu
  iter = topcell.begin_shapes_rec(topcell.layout().layer(LayerTextN))
  i=0
  texts=[] # pya Text, for Verification
  opt_in = [] # dictionary containing everything extracted from the opt_in labels.
  while not(iter.at_end()):
    if iter.shape().is_text():
      text = iter.shape().text
      if text.string.find("opt_in") > -1:
        i+=1
        text2 = iter.shape().text.transformed(iter.itrans())
        texts.append(text2)
        fields = text.string.split("_")
        while len(fields) < 7:
          fields.append('comment')
        opt_in.append ({'opt_in': text.string, 'x': int(text2.x*dbu), 'y': int(text2.y*dbu), 'pol': fields[2], 'wavelength': fields[3], 'type': fields[4], 'deviceID': fields[5], 'params': fields[6:], 'Text': text2 })
        params_txt = ''
        for f in fields[6:]:
          params_txt += ', ' + str(f)
        text_out += "%s, %s, %s, %s, %s, %s%s<br>" %(int(text2.x*dbu), int(text2.y*dbu), fields[2], fields[3],fields[4],fields[5],params_txt )
    iter.next()
  text_out += "<br>*** Number of automated measurement labels: %s.<br>" % i
  return text_out, opt_in



try:
  advance_iterator = next
except NameError:
  def advance_iterator(it):
    return it.next()
    


# XML to Dict parser, from:
# https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary-in-python/10077069
def etree_to_dict(t):
  from collections import defaultdict
  d = {t.tag: {} if t.attrib else None}
  children = list(t)
  if children:
    dd = defaultdict(list)
    for dc in map(etree_to_dict, children):
      for k, v in dc.items():
        dd[k].append(v)
    d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
  if t.attrib:
    d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
  if t.text:
    text = t.text.strip()
    if children or t.attrib:
      if text:
        d[t.tag]['#text'] = text
    else:
      d[t.tag] = text
  return d

def xml_to_dict(t):
  from xml.etree import cElementTree as ET
  try:
    e = ET.XML(t)
  except:
    raise UserWarning("Error in the XML file.")
  return etree_to_dict(e)
  

def eng_str(x):
    import math
    # x input in meters
    # output in meters, engineering notation, rounded to 1 nm
    
    EngExp_notation = 1 # 1 = "1.0e-6", 0 = "1.0u"
    x = round(x*1E9)/1E9
    y = abs(x)
    if y == 0:
      return '0'
    else:
      exponent = int(math.floor(math.log10(y)))
      engr_exponent = exponent - exponent%3
      if engr_exponent == -3:
        str_engr_exponent = "m"
        z = y/10**engr_exponent
      elif engr_exponent == -6:
        str_engr_exponent = "u"
        z = y/10**engr_exponent
      elif engr_exponent == -9:
        str_engr_exponent = "n"
        z = y/10**engr_exponent
      else:
        str_engr_exponent = ""
        z = y/10**engr_exponent
      sign = '-' if x < 0 else ''
      if EngExp_notation:
        return sign+str(z)+'E'+str(engr_exponent)
#      return sign+ '%3.3f' % z +str(str_engr_exponent)
      else:
        return sign+ str(z) +str(str_engr_exponent)



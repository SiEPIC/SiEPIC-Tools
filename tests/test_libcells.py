import SiEPIC  # do this first to trigger extend.py

import pya
from pya import PCellDeclarationHelper, DPath, DPoint, Trans, Path, Point, Text, Polygon

from lytest import contained_pyaCell, difftest_it

from SiEPIC.utils.gds_import import GDSCell

import os

origin = pya.DPoint(0, 0)
ex = pya.DVector(1, 0)
layerspecs = [pya.LayerInfo(1, 0), pya.LayerInfo(2, 0)]


# klayout.db does not initialize technologies, so we have to spoof it
# I have commented out the get_technology_by_name calls
TECHNOLOGY = dict(Waveguide=pya.LayerInfo(1, 0),
                  DevRec=pya.LayerInfo(68, 0),
                  PinRec=pya.LayerInfo(1, 10))

# This is COPIED out of GSiP. It is not a live test, just a proof of concept
# Since it is defined in a lym file, it cannot be imported by python
# This seems not a difficult thing to change
class Waveguide(PCellDeclarationHelper):

  def __init__(self):
    # Important: initialize the super class
    super(Waveguide, self).__init__()
    # declare the parameters
    # TECHNOLOGY = get_technology_by_name('GSiP')
    self.param("path", self.TypeShape, "Path", default = DPath([DPoint(0,0), DPoint(10,0), DPoint(10,10)], 0.5))
    self.param("radius", self.TypeDouble, "Radius", default = 5)
    self.param("width", self.TypeDouble, "Width", default = 0.5)
    self.param("adiab", self.TypeBoolean, "Adiabatic", default = False)
    self.param("bezier", self.TypeDouble, "Bezier Parameter", default = 0.35)
    self.param("layers", self.TypeList, "Layers", default = ['Waveguide'])
    self.param("widths", self.TypeList, "Widths", default =  [0.5])
    self.param("offsets", self.TypeList, "Offsets", default = [0])

  def display_text_impl(self):
    # Provide a descriptive text for the cell
    return "Waveguide_%s" % self.path

  def coerce_parameters_impl(self):
    from SiEPIC.extend import to_itype
    print("GSiP.Waveguide coerce parameters")

    # TECHNOLOGY = get_technology_by_name('GSiP')
    dbu = self.layout.dbu
    wg_width = to_itype(self.width,dbu)
    for lr in range(0, len(self.layers)):
      layer = self.layout.layer(TECHNOLOGY[self.layers[lr]])
      width = to_itype(self.widths[lr],dbu)
      # check to make sure that the waveguide with parameters are consistent in both places
      if self.layout.layer(TECHNOLOGY['Waveguide']) == layer:
        if width != wg_width:
          self.widths[lr] = self.width
      # check to make sure that the DevRec is bigger than the waveguide width
      if self.layout.layer(TECHNOLOGY['DevRec']) == layer:
        if width < wg_width:
          self.widths[lr] = self.width*2

  def can_create_from_shape_impl(self):
    return self.shape.is_path()

  def transformation_from_shape_impl(self):
    return Trans(Trans.R0,0,0)

  def parameters_from_shape_impl(self):
    self.path = self.shape.dpath

  def produce_impl(self):

    from SiEPIC.utils import arc_xy, arc_bezier, angle_vector, angle_b_vectors, inner_angle_b_vectors, translate_from_normal
    from math import cos, sin, pi, sqrt
    import pya
    from SiEPIC.extend import to_itype

    print("GSiP.Waveguide")

    # TECHNOLOGY = get_technology_by_name('GSiP')

    dbu = self.layout.dbu
    wg_width = to_itype(self.width,dbu)
    path = self.path.to_itype(dbu)

    if not (len(self.layers)==len(self.widths) and len(self.layers)==len(self.offsets) and len(self.offsets)==len(self.widths)):
      raise Exception("There must be an equal number of layers, widths and offsets")
    path.unique_points()
    turn=0
    for lr in range(0, len(self.layers)):
      layer = self.layout.layer(TECHNOLOGY[self.layers[lr]])

      width = to_itype(self.widths[lr],dbu)
      offset = to_itype(self.offsets[lr],dbu)

      pts = path.get_points()
      wg_pts = [pts[0]]
      for i in range(1,len(pts)-1):
        turn = ((angle_b_vectors(pts[i]-pts[i-1],pts[i+1]-pts[i])+90)%360-90)/90
        dis1 = pts[i].distance(pts[i-1])
        dis2 = pts[i].distance(pts[i+1])
        angle = angle_vector(pts[i]-pts[i-1])/90
        pt_radius = to_itype(self.radius,dbu)
        # determine the radius, based on how much space is available
        if len(pts)==3:
          pt_radius = min (dis1, dis2, pt_radius)
        else:
          if i==1:
            if dis1 <= pt_radius:
              pt_radius = dis1
          elif dis1 < 2*pt_radius:
            pt_radius = dis1/2
          if i==len(pts)-2:
            if dis2 <= pt_radius:
              pt_radius = dis2
          elif dis2 < 2*pt_radius:
            pt_radius = dis2/2
        # waveguide bends:
        if(self.adiab):
          wg_pts += Path(arc_bezier(pt_radius, 270, 270 + inner_angle_b_vectors(pts[i-1]-pts[i], pts[i+1]-pts[i]), self.bezier, DevRec='DevRec' in self.layers[lr]), 0).transformed(Trans(angle, turn < 0, pts[i])).get_points()
        else:
          wg_pts += Path(arc_xy(-pt_radius, pt_radius, pt_radius, 270, 270 + inner_angle_b_vectors(pts[i-1]-pts[i], pts[i+1]-pts[i]),DevRec='DevRec' in self.layers[lr]), 0).transformed(Trans(angle, turn < 0, pts[i])).get_points()
      wg_pts += [pts[-1]]
      wg_pts = pya.Path(wg_pts, 0).unique_points().get_points()
      wg_polygon = Polygon(translate_from_normal(wg_pts, width/2 + (offset if turn > 0 else - offset))+translate_from_normal(wg_pts, -width/2 + (offset if turn > 0 else - offset))[::-1])
      self.cell.shapes(layer).insert(wg_polygon)

      if self.layout.layer(TECHNOLOGY['Waveguide']) == layer:
        waveguide_length = wg_polygon.area() / self.width * dbu**2

    pts = path.get_points()
    LayerPinRecN = self.layout.layer(TECHNOLOGY['PinRec'])

    t1 = Trans(angle_vector(pts[0]-pts[1])/90, False, pts[0])
    self.cell.shapes(LayerPinRecN).insert(Path([Point(-50, 0), Point(50, 0)], wg_width).transformed(t1))
    self.cell.shapes(LayerPinRecN).insert(Text("pin1", t1, 0.3/dbu, -1))

    t = Trans(angle_vector(pts[-1]-pts[-2])/90, False, pts[-1])
    self.cell.shapes(LayerPinRecN).insert(Path([Point(-50, 0), Point(50, 0)], wg_width).transformed(t))
    self.cell.shapes(LayerPinRecN).insert(Text("pin2", t, 0.3/dbu, -1))

    LayerDevRecN = self.layout.layer(TECHNOLOGY['DevRec'])

    # Compact model information
    angle_vec = angle_vector(pts[0]-pts[1])/90
    halign = 0 # left
    angle=0
    pt2=pts[0]
    pt3=pts[0]
    if angle_vec == 0: # horizontal
      halign = 2 # right
      angle=0
      pt2=pts[0] + Point(0, wg_width)
      pt3=pts[0] + Point(0, -wg_width)
    if angle_vec == 2: # horizontal
      halign = 0 # left
      angle = 0
      pt2=pts[0] + Point(0, wg_width)
      pt3=pts[0] + Point(0, -wg_width)
    if angle_vec == 1: # vertical
      halign = 2 # right
      angle = 1
      pt2=pts[0] + Point(wg_width,0)
      pt3=pts[0] + Point(-wg_width,0)
    if angle_vec == -1: # vertical
      halign = 0 # left
      angle = 1
      pt2=pts[0] + Point(wg_width,0)
      pt3=pts[0] + Point(-wg_width,0)

    t = Trans(angle, False, pts[0])
    text = Text ('Lumerical_INTERCONNECT_library=Design kits/GSiP', t, 0.1/dbu, -1)
    text.halign=halign
    shape = self.cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt2)
    text = Text ('Component=wg_strip_integral_1550', t, 0.1/dbu, -1)
    text.halign=halign
    shape = self.cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt3)
    pts_txt = str([ [round(p.to_dtype(dbu).x,3), round(p.to_dtype(dbu).y,3)] for p in pts ]).replace(', ',',')
    text = Text ( \
      'Spice_param:wg_length=%.3fu wg_width=%.3fu points="%s" radius=%s' %\
        (waveguide_length, self.width, pts_txt,self.radius ), t, 0.1/dbu, -1  )
    text.halign=halign
    shape = self.cell.shapes(LayerDevRecN).insert(text)


# Begin testing
@contained_pyaCell
def PCell_Waveguide(TOP):
    ''' I am having trouble with this because unspecified parameters are not getting default values.
        The initializer of Waveguide does not error, but produce_impl fails.
    '''
    ly = TOP.layout()
    # pcell = Waveguide(cell=TOP, layout=ly)  # this fails
    pcell = Waveguide()  # this works
    # pcell.layout = ly; pcell.cell = TOP  # but then these do nothing
    pcell.produce_impl()

# def test_PCell_Waveguide(): difftest_it(PCell_Waveguide, file_ext='.oas')()


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
example_gds = os.path.join('Examples', 'GSiP', 'RingModTransceiver', 'Layouts')
LOCAL_GDS_DIR = os.path.join(root_dir, example_gds)
gds_name = 'GSiP_RingMod_Transceiver.gds'
cell_name = 'GSiP_RingMod_Transceiver'

@contained_pyaCell
def Fixed_RingMod_Classic(TOP):
    ''' Using klayout's traditional read gds, insert cell method '''
    ly = TOP.layout()
    ringmod = GDSCell(cell_name, filename=gds_name, gds_dir=LOCAL_GDS_DIR)
    ly.read(os.path.join(LOCAL_GDS_DIR, gds_name))
    gdscell2 = ly.cell(cell_name)
    rot_DTrans = pya.DTrans.R0
    TOP.insert(pya.DCellInstArray(gdscell2.cell_index(),
                                  pya.DTrans(rot_DTrans, origin)))

def test_Fixed_RingMod_Classic(): difftest_it(Fixed_RingMod_Classic, file_ext='.oas')()


@contained_pyaCell
def Fixed_RingMod_PCell(TOP):
    ''' SiEPIC's handling of pcells with fixed gds is quite complex.
        @tlima I'm having problems with the pcell call.
        It says its not getting the required positional argument of "layout".

        This test is deactivated
    '''
    ly = TOP.layout()
    ringmod = GDSCell(cell_name, filename=gds_name, gds_dir=LOCAL_GDS_DIR)
    ringmod.pcell(ly, cell=TOP, params=None)

# def test_Fixed_RingMod_PCell(): difftest_it(Fixed_RingMod_PCell, file_ext='.oas')()

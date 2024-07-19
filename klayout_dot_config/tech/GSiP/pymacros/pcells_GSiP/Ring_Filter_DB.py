import pya
from pya import *

class Ring_Filter_DB(pya.PCellDeclarationHelper):
  """
  The PCell declaration for thermally tunable ring filter.
  """
  def __init__(self):
    super(Ring_Filter_DB, self).__init__()

    # declare the parameters
    from SiEPIC.utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name('GSiP')

    self.param("silayer", self.TypeLayer, "Si Layer", default = TECHNOLOGY['Si'])
    self.param("s", self.TypeShape, "", default = pya.DPoint(0, 0))
    self.param("r", self.TypeDouble, "Radius", default = 10)
    self.param("w", self.TypeDouble, "Waveguide Width", default = 0.5)
    self.param("g", self.TypeDouble, "Gap", default = 0.2)
    self.param("gmon", self.TypeDouble, "Gap Monitor", default = 0.5)
    self.param("component_ID", self.TypeInt, "Component_ID (>0)", default = 0)
    self.param("si3layer", self.TypeLayer, "SiEtch2(Rib) Layer", default = TECHNOLOGY['SiEtch2'])
    self.param("vllayer", self.TypeLayer, "VL Layer", default = TECHNOLOGY['VL'])
    self.param("mllayer", self.TypeLayer, "ML Layer", default = TECHNOLOGY['ML'])
    self.param("mhlayer", self.TypeLayer, "MH Layer", default = TECHNOLOGY['M Heater'])
    self.param("textpolygon", self.TypeInt, "Draw text polygon label? 0/1", default = 1)
    self.param("textl", self.TypeLayer, "Text Layer", default = TECHNOLOGY['Text'])
    self.param("pinrec", self.TypeLayer, "PinRec Layer", default = TECHNOLOGY['PinRec'])
    self.param("devrec", self.TypeLayer, "DevRec Layer", default = TECHNOLOGY['DevRec'])

  def display_text_impl(self):
    # Provide a descriptive text for the cell
    return "Ring_Filter_DB(R=" + ('%.3f' % self.r) + ",g=" + ('%g' % (1000*self.g)) + ",gmon=" + ('%g' % (1000*self.gmon)) + ")"

  def can_create_from_shape_impl(self):
    return False
    
  def produce_impl(self):
    # This is the main part of the implementation: create the layout
    from math import pi, cos, sin
    from SiEPIC.extend import to_itype
    
    # fetch the parameters
    dbu = self.layout.dbu
    ly = self.layout
    shapes = self.cell.shapes
    ly.technology_name='GSiP'
    
    LayerSi = self.silayer
    LayerSi3 = ly.layer(self.si3layer)
    LayerSiN = ly.layer(LayerSi)
    LayervlN = ly.layer(self.vllayer)
    LayermlN = ly.layer(self.mllayer)
    LayermhN = ly.layer(self.mhlayer)
    TextLayerN = ly.layer(self.textl)
    LayerPinRecN = ly.layer(self.pinrec)
    LayerDevRecN = ly.layer(self.devrec)

    
    # Define variables for the Modulator
    # Variables for the Si waveguide
    w = to_itype(self.w,dbu)
    r = to_itype(self.r,dbu)
    g = to_itype(self.g,dbu)
    gmon = to_itype(self.gmon,dbu)

    #Variables for the N layer
    w_1 = 2.0/dbu  #same for N, P, N+, P+ layer
    r_n = to_itype(self.r - 1.0,dbu)
    
    #Variables for the VC layer
    w_vc = to_itype(4.0,dbu)
    r_vc1 = to_itype(self.r - 3.75,dbu)
    r_vc2 = to_itype(self.r + 3.75,dbu)
   
    #Variables for the M1 layer
    w_m1_in = r_vc1 + w_vc/2.0 + to_itype(0.5,dbu)
    r_m1_in = r_vc1 + w_vc/2.0 + to_itype(0.5,dbu) /2.0
    w_m1_out = to_itype(6.0,dbu)
    r_m1_out = to_itype(self.r + 4.25,dbu)
    
    #Variables for the VL layer
    r_vl =  w_m1_in/2.0 -  to_itype(2.1,dbu)
    w_via = to_itype(5.0,dbu)
    h_via = to_itype(5.0,dbu)

    # Variables for the SiEtch2 layer  (Slab)
    w_Si3 = w_m1_out + 2*(r_m1_out)
    h_Si3 = w_Si3
    taper_bigend =  to_itype(2,dbu)
    taper_smallend =  to_itype(0.3,dbu)
    taper_length =  to_itype(5,dbu)

    #Variables for the MH layer
    w_mh = to_itype(2.0,dbu)
    r_mh = r
    r_mh_in = r_mh - w_mh/2.0
    
    #Define Ring centre   
    x0 = r + w/2
    y0 = r + g + w 

    ######################
    # Generate the layout:
   
    # Create the ring resonator
    from SiEPIC.utils.layout import layout_ring
    layout_ring(self.cell, LayerSiN, pya.DPoint((self.r+self.w/2), (self.r+self.g+self.w)), self.r, self.w)
    # ring centre:
    t = pya.Trans(pya.Trans.R0,(self.r+self.w/2)/dbu, (self.r+self.g+self.w)/dbu)
#    pcell = ly.create_cell("Ring", "GSiP", { "layer": LayerSi, "radius": self.r, "width": self.w } )
#    self.cell.insert(pya.CellInstArray(pcell.cell_index(), t))
    
    # Create the two waveguides
    wg1 = pya.Box(x0 - (w_Si3 / 2 + taper_length), -w/2, x0 + (w_Si3 / 2 + taper_length), w/2)
    shapes(LayerSiN).insert(wg1)
    y_offset = 2*r + g + gmon + 2*w
    wg2 = pya.Box(x0 - (w_Si3 / 2 + taper_length), y_offset-w/2, x0 + (w_Si3 / 2 + taper_length), y_offset+w/2)
    shapes(LayerSiN).insert(wg2)
    
    #Create the SiEtch2 (Slab) layer
    boxSi3 = pya.Box(x0-w_Si3/2.0, y0 - h_Si3/2.0, x0+w_Si3/2.0, y0 + h_Si3/2.0)
    shapes(LayerSi3).insert(boxSi3)
    pin1pts = [pya.Point(x0-w_Si3/2.0,-taper_bigend/2.0), pya.Point(x0-w_Si3/2.0-taper_length,-taper_smallend/2.0), pya.Point(x0-w_Si3/2.0-taper_length,taper_smallend/2.0), pya.Point(x0-w_Si3/2.0, taper_bigend/2.0)]
    pin2pts = [pya.Point(x0+w_Si3/2.0,-taper_bigend/2.0), pya.Point(x0+w_Si3/2.0+taper_length,-taper_smallend/2.0), pya.Point(x0+w_Si3/2.0+taper_length,taper_smallend/2.0), pya.Point(x0+w_Si3/2.0,+taper_bigend/2.0)]
    pin3pts = [pya.Point(x0-w_Si3/2.0,y_offset-taper_bigend/2.0), pya.Point(x0-w_Si3/2.0-taper_length,y_offset-taper_smallend/2.0), pya.Point(x0-w_Si3/2.0-taper_length,y_offset+taper_smallend/2.0),pya. Point(x0-w_Si3/2.0,y_offset+ taper_bigend/2.0)]
    pin4pts = [pya.Point(x0+w_Si3/2.0,y_offset-taper_bigend/2.0), pya.Point(x0+w_Si3/2.0+taper_length,y_offset-taper_smallend/2.0), pya.Point(x0+w_Si3/2.0+taper_length,y_offset+taper_smallend/2.0), pya.Point(x0+w_Si3/2.0,y_offset+taper_bigend/2.0)]
    shapes(LayerSi3).insert(pya.Polygon(pin1pts))
    shapes(LayerSi3).insert(pya.Polygon(pin2pts))
    shapes(LayerSi3).insert(pya.Polygon(pin3pts))
    shapes(LayerSi3).insert(pya.Polygon(pin4pts))
    
    from SiEPIC.utils import arc
    
    # arc angles
    # doping:
    angle_min_doping = -35
    angle_max_doping = 215
    # VC contact:
    angle_min_VC = angle_min_doping + 8
    angle_max_VC = angle_max_doping - 8
    # M1:
    angle_min_M1 = angle_min_VC - 4
    angle_max_M1 = angle_max_VC + 4
    # MH:
    angle_min_MH = -75.0
    angle_max_MH = 255
    
    #Create the VL Layer, as well as the electrical PinRec geometries
    # heater contacts
    boxVL3 = pya.Box(x0+(r_mh_in)*cos(angle_min_MH/180*pi) + 2.5/dbu, -w/2.0 -  10/dbu, x0 + (r_mh_in)*cos(angle_min_MH/180*pi) + 7.5/dbu, -w/2.0 -  5/dbu)
    shapes(LayervlN).insert(boxVL3)
    shapes(LayerPinRecN).insert(boxVL3)
    shapes(LayerPinRecN).insert(pya.Text ("elec2h2", pya.Trans(pya.Trans.R0,x0+(r_mh_in)*cos(angle_min_MH/180*pi) + 5.0/dbu,-w/2.0 -  7.5/dbu))).text_size = 0.5/dbu
    boxVL4 = pya.Box(x0-(r_mh_in)*cos(angle_min_MH/180*pi)- 7.5/dbu, -w/2.0 -  10/dbu, x0 - (r_mh_in)*cos(angle_min_MH/180*pi) - 2.5/dbu, -w/2.0 -  5/dbu)
    shapes(LayervlN).insert(boxVL4)
    shapes(LayerPinRecN).insert(boxVL4)
    shapes(LayerPinRecN).insert(pya.Text ("elec2h1", pya.Trans(pya.Trans.R0,x0-(r_mh_in)*cos(angle_min_MH/180*pi) - 5.0/dbu,-w/2.0 -  7.5/dbu))).text_size = 0.5/dbu

    #Create the MH Layer
    poly = pya.Path(arc(self.r/dbu, angle_min_MH, angle_max_MH), w_mh).transformed(t).simple_polygon()
    self.cell.shapes(LayermhN).insert(poly)
    boxMH1 = pya.Box(x0+(r_mh_in)*cos(angle_min_MH/180*pi), -w/2.0 -  2.5/dbu, x0 + (r_mh_in)*cos(angle_min_MH/180*pi) + w_mh, y0 +(r_mh_in)*sin(angle_min_MH/180*pi))
    shapes(LayermhN).insert(boxMH1)
    boxMH2 = pya.Box(x0-(r_mh_in)*cos(angle_min_MH/180*pi)  - w_mh, -w/2.0 -  2.5/dbu, x0 - (r_mh_in)*cos(angle_min_MH/180*pi), y0 +(r_mh_in)*sin(angle_min_MH/180*pi))
    shapes(LayermhN).insert(boxMH2)
    boxMH3 = pya.Box(x0+(r_mh_in)*cos(angle_min_MH/180*pi), -w/2.0 -  12.5/dbu, x0 + (r_mh_in)*cos(angle_min_MH/180*pi) + 10/dbu, -w/2.0 -  2.5/dbu)
    shapes(LayermhN).insert(boxMH3)
    boxMH4 = pya.Box(x0-(r_mh_in)*cos(angle_min_MH/180*pi)- 10/dbu, -w/2.0 -  12.5/dbu, x0 - (r_mh_in)*cos(angle_min_MH/180*pi), -w/2.0 -  2.5/dbu)
    shapes(LayermhN).insert(boxMH4)
    
    # Create the pins, as short paths:
    from SiEPIC._globals import PIN_LENGTH as pin_length
    
    shapes(LayerPinRecN).insert(pya.Path([pya.Point(x0 - (w_Si3 / 2 + taper_length) + pin_length/2, 0),
                                          pya.Point(x0 - (w_Si3 / 2 + taper_length) - pin_length/2, 0)], w))
    shapes(LayerPinRecN).insert(pya.Text("opt1", pya.Trans(pya.Trans.R0,x0 - (w_Si3 / 2 + taper_length), 0))).text_size = 0.5/dbu

    shapes(LayerPinRecN).insert(pya.Path([pya.Point(x0 + (w_Si3 / 2 + taper_length) - pin_length/2, 0),
                                          pya.Point(x0 + (w_Si3 / 2 + taper_length) + pin_length/2, 0)], w))
    shapes(LayerPinRecN).insert(pya.Text("opt2", pya.Trans(pya.Trans.R0,x0 + (w_Si3 / 2 + taper_length), 0))).text_size = 0.5/dbu

    shapes(LayerPinRecN).insert(pya.Path([pya.Point(x0 - (w_Si3 / 2 + taper_length) + pin_length/2, y_offset),
                                          pya.Point(x0 - (w_Si3 / 2 + taper_length) - pin_length/2, y_offset)], w))
    shapes(LayerPinRecN).insert(pya.Text("opt3", pya.Trans(pya.Trans.R0,x0 - (w_Si3 / 2 + taper_length), y_offset))).text_size = 0.5/dbu

    shapes(LayerPinRecN).insert(pya.Path([pya.Point(x0 + (w_Si3 / 2 + taper_length) - pin_length/2, y_offset),
                                          pya.Point(x0 + (w_Si3 / 2 + taper_length) + pin_length/2, y_offset)], w))
    shapes(LayerPinRecN).insert(pya.Text("opt4", pya.Trans(pya.Trans.R0,x0 + (w_Si3 / 2 + taper_length), y_offset))).text_size = 0.5/dbu

    # Create the device recognition layer
    shapes(LayerDevRecN).insert(pya.Box(x0 - (w_Si3 / 2 + taper_length), -w/2.0 -  12.5/dbu, x0 + (w_Si3 / 2 + taper_length), y0 + r_m1_out + w_m1_out+h_via ))

    # Compact model information
    shape = shapes(LayerDevRecN).insert(pya.Text('Lumerical_INTERCONNECT_library=Design kits/GSiP', \
      pya.Trans(pya.Trans.R0,0, 0))).text_size = 0.3/dbu
    shapes(LayerDevRecN).insert(pya.Text ('Component=Ring_Filter_DB', \
      pya.Trans(pya.Trans.R0,0, w*2))).text_size = 0.3/dbu
    shapes(LayerDevRecN).insert(pya.Text('Component_ID=%s' % self.component_ID, \
      pya.Trans(pya.Trans.R0,0, w*4))).text_size = 0.3/dbu
    shapes(LayerDevRecN).insert(pya.Text \
      ('Spice_param:radius=%.3fu wg_width=%.3fu gap=%.3fu gap_monitor=%.3fu' %\
      (self.r, self.w, self.g, self.gmon), \
      pya.Trans(pya.Trans.R0,0, -w*2) ) ).text_size = 0.3/dbu
    
    # Add a polygon text description
    from SiEPIC.utils import layout_pgtext
    if self.textpolygon:
      layout_pgtext(self.cell, self.textl, self.w, self.r+self.w, "%.3f-%g" % ( self.r, self.g), 1)

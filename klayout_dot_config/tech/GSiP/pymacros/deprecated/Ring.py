import pya
from pya import *


class Ring(pya.PCellDeclarationHelper):
    def __init__(self):
        # Important: initialize the super class
        super(Ring, self).__init__()

        # declare the parameters
        from SiEPIC.utils import get_technology_by_name

        TECHNOLOGY = get_technology_by_name("GSiP")
        self.param("width", self.TypeDouble, "Width", default=0.5)
        self.param("radius", self.TypeDouble, "Radius", default=5)
        self.param("layer", self.TypeLayer, "Layer", default=TECHNOLOGY["Si"])

    def display_text_impl(self):
        # Provide a descriptive text for the cell
        return "Ring_%s" % self.radius

    def coerce_parameters_impl(self):
        pass

    def produce_impl(self):
        from SiEPIC.utils import arc
        from SiEPIC.extend import to_itype

        dbu = self.layout.dbu

        layer = self.layout.layer(self.layer)
        radius = to_itype(self.radius, dbu)
        width = to_itype(self.width, dbu)

        poly = pya.Polygon(arc(radius + width / 2, 0, 360))
        hole = pya.Polygon(arc(radius - width / 2, 0, 360))
        poly.insert_hole(hole.get_points())
        self.cell.shapes(layer).insert(poly)

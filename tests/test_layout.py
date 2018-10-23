import pya

from lytest import contained_pyaCell, difftest_it

from SiEPIC.utils.layout import layout_rectangle

origin = pya.DPoint(0, 0)
ex = pya.DVector(1, 0)
layerspecs = [pya.LayerInfo(1, 0), pya.LayerInfo(2, 0)]

@contained_pyaCell
def Rectangle(TOP):
    ly = TOP.layout()
    layers = [ly.insert_layer(ls) for ls in layerspecs]

    dpoly = layout_rectangle(TOP, layers[0], origin, 10, 20, ex)
    dpoly.resize(10, ly.dbu)
    dpoly.layout(TOP, layers[1])

def test_Rectangle(): difftest_it(Rectangle, file_ext='.oas')()
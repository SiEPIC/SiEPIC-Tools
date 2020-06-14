import pya

from lytest import contained_pyaCell, difftest_it

from SiEPIC.utils.layout import layout_rectangle, rectangle_dpolygon

origin = pya.DPoint(0, 0)
ex = pya.DVector(1, 0)
layerspecs = [pya.LayerInfo(1, 0), pya.LayerInfo(2, 0)]


@contained_pyaCell
def Basic(TOP):
    ly = TOP.layout()
    layers = [ly.insert_layer(ls) for ls in layerspecs]
    layout_rectangle(TOP, layers[0], origin, 10, 20, ex)

def test_Basic(): difftest_it(Basic, file_ext='.oas')()


@contained_pyaCell
def DpolyManipulations(TOP):
    ly = TOP.layout()
    layers = [ly.insert_layer(ls) for ls in layerspecs]

    r1 = layout_rectangle(TOP, layers[0], origin, 10, 20, ex)
    assert r1.area() == 200.
    r1.resize(10, ly.dbu)
    assert r1.area() == 1200.

    r2 = r1.dup()
    # r3 = r2.dup() #moved(100, 100)
    r2.move(-100, 0)
    # r3.layout(TOP, layers[0])
    r2.layout(TOP, layers[0])
    r2.layout_drc_exclude(TOP, layers[1], ex=ex)

    r1.round_corners(10., N=100)
    r1.transform_and_rotate(pya.DPoint(0, -50), ex=pya.DVector(1, 1))
    r1.clip(y_bounds=(-65., -35.))
    r1.layout(TOP, layers[0])

def test_DpolyManipulations(): difftest_it(DpolyManipulations, file_ext='.oas')()

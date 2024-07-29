"""

Bezier curve test structures

by Lukas Chrostowski, 2024

Example simple script to
 - create a new layout with a top cell
 - create Bezier S-Bend, 90º Bend, taper
 - to do: U-turn

usage:
 - run this script in KLayout Application, or in standalone Python
"""

from pya import *


def test_bezier_bends():
    designer_name = "Test_Bezier_bend"
    top_cell_name = "GSiP_%s" % designer_name

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import (
        connect_cell,
        connect_pins_with_waveguide,
        zoom_out,
        export_layout,
    )
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    import os

    path = os.path.dirname(os.path.realpath(__file__))

    tech_name = "GSiP"

    from packaging import version

    if version.parse(SiEPIC.__version__) < version.parse("0.5.4"):
        raise Exception(
            "Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater."
        )

    if Python_Env == "Script":
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(path, "../../..")))
        import GSiP

    """
    Create a new layout
    """
    cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite=True)
    dbu = ly.dbu
    import numpy as np

    from SiEPIC.utils.geometry import bezier_parallel, bezier_cubic
    from SiEPIC.utils import translate_from_normal
    from SiEPIC.utils import arc_xy, arc_bezier

    accuracy = 0.005

    w = 10
    h = 1
    width = 0.5
    layer = ly.layer(ly.TECHNOLOGY["Waveguide"])
    layer2 = ly.layer(ly.TECHNOLOGY["FloorPlan"])
    print(" Layers: %s, %s" % (ly.TECHNOLOGY["Waveguide"], ly.TECHNOLOGY["FloorPlan"]))

    # Two S-bends, small ∆h
    wg_Dpts = bezier_parallel(pya.DPoint(0, 0), pya.DPoint(w, h), 0)
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer2).insert(wg_polygon)
    a = 0.45
    wg_Dpts = bezier_cubic(
        pya.DPoint(0, 0), pya.DPoint(w, h), 0, 0, a, a, accuracy=accuracy
    )
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer).insert(wg_polygon)

    # Two S-bends, large ∆h
    h = 9
    wg_Dpts = bezier_parallel(pya.DPoint(0, 0), pya.DPoint(w, h), 0)
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer2).insert(wg_polygon)
    a = 0.75
    wg_Dpts = bezier_cubic(
        pya.DPoint(0, 0), pya.DPoint(w, h), 0, 0, a, a, accuracy=accuracy
    )
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer).insert(wg_polygon)

    # 90º bend
    r = 5
    bezier = 0.2
    a = 1 - bezier  # /np.sqrt(2)
    wg_Dpts = bezier_cubic(
        pya.DPoint(0, 0),
        pya.DPoint(r, r),
        0,
        90 / 180 * np.pi,
        a,
        a,
        accuracy=accuracy,
        verbose=True,
    )
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer).insert(wg_polygon)
    wg_pts = arc_bezier(r / dbu, 0, 90, float(bezier))
    wg_Dpts = [p.to_dtype(dbu) for p in wg_pts]
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    ).transformed(pya.Trans(r, 0))
    cell.shapes(layer2).insert(wg_polygon)

    # U-turn bend
    r = 5
    h = 1.6 * r
    a = 2 / (h / r)
    wg_Dpts = bezier_cubic(
        pya.DPoint(0, 0),
        pya.DPoint(0, h),
        0,
        180 / 180 * np.pi,
        a,
        a,
        accuracy=accuracy,
        verbose=True,
    )
    wg_polygon = pya.DPolygon(
        translate_from_normal(wg_Dpts, width / 2)
        + translate_from_normal(wg_Dpts, -width / 2)[::-1]
    )
    cell.shapes(layer).insert(wg_polygon)

    # Save
    filename = os.path.splitext(os.path.basename(__file__))[0]
    file_out = export_layout(
        cell, path, filename + top_cell_name, format="oas", screenshot=True
    )

    # Display in KLayout
    from SiEPIC._globals import Python_Env

    if Python_Env == "Script":
        from SiEPIC.utils import klive

        klive.show(file_out, technology=tech_name, keep_position=True)

    # Plot
    # cell.plot() # in the browser


def test_bezier_tapers():
    designer_name = "Test_Bezier_tapers"
    top_cell_name = "GSiP_%s" % designer_name

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import (
        connect_cell,
        connect_pins_with_waveguide,
        zoom_out,
        export_layout,
    )
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    import os

    path = os.path.dirname(os.path.realpath(__file__))

    tech_name = "GSiP"

    from packaging import version

    if version.parse(SiEPIC.__version__) < version.parse("0.5.4"):
        raise Exception(
            "Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater."
        )

    if Python_Env == "Script":
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(path, "../../..")))
        import GSiP

    """
    Create a new layout
    """
    cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite=True)
    dbu = ly.dbu
    import numpy as np

    from SiEPIC.utils.geometry import bezier_parallel, bezier_cubic
    from SiEPIC.utils import translate_from_normal
    from SiEPIC.utils import arc_xy, arc_bezier

    accuracy = 0.005
    layer = ly.layer(ly.TECHNOLOGY["Waveguide"])

    # Bezier Taper
    taper_length = 20
    width0 = 0.5
    width1 = 3.0
    # a = 0.4 # bezier parameter
    # b = a
    # wg_Dpts =   bezier_cubic(pya.DPoint(0, width0/2), pya.DPoint(taper_length, width1/2), 0, 0, a, b, accuracy = accuracy) + \
    #             bezier_cubic(pya.DPoint(0, -width0/2), pya.DPoint(taper_length, -width1/2), 0, 0, a, b, accuracy = accuracy)[::-1]
    # wg_polygon = pya.DPolygon( wg_Dpts )
    # cell.shapes(layer).insert(wg_polygon)
    # a = 0.5 # bezier parameter
    # b = a
    # wg_Dpts =   bezier_cubic(pya.DPoint(0, width0/2), pya.DPoint(taper_length, width1/2), 0, 0, a, b, accuracy = accuracy) + \
    #             bezier_cubic(pya.DPoint(0, -width0/2), pya.DPoint(taper_length, -width1/2), 0, 0, a, b, accuracy = accuracy)[::-1]
    # wg_polygon = pya.DPolygon( wg_Dpts )
    # cell.shapes(layer).insert(wg_polygon)
    # a = 0.6 # bezier parameter
    # b = a
    # wg_Dpts =   bezier_cubic(pya.DPoint(0, width0/2), pya.DPoint(taper_length, width1/2), 0, 0, a, b, accuracy = accuracy) + \
    #             bezier_cubic(pya.DPoint(0, -width0/2), pya.DPoint(taper_length, -width1/2), 0, 0, a, b, accuracy = accuracy)[::-1]
    # wg_polygon = pya.DPolygon( wg_Dpts )
    # cell.shapes(layer).insert(wg_polygon)

    # matching a Sinusoidal taper, per Sean Lam in EBeam_Beta
    a = 0.37  # bezier parameter
    b = 0.37  # 0.385
    taper_length = 20
    wg_Dpts = (
        bezier_cubic(
            pya.DPoint(0, width0 / 2),
            pya.DPoint(taper_length, width1 / 2),
            0,
            0,
            a,
            b,
            accuracy=accuracy,
        )
        + bezier_cubic(
            pya.DPoint(0, -width0 / 2),
            pya.DPoint(taper_length, -width1 / 2),
            0,
            0,
            a,
            b,
            accuracy=accuracy,
        )[::-1]
    )
    wg_polygon = pya.DPolygon(wg_Dpts)
    cell.shapes(layer).insert(wg_polygon)

    # a = 0.95 # bezier parameter
    # b = 0.05 # bezier parameter
    # wg_Dpts =   bezier_cubic(pya.DPoint(0, width0/2), pya.DPoint(taper_length, width1/2), 0, 0, a, b, accuracy = accuracy) + \
    #             bezier_cubic(pya.DPoint(0, -width0/2), pya.DPoint(taper_length, -width1/2), 0, 0, a, b, accuracy = accuracy)[::-1]
    # wg_polygon = pya.DPolygon( wg_Dpts )
    # cell.shapes(layer).insert(wg_polygon)

    # Save
    filename = os.path.splitext(os.path.basename(__file__))[0]
    file_out = export_layout(
        cell, path, filename + top_cell_name, format="oas", screenshot=True
    )

    # Display in KLayout
    from SiEPIC._globals import Python_Env

    if Python_Env == "Script":
        from SiEPIC.utils import klive

        klive.show(file_out, technology=tech_name, keep_position=True)

    # Plot
    # cell.plot() # in the browser


if __name__ == "__main__":
    # test_bezier_bends()
    test_bezier_tapers()

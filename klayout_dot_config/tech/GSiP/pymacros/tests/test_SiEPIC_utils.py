'''
Unit testing for SiEPIC.utils.__init__

by Lukas Chrostowski, 2024


usage:
 - run this script in KLayout Application, or in standalone Python
'''

from pya import *

def test_load_layout():
    import pya
    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse('0.5.4'):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    # "path" is the folder where this script is located
    import os
    path = os.path.dirname(os.path.realpath(__file__))

    if Python_Env == 'Script':
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys
        sys.path.insert(0,os.path.abspath(os.path.join(path, '../../..')))
        import GSiP

    # Create a new layout
    topcell, ly = new_layout(tech_name, "UnitTesting", overwrite = True)

    # Load layout from file
    from SiEPIC.utils import load_layout
    load_layout(ly, 
                os.path.abspath(os.path.join(path, '../../gds/building_blocks')), 
                'Germanium_Detector_Floating.GDS', 
                single_topcell = True, Verbose = False)


def test_create_cell2():
    import pya
    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse('0.5.4'):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    # "path" is the folder where this script is located
    import os
    path = os.path.dirname(os.path.realpath(__file__))

    if Python_Env == 'Script':
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys
        sys.path.insert(0,os.path.abspath(os.path.join(path, '../../..')))
        import GSiP

    # Create a new layout
    topcell, ly = new_layout(tech_name, "UnitTesting", overwrite = True)

    # Load a cell from the library
    from SiEPIC.utils import create_cell2
    cell_y = create_cell2(ly, 'YBranch_te1550', 'GSiP', load_check=True)

    t = Trans(Trans.R0,0,0)
    topcell.insert(CellInstArray(cell_y.cell_index(), t))


def test_waveguide_length():
    import pya
    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse('0.5.4'):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    # "path" is the folder where this script is located
    import os
    path = os.path.dirname(os.path.realpath(__file__))

    if Python_Env == 'Script':
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys
        sys.path.insert(0,os.path.abspath(os.path.join(path, '../../..')))
        import GSiP

    # Create a new layout
    topcell, ly = new_layout(tech_name, "UnitTesting", overwrite = True)

    # Create waveguide PCell
    cell_wg = ly.create_cell('Waveguide', tech_name, {'layers':['Waveguide','DevRec'], 'widths':[0.5,2], 'offsets':[0,0]})
    print(cell_wg)
    if not cell_wg:
        raise Exception('Waveguide not loaded')
    t = Trans(Trans.R0,0,0)
    topcell.insert(CellInstArray(cell_wg.cell_index(), t))

    # Save
    filename = os.path.splitext(os.path.basename(__file__))[0]
    file_out = export_layout(topcell, path, filename, format='oas', screenshot=True)

    # Display in KLayout
    from SiEPIC._globals import Python_Env
    if Python_Env == 'Script':
        from SiEPIC.utils import klive
        klive.show(file_out, technology=tech_name, keep_position=True)
    os.remove(file_out)


    from SiEPIC.utils import waveguide_length
    wgl = waveguide_length(cell_wg)
    print(wgl)

if __name__ == "__main__":
    test_load_layout()
    test_create_cell2()
    test_waveguide_length()
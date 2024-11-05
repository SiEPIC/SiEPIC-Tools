'''
Instiate all cells in the library

by Lukas Chrostowski, 2024


usage:
 - run this script in KLayout Application, or in standalone Python
'''

from pya import *

def test_all_library_cells(show_klive=False):
    designer_name = 'Test'
    top_cell_name = 'GSiP_%s' % designer_name

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check

    import os
    path = os.path.dirname(os.path.realpath(__file__))

    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse('0.5.4'):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    if Python_Env == 'Script':
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import sys
        sys.path.insert(0,os.path.abspath(os.path.join(path, '../../..')))
        import GSiP

    # Create a new layout
    topcell, ly = new_layout(tech_name, "UnitTesting", overwrite = True)

    # Instantiate all cells
    from SiEPIC.scripts import instantiate_all_library_cells
    instantiate_all_library_cells(topcell, terminator_cells = ['Terminator_TE_1550'], terminator_libraries = ['GSiP'], terminator_waveguide_types = ['Strip'])

    # Check if there are any errors
    for cell_id in topcell.called_cells():
        c = ly.cell(cell_id)
        error_shapes = c.shapes(ly.error_layer())
        for error in error_shapes.each():
            raise Exception('Error in cell: %s, %s' % (c.name, error.text))
        if c.is_empty() or c.bbox().area() == 0:
            raise Exception('Empty cell: %s' % c.name)

    if show_klive:
        topcell.show()

    # Verify
    num_errors = layout_check(cell=topcell, verify_DFT=False, verbose=False, GUI=True)
    if num_errors:
        raise Exception('Number of errors: %s' % num_errors)
    print('Number of errors: %s' % num_errors)


if __name__ == "__main__":
    test_all_library_cells(show_klive=True)

'''
--- Simple MZI ---

by Lukas Chrostowski, 2020-2024

Example simple script to
 - create a new layout with a top cell
 - create an MZI
 - export to OASIS for submission to fabrication

using SiEPIC-Tools function including connect_pins_with_waveguide and connect_cell

usage:
 - run this script in KLayout Application, or in standalone Python
'''

from pya import *

def example_circuit():
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

    '''
    Create a new layout, with a top cell, and Draw the floor plan
    '''    
    cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite = True)
    floorplan(cell, 605e3, 410e3)

    dbu = ly.dbu

    from SiEPIC.scripts import connect_pins_with_waveguide, connect_cell
    waveguide_type='Strip TE 1550 nm'

    # Load cells from library
    cell_ebeam_gc = ly.create_cell('Grating_Coupler_13deg_TE_1550_Oxide', tech_name)
    cell_ebeam_y = ly.create_cell('YBranch_te1550', tech_name)

    # grating couplers, place at absolute positions
    x,y = 60000, 15000
    t = Trans(Trans.R0,x,y)
    instGC1 = cell.insert(CellInstArray(cell_ebeam_gc.cell_index(), t))
    t = Trans(Trans.R0,x,y+127000)
    instGC2 = cell.insert(CellInstArray(cell_ebeam_gc.cell_index(), t))

    # automated test label
    text = Text ("opt_in_TE_1550_device_%s_MZI1" % designer_name, t)
    cell.shapes(ly.layer(ly.TECHNOLOGY['Text'])).insert(text).text_size = 5/dbu

    # Y branches:
    instY1 = connect_cell(instGC1, 'opt_wg', cell_ebeam_y, 'opt1')
    instY1.transform(Trans(20000,0))
    # 'pin1' is not a pin on this cell, but rather 'opt1':
    instY2 = connect_cell(instGC2, 'opt_wg', cell_ebeam_y, 'pin1', relaxed_pinnames=True)
    instY2.transform(Trans(20000,0))

    # move the splitters into a subcell, to make an MZI sub-cell:
    cell_MZI = ly.create_cell('MZI')
    t = Trans(Trans.R0,0,0)
    inst_MZI = cell.insert(CellInstArray(cell_MZI.cell_index(), t))
    instY1.parent_cell = cell_MZI
    instY2.parent_cell = cell_MZI

    # Waveguides:
    connect_pins_with_waveguide(instGC1, 'opt_wg', instY1, 'opt1', waveguide_type=waveguide_type, verbose=True, error_min_bend_radius=False)
    connect_pins_with_waveguide(instGC2, 'opt_wg', instY2, 'opt1', waveguide_type=waveguide_type, verbose=True, error_min_bend_radius=True)
    connect_pins_with_waveguide(instY1, '2', instY2, '3', waveguide_type=waveguide_type, relaxed_pinnames=True)
    connect_pins_with_waveguide(instY1, 'opt3', instY2, 'opt2', waveguide_type=waveguide_type,turtle_B=[25,-90])
    
    # Zoom out
    zoom_out(cell)

    # Delete extra top cells
    from SiEPIC.scripts import delete_extra_topcells
    delete_extra_topcells(ly, cell)

    # Find the automated measurement coordinates:
    from SiEPIC.utils import find_automated_measurement_labels
    text_out, opt_in = find_automated_measurement_labels(cell)
    print(opt_in)

    # calculate areas
    from SiEPIC.scripts import calculate_area
    text = calculate_area(ly,cell)
    print (text)

    # Verify
    num_errors = layout_check(cell=cell, verbose=False, GUI=True)
    print('Number of errors: %s' % num_errors)

    # Netlist
    from SiEPIC.netlist import export_spice
    filename_netlist, filename_subckt = export_spice(cell, opt_in_selection_text=["opt_in_TE_1550_device_%s_MZI1" % designer_name])
    print(filename_netlist)

    # Save
    filename = os.path.splitext(os.path.basename(__file__))[0]
    file_out = export_layout(cell, path, filename, format='oas', screenshot=True)

    # Display in KLayout
    from SiEPIC._globals import Python_Env
    if Python_Env == 'Script':
        from SiEPIC.utils import klive
        klive.show(file_out, technology=tech_name, keep_position=True)

    # Plot 
    cell.plot() # in the browser


    return num_errors

def test_example_circuit():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    import sys
    sys.path.insert(0,os.path.abspath(os.path.join(path, '../../../../python')))
    import SiEPIC

    assert example_circuit() == 0

if __name__ == "__main__":
    example_circuit()


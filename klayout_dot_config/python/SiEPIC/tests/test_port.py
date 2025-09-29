
import pya # KLayout Python API
import SiEPIC  # import module for SiEPIC-Tools, helper functions for KLayout
import siepic_ebeam_pdk  # import module for the SiEPIC-EBeam-PDK technology
import dft_aim_siepic_laser_pic_project1
from SiEPIC.utils.layout import new_layout

tech_name, top_cell_name = 'EBeam', 'Top'
topcell, ly = new_layout(tech_name, top_cell_name)
from SiEPIC.utils.layout import floorplan
floorplan(topcell, 300e3, 160e3)

# Design for Test rules
t = pya.Trans(pya.Trans.R0, 10, 10)
text = pya.Text ("DFT=DFT_AIM_SiEPIC_Laser_PIC_Project1", t)
text.valign = pya.Text.VAlignTop
s = topcell.shapes(ly.layer(ly.TECHNOLOGY['Text'])).insert(text)

# Get Technology
from SiEPIC.utils import get_technology_by_name
TECHNOLOGY = get_technology_by_name(tech_name)

# Use SiEPIC function create_cell2 to add a cell
from SiEPIC.utils import create_cell2
cell = create_cell2(ly, 'GC_SiN_TE_1310_8degOxide_BB', 'EBeam-SiN')
# Instantiate it in the layout with position and rotation
t = pya.Trans(pya.Trans.R180, 180e3, 15e3)
inst_gc1 = topcell.insert(
    pya.CellInstArray(cell.cell_index(), t))
t = pya.Trans(pya.Trans.R180, 180e3,15e3 + 127e3 * 1)
inst_gc2 = topcell.insert(pya.CellInstArray(cell.cell_index(), t))
# Add label for automated testing, on the top grating coupler
t = pya.Trans(pya.Trans.R0, 180e3,15e3 + 127e3 * 1)
optin = "opt_in_TE_1310_device_studentname_MZI"
text = pya.Text (optin, t)
s = topcell.shapes(ly.layer(ly.TECHNOLOGY['Text'])).insert(text)
s.text_size = 10/ly.dbu # font size
cell_taper = create_cell2(ly,
            'taper_SiN_750_800', 'EBeam-SiN')

# Connect a cell, to an instance
from SiEPIC.scripts import connect_cell
inst_gc1t = connect_cell(inst_gc1, 'opt1', cell_taper, 'opt2')
inst_gc2t = connect_cell(inst_gc2, 'opt1', cell_taper, 'opt2')


# Instantiate a Port for the laser input at (0,250)
from SiEPIC.utils import create_cell2
cell = create_cell2(ly, 'port_SiN_800', 'EBeam-SiN')
t = pya.Trans(pya.Trans.R0, pya.Vector(-1e3,20e3) )
inst_port = topcell.insert(
    pya.CellInstArray(cell.cell_index(), t))

# Add a Y-Branch
cell_y = create_cell2(ly, 'ebeam_YBranch_te1310', 'EBeam-SiN')
from SiEPIC.scripts import connect_cell
inst_y = connect_cell(inst_port, 'opt1', cell_y, 'opt1')



from SiEPIC.utils import load_Waveguides_by_Tech
waveguide_types = load_Waveguides_by_Tech(tech_name)

print('Waveguide types:')
wg_type = None
for w in waveguide_types:
    if 'SiN' in w['name']:
        if '1310' in w['name'] and 'w=800' in w['name']:
            wg_type = w['name']

from SiEPIC.scripts import connect_pins_with_waveguide
wg = connect_pins_with_waveguide(inst_y, 'opt3',
                                 inst_gc1t, 'opt1',
                                 waveguide_type=wg_type,
                                )
wg = connect_pins_with_waveguide(inst_y, 'opt2',
                                 inst_gc2t, 'opt1',
                                 waveguide_type=wg_type,
                                )

if 1:
    # Run verification
    from SiEPIC.verification import layout_check
    # Run only DFT verification, skip conflicting verification rules
    num_errors = layout_check(cell=topcell)
    # num_errors = layout_check(cell=topcell, verbose=True, dft_module='dft_aim_siepic_laser_pic_project1', verify_DFT=True)
    assert num_errors==0


if 0:

    print(inst_gc1.find_pins())
    print(inst_port.find_pins())
    #for c in topcell.find_components():
    #    c.display()

    import os
    dirname = os.path.dirname(__file__)
    from SiEPIC.scripts import export_layout
    export_type = 'static'
    filename = 'test_port' # hardcoded filename
    if export_type == 'static':
        # Export for fabrication, removing PCells
        file_out = export_layout(topcell, dirname, filename, format='gds')

    text_subckt, text_main, num_detectors, detector_list = topcell.spice_netlist_export(opt_in_selection_text=[optin])
    print(text_subckt)

    components = topcell.find_components()
    laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs, detector_list = topcell.get_LumericalINTERCONNECT_analyzers_from_opt_in(components, verbose=None, opt_in_selection_text=[optin])
    print(detector_list)
    print(detector_nets)
    for d in detector_nets:
        d.display()
    # topcell.show()



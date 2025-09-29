import pya # KLayout Python API
import SiEPIC  # import module for SiEPIC-Tools, helper functions for KLayout
import siepic_ebeam_pdk  # import module for the SiEPIC-EBeam-PDK technology
import dft_aim_siepic_laser_pic_project1
from SiEPIC.utils.layout import new_layout

tech_name, top_cell_name = 'EBeam', 'Top'
topcell, ly = new_layout(tech_name, top_cell_name)
from SiEPIC.utils.layout import floorplan
floorplan(topcell, 350e3, 450e3)

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
t = pya.Trans(pya.Trans.R180, 280e3, 15e3)
inst_gc1 = topcell.insert(
    pya.CellInstArray(cell.cell_index(), t))
t = pya.Trans(pya.Trans.R180, 280e3,15e3 + 127e3 * 1)
inst_gc2 = topcell.insert(pya.CellInstArray(cell.cell_index(), t))
t = pya.Trans(pya.Trans.R180, 280e3,15e3 + 127e3 * 2)
inst_gc3 = topcell.insert(pya.CellInstArray(cell.cell_index(), t))
t = pya.Trans(pya.Trans.R180, 280e3,15e3 + 127e3 * 3)
inst_gc4 = topcell.insert(pya.CellInstArray(cell.cell_index(), t))
# Add label for automated testing, on the top grating coupler
t = pya.Trans(pya.Trans.R0, 280e3,15e3 + 127e3 * 3)
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
inst_gc3t = connect_cell(inst_gc3, 'opt1', cell_taper, 'opt2')
inst_gc4t = connect_cell(inst_gc4, 'opt1', cell_taper, 'opt2')


# Add a splitter
cell_mmi = create_cell2(ly, 'ebeam_MMI_2x2_5050_te1310', 'EBeam-SiN')
t = pya.Trans(pya.Trans.R90, 30e3,15e3 + 127e3 * 1)
inst_mmi = topcell.insert(pya.CellInstArray(cell_mmi.cell_index(), t))


from SiEPIC.utils import load_Waveguides_by_Tech
waveguide_types = load_Waveguides_by_Tech(tech_name)
wg_type = None
for w in waveguide_types:
    if 'SiN' in w['name']:
        if '1310' in w['name'] and 'w=800' in w['name']:
            wg_type = w['name']

from SiEPIC.scripts import connect_pins_with_waveguide
wg = connect_pins_with_waveguide(inst_mmi, 'opt1',
                                 inst_gc1t, 'opt1',
                                 waveguide_type=wg_type,
                                )
wg = connect_pins_with_waveguide(inst_mmi, 'opt2',
                                 inst_gc2t, 'opt1',
                                 waveguide_type=wg_type,
                                 turtle_A=[60, 90, 120,90,0,0],
                                )
wg = connect_pins_with_waveguide(inst_mmi, 'opt4',
                                 inst_gc3t, 'opt1',
                                 waveguide_type=wg_type,
                                )
wg = connect_pins_with_waveguide(inst_mmi, 'opt3',
                                 inst_gc4t, 'opt1',
                                 waveguide_type=wg_type,
                                )

topcell.show()

if 1:
    # Run verification
    from SiEPIC.verification import layout_check
    # Run only DFT verification, skip conflicting verification rules
    num_errors = layout_check(cell=topcell)
    # num_errors = layout_check(cell=topcell, verbose=True, dft_module='dft_aim_siepic_laser_pic_project1', verify_DFT=True)
    assert num_errors==0

if 1:
    text_subckt, text_main, num_detectors, detector_list = topcell.spice_netlist_export(opt_in_selection_text=[optin])
    print(text_subckt)

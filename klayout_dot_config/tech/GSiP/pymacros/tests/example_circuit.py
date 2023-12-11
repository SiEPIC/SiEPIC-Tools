'''
--- Simple MZI ---

by Lukas Chrostowski, 2020-2023

Example simple script to
 - create a new layout with a top cell
 - create an MZI
 - export to OASIS for submission to fabrication

using SiEPIC-Tools function including connect_pins_with_waveguide and connect_cell

usage:
 - run this script in KLayout Application
'''

designer_name = 'Test'
top_cell_name = 'GSiP_%s' % designer_name

import pya
from pya import *

import SiEPIC
from SiEPIC._globals import Python_Env
from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
from SiEPIC.utils.layout import new_layout, floorplan
from SiEPIC.extend import to_itype
from SiEPIC.verification import layout_check

import os

tech_name = 'GSiP'

if SiEPIC.__version__ < '0.5.4':
    raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

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
instY2 = connect_cell(instGC2, 'opt_wg', cell_ebeam_y, 'opt1')
instY2.transform(Trans(20000,0))

# Waveguides:
connect_pins_with_waveguide(instGC1, 'opt_wg', instY1, 'opt1', waveguide_type=waveguide_type)
connect_pins_with_waveguide(instGC2, 'opt_wg', instY2, 'opt1', waveguide_type=waveguide_type)
connect_pins_with_waveguide(instY1, 'opt2', instY2, 'opt3', waveguide_type=waveguide_type)
connect_pins_with_waveguide(instY1, 'opt3', instY2, 'opt2', waveguide_type=waveguide_type,turtle_B=[25,-90])

# Zoom out
zoom_out(cell)

# Verify
num_errors = layout_check(cell=cell, verbose=False, GUI=True)
print('Number of errors: %s' % num_errors)

# Save
path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.splitext(os.path.basename(__file__))[0]
file_out = export_layout(cell, path, filename, format='oas', screenshot=True)

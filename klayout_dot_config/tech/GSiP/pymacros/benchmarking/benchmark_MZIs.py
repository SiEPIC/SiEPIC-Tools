'''
--- Benchmarking: MZI ---
  
by Lukas Chrostowski, 2024
 

Example simple script to
 - create a new layout with a top cell
 - create many MZI variants
 - export to OASIS for submission to fabrication

using SiEPIC-Tools function including connect_pins_with_waveguide and connect_cell

Use instructions:

Run in Python, e.g., VSCode

pip install required packages:
 - klayout, SiEPIC, GSiP, numpy

To install the GSiP PDK from a GitHub Desktop local copy: 
pip install -e GitHub/SiEPIC-Tools/klayout_dot_config/tech

'''

import time
# Start timer
start_time = time.time()

top_cell_name = 'GSiP_MZI'
export_type = 'static'  # static: for fabrication, PCell: include PCells in file

import pya
from pya import *

import SiEPIC
from SiEPIC._globals import Python_Env
from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
from SiEPIC.utils.layout import new_layout, floorplan
from SiEPIC.extend import to_itype
from SiEPIC.verification import layout_check

import os

if Python_Env == 'Script':
    # For external Python mode, when installed using pip install siepic_ebeam_pdk
    import GSiP

# Calculate and print execution time
execution_time = time.time() - start_time
print(f"Execution time (load PDK and Py modules): {execution_time} seconds")

tech_name = 'GSiP'

'''
Create a new layout using the EBeam technology,
with a top cell
and Draw the floor plan
'''    
cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite = True)
# floorplan(cell, 605e3, 410e3)

dbu = ly.dbu

from SiEPIC.scripts import connect_pins_with_waveguide, connect_cell
waveguide_type='Strip'

# Load cells from library
cell_ebeam_y = ly.create_cell('YBranch_te1550', tech_name)

# Array of MZI, and spacing:
n_x, n_y = 10, 10
dx, dy = 100e3, 100e3
# Each MZI 
dx_mzi = 50e3
dl_ij = 0.5
for i in range(n_x):
    for j in range(n_y):
        
        # Y branches:
        # Version 1: place it at an absolute position:
        t = Trans(dx*i,dy*j)
        instY1 = cell.insert(CellInstArray(cell_ebeam_y.cell_index(), t))
        t = Trans(Trans.R180,dx*i+dx_mzi,dy*j)
        instY2 = cell.insert(CellInstArray(cell_ebeam_y.cell_index(), t))

        # Waveguides:
        connect_pins_with_waveguide(instY1, 'opt2', instY2, 'opt3', waveguide_type=waveguide_type,turtle_A=[5,90, 0, 90], turtle_B=[5,-90, 10+dl_ij*(i*n_y+j),90])
        connect_pins_with_waveguide(instY1, 'opt3', instY2, 'opt2', waveguide_type=waveguide_type,turtle_A=[5,-90, 0, -90], turtle_B=[5,90, 10,-90])

# Zoom out
zoom_out(cell)


# Export for fabrication, removing PCells
path = os.path.dirname(os.path.realpath(__file__))
filename, extension = os.path.splitext(os.path.basename(__file__))
if export_type == 'static':
    file_out = export_layout(cell, path, filename, relative_path = '..', format='oas', screenshot=True)
else:
    file_out = os.path.join(path,'..',filename+'.oas')
    ly.write(file_out)


# Calculate and print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time (generate and save GDS): {execution_time} seconds")

# Verify
Verify = False
if Verify:
    file_lyrdb = os.path.join(path,filename+'.lyrdb')
    num_errors = layout_check(cell = cell, verbose=False, GUI=True, file_rdb=file_lyrdb)
    print('Number of errors: %s' % num_errors)
else:
    file_lyrdb = None
    
# Display the layout in KLayout, using KLayout Package "klive", which needs to be installed in the KLayout Application
if Python_Env == 'Script':
    from SiEPIC.utils import klive
    klive.show(file_out, lyrdb_filename=file_lyrdb, technology=tech_name)

# Calculate and print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

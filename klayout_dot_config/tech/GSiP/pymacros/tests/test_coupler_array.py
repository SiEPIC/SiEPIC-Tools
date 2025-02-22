"""
Test for FaML

by Lukas Chrostowski 2024

"""

def test_coupler_array(show_klive=False):
    '''
    --- Simple MZI, tested using Facet-Attached Micro Lenses (FaML) ---
    
    by Lukas Chrostowski, 2024
    
    Example simple script to
    - use the GSiP technology
    - using KLayout and SiEPIC-Tools, with function including connect_pins_with_waveguide and connect_cell
    - create a new layout with a top cell
    - create a Mach-Zehnder Interferometer (MZI) circuits
    - export to OASIS for submission to fabrication
    - display the layout in KLayout using KLive
    
    Test plan
    - count lenses from the bottom up (bottom is 1, top is 6, in this design)
    - laser input on bottom lens (1), detector on second (2), for alignment
    - MZI1: laser on 3, detector on 4, sweep
    - MZI2: laser on 5, detector on 6, sweep
    

    Use instructions:

    Run in Python, e.g., VSCode

    pip install required packages:
    - klayout, SiEPIC, siepic_ebeam_pdk, numpy

    '''

    designer_name = 'LukasChrostowski'
    top_cell_name = 'EBeam_%s_MZI2_FaML' % designer_name
    export_type = 'static'  # static: for fabrication, PCell: include PCells in file
    #export_type = 'PCell'  # static: for fabrication, PCell: include PCells in file

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide, zoom_out, export_layout
    from SiEPIC.utils.layout import new_layout, floorplan, coupler_array
    from SiEPIC.extend import to_itype
    from SiEPIC.verification import layout_check
    
    import os

    if Python_Env == 'Script':
        # For external Python mode, when installed using pip install siepic_ebeam_pdk
        import GSiP

    print('EBeam_LukasChrostowski_MZI2 layout script')
    
    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse("0.5.4"):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    '''
    Create a new layout using the EBeam technology,
    with a top cell
    and Draw the floor plan
    '''    
    cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite = True)

    waveguide_type1='Strip'

    #######################
    # Circuit #1 – Loopback
    #######################
    # draw two edge couplers for facet-attached micro-lenses
    inst = coupler_array(cell, 
            label = "opt_in_TE_1550_FaML_%s_loopback" % designer_name,
            cell_name = 'Grating_Coupler_13deg_TE_1550_Oxide',
            cell_library = 'GSiP',
             cell_params =  {},
             count=2,
            )    
    # loopback waveguide
    connect_pins_with_waveguide(inst[0], 'opt_wg', inst[1], 'opt_wg', waveguide_type=waveguide_type1)

    # Export for fabrication, removing PCells
    path = os.path.dirname(os.path.realpath(__file__))
    filename, extension = os.path.splitext(os.path.basename(__file__))
    if export_type == 'static':
        file_out = export_layout(cell, path, filename, relative_path = '..', format='oas', screenshot=True)
    else:
        file_out = os.path.join(path,'..',filename+'.oas')
        ly.write(file_out)

    # Verify
    file_lyrdb = os.path.join(path,filename+'.lyrdb')
    num_errors = layout_check(cell = cell, verbose=False, GUI=True, file_rdb=file_lyrdb)
    print('Number of errors: %s' % num_errors)

    # Display the layout in KLayout, using KLayout Package "klive", which needs to be installed in the KLayout Application
    if show_klive:
        if Python_Env == 'Script':
            from SiEPIC.utils import klive
            klive.show(file_out, lyrdb_filename=file_lyrdb, technology=tech_name)
    os.remove(file_out)

    if num_errors > 0:
        raise Exception ('Errors found in test_coupler_array')

if __name__ == "__main__":
    test_coupler_array(show_klive=True)

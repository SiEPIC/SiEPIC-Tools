"""
Test for SiEPIC.utils.top_cell_with_most_subcells_or_shapes

by Lukas Chrostowski 2025

"""

def test_top_cell_with_most_subcells_or_shapes():
    '''

    '''

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.utils.layout import new_layout
    import siepic_ebeam_pdk
    
    import os
    
    tech_name = 'EBeam'

    '''
    Create a new layout with a top cell
    '''    
    cell, ly = new_layout(tech_name, 'example', GUI=True, overwrite = True)


    from SiEPIC.utils import top_cell_with_most_subcells_or_shapes

    topcell = top_cell_with_most_subcells_or_shapes(ly, verbose=True)

    assert topcell.name == 'example'

    # add another top cell to the same layout and check the top cell name has changed
    cell = ly.create_cell('cell')

    topcell = top_cell_with_most_subcells_or_shapes(ly, verbose=True)
    assert topcell.name in ['cell', 'example']

    from SiEPIC.utils.layout import floorplan
    floorplan(cell, 100e3, 100e3)
    
    topcell = top_cell_with_most_subcells_or_shapes(ly, verbose=True)
    assert topcell.name == 'cell'

    # add another top cell to the same layout and check the top cell name has changed
    cell = ly.create_cell('top')
    topcell = top_cell_with_most_subcells_or_shapes(ly, verbose=True)
    assert topcell.name == 'top'

if __name__ == "__main__":
    test_top_cell_with_most_subcells_or_shapes()

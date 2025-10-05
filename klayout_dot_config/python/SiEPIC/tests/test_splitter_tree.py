"""
Test for SiEPIC.utils.layout.y_splitter_tree

by Lukas Chrostowski 2025

"""

def test_y_splitter_tree():
    '''

    '''

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.utils.layout import new_layout
    import siepic_ebeam_pdk
    
    import os
    
    from SiEPIC.utils.layout import y_splitter_tree
    
    tech_name = 'EBeam'

    '''
    Create a new layout with a top cell
    '''    
    cell, ly = new_layout(tech_name, 'top', GUI=True, overwrite = True)

    from SiEPIC.utils.layout import add_time_stamp
    add_time_stamp(cell, layerinfo=pya.LayerInfo(10,0))    

    library = "EBeam-SiN"
    waveguide_type='SiN Strip TE 1310 nm, w=800 nm'

    from SiEPIC.utils import create_cell2
    cell_y = create_cell2(ly, 'ebeam_YBranch_te1310', library)

    max_tree_depth =  7
    for tree_depth in range(1, max_tree_depth):
        inst_tree_in, inst_tree_out, cell_tree = y_splitter_tree(cell, tree_depth=tree_depth, y_splitter_cell=cell_y, library=library, wg_type=waveguide_type, draw_waveguides=True)
        t = pya.Trans(pya.Trans.R0, cell.bbox().right, 0)
        cell.insert(pya.CellInstArray(cell_tree.cell_index(), t))

    cell.show()    

    # Scan the cell for all shapes on the layer 999/0. Assert that there are no shapes
    count_errors = 0
    layer_index = ly.layer(pya.LayerInfo(4, 0))
    print(f"Layer index: {layer_index}")
    for shape_obj in cell.each_shape(layer_index):
        # shape_obj is a Shape object
        # You can now work with the shape, e.g., get its polygon
        polygon = shape_obj.polygon
        print(polygon)
        count_errors += 1

    assert count_errors == 0, "There are shapes on the layer 999/0"

#    assert topcell.name == 'cell'

if __name__ == "__main__":
    test_y_splitter_tree()

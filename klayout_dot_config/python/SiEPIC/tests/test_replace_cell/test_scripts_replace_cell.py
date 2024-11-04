"""
Test for SiEPIC.scripts.replace_cell

by Lukas Chrostowski 2024

"""

def test_replace_cell():
    '''

    '''

    import pya

    import SiEPIC
    from SiEPIC._globals import Python_Env
    from SiEPIC.utils.layout import new_layout
    
    import os

    if Python_Env == 'Script':
        # For external Python mode, when installed using pip install siepic_ebeam_pdk
        import GSiP
    
    tech_name = 'GSiP'

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse("0.5.4"):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")


    from SiEPIC.scripts import replace_cell, delete_extra_topcells

    # The circuit layout that contains BB cells
    path = os.path.dirname(os.path.realpath(__file__))
    file_in = os.path.join(path,'example_bb.gds')
    layout = pya.Layout()
    layout.read(file_in)
    top_cell = layout.top_cell()

    # the individual BB reference cell
    file_bb = os.path.join(path,'ip_library_bb.gds')
    ly_bb = pya.Layout()
    ly_bb.read(file_bb)
    cell_bb = ly_bb.top_cell()
  
    # the individual WB cell
    file_wb = os.path.join(path,'ip_library_wb.gds')
    '''
    ly_wb = pya.Layout()
    ly_wb.read(file_wb)
    cell_wb = ly_wb.top_cell()
    '''
    def check_bb_geometries(layout):
        '''
        check if there are any Black Box layers in the layout
        '''
        layer_bb = layout.layer(pya.LayerInfo(998,0))  # hard coded for the GSiP PDK
        r1 = pya.Region(top_cell.begin_shapes_rec(layer_bb))
        diff_count = 0
        if not r1.is_empty():
            diff_count = r1.size()
            print(
                f" - SiEPIC.scripts.layout_diff: {r1.size()} Black Box geometry(ies) found in {top_cell.name} on layer {layout.get_info(layer_bb)}."
            )
        return diff_count

    
    # Check -- exact replacement (without $)
    if 1:
        text_out, count, error = replace_cell(layout,
                                cell_ref_bb = cell_bb,
                                cell_y_file = file_wb,  
                                Exact = True, 
                                run_layout_diff = False,
                                debug = True,
                                )
        print('replaced %s' %count)
        assert count == 1
        assert check_bb_geometries(layout) == 1

        file_out = os.path.join(path,'example_replaced.gds')
        delete_extra_topcells(layout, top_cell.name)
        layout.write(file_out)
        try:
            # Display the layout in KLayout, using KLayout Package "klive", which needs to be installed in the KLayout Application
            if Python_Env == 'Script':
                from SiEPIC.utils import klive
                klive.show(file_out, technology='EBeam')
        except:
            pass
        os.remove(file_out)


    # Check -- non-exact replacement (with $)
    if 1:
        layout = pya.Layout()
        layout.read(file_in)
        top_cell = layout.top_cell()
        text_out, count, error = replace_cell(layout,
                                cell_ref_bb = cell_bb,
                                cell_y_file = file_wb,  
                                Exact = False, RequiredCharacter='$',
                                run_layout_diff = False,
                                debug = True,
                                )
        print('replaced %s' %count)
        assert count == 2
        assert check_bb_geometries(layout) == 0

        file_out = os.path.join(path,'example_replaced2.gds')
        delete_extra_topcells(layout, top_cell.name)
        layout.write(file_out)
        try:
            # Display the layout in KLayout, using KLayout Package "klive", which needs to be installed in the KLayout Application
            if Python_Env == 'Script':
                from SiEPIC.utils import klive
                klive.show(file_out, technology='EBeam')
        except:
            pass
        os.remove(file_out)

    # Check -- Run BB reference vs. design layout difference, non-exact replacement (with $)
    if 1:
        layout = pya.Layout()
        layout.read(file_in)
        top_cell = layout.top_cell()
        text_out, count, error = replace_cell(layout,
                                cell_y_file = file_wb,  
                                Exact = False, RequiredCharacter='$',
                                cell_ref_bb = cell_bb,
                                run_layout_diff = True,
                                debug = True,
                                )
        print('replaced %s' %count)
        assert count == 2
        assert check_bb_geometries(layout) == 0
        assert error == False

    # Check -- Run BB reference (changed) vs. design layout difference, non-exact replacement (with $)
    if 1:
        layout = pya.Layout()
        layout.read(file_in)
        top_cell = layout.top_cell()
        # the (changed) BB reference cell
        file_bb = os.path.join(path,'ip_library_bb2.gds')
        ly_bb = pya.Layout()
        ly_bb.read(file_bb)
        cell_bb = ly_bb.top_cell()
        text_out, count, error = replace_cell(layout,
                                cell_ref_bb = cell_bb,
                                cell_y_file = file_wb,  
                                Exact = False, RequiredCharacter='$',
                                run_layout_diff = True,
                                debug = True,
                                )
        assert error == True

if __name__ == "__main__":
    test_replace_cell()

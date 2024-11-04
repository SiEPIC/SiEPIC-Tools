"""
Test for SiEPIC.scripts.layout_diff

by Lukas Chrostowski 2024

"""

def test_layout_diff():
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

    '''
    Create a new layout using the EBeam technology,
    with a top cell
    '''    
    cell, ly = new_layout(tech_name, 'top', GUI=True, overwrite = True)

    waveguide_type_delay='SiN routing TE 1550 nm (compound waveguide)'

    # Load cells from library
    cell1 = ly.create_cell('Ring_Modulator_DB', 'GSiP',
                                    {'r':10,
                                    })
    cell2 = ly.create_cell('Ring_Modulator_DB', 'GSiP',
                                    {'r':11,
                                    })

    from SiEPIC.scripts import layout_diff
    
    num_diff = layout_diff(cell1, cell2)
    print(num_diff)
    assert num_diff == 123

    cell3 = cell1.dup()
    num_diff = layout_diff(cell1, cell3)
    print(num_diff)
    assert num_diff == 0

if __name__ == "__main__":
    test_layout_diff()

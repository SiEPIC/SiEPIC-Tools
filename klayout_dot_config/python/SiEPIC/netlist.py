# SiEPIC-Tools
# Netlist extraction

import pya

def export_spice_layoutview(verbose=False,opt_in_selection_text=[], require_save=True):
    '''From the Layout View open in the GUI, 
    extract the netlist, and export a SPICE file
    '''

    print ('*** circuit_simulation(), opt_in: %s' % opt_in_selection_text)
    if verbose:
        print('*** circuit_simulation()')
  
    # check for supported operating system, tested on:
    # Windows 7, 10
    # OSX Monterey
    # Linux
    import sys
    if not any([sys.platform.startswith(p) for p in {"win","linux","darwin"}]):
        raise Exception("Unsupported operating system: %s" % sys.platform)
  
    from . import _globals
    from SiEPIC.utils import get_layout_variables
    TECHNOLOGY, lv, layout, topcell = get_layout_variables()
  
    
    # Save the layout prior to running simulations, if there are changes.
    if require_save:
        import pya
        mw = pya.Application.instance().main_window()
        if mw.manager().has_undo():
            mw.cm_save()
        layout_filename = mw.current_view().active_cellview().filename()
        if len(layout_filename) == 0:
            raise Exception("Please save your layout before running the simulation")
        
    return export_spice(topcell = topcell, verbose=verbose, opt_in_selection_text=opt_in_selection_text)


def export_spice(topcell=None, verbose=False,opt_in_selection_text=[]):
    '''From the layout, extract the netlist, and export a SPICE file
    '''
    # Output the Spice netlist:
    text_Spice, text_Spice_main, num_detectors, detector_list = \
        topcell.spice_netlist_export(verbose=verbose, opt_in_selection_text=opt_in_selection_text)
    if not text_Spice:
        raise Exception("No netlist available. Cannot run simulation.")
        return
    if verbose:   
        print(text_Spice)
    
    circuit_name = topcell.name.replace('.','') # remove "."
    if '_' in circuit_name[0]:
        circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _
    
    from . import _globals
    tmp_folder = _globals.TEMP_FOLDER
    import os
    filename = os.path.join(tmp_folder, '%s_main.spi' % circuit_name)
    filename_subckt = os.path.join(tmp_folder,  '%s.spi' % circuit_name)
    filename2 = os.path.join(tmp_folder, '%s.lsf' % circuit_name)
    filename_icp = os.path.join(tmp_folder, '%s.icp' % circuit_name)
    
    text_Spice_main += '.INCLUDE "%s"\n\n' % (filename_subckt)
    
    # Write the Spice netlist to file
    file = open(filename, 'w')
    file.write (text_Spice)
    file.write (text_Spice_main)
    file.close()

    return filename, filename_subckt
    
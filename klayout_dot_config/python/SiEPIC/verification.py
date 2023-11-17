#################################################################################
#                SiEPIC Tools - verification                                    #
#################################################################################
'''
by Lukas Chrostowski, 2023

'''


def layout_check(cell=None, verbose=False, GUI=False, timing=True):
    '''Functional Verification:
    Verification of things that are specific to photonic integrated circuits, including
    - Waveguides: paths, radius, bend points, Manhattan
    - Component checking: overlapping, avoiding crosstalk
    - Connectivity check: disconnected pins, mismatched pins
    - Simulation model check
    - Design for Test: Specific for each technology, check of optical IO position, direction, pitch, etc.

    Description: https://github.com/SiEPIC/SiEPIC-Tools/wiki/SiEPIC-Tools-Menu-descriptions#functional-layout-check

    Tools that can create layouts that are compatible with this Verification:
        - KLayout SiEPIC-Tools, and various PDKs such as
            https://github.com/SiEPIC/SiEPIC_EBeam_PDK
        - GDSfactory 
            "UBCPDK" https://github.com/gdsfactory/ubc
            based on https://github.com/SiEPIC/SiEPIC_EBeam_PDK
        - Luceda
            https://academy.lucedaphotonics.com/pdks/siepic/siepic.html
            https://academy.lucedaphotonics.com/pdks/siepic_shuksan/siepic_shuksan.html
    
    Limitations:
    - we assume that the layout was created based on the standard defined in SiEPIC-Tools in KLayout
      https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout
    - The layout can contain PCells, or with $$$CONTEXT_INFO$$$ removed, i.e., fixed cells
    - The layout cannot have been flattened. This allows us to isolate individual components
      by their instances and cells.
    - Parameters from cells can be extracted from the PCell, or from the text labels in the cell
    - Working with a flattened layout would be harder, and require:
       - reading parameters from the text labels (OK)
       - find_components would need to look within the DevRec layer, rather than in the selected cell
       - when pins are connected, we have two overlapping ones, so detecting them would be problematic;
         This could be solved by putting the pins inside the cells, rather than sticking out.    
    '''

    if verbose:
        print("*** layout_check()")

    if timing:
        print("*** layout_check(), timing. ")
        from time import time
        time1 = time()
        # print('Time elapsed: %s' % (time() - time1))    

        
    import pya
    try:
        from . import _globals
        from .utils import get_technology, find_paths, find_automated_measurement_labels, angle_vector
        from .utils import advance_iterator
        from ._globals import KLAYOUT_VERSION
        from .scripts import trim_netlist
    except:
        from SiEPIC import _globals
        from SiEPIC.utils import get_technology, find_paths, find_automated_measurement_labels, angle_vector
        from SiEPIC.utils import advance_iterator
        from SiEPIC._globals import KLAYOUT_VERSION
        from SiEPIC.scripts import trim_netlist
        
    TECHNOLOGY = get_technology()
    dbu = TECHNOLOGY['dbu']

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise Exception("No view selected")
    if cell is None:
        ly = lv.active_cellview().layout()
        if ly == None:
            raise Exception("No active layout")
        cell = lv.active_cellview().cell
        if cell == None:
            raise Exception("No active cell")
    else:
        ly = cell.layout()
    cv = lv.active_cellview()

    if not TECHNOLOGY['technology_name']:
        if GUI:
            v = pya.MessageBox.warning("Errors", "SiEPIC-Tools verification requires a technology to be chosen.  \n\nThe active technology is displayed on the bottom-left of the KLayout window, next to the T. \n\nChange the technology using KLayout File | Layout Properties, then choose Technology and find the correct one (e.g., EBeam, GSiP).", pya.MessageBox.Ok)
            return
        else:
            raise Exception("SiEPIC-Tools verification requires a technology to be chosen.  \n\nThe active technology is displayed on the bottom-left of the KLayout window, next to the T. \n\nChange the technology using KLayout File | Layout Properties, then choose Technology and find the correct one (e.g., EBeam, GSiP).")

    # Get the components and nets for the layout
    nets, components = cell.identify_nets(verbose=False)

    if verbose:
        print("* Display list of components:")
        [c.display() for c in components]

    if not components:
        if GUI:
            v = pya.MessageBox.warning(
                "Errors", "No components found (using SiEPIC-Tools DevRec and PinRec definitions). Cannot perform Verification.", pya.MessageBox.Ok)
            return
        else:
            raise Exception("No components found (using SiEPIC-Tools DevRec and PinRec definitions). Cannot perform Verification.")

    if timing:
        print("*** layout_check(), timing; done nets (%s), components (%s) " % (len(nets), len(components)))
        print('    Time elapsed: %s' % (time() - time1))    


    # Create a Results Database
    rdb_i = lv.create_rdb("SiEPIC-Tools Verification: %s technology" %
                          TECHNOLOGY['technology_name'])
    rdb = lv.rdb(rdb_i)
    rdb.top_cell_name = cell.name
    rdb_cell = rdb.create_cell(cell.name)

    # Waveguide checking
    rdb_cell = next(rdb.each_cell())
    rdb_cat_id_wg = rdb.create_category("Waveguide")
    rdb_cat_id_wg_path = rdb.create_category(rdb_cat_id_wg, "Path")
    rdb_cat_id_wg_path.description = "Waveguide path: Only 2 points allowed in a path. Convert to a Waveguide if necessary."
    rdb_cat_id_wg_radius = rdb.create_category(rdb_cat_id_wg, "Radius")
    rdb_cat_id_wg_radius.description = "Not enough space to accommodate the desired bend radius for the waveguide."
    rdb_cat_id_wg_bendpts = rdb.create_category(rdb_cat_id_wg, "Bend points")
    rdb_cat_id_wg_bendpts.description = "Waveguide bend should have more points per circle."
    rdb_cat_id_wg_manhattan = rdb.create_category(rdb_cat_id_wg, "Manhattan")
    rdb_cat_id_wg_manhattan.description = "The first and last waveguide segment need to be Manhattan (vertical or horizontal) so that they can connect to device pins."

    # Component checking
    rdb_cell = next(rdb.each_cell())
    rdb_cat_id_comp = rdb.create_category("Component")
    rdb_cat_id_comp_flat = rdb.create_category(rdb_cat_id_comp, "Flattened component")
    rdb_cat_id_comp_flat.description = "SiEPIC-Tools Verification, Netlist extraction, and Simulation only functions on hierarchical layouts, and not on flattened layouts.  Add to the discussion here: https://github.com/lukasc-ubc/SiEPIC-Tools/issues/37"
    rdb_cat_id_comp_overlap = rdb.create_category(rdb_cat_id_comp, "Overlapping component")
    rdb_cat_id_comp_overlap.description = "Overlapping components (defined as overlapping DevRec layers; touching is ok)"
    rdb_cat_id_comp_shapesoutside = rdb.create_category(rdb_cat_id_comp, "Shapes outside component")
    rdb_cat_id_comp_shapesoutside.description = "Shapes for device layers need to be inside a component. At minimum, they must be inside a cell and inside a DevRec shape. Read more about requirements for components: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout"
    rdb_cat_id_comp_pinerrors = rdb.create_category(rdb_cat_id_comp, "Invalid Pin")
    rdb_cat_id_comp_pinerrors.description = "Invalid pin found. Read more about requirements for components: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout"

    # Connectivity checking
    rdb_cell = next(rdb.each_cell())
    rdb_cat_id = rdb.create_category("Connectivity")
    rdb_cat_id_discpin = rdb.create_category(rdb_cat_id, "Disconnected pin")
    rdb_cat_id_discpin.description = "Disconnected pin"
    rdb_cat_id_mismatchedpin = rdb.create_category(rdb_cat_id, "Mismatched pin")
    rdb_cat_id_mismatchedpin.description = "Mismatched pin widths"

    # Simulation checking
    # disabled by lukasc, 2021/05
    if 0:
        rdb_cell = next(rdb.each_cell())
        rdb_cat_id = rdb.create_category("Simulation")
        rdb_cat_id_sim_nomodel = rdb.create_category(rdb_cat_id, "Missing compact model")
        rdb_cat_id_sim_nomodel.description = "A compact model for this component was not found. Possible reasons: 1) Please run SiEPIC | Simulation | Setup Lumerical INTERCONNECT and CML, to make sure that the Compact Model Library is installed in INTERCONNECT, and that KLayout has a list of all component models. 2) the library does not have a compact model for this component. "

    # Design for Test checking
    from SiEPIC.utils import load_DFT
    DFT = load_DFT()
    if DFT:
        if verbose:
            print(DFT)
        rdb_cell = next(rdb.each_cell())
        rdb_cat_id = rdb.create_category("Design for test")
        rdb_cat_id_optin_unique = rdb.create_category(rdb_cat_id, "opt_in label: same")
        rdb_cat_id_optin_unique.description = "Automated test opt_in labels should be unique."
        rdb_cat_id_optin_missing = rdb.create_category(rdb_cat_id, "opt_in label: missing")
        rdb_cat_id_optin_missing.description = "Automated test opt_in labels are required for measurements on the Text layer. \n\nDetails on the format for the opt_in labels can be found at https://github.com/lukasc-ubc/SiEPIC-Tools/wiki/SiEPIC-Tools-Menu-descriptions#connectivity-layout-check"
        rdb_cat_id_optin_toofar = rdb.create_category(rdb_cat_id, "opt_in label: too far away")
        rdb_cat_id_optin_toofar.description = "Automated test opt_in labels must be placed at the tip of the grating coupler, namely near the (0,0) point of the cell."
        rdb_cat_id_optin_wavelength = rdb.create_category(rdb_cat_id, "opt_in label: wavelength")
        if type(DFT['design-for-test']['tunable-laser']) == list:
            DFT_wavelengths = [w['wavelength'] for w in DFT['design-for-test']['tunable-laser']]
        else:
            DFT_wavelengths = DFT['design-for-test']['tunable-laser']['wavelength']
        rdb_cat_id_optin_wavelength.description = "Automated test opt_in labels must have a wavelength for a laser specified in the DFT.xml file: %s.  \n\nDetails on the format for the opt_in labels can be found at https://github.com/lukasc-ubc/SiEPIC-Tools/wiki/SiEPIC-Tools-Menu-descriptions#connectivity-layout-check" % DFT_wavelengths
        if type(DFT['design-for-test']['tunable-laser']) == list:
            DFT_polarizations = [p['polarization']
                                 for p in DFT['design-for-test']['tunable-laser']]
        else:
            if 'polarization' in DFT['design-for-test']['tunable-laser']:
                DFT_polarizations = DFT['design-for-test']['tunable-laser']['polarization']
            else:
                DFT_polarizations = "TE or TM"
        rdb_cat_id_optin_polarization = rdb.create_category(
            rdb_cat_id, "opt_in label: polarization")
        rdb_cat_id_optin_polarization.description = "Automated test opt_in labels must have a polarization as specified in the DFT.xml file: %s. \n\nDetails on the format for the opt_in labels can be found at https://github.com/lukasc-ubc/SiEPIC-Tools/wiki/SiEPIC-Tools-Menu-descriptions#connectivity-layout-check" % DFT_polarizations
#    rdb_cat_id_GCpitch = rdb.create_category(rdb_cat_id, "Grating Coupler pitch")
#    rdb_cat_id_GCpitch.description = "Grating couplers must be on a %s micron pitch, vertically arranged, as specified in the DFT.xml." % (float(DFT['design-for-test']['grating-couplers']['gc-pitch']))
        rdb_cat_id_GCorient = rdb.create_category(rdb_cat_id, "Grating coupler orientation")
        rdb_cat_id_GCorient.description = "The grating coupler is not oriented (rotated) the correct way for automated testing."
        rdb_cat_id_GCarrayconfig = rdb.create_category(rdb_cat_id, "Fibre array configuration")
        array_angle = (float(DFT['design-for-test']['grating-couplers']['gc-array-orientation']))%360.0
        if array_angle==0:
          dir1 = ' right of '
          dir2 = ' left of '
          dir3 = 'horizontally'
        elif array_angle==90:
          dir1 = ' above '
          dir2 = ' below '
          dir3 = 'vertically'
        elif array_angle == 180:
          dir1 = ' left of '
          dir2 = ' right of '
          dir3 = 'horizontally'
        else:
          dir1 = ' below '
          dir2 = ' above '
          dir3 = 'vertically'
        
        rdb_cat_id_GCarrayconfig.description = "Circuit must be connected such that there is at most %s Grating Coupler(s) %s the opt_in label (laser injection port) and at most %s Grating Coupler(s) %s the opt_in label. \nGrating couplers must be on a %s micron pitch, %s arranged." % (
            int(DFT['design-for-test']['grating-couplers']['detectors-above-laser']), dir1,int(DFT['design-for-test']['grating-couplers']['detectors-below-laser']), dir2,float(DFT['design-for-test']['grating-couplers']['gc-pitch']),dir3)

    else:
        if verbose:
            print('  No DFT rules found.')

    if timing:
        print("*** layout_check(), timing; done DFT")
        print('    Time elapsed: %s' % (time() - time1))    

    paths = find_paths(TECHNOLOGY['Waveguide'], cell=cell)
    for p in paths:
        if verbose:
            print("%s, %s" % (type(p), p))
        # Check for paths with > 2 vertices
        Dpath = p.to_dtype(dbu)
        if Dpath.num_points() > 2:
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_path.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(Dpath.polygon()))

    if timing:
        print("*** layout_check(), timing; done invalid Waveguide paths")
        print('    Time elapsed: %s' % (time() - time1))    

    '''
    check for invalid pins
    '''
    # Get all the pins, and possible pin errors
    pins, pin_errors = cell.find_pins()
    if pin_errors:
        for p in pin_errors:
            if p[0].polygon():
                print (p)
                print(p[0].polygon())
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_comp_pinerrors.rdb_id())
                rdb_item.add_value(pya.RdbItemValue(p[0].polygon().to_dtype(dbu)))
                # .transformed(p[1].to_trans().to_itrans(dbu))

    if timing:
        print("*** layout_check(), timing; done invalid pins ")
        print('    Time elapsed: %s' % (time() - time1))    

    

    '''
    Shapes need to be inside a component, for device layers.
        EBeam PDK:
            - Si 1\0
            - SiN 1\5
            - M1_heater 11\0
    Read more about requirements for components: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout
    Method:
        - find all shapes: make a list
        - find all components, and their shapes: make a list
        - substract the two lists, and produce errors
    rdb_cat_id_comp_shapesoutside
    '''
    from SiEPIC.utils import load_Verification
    verification = load_Verification()
    if verification:
        print(verification)
        # define device-only layers
        try:
            deviceonly_layers = eval(verification['verification']['shapes-inside-components']['deviceonly-layers'])
        except:
            deviceonly_layers = [ [1,0] ]
        deviceonly_layers_ids = [ly.find_layer(*l) for l in deviceonly_layers if ly.find_layer(*l) is not None]
        # print(deviceonly_layers_ids)
        if verbose:
            print(" - checking that shapes are inside components" )
      
        # get cells for all components
        cells_components = []
        for i in range(0, len(components)):
            cells_components.append ( components[i].cell )
        
        # get shapes from layout that aren't part of the component cells
        if verbose:
            # display all shapes
            iter1 = pya.RecursiveShapeIterator(ly, cell, deviceonly_layers_ids )
            while not iter1.at_end():
                print("   - %s" % iter1.shape())
                iter1.next()        
        iter1 = pya.RecursiveShapeIterator(ly, cell, deviceonly_layers_ids )
        iter1.unselect_cells([c.cell_index() for c in cells_components if c is not None] )
        extra_shapes = []
        while not iter1.at_end():
            # make sure the list has unique elements
            if [iter1.shape(), iter1.itrans()] not in extra_shapes:
                extra_shapes.append([iter1.shape(), iter1.itrans()])
            iter1.next()
        if verbose:
            print(" - found %s shape(s) not belonging to components " % len(extra_shapes) )
            for e in extra_shapes:
                print( "   - %s, %s" % (e[0], e[1]) )
        # add shapes into the results database
        for e in extra_shapes:
            if e[0].dpolygon:
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_comp_shapesoutside.rdb_id())
                rdb_item.add_value(pya.RdbItemValue(e[0].dpolygon.transformed(e[1].to_trans().to_itrans(dbu))))

    if timing:
        print("*** layout_check(), timing; done shapes in component ")
        print('    Time elapsed: %s' % (time() - time1))    


    # Experimental, attempt to break up the circuit into regions connected by DevRec layers
    region = pya.Region()
    for i in range(0, len(components)):
        c = components[i]
        region += pya.Region(c.polygon)
    print ('DevRec Regions: original %s, merged %s' % (region.count(), region.merge().count()))
    '''
    Approach: create lists of components for each merged region, then do the verification on a per-merged-region basis
    reduce the O(n**2) to O((n/10)**2)  (assuming on average 10 components per circuit)
    '''

    if timing:
        print("*** layout_check(), timing; counting merged DevRec regions")
        print('    Time elapsed: %s' % (time() - time1))    

        
    '''
    Component checks:
    '''
    for i in range(0, len(components)):
        c = components[i]
        # the following only works for layouts where the Waveguide is still a PCells (not flattened)
        # basic_name is assigned in Cell.find_components, by reading the PCell parameter
        # if the layout is flattened, we don't have an easy way to get the path
        # it could be done perhaps as a parameter (points)
        if c.basic_name == "Waveguide" and c.cell.is_pcell_variant():
            pcell_params = c.cell.pcell_parameters_by_name()
            Dpath = pcell_params['path']
            if 'radius' in pcell_params:
                radius = pcell_params['radius']
            else:
                radius = 5
            if verbose:
                print(" - Waveguide: cell: %s, %s" % (c.cell.name, radius))

            # Radius check:
            if not Dpath.radius_check(radius):
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_radius.rdb_id())
                rdb_item.add_value(pya.RdbItemValue( "The minimum radius is set at %s microns for this waveguide." % (radius) ))
                rdb_item.add_value(pya.RdbItemValue(Dpath))

            # Check for waveguides with too few bend points

            # Check if waveguide end segments are Manhattan; this ensures they can connect to a pin
            if not Dpath.is_manhattan_endsegments():
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_manhattan.rdb_id())
                rdb_item.add_value(pya.RdbItemValue(Dpath))

        if c.basic_name == "Flattened":
            if verbose:
                print(" - Component: Flattened: %s" % (c.polygon))
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_comp_flat.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(c.polygon.to_dtype(dbu)))

        # check all the component's pins to check if they are assigned a net:
        for pin in c.pins:
            if pin.type == _globals.PIN_TYPES.OPTICAL and pin.net.idx == None:
                # disconnected optical pin
                if verbose:
                    print(" - Found disconnected pin, type %s, at (%s)" % (pin.type, pin.center))
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_discpin.rdb_id())
                rdb_item.add_value(pya.RdbItemValue(pin.path.to_dtype(dbu)))

        # Verification: overlapping components (DevRec)
            # automatically takes care of waveguides crossing other waveguides & components
        # Region: put in two DevRec polygons (in raw), measure area, merge, check if are is the same
        #  checks for touching but not overlapping DevRecs
        for i2 in range(i + 1, len(components)):
            
            c2 = components[i2]
            r1 = pya.Region(c.polygon)
            r2 = pya.Region(c2.polygon)
            polygon_and = [p for p in r1 & r2]
            if polygon_and:
                print(" - Found overlapping components: %s, %s" % (c.component, c2.component))
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_comp_overlap.rdb_id())
                if c.component == c2.component:
                    rdb_item.add_value(pya.RdbItemValue(
                        "There are two identical components overlapping: " + c.component))
                for p in polygon_and:
                    rdb_item.add_value(pya.RdbItemValue(p.to_dtype(dbu)))
                # check if these components have the same name; possibly a copy and paste error
        if DFT:
            # DFT verification
            # GC facing the right way
            if c.basic_name:
                ci = c.basic_name  # .replace(' ','_').replace('$','_')
                gc_orientation_error = False
                for gc in DFT['design-for-test']['grating-couplers']['gc-orientation'].keys():
                    DFT_GC_angle = int(DFT['design-for-test']['grating-couplers']['gc-orientation'][gc])
                    if ci.startswith(gc) and c.trans.angle != DFT_GC_angle:
                        if verbose:
                            print(" - Found DFT error, GC facing the wrong way: %s, %s" %
                                  (c.component, c.trans.angle))
                        rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_GCorient.rdb_id())
                        rdb_item.add_value(pya.RdbItemValue( "Cell %s should be %s degrees" % (ci,DFT_GC_angle) ))
                        rdb_item.add_value(pya.RdbItemValue(c.polygon.to_dtype(dbu)))


        # Pre-simulation check: do components have models?
        # disabled by lukasc, 2021/05
        if 0 and not c.has_model():
            if verbose:
                print(" - Missing compact model, for component: %s" % (c.component))
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_sim_nomodel.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(c.polygon.to_dtype(dbu)))

    if timing:
        print("*** layout_check(), timing; done components check ")
        print('    Time elapsed: %s' % (time() - time1))    

    if DFT:
        # DFT verification

        text_out, opt_in = find_automated_measurement_labels(cell)

        '''
    # opt_in labels missing: 0 labels found. draw box around the entire circuit.
    # replaced with code below that finds each circuit separately
    if len(opt_in) == 0:
      rdb_item = rdb.create_item(rdb_cell.rdb_id(),rdb_cat_id_optin_missing.rdb_id())
      rdb_item.add_value(pya.RdbItemValue( pya.Polygon(cell.bbox()).to_dtype(dbu) ) )
    '''

        # dataset for all components found connected to opt_in labels; later
        # subtract from all components to find circuits with missing opt_in
        components_connected_opt_in = []

        # opt_in labels
        for ti1 in range(0, len(opt_in)):
            if 'opt_in' in opt_in[ti1]:
                t = opt_in[ti1]['Text']
                box_s = 1000
                box = pya.Box(t.x - box_s, t.y - box_s, t.x + box_s, t.y + box_s)
                # opt_in labels check for unique
                for ti2 in range(ti1 + 1, len(opt_in)):
                    if 'opt_in' in opt_in[ti2]:
                        if opt_in[ti1]['opt_in'] == opt_in[ti2]['opt_in']:
                            if verbose:
                                print(" - Found DFT error, non unique text labels: %s, %s, %s" %
                                      (t.string, t.x, t.y))
                            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_optin_unique.rdb_id())
                            rdb_item.add_value(pya.RdbItemValue(t.string))
                            rdb_item.add_value(pya.RdbItemValue(pya.Polygon(box).to_dtype(dbu)))
    
                # opt_in format check:
                if not opt_in[ti1]['wavelength'] in DFT_wavelengths:
                    if verbose:
                        print(" - DFT error: wavelength")
                    rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_optin_wavelength.rdb_id())
                    rdb_item.add_value(pya.RdbItemValue(pya.Polygon(box).to_dtype(dbu)))
    
                if not (opt_in[ti1]['pol'] in DFT_polarizations):
                    if verbose:
                        print(" - DFT error: polarization")
                    rdb_item = rdb.create_item(
                        rdb_cell.rdb_id(), rdb_cat_id_optin_polarization.rdb_id())
                    rdb_item.add_value(pya.RdbItemValue(pya.Polygon(box).to_dtype(dbu)))
    
                # find the GC closest to the opt_in label.
                components_sorted = sorted([c for c in components if [p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO]],
                                           key=lambda x: x.trans.disp.to_p().distance(pya.Point(t.x, t.y).to_dtype(1)))
                # GC too far check:
                if components_sorted:
                    dist_optin_c = components_sorted[0].trans.disp.to_p(
                    ).distance(pya.Point(t.x, t.y).to_dtype(1))
                    if verbose:
                        print(" - Found opt_in: %s, nearest GC: %s.  Locations: %s, %s. distance: %s" % (opt_in[ti1][
                              'Text'], components_sorted[0].instance,  components_sorted[0].center, pya.Point(t.x, t.y), dist_optin_c * dbu))
                    if dist_optin_c > float(DFT['design-for-test']['opt_in']['max-distance-to-grating-coupler']) * 1000:
                        if verbose:
                            print(" - opt_in label too far from the nearest grating coupler: %s, %s" %
                                  (components_sorted[0].instance, opt_in[ti1]['opt_in']))
                        rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_optin_toofar.rdb_id())
                        rdb_item.add_value(pya.RdbItemValue(pya.Polygon(box).to_dtype(dbu)))
    
                    # starting with each opt_in label, identify the sub-circuit, then GCs, and
                    # check for GC spacing
                    trimmed_nets, trimmed_components = trim_netlist(
                        nets, components, components_sorted[0])
                    components_connected_opt_in = components_connected_opt_in + trimmed_components
                    detector_GCs = [c for c in trimmed_components if [p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO] if (
                        c.trans.disp - components_sorted[0].trans.disp).to_p() != pya.DPoint(0, 0)]
                    if verbose:
                        print("   N=%s, detector GCs: %s" %
                              (len(detector_GCs), [c.display() for c in detector_GCs]))
                    vect_optin_GCs = [(c.trans.disp - components_sorted[0].trans.disp).to_p()
                                      for c in detector_GCs]
                    # for vi in range(0,len(detector_GCs)):
                    #  if round(angle_vector(vect_optin_GCs[vi])%180)!=int(DFT['design-for-test']['grating-couplers']['gc-array-orientation']):
                    #    if verbose:
                    #      print( " - DFT GC pitch or angle error: angle %s, %s"  % (round(angle_vector(vect_optin_GCs[vi])%180), opt_in[ti1]['opt_in']) )
                    #    rdb_item = rdb.create_item(rdb_cell.rdb_id(),rdb_cat_id_GCpitch.rdb_id())
                    #    rdb_item.add_value(pya.RdbItemValue( detector_GCs[vi].polygon.to_dtype(dbu) ) )
    
                    # find the GCs in the circuit that don't match the testing configuration
                    import numpy as np
                    array_angle = float(DFT['design-for-test']['grating-couplers']['gc-array-orientation'])
                    pitch = float(DFT['design-for-test']['grating-couplers']['gc-pitch'])
                    sx = np.round(np.cos(array_angle/180*np.pi))
                    sy = np.round(np.sin(array_angle/180*np.pi))
                    
                    for d in list(range(int(DFT['design-for-test']['grating-couplers']['detectors-above-laser']) + 0, 0, -1)) + list(range(-1, -int(DFT['design-for-test']['grating-couplers']['detectors-below-laser']) - 1, -1)):
                        if pya.DPoint(d * sx* pitch * 1000, d *sy* pitch * 1000) in vect_optin_GCs:
                            del_index = vect_optin_GCs.index(pya.DPoint(
                                d * sx* pitch * 1000, d *sy* pitch * 1000))
                            del vect_optin_GCs[del_index]
                            del detector_GCs[del_index]
                    for vi in range(0, len(vect_optin_GCs)):
                        if verbose:
                            print(" - DFT GC array config error: %s, %s" %
                                  (components_sorted[0].instance, opt_in[ti1]['opt_in']))
                        rdb_item = rdb.create_item(
                            rdb_cell.rdb_id(), rdb_cat_id_GCarrayconfig.rdb_id())
                        rdb_item.add_value(pya.RdbItemValue(
                            "The label having the error is: " + opt_in[ti1]['opt_in']))
                        rdb_item.add_value(pya.RdbItemValue(detector_GCs[vi].polygon.to_dtype(dbu)))
                        rdb_item.add_value(pya.RdbItemValue(pya.Polygon(box).to_dtype(dbu)))

        # subtract components connected to opt_in labels from all components to
        # find circuits with missing opt_in
        components_without_opt_in = [
            c for c in components if not (c in components_connected_opt_in)]
        i = 0  # to avoid getting stuck in the loop in case of an error
        while components_without_opt_in and i < 50:
            # find the first GC
            components_GCs = [c for c in components_without_opt_in if [
                p for p in c.pins if p.type == _globals.PIN_TYPES.OPTICALIO]]
            if components_GCs:
                trimmed_nets, trimmed_components = trim_netlist(
                    nets, components, components_GCs[0])
                # circuit without opt_in label, generate error
                r = pya.Region()
                for c in trimmed_components:
                    r.insert(c.polygon)
                for p in r.each_merged():
                    rdb_item = rdb.create_item(
                        rdb_cell.rdb_id(), rdb_cat_id_optin_missing.rdb_id())
                    rdb_item.add_value(pya.RdbItemValue(p.to_dtype(dbu)))
                # remove from the list of components without opt_in:
                components_without_opt_in = [
                    c for c in components_without_opt_in if not (c in trimmed_components)]
                i += 1
            else:
                break

        # GC spacing between separate GC circuits (to avoid measuring the wrong one)

    if timing:
        print("*** layout_check(), timing; done DFT ")
        print('    Time elapsed: %s' % (time() - time1))    

    for n in nets:
        # Verification: optical pin width mismatches
        if n.type == _globals.PIN_TYPES.OPTICAL and not n.idx == None:
            pin_paths = [p.path for p in n.pins]
            if pin_paths[0].width != pin_paths[-1].width:
                if verbose:
                    print(" - Found mismatched pin widths: %s" % (pin_paths[0]))
                r = pya.Region([pin_paths[0].to_itype(1).polygon(),
                                pin_paths[-1].to_itype(1).polygon()])
                polygon_merged = advance_iterator(r.each_merged())
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_mismatchedpin.rdb_id())
                rdb_item.add_value(pya.RdbItemValue( "Pin widths: %s, %s" % (pin_paths[0].width, pin_paths[-1].width)  ))
                rdb_item.add_value(pya.RdbItemValue(polygon_merged.to_dtype(dbu)))

    if timing:
        print("*** layout_check(), timing; done pin mismatch ")
        print('    Time elapsed: %s' % (time() - time1))    

    # displays results in Marker Database Browser, using Results Database (rdb)
    if rdb.num_items() > 0:
        if GUI:
            v = pya.MessageBox.warning(
                "Errors", "%s layout errors detected.  \nPlease review errors using the 'Marker Database Browser'." % rdb.num_items(), pya.MessageBox.Ok)
            lv.show_rdb(rdb_i, cv.cell_index)
    else:
        if GUI:
            v = pya.MessageBox.warning("Errors", "No layout errors detected.", pya.MessageBox.Ok)

    # Save results of verification as a Text label on the cell. Include OS,
    # SiEPIC-Tools and PDK version info.
    LayerTextN = cell.layout().layer(TECHNOLOGY['Text'])
    iter1 = cell.begin_shapes_rec(LayerTextN)
    while not(iter1.at_end()):
        if iter1.shape().is_text():
            text = iter1.shape().text
            if text.string.find("SiEPIC-Tools verification") > -1:
                text_SiEPIC = text
                print(" * Previous label: %s" % text_SiEPIC)
                iter1.shape().delete()
        iter1.next()

    import SiEPIC.__init__
    import sys
    from time import strftime
    text = pya.DText("SiEPIC-Tools verification: %s errors\n%s\nSiEPIC-Tools v%s\ntechnology: %s\n%s\nPython: %s, %s\n%s" % (rdb.num_items(), strftime("%Y-%m-%d %H:%M:%S"),
                                                                                                                             SiEPIC.__init__.__version__, TECHNOLOGY['technology_name'], sys.platform, sys.version.split('\n')[0], sys.path[0], pya.Application.instance().version()), pya.DTrans(cell.dbbox().p1))
    shape = cell.shapes(LayerTextN).insert(text)
    shape.text_size = 0.1 / dbu

    if timing:
        print("*** layout_check(), timing; all done. ")
        print('    Time elapsed: %s' % (time() - time1))    

if __name__ == "__main__":
    print('SiEPIC-Tools functional verification')
    from SiEPIC.utils import get_layout_variables, load_Waveguides_by_Tech
    TECHNOLOGY, lv, layout, cell = get_layout_variables()  
    layout_check(cell=cell, verbose=True)
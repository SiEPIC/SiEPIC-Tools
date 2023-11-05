
#################################################################################
#                SiEPIC Tools - metal_menu_helper                               #
#################################################################################
'''

Overview:
Functions to control the adding of pins to paths on metal layers. The functions follow essentially the same
process as the functions which they were derived from. All original functions and changes are specified above each function declaration.

Author - Karl McNulty (Rochester Institute of Technology)

List of Functions:
get_layout_variables_m
select_paths_m
select_wireguides
path_to_wireguide
wireguide_to_path

'''

import pya
import os

# ORIGINAL FUNCTION - get_layout_variables()
# The changes to the original function are as follows:
# 1) TECHNOLOGY is found using the "get_technology_by_name()" function instead of the "get_technology()" function
def get_layout_variables_m():
    from .utils import get_technology_by_name

    # grab the current technology name being used
    lv = pya.Application.instance().main_window().current_view()
    tech_name = lv.active_cellview().technology

    TECHNOLOGY = get_technology_by_name(tech_name) # get the technology info

    # Configure variables to find in the presently selected cell:
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        print("No view selected")
        raise UserWarning("No view selected. Make sure you have an open layout.")
    # Find the currently selected layout.
    ly = pya.Application.instance().main_window().current_view().active_cellview().layout()
    if ly == None:
        raise UserWarning("No layout. Make sure you have an open layout.")
    # find the currently selected cell:
    cv = pya.Application.instance().main_window().current_view().active_cellview()
    cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    if cell == None:
        raise UserWarning("No cell. Make sure you have an open layout.")

    return TECHNOLOGY, lv, ly, cell

# ORIGINAL FUNCTION - select_paths()
# The changes to the original function are as follows:
# 1) The function takes in a list of layer pairs (LayerInfo, str(LayerName)) which the function will look for paths with those layers
#    when no selection is made (no component is specifically clicked on within KLayout). The paths with those layers will be converted
#    to wireguides.
# 2) A overarching for loop is created to iterate through all the desired layers (within the 'layers' parameter), in order
#    to find all paths within the layout which are drawn in that layer. Note that this functionality only occurs when
#    no components are selected in KLayout (just hit '3' without selecting a component and all metal layer paths will be converted
#    to wireguides)
# 3) A final "for" loop is added at the end of the empty section path (no objects are selected) in order to apply a property called
#    "LayerName" to the object's shape. This property is used in the "path_to_wireguide()" function to create the pcell in the correct layer 
# 4) Another parameter is added in the "if" statement determining if an object is a path. The added parameter checks to make sure the object
#    is also not a pcell (as wireguides are still considered paths by the "is_path()" function)
# 5) Changed all "waveguide" names and variables to "wireguides"
def select_paths_m(layers, cell=None, verbose=None):

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

    selection = lv.object_selection
    if verbose:
        print("SiEPIC.utils.select_paths: selection, before: %s" % lv.object_selection)
    if selection == []:
        for layer, layer_str_name in layers: # run through all the layers inputted into the function (allows for multiple path layer selections)
            itr = cell.begin_shapes_rec(ly.layer(layer))
            while not(itr.at_end()):
                if verbose:
                    print("SiEPIC.utils.select_paths: itr: %s" % itr)
                if itr.shape().is_path() and not itr.cell().pcell_parameters(): # must be a path and not a pcell
                    print('Name: ' + str(itr.cell().pcell_parameters()))
                    if verbose:
                        print("SiEPIC.utils.select_paths: path: %s" % itr.shape())
                    selection.append(pya.ObjectInstPath())
                    selection[-1].layer = ly.layer(layer)
                    selection[-1].shape = itr.shape()
                    selection[-1].top = cell.cell_index()
                    selection[-1].cv_index = 0
                itr.next()
            #for o in selection: # set the LayerName property if the object (path) does not already have a LayerName
            #    if not o.shape.property('LayerName'): # check for LayerName property
            #        o.shape.set_property('LayerName', layer_str_name) # add LayerName propery if not present
        lv.object_selection = selection # set the return item equal to selection after all loops are finished
    else:
        lv.object_selection = [o for o in selection if (not o.is_cell_inst() and not o.shape.cell.pcell_parameters()) and o.shape.is_path()] # must be a path and not a pcell
    if verbose:
        print("SiEPIC.utils.select_paths: selection, after: %s" % lv.object_selection)
    return lv.object_selection	

# ORIGINAL FUNCTION - select_waveguides()
# The changes to the original function are as follows:
# 1) The name for identification of correct guides to find is changed to Wireguide (in order to find wireguides and
#    not waveguides)
# 2) Changed all "waveguide" names and variables to "wireguides"
def select_wireguides(cell=None):
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

    selection = lv.object_selection
    if selection == []:
        for instance in cell.each_inst():
            if instance.cell.basic_name() == "Wireguide": # replace "Waveguide" with "Wireguide"
                selection.append(pya.ObjectInstPath())
                selection[-1].top = cell.cell_index()
                selection[-1].append_path(pya.InstElement.new(instance))
        lv.object_selection = selection
    else:
        lv.object_selection = [o for o in selection if o.is_cell_inst(
        ) and o.inst().cell.basic_name() == "Wireguide"]

    return lv.object_selection
	

# ORIGINAL FUNCTION - path_to_waveguide()
# The changes to the original function are as follows:
# 1) No GUI is necessary for the creation of the wireguides. All parameters (params) are specified below, as the wireguides
#    do not require much need for user input. Note that the width of the wireguide is set to be whatever the path width the 
#    wireguide was created from (so the wireguides maintain the width of the path drawn)
# 2) The selected paths are chosen based upon layers indicated in the xml layer files (looks for "Wireguide_...")
# 3) The inserted component (though essentially the same as a waveguide pcell) is changed to be named Wireguide (and a corresponding pcell
#    declaration is made within the instance library in KLayout)
# 4) The name of the layer that the wireguide is found by two methods:
#    a) If no mouse selection is made, the layer name is found from the 'LayerName' property saved within the object's shape (which was
#       done in the"select_paths_m()" function)
#    b) If a mouse selection is made, the layer name is found by searching through all the layout layers and finding the layer that was
#       selected by the mouse. The string name is then taken for that found layer and saved in a local variable
# 5) Changed all "waveguide" names and variables to "wireguides"
def path_to_wireguide(cell=None, lv_commit=True, verbose=False, select_wireguides=False):
    from . import _globals
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables_m()

    if not cell:
        cell = top_cell

    if verbose:
        print("SiEPIC.scripts path_to_wireguide()" )

    if lv_commit:
        lv.transaction("Path to Wireguide")

    layers_to_select = list()
    for key in TECHNOLOGY.keys():
        if 'Wireguide' in key:
            layers_to_select.append((TECHNOLOGY[key], key))
            print(key)

    selected_paths = select_paths_m(layers_to_select, cell, verbose=verbose) # find all paths on the selected layers

    if verbose:
        print("SiEPIC.scripts path_to_wireguide(): selected_paths = %s" % selected_paths)
    selection = []

    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    
    params = {'radius' : 0, 'width' : 2, 'adiabatic' : False, 'bezier' : '0', 'offset' : 0} # set parameters

    if verbose:
        print("SiEPIC.scripts path_to_wireguide(): params = %s" % params)

    for obj in selected_paths:
        path = obj.shape.path

        dbu = obj.layout().dbu # get the database units
        params['width'] = path.width * dbu # adjust the width and save for wireguide creation
        if obj.shape.property('LayerName'): # check if obj has LayerName (for no mouse selections)
            input_layer_name = obj.shape.property('LayerName') # save layer name for wireguide creation
        else: # find the name of the layer if not specified (for mouse selections)
            lv  = pya.Application.instance().main_window().current_view()
            ly  = lv.active_cellview().layout()
            dbu = ly.dbu
            lp_found = None
            iter = lv.begin_layers() # loop through all layers
            while not iter.at_end():
                lp = iter.current()
                if lp.cellview() == obj.cv_index and lp.layer_index() == obj.layer: # find specified layer within the layer loop
                    lp_found = lp # save layer properties
                iter.next()
            input_layer_name = lp_found.name # save layer name for wireguide creation

        path.unique_points()
        if not path.is_manhattan_endsegments():
            warning.setText(
                "Warning: Wireguide segments (first, last) are not Manhattan (vertical, horizontal).")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return
        if not path.is_manhattan():
            warning.setText(
                "Error: Wireguide segments are not Manhattan (vertical, horizontal). This is not supported in SiEPIC-Tools.")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return
        if not path.radius_check(params['radius'] / TECHNOLOGY['dbu']):
            warning.setText(
                "Warning: One of the wireguide segments has insufficient length to accommodate the desired bend radius.")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return

        pins,_ = cell.find_pins()
        path.snap_m(pins)
        Dpath = path.to_dtype(TECHNOLOGY['dbu'])
        width_devrec = params['width'] + _globals.WG_DEVREC_SPACE * 2 # get DevRec width based on the wireguide width found earlier
        try:
            pcell = ly.create_cell("Wireguide", TECHNOLOGY['technology_name'], {"path": Dpath, # input parameters
                                                                                "radius": params['radius'],
                                                                                "width": params['width'],
                                                                                "adiab": params['adiabatic'],
                                                                                "bezier": params['bezier'],
                                                                                "layers": [input_layer_name] + ['DevRec'], # set the layer as the same as the path that the wireguide came from (along with DevRec)
                                                                                "widths": [params['width']] + [width_devrec],
                                                                                "offsets": [params['offset']] + [0]})
            print("SiEPIC.metal_menu_helper.path_to_wireguide(): Wireguide from %s, %s" %
                  (TECHNOLOGY['technology_name'], pcell))
        except:
            pass
        if not pcell:
            try:
                pcell = ly.create_cell("Wireguide", "SiEPIC General", {"path": Dpath, # input parameters
                                                                       "radius": params['radius'],
                                                                       "width": params['width'],
                                                                       "adiab": params['adiabatic'],
                                                                       "bezier": params['bezier'],
                                                                       "layers": [input_layer_name] + ['DevRec'], # set the layer as the same as the path that the wireguide came from (along with DevRec)
                                                                       "widths": [params['width']] + [width_devrec],
                                                                       "offsets": [params['offset']] + [0]})
                print("SiEPIC.metal_menu_helper.path_to_wireguide(): Wireguide from SiEPIC General, %s" % pcell)
            except:
                pass
        if not pcell:
            raise Exception(
                "'Wireguide' in 'SiEPIC General' library is not available. Check that the library was loaded successfully.")
        selection.append(pya.ObjectInstPath())
        selection[-1].top = obj.top
        selection[-1].append_path(pya.InstElement.new(cell.insert(
            pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))))

        obj.shape.delete()
	
    lv.clear_object_selection()
    if select_wireguides:
        lv.object_selection = selection
    if lv_commit:
        lv.commit()

# ORIGINAL FUNCTION - waveguide_to_path()
# The changes to the original function are as follows:
# 1) Have the path maintain the same layer as it previuosly had as a Wireguide
# 2) Changed all "waveguide" names and variables to "wireguides"
def wireguide_to_path(cell=None):
    from . import _globals
    from .utils import get_layout_variables

    if cell is None:
        TECHNOLOGY, lv, ly, cell = get_layout_variables()
    else:
        TECHNOLOGY, lv, _, _ = get_layout_variables()
        ly = cell.layout()

    lv.transaction("wireguide to path")

    # record objects to delete:
    to_delete = []

    wireguides = select_wireguides(cell)
    selection = []
    for obj in wireguides:
        # path from wireguide guiding shape
        wireguide = obj.inst()
        layer_list =  wireguide.cell.pcell_parameter('layers') # get the list of layers in the wireguide pcell (should only be one)
        original_layer = ly.layer(TECHNOLOGY[layer_list[0]]) # convert layer to understandable type for future functions

        from ._globals import KLAYOUT_VERSION

        if KLAYOUT_VERSION > 24:
            path = wireguide.cell.pcell_parameters_by_name()['path']
        else:
            # wireguide path and width from Wireguide PCell
            path1 = wireguide.cell.pcell_parameters_by_name()['path']
            path = pya.Path()
            path.width = wireguide.cell.pcell_parameters_by_name()['width'] / TECHNOLOGY['dbu']
            pts = []
            for pt in [pt1 for pt1 in (path1).each_point()]:
                if type(pt) == pya.Point:
                    # for instantiated PCell
                    pts.append(pya.Point())
                else:
                    # for wireguide from path
                    pts.append(pya.Point().from_dpoint(pt * (1 / TECHNOLOGY['dbu'])))
            path.points = pts

        selection.append(pya.ObjectInstPath())
        selection[-1].layer = original_layer
        # DPath.transformed requires DTrans. wireguide.trans is a Trans object
        if KLAYOUT_VERSION > 24:
            selection[-1].shape = cell.shapes(original_layer).insert(
                path.transformed(wireguide.trans.to_dtype(TECHNOLOGY['dbu'])))
        else:
            selection[-1].shape = cell.shapes(original_layer).insert(
                path.transformed(pya.Trans(wireguide.trans.disp.x, wireguide.trans.disp.y)))

        selection[-1].top = obj.top
        selection[-1].cv_index = obj.cv_index

        # deleting the instance was ok, but would leave the cell which ends up as
        # an uninstantiated top cell
        to_delete.append(obj.inst())

    # deleting instance or cell should be done outside of the for loop,
    # otherwise each deletion changes the instance pointers in KLayout's
    # internal structure
    [t.delete() for t in to_delete]

    # Clear the layout view selection, since we deleted some objects (but
    # others may still be selected):
    lv.clear_object_selection()
    # Select the newly added objects
    lv.object_selection = selection
    # Record a transaction, to enable "undo"
    lv.commit()

# ORIGINAL FUNCTION - snap_component()
# The changes to the original function are as follows:
# 1) Change the pin.TYPE to ELECTRICAL to enable electrical pin snapping instead of OPTICAL pin snapping 
def snap_metal_component():
    print("*** snap_component, move selected object to snap onto the transient: ")

    from . import _globals

    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    # Define layers based on PDK:
    LayerSiN = TECHNOLOGY['Waveguide']
    LayerPinRecN = TECHNOLOGY['PinRec']
    LayerDevRecN = TECHNOLOGY['DevRec']
    LayerFbrTgtN = TECHNOLOGY['FbrTgt']
    LayerErrorN = TECHNOLOGY['Errors']

    # we need two objects.  One is selected, and the other is a transient selection
    if lv.has_transient_object_selection() == False:
        print("No transient selection")
        import sys
        if sys.platform.startswith('darwin'):
            v = pya.MessageBox.warning(
                "No transient selection", "Hover the mouse (transient selection) over the object to which you wish to snap to.\nIf it still doesn't work, please ensure that 'Transient mode' selection is enabled in the KLayout menu KLayout - Preferences - Applications - Selection.", pya.MessageBox.Ok)
        else:
            v = pya.MessageBox.warning(
                "No transient selection", "Hover the mouse (transient selection) over the object to which you wish to snap to.\nIf it still doesn't work, please ensure that 'Transient mode' selection is enabled in the KLayout menu File - Settings - Applications - Selection.", pya.MessageBox.Ok)
    else:
        # find the transient selection:
        o_transient_iter = lv.each_object_selected_transient()
        o_transient = next(o_transient_iter)  # returns ObjectInstPath[].

        # Find the selected objects
        o_selection = lv.object_selection   # returns ObjectInstPath[].

        if len(o_selection) < 1:
            v = pya.MessageBox.warning(
                "No selection", "Select the object you wish to be moved.", pya.MessageBox.Ok)
        if len(o_selection) > 1:
            v = pya.MessageBox.warning(
                "Too many selected", "Select only one object you wish to be moved.", pya.MessageBox.Ok)
        else:
            o_selection = o_selection[0]
            if o_selection.is_cell_inst() == False:
                v = pya.MessageBox.warning(
                    "No selection", "The selected object must be an instance (not primitive polygons)", pya.MessageBox.Ok)
            elif o_transient.is_cell_inst() == False:
                v = pya.MessageBox.warning(
                    "No selection", "The selected object must be an instance (not primitive polygons)", pya.MessageBox.Ok)
            elif o_selection.inst().is_regular_array():
                v = pya.MessageBox.warning(
                    "Array", "Selection was an array. \nThe array was 'exploded' (Edit | Selection | Resolve Array). \nPlease select the objects and try again.", pya.MessageBox.Ok)
                # Record a transaction, to enable "undo"
                lv.transaction("Object snapping - exploding array")
                o_selection.inst().explode()
                # Record a transaction, to enable "undo"
                lv.commit()
            elif o_transient.inst().is_regular_array():
                v = pya.MessageBox.warning(
                    "Array", "Selection was an array. \nThe array was 'exploded' (Edit | Selection | Resolve Array). \nPlease select the objects and try again.", pya.MessageBox.Ok)
                # Record a transaction, to enable "undo"
                lv.transaction("Object snapping - exploding array")
                o_transient.inst().explode()
                # Record a transaction, to enable "undo"
                lv.commit()
            elif o_transient == o_selection:
                v = pya.MessageBox.warning(
                    "Same selection", "We need two different objects: one selected, and one transient (hover mouse over).", pya.MessageBox.Ok)
            else:
                # we have two instances, we can snap them together:

                # Find the pins within the two cell instances:
                pins_transient = o_transient.inst().find_pins(verbose=True)
                pins_selection = o_selection.inst().find_pins(verbose=True)
                print("all pins_transient (x,y): %s" %
                      [[point.x, point.y] for point in [pin.center for pin in pins_transient]])
                print("all pins_selection (x,y): %s" %
                      [[point.x, point.y] for point in [pin.center for pin in pins_selection]])

                # create a list of all pin pairs for comparison;
                # pin pairs must have a 180 deg orientation (so they can be connected);
                # then sort to find closest ones
                # nested list comprehension, tutorial:
                # https://spapas.github.io/2016/04/27/python-nested-list-comprehensions/
                pin_pairs = sorted([[pin_t, pin_s]
                                    for pin_t in pins_transient
                                    for pin_s in pins_selection
                                    if (abs(pin_t.rotation - pin_s.rotation) % 360 - 180) < 1 and pin_t.type == _globals.PIN_TYPES.ELECTRICAL and pin_s.type == _globals.PIN_TYPES.ELECTRICAL],
                                   key=lambda x: x[0].center.distance(x[1].center))

                if pin_pairs:
                    print("shortest pins_transient & pins_selection (x,y): %s" %
                          [[point.x, point.y] for point in [pin.center for pin in pin_pairs[0]]])
                    print("shortest distance: %s" % pin_pairs[0][
                          0].center.distance(pin_pairs[0][1].center))

                    trans = pya.Trans(pya.Trans.R0, pin_pairs[0][
                                      0].center - pin_pairs[0][1].center)
                    print("translation: %s" % trans)

                    # Record a transaction, to enable "undo"
                    lv.transaction("Object snapping")
                    # Move the selected object
                    o_selection.inst().transform(trans)
                    # Record a transaction, to enable "undo"
                    lv.commit()
                else:
                    v = pya.MessageBox.warning("Snapping failed",
                                               "Snapping failed. \nNo matching pins found. \nNote that pins must have exactly matching orientations (180 degrees)", pya.MessageBox.Ok)
                    return

                pya.Application.instance().main_window().message(
                    'SiEPIC snap_components: moved by %s.' % trans, 2000)

                return
# end def snap_component()
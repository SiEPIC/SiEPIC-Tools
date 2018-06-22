
#################################################################################
#                SiEPIC Tools - scripts                                         #
#################################################################################
'''

path_to_waveguide
roundpath_to_waveguide
waveguide_to_path
waveguide_length
waveguide_length_diff
waveguide_heal
auto_route
snap_component
delete_top_cells
compute_area
calibreDRC
auto_coord_extract
calculate_area
trim_netlist
layout_check
open_PDF_files
open_folder
user_select_opt_in
fetch_measurement_data_from_github
measurement_vs_simulation
resize waveguide
'''


import pya


def path_to_waveguide(params=None, cell=None, lv_commit=True, GUI=False, verbose=False):
    from . import _globals
    from .utils import select_paths, get_layout_variables
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
    if not cell:
        cell = top_cell

    if lv_commit:
        lv.transaction("Path to Waveguide")

    if params is None:
        params = _globals.WG_GUI.get_parameters(GUI)
    if params is None:
        return
    if verbose:
        print("SiEPIC.scripts path_to_waveguide(): params = %s" % params)
    selected_paths = select_paths(TECHNOLOGY['Waveguide'], cell, verbose=verbose)
    selection = []

    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    for obj in selected_paths:
        path = obj.shape.path
        path.unique_points()
        if not path.is_manhattan_endsegments():
            warning.setText(
                "Warning: Waveguide segments (first, last) are not Manhattan (vertical, horizontal).")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return
        if not path.is_manhattan():
            warning.setText(
                "Error: Waveguide segments are not Manhattan (vertical, horizontal). This is not supported in SiEPIC-Tools.")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return
        if not path.radius_check(params['radius'] / TECHNOLOGY['dbu']):
            warning.setText(
                "Warning: One of the waveguide segments has insufficient length to accommodate the desired bend radius.")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return

        path.snap(cell.find_pins())
        Dpath = path.to_dtype(TECHNOLOGY['dbu'])
        width_devrec = max([wg['width'] for wg in params['wgs']]) + _globals.WG_DEVREC_SPACE * 2
        try:
            pcell = ly.create_cell("Waveguide", TECHNOLOGY['technology_name'], {"path": Dpath,
                                                                                "radius": params['radius'],
                                                                                "width": params['width'],
                                                                                "adiab": params['adiabatic'],
                                                                                "bezier": params['bezier'],
                                                                                "layers": [wg['layer'] for wg in params['wgs']] + ['DevRec'],
                                                                                "widths": [wg['width'] for wg in params['wgs']] + [width_devrec],
                                                                                "offsets": [wg['offset'] for wg in params['wgs']] + [0]})
            print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s" %
                  (TECHNOLOGY['technology_name'], pcell))
        except:
            pass
        if not pcell:
            try:
                pcell = ly.create_cell("Waveguide", "SiEPIC General", {"path": Dpath,
                                                                       "radius": params['radius'],
                                                                       "width": params['width'],
                                                                       "adiab": params['adiabatic'],
                                                                       "bezier": params['bezier'],
                                                                       "layers": [wg['layer'] for wg in params['wgs']] + ['DevRec'],
                                                                       "widths": [wg['width'] for wg in params['wgs']] + [width_devrec],
                                                                       "offsets": [wg['offset'] for wg in params['wgs']] + [0]})
                print("SiEPIC.scripts.path_to_waveguide(): Waveguide from SiEPIC General, %s" % pcell)
            except:
                pass
        if not pcell:
            raise Exception(
                "'Waveguide' in 'SiEPIC General' library is not available. Check that the library was loaded successfully.")
        selection.append(pya.ObjectInstPath())
        selection[-1].top = obj.top
        selection[-1].append_path(pya.InstElement.new(cell.insert(
            pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))))

        obj.shape.delete()

    lv.clear_object_selection()
    lv.object_selection = selection
    if lv_commit:
        lv.commit()

'''
convert a KLayout ROUND_PATH, which was used to make a waveguide
in SiEPIC_EBeam_PDK versions up to v0.1.41, back to a Path, then waveguide.
This allows the user to migrate designs to the new Waveguide PCell.
'''


def roundpath_to_waveguide(verbose=False):

    from . import _globals
    from .utils import get_layout_variables
#  from .scripts import path_to_waveguide
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    dbu = TECHNOLOGY['dbu']

    # Record a transaction, to enable "undo"
    lv.transaction("ROUND_PATH to Waveguide")

    if verbose:
        print("SiEPIC.scripts.roundpath_to_waveguide()")

    # record objects to delete:
    to_delete = []
    # new objects will become selected after the waveguide-to-path operation
    new_selection = []
    # Find the selected objects
    object_selection = lv.object_selection   # returns ObjectInstPath[].

    Waveguide_Types = ["ROUND_PATH"]

    if object_selection == []:
        if verbose:
            print("Nothing selected.  Automatically selecting waveguides.")
        # find all instances, specifically, Waveguides:
        for inst in cell.each_inst():
            if verbose:
                print("Cell: %s" % (inst.cell.basic_name()))
            if inst.cell.basic_name() in Waveguide_Types:
                n = len(object_selection)
                object_selection.append(pya.ObjectInstPath())
                object_selection[n].top = cell.cell_index()
                object_selection[n].append_path(pya.InstElement.new(inst))
        # Select the newly added objects
        lv.object_selection = object_selection

    is_ROUNDPATH = False
    for o in object_selection:
        # Find the selected waveguides
        if o.is_cell_inst():
            if verbose:
                print("Selected object is a cell.")
            oinst = o.inst()
            if oinst.is_pcell():
                c = oinst.cell
                # and c.pcell_parameters_by_name()['layer'] == LayerSi:
                if c.basic_name() in Waveguide_Types:
                    LayerSiN = c.pcell_parameters_by_name()['layer']
                    radius = c.pcell_parameters_by_name()['radius']
                    if verbose:
                        print("%s on Layer %s." % (c.basic_name(), LayerSiN))
                    is_ROUNDPATH = True
                    trans = oinst.trans

        elif o.shape:
            if verbose:
                print("Selected object is a shape.")
            c = o.shape.cell
            # and c.pcell_parameters_by_name()['layer'] == LayerSi:
            if c.basic_name() in Waveguide_Types and c.is_pcell_variant():
                # we have a waveguide GUIDING_LAYER selected
                LayerSiN = c.pcell_parameters_by_name()['layer']
                radius = c.pcell_parameters_by_name()['radius']
                if verbose:
                    print("Selected object is a GUIDING_LAYER in %s on Layer %s." %
                          (c.basic_name(), LayerSiN))
                trans = o.source_trans().s_trans()
                o_instpathlen = o.path_length()
                oinst = o.path_nth(o_instpathlen - 1).inst()
                is_ROUNDPATH = True

        # We now have a waveguide ROUND_PATH PCell, with variables: o
        # (ObjectInstPath), oinst (Instance), c (Cell)
        if is_ROUNDPATH == True:
            path_obj = c.pcell_parameters_by_name()['path']
            if verbose:
                print(path_obj)
            wg_width = path_obj.width / dbu
            # convert wg_path (in microns) to database numbers

            from ._globals import KLAYOUT_VERSION
            if KLAYOUT_VERSION > 24:
                new_wg = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(
                    path_obj.transformed(trans.to_dtype(TECHNOLOGY['dbu'])))
            else:
                v = pya.MessageBox.warning(
                    "KLayout 0.25 or higher required.", "ROUND_PATH to Waveguide is implemented using KLayout 0.25 or higher functionality.", pya.MessageBox.Ok)
                return

            # Leave the newly created path selected, to convert after
            new_selection.append(pya.ObjectInstPath())
            new_selection[-1].layer = ly.layer(TECHNOLOGY['Waveguide'])
            new_selection[-1].shape = new_wg
            new_selection[-1].top = o.top
            new_selection[-1].cv_index = o.cv_index

            to_delete.append(oinst)  # delete the instance; leaves behind the cell if it's not used

    # Clear the layout view selection:
    lv.clear_object_selection()
    # Select the newly added objects
    lv.object_selection = new_selection

    # Convert the selected paths to a Waveguide:
    path_to_waveguide(lv_commit=False, verbose=verbose)

    # Record a transaction, to enable "undo"
    lv.commit()

    if not(is_ROUNDPATH):
        v = pya.MessageBox.warning(
            "No ROUND_PATH selected", "No ROUND_PATH selected.\nPlease select a ROUND_PATH. \nIt will get converted to a path.", pya.MessageBox.Ok)


def waveguide_to_path(cell=None):
    from . import _globals
    from .utils import select_waveguides, get_layout_variables

    if cell is None:
        TECHNOLOGY, lv, ly, cell = get_layout_variables()
    else:
        TECHNOLOGY, lv, _, _ = get_layout_variables()
        ly = cell.layout()

    lv.transaction("waveguide to path")

    # record objects to delete:
    to_delete = []

    waveguides = select_waveguides(cell)
    selection = []
    for obj in waveguides:
        # path from waveguide guiding shape
        waveguide = obj.inst()

        from ._globals import KLAYOUT_VERSION

        if KLAYOUT_VERSION > 24:
            path = waveguide.cell.pcell_parameters_by_name()['path']
        else:
            # waveguide path and width from Waveguide PCell
            path1 = waveguide.cell.pcell_parameters_by_name()['path']
            path = pya.Path()
            path.width = waveguide.cell.pcell_parameters_by_name()['width'] / TECHNOLOGY['dbu']
            pts = []
            for pt in [pt1 for pt1 in (path1).each_point()]:
                if type(pt) == pya.Point:
                    # for instantiated PCell
                    pts.append(pya.Point())
                else:
                    # for waveguide from path
                    pts.append(pya.Point().from_dpoint(pt * (1 / TECHNOLOGY['dbu'])))
            path.points = pts

        selection.append(pya.ObjectInstPath())
        selection[-1].layer = ly.layer(TECHNOLOGY['Waveguide'])
        # DPath.transformed requires DTrans. waveguide.trans is a Trans object
        if KLAYOUT_VERSION > 24:
            selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(
                path.transformed(waveguide.trans.to_dtype(TECHNOLOGY['dbu'])))
        else:
            selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(
                path.transformed(pya.Trans(waveguide.trans.disp.x, waveguide.trans.disp.y)))

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


def waveguide_length():

    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    import SiEPIC.utils

    selection = lv.object_selection
    if len(selection) == 1 and selection[0].inst().is_pcell() and "Waveguide" in selection[0].inst().cell.basic_name():
        cell = selection[0].inst().cell
        area = SiEPIC.utils.advance_iterator(cell.each_shape(
            cell.layout().layer(TECHNOLOGY['Waveguide']))).polygon.area()
        width = cell.pcell_parameters_by_name()['width'] / cell.layout().dbu
        pya.MessageBox.warning("Waveguide Length", "Waveguide length (um): %s" %
                               str(area / width * cell.layout().dbu), pya.MessageBox.Ok)
    else:
        pya.MessageBox.warning("Selection is not a waveguide",
                               "Select one waveguide you wish to measure.", pya.MessageBox.Ok)


def waveguide_length_diff():

    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    import SiEPIC.utils

    selection = lv.object_selection
    if len(selection) == 2 and selection[0].inst().is_pcell() and "Waveguide" in selection[0].inst().cell.basic_name() and selection[1].inst().is_pcell() and "Waveguide" in selection[1].inst().cell.basic_name():
        cell = selection[0].inst().cell
        area1 = SiEPIC.utils.advance_iterator(cell.each_shape(
            cell.layout().layer(TECHNOLOGY['Waveguide']))).polygon.area()
        width1 = cell.pcell_parameters_by_name()['width'] / cell.layout().dbu
        cell = selection[1].inst().cell
        area2 = SiEPIC.utils.advance_iterator(cell.each_shape(
            cell.layout().layer(TECHNOLOGY['Waveguide']))).polygon.area()
        width2 = cell.pcell_parameters_by_name()['width'] / cell.layout().dbu
        pya.MessageBox.warning("Waveguide Length Difference", "Difference in waveguide lengths (um): %s" % str(
            abs(area1 / width1 - area2 / width2) * cell.layout().dbu), pya.MessageBox.Ok)
    else:
        pya.MessageBox.warning("Selection are not a waveguides",
                               "Select two waveguides you wish to measure.", pya.MessageBox.Ok)


def waveguide_heal():
    print("waveguide_heal")


def auto_route():
    print("auto_route")


'''
SiEPIC-Tools: Snap Component

by Lukas Chrostowski (c) 2016-2017

This Python function implements snapping of one component to another.

Usage:
- Click to select the component you wish to move (selected)
- Hover the mouse over the component you wish to align to (transient)
- Shift-O to run this script
- The function will find the closest pins between these components, and move the selected component

Version history:

Lukas Chrostowski           2016/03/08
 - Initial version

Lukas Chrostowski           2017/12/16
 - Updating to SiEPIC-Tools 0.3.x module based approach rather than a macro
   and without optical components database
 - Strict assumption that pin directions in the component are consistent, namely
   they indicate which way the signal is LEAVING the component
   (path starts with the point inside the DevRec, then the point outside)
   added to wiki https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK/wiki/Component-and-PCell-Layout
   This avoids the issue of components ending up on top of each other incorrectly.
   Ensures that connections are valid

Lukas Chrostowski           2017/12/17
 - removing redundant code, and replacing with Brett's functions:
   - Cell.find_pins, rather than code within.

'''


def snap_component():
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
                                    if (abs(pin_t.rotation - pin_s.rotation) % 360 - 180) < 1 and pin_t.type == _globals.PIN_TYPES.OPTICAL and pin_s.type == _globals.PIN_TYPES.OPTICAL],
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


# keep the selected top cell; delete everything else
def delete_top_cells():

    def delete_cells(ly, cell):
        if cell in ly.top_cells():
            ly.delete_cells([tcell for tcell in ly.each_top_cell() if tcell != cell.cell_index()])
        if len(ly.top_cells()) > 1:
            delete_cells(ly, cell)

    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    if cell in ly.top_cells():
        lv.transaction("Delete extra top cells")
        delete_cells(ly, cell)
        lv.commit()
    else:
        v = pya.MessageBox.warning(
            "No top cell selected", "No top cell selected.\nPlease select a top cell to keep\n(not a sub-cell).", pya.MessageBox.Ok)


def compute_area():
    print("compute_area")


def calibreDRC(params=None, cell=None):
    import sys
    import os
    import pipes
    import codecs
    from . import _globals
    from .utils import get_layout_variables

    if cell is None:
        _, lv, ly, cell = get_layout_variables()
    else:
        _, lv, _, _ = get_layout_variables()
        ly = cell.layout()

    # the server can be configured via ~/.ssh/config, and named "drc"
    server = "drc"

    if not params:
        from .utils import load_Calibre
        CALIBRE = load_Calibre()
        params = {}
        params['pdk'] = CALIBRE['Calibre']['remote_pdk_location']
        params['calibre'] = CALIBRE['Calibre']['remote_calibre_script']
        params['remote_calibre_rule_deck_main_file'] = CALIBRE[
            'Calibre']['remote_calibre_rule_deck_main_file']
        params['remote_additional_commands'] = CALIBRE['Calibre']['remote_additional_commands']
        if 'server' in CALIBRE['Calibre'].keys():
            server = CALIBRE['Calibre']['server']

    if any(value == '' for key, value in params.items()):
        raise Exception("Missing information")

    lv.transaction("Calibre DRC")
    import time
    progress = pya.RelativeProgress("Calibre DRC", 5)
    progress.format = "Saving Layout to Temporary File"
    progress.set(1, True)
    time.sleep(1)
    pya.Application.instance().main_window().repaint()

    # Local temp folder:
    local_path = _globals.TEMP_FOLDER
    print("SiEPIC.scripts.calibreDRC; local tmp folder: %s" % local_path)
    local_file = os.path.basename(lv.active_cellview().filename())
    if not local_file:
        local_file = 'layout'
    local_pathfile = os.path.join(local_path, local_file)

    # Layout path and filename:
    mw = pya.Application.instance().main_window()
    layout_path = mw.current_view().active_cellview().filename()  # /path/file.gds
    layout_filename = os.path.basename(layout_path)               # file.gds
    layout_basefilename = layout_filename.split('.')[0]           # file
    import getpass
    remote_path = "/tmp/%s_%s" % (getpass.getuser(), layout_basefilename)

    results_file = layout_basefilename + ".rve"
    results_pathfile = os.path.join(os.path.dirname(local_pathfile), results_file)
    tmp_ly = ly.dup()
    [c.flatten(True) for c in tmp_ly.each_cell()]
    opts = pya.SaveLayoutOptions()
    opts.format = "GDS2"
    tmp_ly.write(local_pathfile, opts)

    with codecs.open(os.path.join(local_path, 'run_calibre'), 'w', encoding="utf-8") as file:
        cal_script = '#!/bin/tcsh \n'
        cal_script += 'source %s \n' % params['calibre']
        cal_script += '%s \n' % params['remote_additional_commands']
        cal_script += '$MGC_HOME/bin/calibre -drc -hier -turbo -nowait drc.cal \n'
        file.write(cal_script)

    with codecs.open(os.path.join(local_path, 'drc.cal'), 'w', encoding="utf-8") as file:
        cal_deck = 'LAYOUT PATH  "%s"\n' % os.path.basename(local_pathfile)
        cal_deck += 'LAYOUT PRIMARY "%s"\n' % cell.name
        cal_deck += 'LAYOUT SYSTEM GDSII\n'
        cal_deck += 'DRC RESULTS DATABASE "drc.rve" ASCII\n'
        cal_deck += 'DRC MAXIMUM RESULTS ALL\n'
        cal_deck += 'DRC MAXIMUM VERTEX 4096\n'
        cal_deck += 'DRC CELL NAME YES CELL SPACE XFORM\n'
        cal_deck += 'VIRTUAL CONNECT COLON NO\n'
        cal_deck += 'VIRTUAL CONNECT REPORT NO\n'
        cal_deck += 'DRC ICSTATION YES\n'
        cal_deck += 'INCLUDE "%s/%s"\n' % (params['pdk'],
                                           params['remote_calibre_rule_deck_main_file'])
        file.write(cal_deck)

    import platform
    version = platform.python_version()
    out = ''
    if version.find("2.") > -1:
        import commands
        cmd = commands.getstatusoutput

        progress.set(2, True)
        progress.format = "Uploading Layout and Scripts"
        pya.Application.instance().main_window().repaint()

        out += cmd('ssh %s "mkdir -p %s"' % (server, remote_path))[1]
        out += cmd('cd "%s" && scp "%s" %s:%s' % (local_path, local_file, server, remote_path))[1]
        out += cmd('cd "%s" && scp "%s" %s:%s' %
                   (local_path, 'run_calibre', server, remote_path))[1]
        out += cmd('cd "%s" && scp "%s" %s:%s' % (local_path, 'drc.cal', server, remote_path))[1]

        progress.set(3, True)
        progress.format = "Checking Layout for Errors"
        pya.Application.instance().main_window().repaint()

        out += cmd('ssh %s "cd %s && source run_calibre"' % (server, remote_path))[1]

        progress.set(4, True)
        progress.format = "Downloading Results"
        pya.Application.instance().main_window().repaint()

        out += cmd('cd "%s" && scp %s:%s "%s"' %
                   (local_path, server, remote_path + "/drc.rve", results_file))[1]

        progress.set(5, True)
        progress.format = "Finishing"
        pya.Application.instance().main_window().repaint()

    elif version.find("3.") > -1:
        import subprocess
        cmd = subprocess.check_output

        progress.format = "Uploading Layout and Scripts"
        progress.set(2, True)
        pya.Application.instance().main_window().repaint()

        try:
            out += cmd('ssh %s "mkdir -p %s"' % (server, remote_path), shell=True).decode('utf-8')
            out += cmd('cd "%s" && scp "%s" %s:%s' %
                       (local_path, local_file, server, remote_path), shell=True).decode('utf-8')
            out += cmd('cd "%s" && scp "%s" %s:%s' % (local_path, 'run_calibre',
                                                      server, remote_path), shell=True).decode('utf-8')
            out += cmd('cd "%s" && scp "%s" %s:%s' %
                       (local_path, 'drc.cal', server, remote_path), shell=True).decode('utf-8')

            progress.format = "Checking Layout for Errors"
            progress.set(3, True)
            pya.Application.instance().main_window().repaint()

            out += cmd('ssh %s "cd %s && source run_calibre"' %
                       (server, remote_path), shell=True).decode('utf-8')

            progress.format = "Downloading Results"
            progress.set(4, True)
            pya.Application.instance().main_window().repaint()

            out += cmd('cd "%s" && scp %s:%s "%s"' % (local_path, server, remote_path +
                                                      "/drc.rve", results_file), shell=True).decode('utf-8')
        except subprocess.CalledProcessError as e:
            out += '\nError running ssh or scp commands. Please check that these programs are available.\n'
            out += str(e.output)
#        if e.output.startswith('error: {'):
#          import json
#          error = json.loads(e.output[7:])
#          print (error['code'])
#          print (error['message'])
    progress.format = "Finishing"
    progress.set(5, True)

    print(out)
    progress._destroy()
    if os.path.exists(results_pathfile):
        rdb_i = lv.create_rdb("Calibre Verification")
        rdb = lv.rdb(rdb_i)
        rdb.load(results_pathfile)
        rdb.top_cell_name = cell.name
        rdb_cell = rdb.create_cell(cell.name)
        lv.show_rdb(rdb_i, lv.active_cellview().cell_index)
    else:
        pya.MessageBox.warning(
            "Errors", "Something failed during the server Calibre DRC check: %s" % out,  pya.MessageBox.Ok)

    pya.Application.instance().main_window().update()
    lv.commit()


def auto_coord_extract():
    from .utils import get_technology
    TECHNOLOGY = get_technology()

    def gen_ui():
        global wdg
        if 'wdg' in globals():
            if wdg is not None and not wdg.destroyed():
                wdg.destroy()
        global wtext

        def button_clicked(checked):
            """ Event handler: "OK" button clicked """
            wdg.destroy()

        wdg = pya.QDialog(pya.Application.instance().main_window())

        wdg.setAttribute(pya.Qt.WA_DeleteOnClose)
        wdg.setWindowTitle("SiEPIC-Tools: Automated measurement coordinate extraction")

        wdg.resize(1000, 500)
        wdg.move(1, 1)

        grid = pya.QGridLayout(wdg)

        windowlabel1 = pya.QLabel(wdg)
        windowlabel1.setText("output:")
        wtext = pya.QTextEdit(wdg)
        wtext.enabled = True
        wtext.setText('')

        ok = pya.QPushButton("OK", wdg)
        ok.clicked(button_clicked)   # attach the event handler
#    netlist = pya.QPushButton("Save", wdg) # not implemented

        grid.addWidget(windowlabel1, 0, 0, 1, 3)
        grid.addWidget(wtext, 1, 1, 3, 3)
#    grid.addWidget(netlist, 4, 2)
        grid.addWidget(ok, 4, 3)

        grid.setRowStretch(3, 1)
        grid.setColumnStretch(1, 1)

        wdg.show()

    # Create a GUI for the output:
    gen_ui()
    wtext.insertHtml('<br>* Automated measurement coordinates:<br><br>')

    # Find the automated measurement coordinates:
    from .utils import find_automated_measurement_labels
    cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    text_out, opt_in = find_automated_measurement_labels(cell)
    wtext.insertHtml(text_out)


def calculate_area():
    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    dbu = TECHNOLOGY['dbu']

    total = cell.each_shape(ly.layer(TECHNOLOGY['FloorPlan'])).__next__().polygon.area()
    area = 0
    itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['Waveguide']))
    while not itr.at_end():
        area += itr.shape().area()
        itr.next()
    print("Waveguide area: %s mm^2, chip area: %s mm^2, percentage: %s %%" % (area/1e6*dbu*dbu,total/1e6*dbu*dbu, area/total*100))

    if 0:
        area = 0
        itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['SiEtch1']))
        while not itr.at_end():
            area += itr.shape().area()
            itr.next()
        print(area / total)
    
        area = 0
        itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['SiEtch2']))
        while not itr.at_end():
            area += itr.shape().area()
            itr.next()
        print(area / total)


"""
SiEPIC-Tools: Trim Netlist
by Jaspreet Jhoja (c) 2016-2017

This Python function facilitates trimming of netlist based on a selected component.
Version history:

Jaspreet Jhoja           2017/12/29
 - Initial version
"""
# Inputs, and example of how to generate them:
# nets, components = topcell.identify_nets()
# selected_component = components[5]   (elsewhere the desired component is selected)


def trim_netlist(nets, components, selected_component, verbose=None):
    selected = selected_component
    #>17        <2
    # nets[0].pins[0].component.idx
    trimmed_net = []
    net_idx = [[each.pins[0].component.idx, each.pins[1].component.idx] for each in nets]
    len_net_idx = len(net_idx)
    count = 0
    trimmed_nets, trimmed_components = [], []
    while count < (len_net_idx - 1):
        for i in range(count + 1, len_net_idx):  # i keep track of nets from next net to last net
            # first set is formed of elements from current to backwards
            first_set = set(net_idx[count])
            # second set is formed of elements from current + 1 to forward
            second_set = set(net_idx[i])
            if len(first_set.intersection(second_set)) > 0:  # if there are common elements between two sets
                net_idx.pop(i)  # remove the nets from the list
                net_idx.pop(count)  # remove the count net as well
                # merged them and add to the list
                net_idx.append(list(first_set.union(second_set)))
                len_net_idx -= 1  # 2 removed 1 added so reduce 1
                count -= 1  # readjust count as the elements have shifted to left
                break
        count += 1
    for net in net_idx:
        if(selected.idx in net):
            trimmed_components = [each for each in components if each.idx in net]
            trimmed_nets = [each for each in nets if (
                each.pins[0].component.idx in net or each.pins[1].component.idx in net)]
            if verbose:
                print("success - netlist trimmed")

    return trimmed_nets, trimmed_components


'''
Verification:

Limitations:
- we assume that the layout was created by SiEPIC-Tools in KLayout, that PCells are there,
  and that the layout hasn't been flattened. This allows us to isolate individual components,
  and get their parameters. Working with a flattened layout would be harder, and require:
   - reading parameters from the text labels (OK)
   - find_components would need to look within the DevRec layer, rather than in the selected cell
   - when pins are connected, we have two overlapping ones, so detecting them would be problematic;
     This could be solved by putting the pins inside the cells, rather than sticking out.

'''


def layout_check(cell=None, verbose=False):
    if verbose:
        print("*** layout_check()")

    from . import _globals
    from .utils import get_technology, find_paths, find_automated_measurement_labels, angle_vector
    from .utils import advance_iterator
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
        cv = lv.active_cellview()
    else:
        ly = cell.layout()

    if not TECHNOLOGY['technology_name']:
        v = pya.MessageBox.warning("Errors", "SiEPIC-Tools verification requires a technology to be chosen.  \n\nThe active technology is displayed on the bottom-left of the KLayout window, next to the T. \n\nChange the technology using KLayout File | Layout Properties, then choose Technology and find the correct one (e.g., EBeam, GSiP).", pya.MessageBox.Ok)
        return

    # Get the components and nets for the layout
    nets, components = cell.identify_nets(verbose=False)
    if verbose:
        print("* Display list of components:")
        [c.display() for c in components]

    if not components:
        v = pya.MessageBox.warning(
            "Errors", "No components found (using SiEPIC-Tools DevRec and PinRec definitions).", pya.MessageBox.Ok)
        return

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

    # Connectivity checking
    rdb_cell = next(rdb.each_cell())
    rdb_cat_id = rdb.create_category("Connectivity")
    rdb_cat_id_discpin = rdb.create_category(rdb_cat_id, "Disconnected pin")
    rdb_cat_id_discpin.description = "Disconnected pin"
    rdb_cat_id_mismatchedpin = rdb.create_category(rdb_cat_id, "Mismatched pin")
    rdb_cat_id_mismatchedpin.description = "Mismatched pin widths"

    # Simulation checking
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
        rdb_cat_id_GCarrayconfig.description = "Circuit must be connected such that there is at most %s Grating Coupler(s) above the opt_in label (laser injection port) and at most %s Grating Coupler(s) below the opt_in label. \n\nGrating couplers must be on a %s micron pitch, vertically arranged." % (
            int(DFT['design-for-test']['grating-couplers']['detectors-above-laser']), int(DFT['design-for-test']['grating-couplers']['detectors-below-laser']), float(DFT['design-for-test']['grating-couplers']['gc-pitch']))
    else:
        if verbose:
            print('  No DFT rules found.')

    paths = find_paths(TECHNOLOGY['Waveguide'], cell=cell)
    for p in paths:
        if verbose:
            print("%s, %s" % (type(p), p))
        # Check for paths with > 2 vertices
        Dpath = p.to_dtype(dbu)
        if Dpath.num_points() > 2:
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_path.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(Dpath.polygon()))

    for i in range(0, len(components)):
        c = components[i]
        # the following only works for layouts where the Waveguide is still a PCells (not flattened)
        # basic_name is assigned in Cell.find_components, by reading the PCell parameter
        # if the layout is flattened, we don't have an easy way to get the path
        # it could be done perhaps as a parameter (points)
        if c.basic_name == "Waveguide" and c.cell.is_pcell_variant():
            Dpath = c.cell.pcell_parameters_by_name()['path']
            radius = c.cell.pcell_parameters_by_name()['radius']
            if verbose:
                print(" - Waveguide: cell: %s, %s" % (c.cell.name, radius))

            # Radius check:
            if not Dpath.radius_check(radius):
                rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_radius.rdb_id())
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
                        "There are two identical components overlapping: \n" + c.component + "\n"))
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
                    if ci == gc and c.trans.angle != int(DFT['design-for-test']['grating-couplers']['gc-orientation'][gc]):
                        gc_orientation_error = True
                if gc_orientation_error:
                    if verbose:
                        print(" - Found DFT error, GC facing the wrong way: %s, %s" %
                              (c.component, c.trans.angle))
                    rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_GCorient.rdb_id())
                    rdb_item.add_value(pya.RdbItemValue(c.polygon.to_dtype(dbu)))

        # Pre-simulation check: do components have models?
        if not c.has_model():
            if verbose:
                print(" - Missing compact model, for component: %s" % (c.component))
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_sim_nomodel.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(c.polygon.to_dtype(dbu)))

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
            t = opt_in[ti1]['Text']
            box_s = 1000
            box = pya.Box(t.x - box_s, t.y - box_s, t.x + box_s, t.y + box_s)
            # opt_in labels check for unique
            for ti2 in range(ti1 + 1, len(opt_in)):
                if opt_in[ti1]['opt_in'] == opt_in[ti2]['opt_in']:
                    if verbose:
                        print(" - Found DFT error, non unique text labels: %s, %s, %s" %
                              (t.string, t.x, t.y))
                    rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_optin_unique.rdb_id())
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
            from ._globals import KLAYOUT_VERSION
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
                for d in list(range(int(DFT['design-for-test']['grating-couplers']['detectors-above-laser']) + 0, 0, -1)) + list(range(-1, -int(DFT['design-for-test']['grating-couplers']['detectors-below-laser']) - 1, -1)):
                    if pya.DPoint(0, d * float(DFT['design-for-test']['grating-couplers']['gc-pitch']) * 1000) in vect_optin_GCs:
                        del_index = vect_optin_GCs.index(pya.DPoint(
                            0, d * float(DFT['design-for-test']['grating-couplers']['gc-pitch']) * 1000))
                        del vect_optin_GCs[del_index]
                        del detector_GCs[del_index]
                for vi in range(0, len(vect_optin_GCs)):
                    if verbose:
                        print(" - DFT GC array config error: %s, %s" %
                              (components_sorted[0].instance, opt_in[ti1]['opt_in']))
                    rdb_item = rdb.create_item(
                        rdb_cell.rdb_id(), rdb_cat_id_GCarrayconfig.rdb_id())
                    rdb_item.add_value(pya.RdbItemValue(
                        "The label having the error is: \n" + opt_in[ti1]['opt_in'] + "\n"))
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
                rdb_item.add_value(pya.RdbItemValue(polygon_merged.to_dtype(dbu)))

    # displays results in Marker Database Browser, using Results Database (rdb)
    if rdb.num_items() > 0:
        v = pya.MessageBox.warning(
            "Errors", "%s layout errors detected.  \nPlease review errors using the 'Marker Database Browser'." % rdb.num_items(), pya.MessageBox.Ok)
        lv.show_rdb(rdb_i, cv.cell_index)
    else:
        v = pya.MessageBox.warning("Errors", "No layout errors detected.", pya.MessageBox.Ok)

    # Save results of verification as a Text label on the cell. Include OS,
    # SiEPIC-Tools and PDF version info.
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

'''
Open all PDF files using an appropriate viewer
'''


def open_PDF_files(files, files_list):
    import sys
    if sys.platform.startswith('darwin'):
        # open all the files in a single Preview application.
        # open in one window with tabs: https://support.apple.com/en-ca/guide/mac-help/mchlp2469
        # System Preferences - Dock - Prefer tabs when opening documents - Always
        runcmd = '/usr/bin/open -n -a /Applications/Preview.app %s' % files
        if int(sys.version[0]) > 2:
            import subprocess
            print("Running in shell: %s" % runcmd)
            subprocess.Popen(['/usr/bin/open', '-n', '-a', '/Applications/Preview.app', files])
        else:
            import commands
            print("Running in shell: %s" % runcmd)
            print(commands.getstatusoutput(runcmd))
    if sys.platform.startswith('win'):
        import os
        for f in files_list:
            os.startfile(f)
'''
Open the folder using an appropriate file finder / explorer
'''


def open_folder(folder):
    import sys
    if sys.platform.startswith('darwin'):
        runcmd = '/usr/bin/open %s' % folder
        print("Running in shell: %s" % runcmd)
        if int(sys.version[0]) > 2:
            import subprocess
            subprocess.Popen(['/usr/bin/open', folder])
        else:
            import commands
            print(commands.getstatusoutput(runcmd))

    if sys.platform.startswith('win'):
        import subprocess
        print("running in windows explorer, %s" % folder)
        print(subprocess.Popen(r'explorer /select,"%s"' % folder))

'''
User to select opt_in labels, either:
 - Text object selection in the layout
 - GUI with drop-down menu from all labels in the layout
 - argument to the function, opt_in_selection_text, array of opt_in labels (strings)
'''


def user_select_opt_in(verbose=None, option_all=True, opt_in_selection_text=[]):
    from .utils import find_automated_measurement_labels
    text_out, opt_in = find_automated_measurement_labels()
    if not opt_in:
        print(' No opt_in labels found in the layout')
        return False, False

    # optional argument to this function
    if not opt_in_selection_text:

        # First check if any opt_in labels are selected
        from .utils import selected_opt_in_text
        oinstpaths = selected_opt_in_text()
        for oi in oinstpaths:
            opt_in_selection_text.append(oi.shape.text.string)

        if opt_in_selection_text:
            if verbose:
                print(' user selected opt_in labels')
        else:
            # If not, scan the cell and find all the labels
            if verbose:
                print(' starting GUI to select opt_in labels')

            # GUI to ask which opt_in measurement to fetch
            opt_in_labels = [o['opt_in'] for o in opt_in]
            if option_all:
                opt_in_labels.insert(0, 'All opt-in labels')
            opt_in_selection_text = pya.InputDialog.ask_item(
                "opt_in selection", "Choose one of the opt_in labels.",  opt_in_labels, 0)
            if not opt_in_selection_text:  # user pressed cancel
                if verbose:
                    print(' user cancel!')
                return False, False
            if opt_in_selection_text == 'All opt-in labels':
                opt_in_selection_text = [o['opt_in'] for o in opt_in]
                if verbose:
                    print('  selecting all opt_in labels')
            else:
                opt_in_selection_text = [opt_in_selection_text]

    # find opt_in Dict entries matching the opt_in text labels
    opt_in_dict = []
    for o in opt_in:
        for t in opt_in_selection_text:
            if o['opt_in'] == t:
                opt_in_dict.append(o)

    return opt_in_selection_text, opt_in_dict

'''
Fetch measurement data from GitHub

Identify opt_in circuit, using one of:
 - selected opt_in Text objects
 - GUI
    - All - first option
    - Individual - selected

Query GitHub to find measurement data for opt_in label(s)

Get data, one of:
 - All
 - Individual
'''


def fetch_measurement_data_from_github(verbose=None, opt_in_selection_text=[]):
    import pya
    from . import _globals
    tmp_folder = _globals.TEMP_FOLDER
    from .github import github_get_filenames, github_get_files, github_get_file

    user = 'lukasc-ubc'
    repo = 'edX-Phot1x'
    extension = 'pdf'

    if verbose:
        print('Fetch measurement data from GitHub')

    if opt_in_selection_text:
        folder_flatten_option = True
    else:
        folder_flatten_option = None

    if opt_in_selection_text:
        include_path = False

    from .scripts import user_select_opt_in
    opt_in_selection_text, opt_in_dict = user_select_opt_in(
        verbose=verbose, opt_in_selection_text=opt_in_selection_text)

    if verbose:
        print(' opt_in labels: %s' % opt_in_selection_text)
        print(' Begin looping through labels')

    all_measurements = 0
    savefilepath = []

    # Loop through the opt_in text labels
    if not opt_in_selection_text:
        pya.MessageBox.warning("opt_in labels not found",
                               "Error: opt_in labels not found", pya.MessageBox.Ok)
        return

    for ot in opt_in_selection_text:

        fields = ot.split("_")
        search_for = ''
        # Search for device_xxx_xxx_...
#    for i in range(4,min(7,len(fields))):
        for i in range(4, len(fields)):
            search_for += fields[i] + '_'
        if verbose:
            print("  searching for: %s" % search_for)

        filenames = github_get_filenames(
            extension=extension, user=user, repo=repo, filesearch=search_for, verbose=verbose)

        if len(filenames) == 0:
            print(' measurement not found!')

            # Search for xxx_xxx_... (don't include the "device" part)
            search_for = ''
#      for i in range(5,min(7,len(fields))):
            for i in range(5, len(fields)):
                search_for += fields[i] + '_'
            if verbose:
                print("  searching for: %s" % search_for)

            filenames = github_get_filenames(
                extension=extension, user=user, repo=repo, filesearch=search_for, verbose=verbose)

            if len(filenames) == 0:
                pya.MessageBox.warning("Measurement data not found", "Measurement data not found; searched for: %s on GitHub %s/%s" % (
                    search_for, user, repo), pya.MessageBox.Ok)
                print(' measurement not found!')
                return

        if len(filenames) == 1:
            measurements_text = filenames[0][1].replace('%20', ' ')
        elif len(filenames) > 1:
            if all_measurements == 0:
                # GUI to ask which opt_in measurement to fetch
                measurements = [f[1].replace('%20', ' ') for f in filenames]
                measurements.insert(0, 'All measurements')
                measurements_text = pya.InputDialog.ask_item(
                    "opt_in selection", "Choose one of the data files for opt_in = %s, to fetch experimental data.\n" % search_for,  measurements, 0)
                if not measurements_text:  # user pressed cancel
                    if verbose:
                        print(' user cancel!')
                    return
                if measurements_text == 'All measurements':
                    if verbose:
                        print('  all measurements')
                    all_measurements = 1

        if not folder_flatten_option:
            # GUI to ask if we want to keep the folder tree
            options = ['Flatten folder tree', 'Replicate folder tree']
            folder_flatten_option = pya.InputDialog.ask_item(
                "folder tree", "Do you wish to place all files in the same folder (flatten folder tree), or recreate the folder tree structure?",  options, 0)
            if folder_flatten_option == 'Replicate folder tree':
                include_path = True
            else:
                include_path = False

        # Download file(s)
        if all_measurements == 1:
            savefilepath = github_get_files(user=user, repo=repo, filename_search=search_for,
                                            save_folder=tmp_folder,  include_path=include_path, verbose=verbose)
        else:  # find the single data set to download (both pdf and mat files)
            for f in filenames:
                if f[1] == measurements_text.replace(' ', '%20'):
                    file_selection = f
            if verbose:
                print('   File selection: %s' % file_selection)
            import os
            savefilepath = github_get_files(user=user, repo=repo, filename_search=file_selection[0].replace(
                '.' + extension, '.'), save_folder=tmp_folder,  include_path=include_path, verbose=verbose)

    # this launches open_PDF once for all files at the end:
    if savefilepath:
        if verbose:
            print('All files: %s' % (savefilepath))

        filenames = ''
        for s in savefilepath:
            filenames += s + ' '

        if verbose or not opt_in_selection_text:
            open_PDF_files(filenames, savefilepath)
            open_folder(tmp_folder)

    if not opt_in_selection_text:
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        if savefilepath:
            warning.setText("Measurement Data: successfully downloaded files.")
        else:
            warning.setText("Measurement Data: 0 files downloaded.")
        pya.QMessageBox_StandardButton(warning.exec_())

    return filenames, savefilepath


'''
Identify opt_in circuit, using one of:
 - selected opt_in Text objects
 - GUI
    - All - first option
    - Individual - selected

Fetch measurement data from GitHub
Run simulation

Plot data together

'''


def measurement_vs_simulation(verbose=None):
    import pya
    from . import _globals
    tmp_folder = _globals.TEMP_FOLDER
    from .scripts import fetch_measurement_data_from_github
    from .scripts import user_select_opt_in
    from .lumerical.interconnect import circuit_simulation

    if verbose:
        print('measurement_vs_simulation()')

    opt_in_selection_text, opt_in_dict = user_select_opt_in(verbose=verbose)

    if verbose:
        print(' opt_in labels: %s' % opt_in_selection_text)
        print(' Begin looping through labels')

    # Loop through the opt_in text labels
    for ot in opt_in_selection_text:

            # Fetch github data:
        files, savefilepath = fetch_measurement_data_from_github(
            verbose=verbose, opt_in_selection_text=[ot])

        # simulate:
        circuit_simulation(verbose=verbose, opt_in_selection_text=[
                           ot], matlab_data_files=savefilepath)

    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Ok)
    if savefilepath:
        warning.setText(
            "Measurement versus Simulation: successfully downloaded files and simulated.")
    else:
        warning.setText("Measurement Data: 0 files downloaded.")
    pya.QMessageBox_StandardButton(warning.exec_())

    return files, savefilepath


"""
    SiEPIC-Tools: Resize Waveguide
    Author: Jaspreet Jhoja(2016 - 2018)

    This Python file implements a waveguide resizing tool.
    Version history:
       Jaspreet Jhoja 2018/02/13
        - Resizes Waveguides with selection
        - Users are required to press Ctrl + Shift + R
"""


def resize_waveguide():
    import pya
    import sys
    import copy
    from pya import QFont, QWidget, Qt, QVBoxLayout, QFrame, QLabel, QComboBox, QLineEdit, QPushButton, QGridLayout, QSplitter, QTextEdit
    from SiEPIC import utils
    TECHNOLOGY, lv, ly, cell = utils.get_layout_variables()
    from SiEPIC.scripts import path_to_waveguide

    net, comp = cell.identify_nets()
    global points, copy_pts, diff
    objlist = []  # list of data objects
    clear = []  # 12 clears should be there
    topcell = None
    layout = None
    dbu = None
    topcell = cell

    # fetch the database parameters
    dbu = ly.dbu

    # Define layers based on PDK_functions:
    LayerSiN = ly.layer(TECHNOLOGY['Waveguide'])
    LayerPinRecN = ly.layer(TECHNOLOGY['PinRec'])
    LayerDevRecN = ly.layer(TECHNOLOGY['DevRec'])

    # delete instances
    to_delete = []

    # extract the circuit netlist from the physical layout:
    optical_selection = utils.select_waveguides(cell)
    if(len(optical_selection) > 1 or len(optical_selection) == 0):
        pya.MessageBox.warning("Message", "No waveguide is selected", pya.MessageBox.Ok)
    else:
        wg_obj = optical_selection[0]
        if wg_obj.is_cell_inst():
            oinst = wg_obj.inst()
            if oinst.is_pcell():
                c = oinst.cell
                trans = oinst.trans

        path_obj = c.pcell_parameters_by_name()['path']

        if(path_obj.points <= 2):
            v = pya.MessageBox.warning(
                "Message", "Cannot perform this operation on the selected cell/path.\n Hint: Select a cell/path with more than 2 vertices.", pya.MessageBox.Ok)

        else:

            # PCell_get_parameters ( c ) #This line causes syntax error Do not uncomment this line.

            #  path_length
            wg_width = path_obj.width  # in microns
            # calculate the length of the waveguide using the area / width
            iter2 = c.begin_shapes_rec(LayerSiN)
            if iter2.shape().is_polygon():
                area = iter2.shape().polygon.area()
                path_length = area / wg_width * dbu * dbu
            else:
                print("## ROUND_PATH waveguide, not polygon; bug in code? ##")
                path_length = 0  # or path_obj.length()

            escape = 0

            points_obj = path_obj.to_dtype(1 / dbu).get_dpoints()
            points = [[each.x * dbu, each.y * dbu] for each in points_obj]

            # Separate the segments of the waveguide
            segments_all = []
            for i in range(len(points)):
                if(i > 0):
                    pair = [points[i - 1], points[i]]
                    segments_all.append(pair)

            # don't allow any modifications to first and last segment
            global segments
            segments = segments_all[1:-1]

            # Check segment orientation
            global seg_orientation
            seg_orientation = []
            for each in segments:
                if(each[0][0] == each[1][0]):
                    seg_orientation.append("vertical")
                elif(each[0][1] == each[1][1]):
                    seg_orientation.append("horizontal")

            # prop variable which determines the segment propagation
            prop_points = points
            seg_propagation = []
            #+x, -x , +y , -y
            for each in segments:
                print(each[0])
                index = prop_points.index(each[0])
                prop = ""

                if(index == 0):
                    index = index + 1
                    element_idx = index + 1
                   # look at the second index
                else:
                    element_idx = index - 1

                x1 = prop_points[index][0]
                y1 = prop_points[index][1]
                x2 = prop_points[element_idx][0]
                y2 = prop_points[element_idx][1]

                if(x1 == x2):
                    if(y1 < y2):
                        prop = "+y"
                    elif(y1 > y2):
                        prop = "-y"
                        # their x have same value means they are propagating along y axis
                elif(y1 == y2):
                    if(x1 < x2):
                        prop = "-x"
                    elif(x1 > x2):
                        prop = "+x"
                print(index)
                print(element_idx)
                print(prop)
                seg_propagation.append(prop)
                # y have same values along x axis

            global wdg, hbox, lframe1, titlefont, lf1title, parameters, lf1label1, lf1label2, lf1label3, lf1title2, lf1text3, lf1form, lframe1, leftsplitter, splitter1, container, ok
            wdg = QWidget()
            #wdg = QDialog(pya.Application.instance().main_window())
            wdg.setAttribute(pya.Qt.WA_DeleteOnClose)
            wdg.setWindowTitle("Waveguide resizer")

            if sys.platform.startswith('linux'):
                    # Linux-specific code here...
                titlefont = QFont("Arial", 11, QFont.Bold, False)

            elif sys.platform.startswith('darwin'):
                    # OSX specific
                titlefont = QFont("Arial", 13, QFont.Bold, False)

            elif sys.platform.startswith('win'):
                titlefont = QFont("Arial", 9, QFont.Bold, False)

            #titlefont = QFont("Arial", 9, QFont.Bold, False)
            hbox = QVBoxLayout(wdg)

            wdg.setFixedSize(650, 250)

            def selection(self):
                # make a list of these to show them
                global segments, seg_orientation, lf1label1, lf1label2
        #    lf1text1.setText(str(abs(segments[parameters.currentIndex][0][0] - segments[parameters.currentIndex][1][0])*dbu + abs(segments[parameters.currentIndex][0][1] - segments[parameters.currentIndex][1][1])*dbu))
                lf1label1.setText('     Segment length: %s microns' % str((abs(segments[parameters.currentIndex][0][
                                  0] - segments[parameters.currentIndex][1][0]) + abs(segments[parameters.currentIndex][0][1] - segments[parameters.currentIndex][1][1]))))
        #    lf1text2.setText(str(seg_orientation[parameters.currentIndex]))
                lf1label2.setText('     Segment orientation: %s' %
                                  str(seg_orientation[parameters.currentIndex]))

            # Left Frame top section
            lframe1 = QFrame()
            lframe1.setFrameShape(QFrame.StyledPanel)
            lframe1.setStyleSheet("background-color: white;")
            lf1title = QLabel('Current waveguide length (microns): %s' % str(path_length))
            parameters = QComboBox()
            # add vertices as params
            params = []
            for each in range(len(segments)):
                #    params.append("segment %s  points:  %s  - %s" %(str(each), str(tuple(segments[each][0])), str(tuple(segments[each][1]))))
                params.append("segment %s, points: (%s, %s) - (%s, %s)" % (str(each), segments[each][
                              0][0], segments[each][0][1], segments[each][1][0], segments[each][1][1]))

            parameters.addItems(params)
            parameters.currentIndexChanged(selection)
            parameters.setFixedWidth(400)
            parameters.setStyleSheet("background-color: white;")
            lf1label1 = QLabel('Segment length: ')
            lf1label2 = QLabel('Segment orientation: ')
            lf1label3 = QLabel('New target waveguide length (microns): ')
            lf1title2 = QLabel('Chose the segment you wish to be moved:')
            lf1text3 = QLineEdit()
            lf1text3.setAccessibleName('lf1text3')
            lf1text3.setText(str(path_length))

            def button(self):
                # don't want to change the layout while the GUI is open; leads to view problems.
                wdg.close()

                global points, copy_pts, diff

                if lf1text3.text == "":
                    return

                # Record a transaction, to enable "undo"
                lv.transaction("Waveguide resizing")

                # get current index and segment propagation
                index = parameters.currentIndex
                copy_pts = copy.deepcopy(points)
                p1 = copy_pts[copy_pts.index(segments[index][0])]
                p2 = copy_pts[copy_pts.index(segments[index][1])]
                diff = float(lf1text3.text) - path_length
        #    diff = diff/dbu
                prop = seg_propagation[index]
                if(prop == "+x" or prop == "-x"):
                    if(prop == "-x"):
                        diff = diff * -1
                    print("moving x")
                    # perform the action based on diff value
                    copy_pts[copy_pts.index(segments[index][0])][0] = copy_pts[
                        copy_pts.index(segments[index][0])][0] + diff / 2
                    copy_pts[copy_pts.index(segments[index][1])][0] = copy_pts[
                        copy_pts.index(segments[index][1])][0] + diff / 2
                elif(prop == "+y" or prop == "-y"):
                    if(prop == "+y"):
                        diff = diff * -1
                    print("moving y")
                    copy_pts[copy_pts.index(segments[index][0])][1] = copy_pts[
                        copy_pts.index(segments[index][0])][1] + diff / 2
                    copy_pts[copy_pts.index(segments[index][1])][1] = copy_pts[
                        copy_pts.index(segments[index][1])][1] + diff / 2
                print("pushed", p1, p2)
        #    path_obj = cell.pcell_parameters_by_name()['path']

                dpoints = [pya.DPoint(each[0], each[1]) for each in copy_pts]
                dpath = pya.DPath(dpoints, wg_width)
                # replace old path with the new one
                oinst.change_pcell_parameter("path", dpath)
                lv.commit()
                wdg.destroy()  # destroy GUI when we are completely done.

            ok = QPushButton("OK")
            ok.clicked(button)

            lf1form = QGridLayout()
            lf1form.addWidget(lf1title, 0, 0)
            lf1form.addWidget(lf1label3, 1, 0)
            lf1form.addWidget(lf1text3, 1, 1)
            lf1form.addWidget(lf1title2, 2, 0)
            lf1form.addWidget(parameters, 3, 0)
            lf1form.addWidget(lf1label1, 4, 0)
        #  lf1form.addWidget(lf1text1, 4,1)
            lf1form.addWidget(lf1label2, 5, 0)
        #  lf1form.addWidget(lf1text2, 5,1)
            lf1form.addWidget(ok, 7, 1)
            lframe1.setLayout(lf1form)
            leftsplitter = QSplitter(Qt.Vertical)
            leftsplitter.addWidget(lframe1)
            leftsplitter.setSizes([500, 400, 10])
            splitter1 = QSplitter(Qt.Horizontal)
            textedit = QTextEdit()
            splitter1.addWidget(leftsplitter)
            splitter1.setSizes([400, 500])
            container = QWidget()
            hbox.addWidget(splitter1)
            objlist.append(lf1text3)
            selection(None)
            wdg.show()

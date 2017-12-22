import pya

def waveguide_from_path(params = None, cell = None):
  from . import _globals
  from .utils import select_paths, get_technology
  TECHNOLOGY = get_technology()

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
  
  status = _globals.WG_GUI.return_status()
  if status is None and params is None:
    _globals.WG_GUI.show()
  else:
    lv.transaction("waveguide from path")

    if status is False: return
    if params is None: params = _globals.WG_GUI.get_parameters()  
    
    selected_paths = select_paths(TECHNOLOGY['Waveguide'], cell)
    selection = []
  
    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    for obj in selected_paths:
      path = obj.shape.path
      if not path.is_manhattan():
        warning.setText("Warning: Waveguide segments (first, last) are not Manhattan (vertical, horizontal).")
        warning.setInformativeText("Do you want to Proceed?")
        if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
          return
      if not path.radius_check(params['radius']/TECHNOLOGY['dbu']):
        warning.setText("Warning: One of the waveguide segments has insufficient length to accommodate the desired bend radius.")
        warning.setInformativeText("Do you want to Proceed?")
        if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
          return
      
      path.snap(cell.find_pins())
      path = pya.DPath(path.get_dpoints(), path.width) * TECHNOLOGY['dbu']
      path.width = path.width * TECHNOLOGY['dbu']
      width_devrec = max([wg['width'] for wg in params['wgs']]) + _globals.WG_DEVREC_SPACE * 2
      pcell = ly.create_cell("Waveguide", "SiEPIC General", { "path": path,
                                                                     "radius": params['radius'],
                                                                     "width": params['width'],
                                                                     "adiab": params['adiabatic'],
                                                                     "bezier": params['bezier'],
                                                                     "layers": [wg['layer'] for wg in params['wgs']] + [TECHNOLOGY['DevRec']],
                                                                     "widths": [wg['width'] for wg in params['wgs']] + [width_devrec],
                                                                     "offsets": [wg['offset'] for wg in params['wgs']] + [0]} )
      if pcell==None:
        raise Exception("'Waveguide' in 'SiEPIC General' library is not available. Check that the library was loaded successfully.")
      selection.append(pya.ObjectInstPath())
      selection[-1].top = obj.top
      print(pcell)
      selection[-1].append_path(pya.InstElement.new(cell.insert(pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))))
      
      obj.shape.delete()
    
    lv.clear_object_selection()
    lv.object_selection = selection
    lv.commit()


def waveguide_to_path(cell = None):
  from SiEPIC import _globals
  from SiEPIC.utils import select_waveguides, get_technology
  TECHNOLOGY = get_technology()
  
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
    
  lv.transaction("waveguide to path")

  # record objects to delete:
  to_delete = []
  
  waveguides = select_waveguides(cell)
  selection = []
  for obj in waveguides:
    # path from guiding shape
    waveguide = obj.inst()
    # Determine whether we have Python 2 or Python 3
    import sys
    
#    if sys.version_info[0] == 3:
#      # for some reason this doesn't work on Instantiated PCells, but it does on W-generated ones!
#      path1 = waveguide.cell.shapes(waveguide.layout().guiding_shape_layer()).each().__next__().path
#    elif sys.version_info[0] == 2:
#      # Python 2 & 3 fix:
    from SiEPIC.utils import advance_iterator
    itr = waveguide.cell.shapes(waveguide.layout().guiding_shape_layer()).each()
    path1 = advance_iterator(itr)

    # waveguide width from Waveguide PCell
    c = waveguide.cell
    width = c.pcell_parameters_by_name()['width']

    # modify path (doesn't work in 0.24.10 / Python2); neither does dup()
    # perhaps because this path belongs to a PCell.
    #path1.width = int(width/TECHNOLOGY['dbu'])  # 
    
    # instead create a new path:
    path = pya.Path()
    path.width = width/TECHNOLOGY['dbu']
    path.points=[pts for pts in path1.each_point()]

    selection.append(pya.ObjectInstPath())
    selection[-1].layer = ly.layer(TECHNOLOGY['Waveguide'])
    selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(path.transformed(waveguide.trans))
    selection[-1].top = obj.top
    selection[-1].cv_index = obj.cv_index
    
    if 1:
      # deleting the instance was ok, but would leave the cell which ends up as an uninstantiated top cell
      # obj.inst().delete()
      to_delete.append(obj.inst())
    else:
      # delete the cell (KLayout also removes the PCell)
      # deleting it removes the cell entirely (which may be used elsewhere ?)
      # intermittent crashing...
      to_delete.append(waveguide.cell) 


  # deleting instance or cell should be done outside of the for loop, otherwise each deletion changes the instance pointers in KLayout's internal structure
  [t.delete() for t in to_delete]
  #  for t in to_delete:
  #    t.delete()

  # Clear the layout view selection, since we deleted some objects (but others may still be selected):
  lv.clear_object_selection()
  # Select the newly added objects
  lv.object_selection = selection
  # Record a transaction, to enable "undo"
  lv.commit()


def waveguide_length():

  from .utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
    
  selection = lv.object_selection
  if len(selection) == 1 and selection[0].inst().is_pcell() and "Waveguide" in selection[0].inst().cell.basic_name():
    cell = selection[0].inst().cell
    area = cell.each_shape(cell.layout().layer(TECHNOLOGY['DevRec'])).__next__().polygon.area()
    width = 3*cell.pcell_parameters_by_name()['width']/cell.layout().dbu
    pya.MessageBox.warning("Waveguide Length", "Waveguide length (um): %s" % str(area/width*cell.layout().dbu), pya.MessageBox.Ok)
  else:
    pya.MessageBox.warning("Selection is not a waveguide", "Select one waveguide you wish to measure.", pya.MessageBox.Ok)
  
def waveguide_length_diff():

  from .utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
    
  selection = lv.object_selection
  if len(selection) == 2 and selection[0].inst().is_pcell() and "Waveguide" in selection[0].inst().cell.basic_name() and selection[1].inst().is_pcell() and "Waveguide" in selection[1].inst().cell.basic_name():
    cell = selection[0].inst().cell
    area1 = cell.each_shape(cell.layout().layer(TECHNOLOGY['DevRec'])).__next__().polygon.area()
    width1 = 3*cell.pcell_parameters_by_name()['width']/cell.layout().dbu
    cell = selection[1].inst().cell
    area2 = cell.each_shape(cell.layout().layer(TECHNOLOGY['DevRec'])).__next__().polygon.area()
    width2 = 3*cell.pcell_parameters_by_name()['width']/cell.layout().dbu
    pya.MessageBox.warning("Waveguide Length Difference", "Difference in waveguide lengths (um): %s" % str(abs(area1/width1 - area2/width2)*cell.layout().dbu), pya.MessageBox.Ok)
  else:
    pya.MessageBox.warning("Selection are not a waveguides", "Select two waveguides you wish to measure.", pya.MessageBox.Ok)

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
    v = pya.MessageBox.warning("No transient selection", "Hover the mouse (transient selection) over the object to which you wish to snap to.\nEnsure transient selection is enabled in Settings - Applications - Selection.", pya.MessageBox.Ok)
  else:
    # find the transient selection:
    o_transient_iter = lv.each_object_selected_transient()
    o_transient = next(o_transient_iter)  # returns ObjectInstPath[].

    # Find the selected objects
    o_selection = lv.object_selection   # returns ObjectInstPath[].

    if len(o_selection) < 1:
      v = pya.MessageBox.warning("No selection", "Select the object you wish to be moved.", pya.MessageBox.Ok)
    if len(o_selection) > 1:
      v = pya.MessageBox.warning("Too many selected", "Select only one object you wish to be moved.", pya.MessageBox.Ok)
    else:
      o_selection = o_selection[0]
      if o_selection.is_cell_inst()==False:
        v = pya.MessageBox.warning("No selection", "The selected object must be an instance (not primitive polygons)", pya.MessageBox.Ok)
      elif o_transient.is_cell_inst()==False:
        v = pya.MessageBox.warning("No selection", "The selected object must be an instance (not primitive polygons)", pya.MessageBox.Ok)
      elif o_selection.inst().is_regular_array():
        v = pya.MessageBox.warning("Array", "Selection was an array. \nThe array was 'exploded' (Edit | Selection | Resolve Array). \nPlease select the objects and try again.", pya.MessageBox.Ok)
        # Record a transaction, to enable "undo"
        lv.transaction("Object snapping - exploding array")
        o_selection.inst().explode()
        # Record a transaction, to enable "undo"
        lv.commit()
      elif o_transient.inst().is_regular_array():
        v = pya.MessageBox.warning("Array", "Selection was an array. \nThe array was 'exploded' (Edit | Selection | Resolve Array). \nPlease select the objects and try again.", pya.MessageBox.Ok)
        # Record a transaction, to enable "undo"
        lv.transaction("Object snapping - exploding array")
        o_transient.inst().explode()
        # Record a transaction, to enable "undo"
        lv.commit()      
      elif o_transient == o_selection:
        v = pya.MessageBox.warning("Same selection", "We need two different objects: one selected, and one transient (hover mouse over).", pya.MessageBox.Ok)
      else: 
        # we have two instances, we can snap them together:

        # Find the pins within the two cell instances:     
        pins_transient = o_transient.inst().find_pins()
        pins_selection = o_selection.inst().find_pins()
        print("all pins_transient (x,y): %s" % [[point.x, point.y] for point in [pin.center for pin in pins_transient]] )
        print("all pins_selection (x,y): %s" % [[point.x, point.y] for point in [pin.center for pin in pins_selection]] )

        # create a list of all pin pairs for comparison;
        # pin pairs must have a 180 deg orientation (so they can be connected);
        # then sort to find closest ones
        # nested list comprehension, tutorial: https://spapas.github.io/2016/04/27/python-nested-list-comprehensions/
        pin_pairs = sorted( [ [pin_t, pin_s] 
          for pin_t in pins_transient \
          for pin_s in pins_selection \
          if ((pin_t.rotation - pin_s.rotation)%360) == 180 and pin_t.type == _globals.PIN_TYPES.OPTICAL and pin_s.type == _globals.PIN_TYPES.OPTICAL ],
          key=lambda x: x[0].center.distance(x[1].center) )

        if pin_pairs:
          print("shortest pins_transient & pins_selection (x,y): %s" % [[point.x, point.y] for point in [pin.center for pin in pin_pairs[0]]] )
          print("shortest distance: %s" % pin_pairs[0][0].center.distance(pin_pairs[0][1].center) )

          trans = pya.Trans(pya.Trans.R0, pin_pairs[0][0].center - pin_pairs[0][1].center)
          print("translation: %s" % trans )

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
    v = pya.MessageBox.warning("No top cell selected", "No top cell selected.\nPlease select a top cell to keep\n(not a sub-cell).", pya.MessageBox.Ok)
  
def compute_area():
  print("compute_area")
  
def calibreDRC(params = None, cell = None):
  from . import _globals

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
  
  status = _globals.DRC_GUI.return_status()
  if status is None and params is None:
    _globals.DRC_GUI.show()
  else:
    if status is False: return
    if params is None: params = _globals.DRC_GUI.get_parameters()
    
    if any(value == '' for key, value in params.items()):
      raise Exception("Missing information")

    lv.transaction("calibre drc")
    
    import time
    progress = pya.RelativeProgress("Calibre DRC", 5)
    progress.format = "Saving Layout to Temporary File"
    progress.set(1, True)
    time.sleep(1)
    pya.Application.instance().main_window().repaint()
    
    # Python version
    import sys, os, pipes, codecs
    if sys.platform.startswith('win'):
      local_path = os.path.join(os.environ['USERPROFILE'], 'AppData', 'Local', 'Temp')
    else:
      local_path = "/tmp"
    local_file = os.path.basename(lv.active_cellview().filename())
    local_pathfile = os.path.join(local_path, local_file)
    
    remote_path = "/tmp"
    
    results_file = os.path.basename(local_pathfile) + ".rve"
    results_pathfile = os.path.join(os.path.dirname(local_pathfile), results_file)
    tmp_ly = ly.dup()
    [cell.flatten(True) for cell in tmp_ly.each_cell()]
    opts = pya.SaveLayoutOptions()
    opts.format = "GDS2"
    tmp_ly.write(local_pathfile, opts)
    
    with codecs.open(os.path.join(local_path, 'run_calibre'), 'w', encoding="utf-8") as file:
      cal_script  = '#!/bin/tcsh \n'
      cal_script += 'source %s \n' % params['calibre']
      cal_script += 'setenv SIEPIC_IME_PDK %s \n' % params['pdk']
      cal_script += '$MGC_HOME/bin/calibre -drc -hier -turbo -nowait drc.cal \n'
      file.write(cal_script)

    with codecs.open(os.path.join(local_path, 'drc.cal'), 'w', encoding="utf-8") as file:
      cal_deck  = 'LAYOUT PATH  "%s"\n' % os.path.basename(local_pathfile)
      cal_deck += 'LAYOUT PRIMARY "%s"\n' % cell.name
      cal_deck += 'LAYOUT SYSTEM GDSII\n'
      cal_deck += 'DRC RESULTS DATABASE "drc.rve" ASCII\n'
      cal_deck += 'DRC MAXIMUM RESULTS ALL\n'
      cal_deck += 'DRC MAXIMUM VERTEX 4096\n'
      cal_deck += 'DRC CELL NAME YES CELL SPACE XFORM\n'
      cal_deck += 'VIRTUAL CONNECT COLON NO\n'
      cal_deck += 'VIRTUAL CONNECT REPORT NO\n'
      cal_deck += 'DRC ICSTATION YES\n'
      cal_deck += 'INCLUDE "%s/calibre_rule_decks/CMC_SiEPIC_IMESP.drc.cal"\n' % params['pdk']
      file.write(cal_deck)

    version = sys.version
    if version.find("2.") > -1:
      import commands

      progress.set(2, True)
      progress.format = "Uploading Layout and Scripts"
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s "%s" %s:%s' % (os.path.dirname(local_pathfile), params['port'], os.path.basename(local_pathfile), params['url'], remote_path))
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s %s:%s' % (local_path, params['port'], 'run_calibre', params['url'], remote_path))
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s %s:%s' % (local_path, params['port'], 'drc.cal', params['url'], remote_path))

      progress.set(3, True)
      progress.format = "Checking Layout for Errors"
      pya.Application.instance().main_window().repaint()
    
      out = cmd('ssh -i C:/Users/bpoul/.ssh/drc -p%s %s "%s"' % (params['port'], params['url'], "cd " + remote_path +" && source run_calibre"))

      progress.set(4, True)
      progress.format = "Downloading Results"
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s:%s %s' % (os.path.dirname(local_pathfile), params['port'], params['url'], remote_path + "/drc.rve", results_file))

      progress.set(5, True)
      progress.format = "Finishing"
      pya.Application.instance().main_window().repaint()
      
    elif version.find("3.") > -1:
      import subprocess
      cmd = subprocess.check_output

      progress.format = "Uploading Layout and Scripts"      
      progress.set(2, True)
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i %s -P%s "%s" %s:%s' % (os.path.dirname(local_pathfile), params['identity'], params['port'], os.path.basename(local_pathfile), params['url'], remote_path), shell=True)
      out = cmd('cd "%s" && scp -i %s -P%s %s %s:%s' % (local_path, params['identity'], params['port'], 'run_calibre', params['url'], remote_path), shell=True)
      out = cmd('cd "%s" && scp -i %s -P%s %s %s:%s' % (local_path, params['identity'], params['port'], 'drc.cal', params['url'], remote_path), shell=True)

      progress.format = "Checking Layout for Errors"
      progress.set(3, True)
      pya.Application.instance().main_window().repaint()

      out = cmd('ssh -i %s -p%s %s "%s"' % (params['identity'], params['port'], params['url'], "cd " + remote_path +" && source run_calibre"), shell=True)
      
      progress.format = "Downloading Results"
      progress.set(4, True)
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i %s -P%s %s:%s %s' % (os.path.dirname(local_pathfile), params['identity'], params['port'], params['url'], remote_path + "/drc.rve", results_file), shell=True)

      progress.format = "Finishing"
      progress.set(5, True)
      
    progress._destroy()
    if os.path.exists(results_pathfile):
      rdb_i = lv.create_rdb("Calibre Verification")
      rdb = lv.rdb(rdb_i)
      rdb.load (results_pathfile)
      rdb.top_cell_name = cell.name
      rdb_cell = rdb.create_cell(cell.name)
      lv.show_rdb(rdb_i, lv.active_cellview().cell_index)
    else:
      pya.MessageBox.warning("Errors", "Something failed during the server Calibre DRC check.",  pya.MessageBox.Ok)

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
    netlist = pya.QPushButton("Save", wdg) # not implemented
  
    grid.addWidget(windowlabel1, 0, 0, 1, 3)
    grid.addWidget(wtext, 1, 1, 3, 3)
    grid.addWidget(netlist, 4, 2)
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
  t = find_automated_measurement_labels(cell, cell.layout().layer(TECHNOLOGY['LayerText']))
  wtext.insertHtml (t)

def calculate_area():
  from .utils import get_technology
  TECHNOLOGY = get_technology()

  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  ly = lv.active_cellview().layout()
  if ly == None:
    raise Exception("No active layout")
  cell = lv.active_cellview().cell
  if cell == None:
    raise Exception("No active cell")
    
  total = cell.each_shape(ly.layer(TECHNOLOGY['FloorPlan'])).__next__().polygon.area()
  area = 0
  itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['LayerSi']))
  while not itr.at_end():
    area += itr.shape().area()
    itr.next()
  print(area/total)
  
  area = 0
  itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['SiEtch1']))
  while not itr.at_end():
    area += itr.shape().area()
    itr.next()
  print(area/total)
  
  area = 0
  itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['SiEtch2']))
  while not itr.at_end():
    area += itr.shape().area()
    itr.next()
  print(area/total)

def layout_check():
  print("layout_check")
  
def text_netlist_check():
  print("text_netlist_check")
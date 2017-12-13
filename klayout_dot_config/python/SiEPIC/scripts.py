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
    
    selected_paths = select_paths(TECHNOLOGY['Si'], cell)
    selection = []
  
    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    for obj in selected_paths:
      path = obj.shape.path
      if not path.is_manhattan():
        warning.setText("Warning: Waveguide segments (first, last) are not Manhattan (vertical, horizontal).")
        warning.setInformativeText("Do you want to Proceed?")
        if(warning.exec_() == pya.QMessageBox.Cancel):
          return
      if not path.radius_check(params['radius']):
        warning.setText("Warning: One of the waveguide segments has insufficient length to accommodate the desired bend radius.")
        warning.setInformativeText("Do you want to Proceed?")
        if(warning.exec_() == pya.QMessageBox.Cancel):
          return
      
      path.snap(_globals.NET.refresh().pins)
      path = pya.DPath(path.get_dpoints(), path.width) * TECHNOLOGY['dbu']
      path.width = path.width * TECHNOLOGY['dbu']
      pcell = ly.create_cell("Waveguide", "SiEPIC GSiP Library", { "path": path,
                                                                     "radius": params['radius'],
                                                                     "width": params['width'],
                                                                     "adiab": params['adiabatic'],
                                                                     "bezier": params['bezier'],
                                                                     "layers": [wg['layer'] for wg in params['wgs']] + [TECHNOLOGY['DevRec']],
                                                                     "widths": [wg['width'] for wg in params['wgs']] + [2*params['width']],
                                                                     "offsets": [wg['offset'] for wg in params['wgs']] + [0]} )
      selection.append(pya.ObjectInstPath())
      selection[-1].top = obj.top
      selection[-1].append_path(pya.InstElement.new(cell.insert(pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))))
      
      obj.shape.delete()
    
    lv.clear_object_selection()
    lv.object_selection = selection
    lv.commit()
    
def waveguide_to_path(cell = None):
  from . import _globals
  from .utils import select_waveguides, get_technology
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
  
  waveguides = select_waveguides(cell)
  selection = []
  for obj in waveguides:
    waveguide = obj.inst()
    path = waveguide.cell.shapes(waveguide.layout().guiding_shape_layer()).each().__next__().path
    path.width = 0.5/TECHNOLOGY['dbu']
    
    selection.append(pya.ObjectInstPath())
    selection[-1].layer = ly.layer(TECHNOLOGY['LayerSi'])
    selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['LayerSi'])).insert(path)
    selection[-1].top = obj.top
    selection[-1].cv_index = obj.cv_index
    
    obj.inst().delete()

  lv.clear_object_selection()
  lv.object_selection = selection
  lv.commit()
  
def waveguide_length():
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  
  ly = pya.Application.instance().main_window().current_view().active_cellview().layout() 
  if ly == None:
    raise Exception("No active layout")
    
  selection = lv.object_selection
  if len(selection) == 1 and selection[0].inst().is_pcell() and "Waveguide" in selection[0].inst().cell.basic_name():
    cell = selection[0].inst().cell
    area = cell.each_shape(cell.layout().layer(TECHNOLOGY['DevRec'])).__next__().polygon.area()
    width = 3*cell.pcell_parameters_by_name()['width']/cell.layout().dbu
    pya.MessageBox.warning("Waveguide Length", "Waveguide length (um): %s" % str(area/width*cell.layout().dbu), pya.MessageBox.Ok)
  else:
    pya.MessageBox.warning("Selection is not a waveguide", "Select one waveguide you wish to measure.", pya.MessageBox.Ok)
  
def waveguide_length_diff():
  from .utils import get_technology
  TECHNOLOGY = get_technology()
  
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  
  ly = pya.Application.instance().main_window().current_view().active_cellview().layout() 
  if ly == None:
    raise Exception("No active layout")
    
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
  
def snap_component():
  print("snap_component")
  
def delete_top_cells():

  def delete_cells(ly, cell):
    if cell in ly.top_cells():
      ly.delete_cells([tcell for tcell in ly.each_top_cell() if tcell != cell.cell_index()])
    if len(ly.top_cells()) > 1:
      delete_cells(ly, cell)
    
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")

  ly = pya.Application.instance().main_window().current_view().active_cellview().layout() 
  if ly == None:
    raise Exception("No active layout")
  
  cell = pya.Application.instance().main_window().current_view().active_cellview().cell
  if cell == None:
    raise Exception("No active cell")

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
    wdg.setWindowTitle("SiEPIC-EBeam-PDK: Automated measurement coordinate extraction")
  
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
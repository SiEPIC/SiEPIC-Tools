import pya

#Define net class for pin snapping, searching and connecting for netlists
class Net():

  def __init__(self):
    self.connections = []
    self.pins = set()
    
  def add(self, pin):
    self.pins.add(pin)
    return self
    
  def remove(self, pin):
    self.pins.remove(pin)
    self.connections = [(i, j) for i, j in self.connections if (i != pin or j != pin)]
    return self
  
  def connect(self, pin1 = None, pin2 = None):
    if pin1 is None and pin2 is None:
      if len(self.pins) > 1:
        pin1 = self.pins[-1]
        pin2 = self.pins[-2]
    
    if len([(i, j) for i, j in self.connections if (i == pin1 or j == pin1 or i == pin2 or j == pin2)]) > 0:
      raise Exception("Pins connections cannot be more than one to one")
    else:
      self.connections.append((pin1, pin2))
      self.connections.append((pin2, pin1))
    return self
  
  def refresh(self):
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
      raise Exception("No view selected")
    ly = lv.active_cellview().layout()
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
    
    self.connections = []
    self.pins = set()

    from .utils import get_technology
    from . import _globals
    
    TECHNOLOGY = get_technology()
    it = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['PinRec']))
    while not(it.at_end()):
      if it.shape().is_path():
        self.pins.add(Pin(it.shape().path.transformed(it.itrans()), _globals.PIN_TYPES.OPTICAL))
      it.next()
    return self


'''
Pin:
This is a class that describes optical pins on components and waveguides.
A pin consists of:
 - a Path with 2 points, with it's vector giving the direction
 - a Text label giving it's name
 - a type: OPTICAL, I/O, ELECTRICAL
A pin can associated with:
 - a component
 - a net

Uses:
 - Waveguide snapping to nearest pin (in waveguide_from_path, and path.snap)
    - does not need info about component, net...
 - Component snapping to nearest pin (in snap_component)
    - does not need info about component, net...
 - Netlist extraction
    - needs connectivity: which component & net the pin belongs to
    
Pin defs:
 - transform: to move the pin
 
'''
class Pin():

  def __init__(self, path, _type, idx=None, component_idx=None, net=None, pin_name=None ):
    from .utils import angle_vector
    pts = path.get_points()
    self.path = path            # the pin's Path
    self.center = (pts[0]+pts[1])*0.5  # center of the pin: a Point
    self.rotation = angle_vector(pts[0]-pts[1]) # direction / angle of the pin
    self.type = _type           # one of PIN_TYPES, defined in SiEPIC._globals 
    self.idx = idx              # pin number, index, should be unique, 0, 1,
    self.component_idx = component_idx  # which component index this pin belongs to
    self.net = net               # which net this pin is connected to
    self.pin_name = pin_name     # label read from the cell layout (PinRec text)
    
    
  def transform(self, trans):
    from .utils import angle_vector
    self.path = self.path.transformed(trans)
    pts = self.path.get_points()
    self.center = (pts[0]+pts[1])*0.5
    self.rotation = angle_vector(pts[0]-pts[1])
    return self

class WaveguideGUI():

  def __init__(self):
    import os
  
    ui_file = pya.QFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "files", "waveguide_gui.ui"))
    ui_file.open(pya.QIODevice().ReadOnly)
    self.window = pya.QFormBuilder().load(ui_file, pya.Application.instance().main_window())
    ui_file.close
    
    self.window.setFixedSize(pya.Application.instance().desktop().screenGeometry().width/6, pya.Application.instance().desktop().screenGeometry().height/2)
    
    table = self.window.findChild('layerTable')
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels([ "Layer","Width","Offset"])
    table.setColumnWidth(0, 140)
    table.setColumnWidth(1, 50)
    table.setColumnWidth(2, 50)

    #Button Bindings
    self.window.findChild('ok').clicked(self.ok)
    self.window.findChild('cancel').clicked(self.close)
    self.window.findChild('numLayers').valueChanged(self.updateTable)
    self.window.findChild('radioStrip').toggled(self.updateFields)
    self.window.findChild('radioRidge').toggled(self.updateFields)
    self.window.findChild('radioSlot').toggled(self.updateFields)
    self.window.findChild('radioCustom').toggled(self.updateFields)
    self.window.findChild('adiabatic').toggled(self.updateFields)
    self.window.findChild('radioStrip').click()
    self.status = None
    self.layers = []
    
  def updateTable(self, val):
    table = self.window.findChild("layerTable")
    cur = table.rowCount
    if cur > val:
      for i in range(val, cur):
        table.removeRow(i)
    else:
      for i in range(cur, val):
        table.insertRow(i)
        item = pya.QComboBox(table)
        item.clear()
        item.addItems(self.layers)
        table.setCellWidget(i, 0, item)
        item = pya.QLineEdit(table)
        item.setText("0.5")
        table.setCellWidget(i, 1, item)
        item = pya.QLineEdit(table)
        item.setText("0")
        table.setCellWidget(i, 2, item)
        
  def updateFields(self, val):
  
    if self.window.findChild('radioStrip').isChecked():
      self.window.findChild('stripWidth').setEnabled(True)
      self.window.findChild('stripLayer').setEnabled(True)
    else:
      self.window.findChild('stripWidth').setEnabled(False)
      self.window.findChild('stripLayer').setEnabled(False)
      
    if self.window.findChild('radioRidge').isChecked():
      self.window.findChild('ridgeWidth1').setEnabled(True)
      self.window.findChild('ridgeWidth2').setEnabled(True)
      self.window.findChild('ridgeLayer1').setEnabled(True)
      self.window.findChild('ridgeLayer2').setEnabled(True)
    else:
      self.window.findChild('ridgeWidth1').setEnabled(False)
      self.window.findChild('ridgeWidth2').setEnabled(False)
      self.window.findChild('ridgeLayer1').setEnabled(False)
      self.window.findChild('ridgeLayer2').setEnabled(False)
      
    if self.window.findChild('radioSlot').isChecked():
      self.window.findChild('slotWidth1').setEnabled(True)
      self.window.findChild('slotWidth2').setEnabled(True)
      self.window.findChild('slotLayer').setEnabled(True)
    else:
      self.window.findChild('slotWidth1').setEnabled(False)
      self.window.findChild('slotWidth2').setEnabled(False)
      self.window.findChild('slotLayer').setEnabled(False)
      
    if self.window.findChild('radioCustom').isChecked():
      self.window.findChild('numLayers').setEnabled(True)
      self.window.findChild('layerTable').setEnabled(True)
    else:
      self.window.findChild('numLayers').setEnabled(False)
      self.window.findChild('layerTable').setEnabled(False)
      
    if self.window.findChild('adiabatic').isChecked():
      self.window.findChild('bezier').setEnabled(True)
    else:
      self.window.findChild('bezier').setEnabled(False)

  def updateLayers(self, val):
    self.window.findChild("stripLayer").clear()
    self.window.findChild("ridgeLayer1").clear()
    self.window.findChild("ridgeLayer2").clear()
    self.window.findChild("slotLayer").clear()
    self.layers = []
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
      raise Exception("No view selected")

    itr = lv.begin_layers()
    while True:
      if itr == lv.end_layers():
        break
      else:
        self.layers.append(itr.current().name + " - " + itr.current().source.split('@')[0])
        itr.next()
    self.window.findChild("stripLayer").addItems(self.layers)
    self.window.findChild("ridgeLayer1").addItems(self.layers)
    self.window.findChild("ridgeLayer2").addItems(self.layers)
    self.window.findChild("slotLayer").addItems(self.layers)
    
    self.window.findChild("ridgeLayer2").setCurrentIndex(2)

  def show(self):
    self.updateLayers(0)
    self.updateTable(0)
    self.window.show()
  
  def close(self, val):
    self.status = False
    self.window.close()
    from . import scripts
    scripts.waveguide_from_path()

  def ok(self, val):
    self.status = True
    self.window.close()
    from . import scripts
    scripts.waveguide_from_path()
    
  def return_status(self):
    status = self.status
    self.status = None
    return status
  
  def get_parameters(self):
    params = { 'radius': float(self.window.findChild('radius').text),
               'width': 0,
               'adiabatic': self.window.findChild('adiabatic').isChecked(),
               'bezier': float(self.window.findChild('bezier').text),
               'wgs':[] }

    if self.window.findChild('radioStrip').isChecked():
    
      layer = self.window.findChild('stripLayer').currentText
      layer = layer.split(' ')[-1].split('/')
      params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': float(self.window.findChild('stripWidth').text), 'offset': 0})
      params['width'] = params['wgs'][0]['width']
      
    elif self.window.findChild('radioRidge').isChecked():
      layer = self.window.findChild('ridgeLayer1').currentText
      layer = layer.split(' ')[-1].split('/')
      params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': float(self.window.findChild('ridgeWidth1').text), 'offset': 0})
      layer = self.window.findChild('ridgeLayer2').currentText
      layer = layer.split(' ')[-1].split('/')
      params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': float(self.window.findChild('ridgeWidth2').text), 'offset': 0})
      params['width'] = params['wgs'][0]['width']
      
    elif self.window.findChild('radioSlot').isChecked():
      w1 = float(self.window.findChild('slotWidth1').text)
      w2 = float(self.window.findChild('slotWidth2').text)
      layer = self.window.findChild('slotLayer').currentText
      layer = layer.split(' ')[-1].split('/')
      params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': (w1-w2)/2,'offset': (w1+w2)/4})
      params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': (w1-w2)/2,'offset': -(w1+w2)/4})
      params['width'] = w1
    elif self.window.findChild('radioCustom').isChecked():
      table = self.window.findChild('layerTable')
      for i in range(0, int(self.window.findChild('numLayers').value)):
        layer = table.cellWidget(i,0).currentText
        layer = layer.split(' ')[-1].split('/')
        params['wgs'].append({'layer': pya.LayerInfo(int(layer[0]), int(layer[1])), 'width': float(table.cellWidget(i,1).text), 'offset': float(table.cellWidget(i,2).text)})
        w = (params['wgs'][-1]['width']/2+params['wgs'][-1]['offset'])*2
        if params['width'] < w:
          params['width'] = w
    return params

class CalibreGUI():
  def __init__(self):
    import os
  
    ui_file = pya.QFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "files", "calibre_drc_gui.ui"))
    ui_file.open(pya.QIODevice().ReadOnly)
    self.window = pya.QFormBuilder().load(ui_file, pya.Application.instance().main_window())
    ui_file.close
    
    #Button Bindings
    self.window.findChild('connect').clicked(self.ok)
    self.window.findChild('cancel').clicked(self.close)
    self.status = None
    
  def show(self):
    self.window.show()
  
  def close(self, val):
    self.status = False
    self.window.close()
    from . import scripts
    scripts.calibreDRC()

  def ok(self, val):
    self.status = True
    self.window.close()
    from . import scripts
    scripts.calibreDRC()
    
  def return_status(self):
    status = self.status
    self.status = None
    return status
  
  def get_parameters(self):
    return {'url': self.window.findChild('url').text,
            'port': self.window.findChild('port').text,
            'pdk': self.window.findChild('pdk').text,
            'calibre': self.window.findChild('calibre').text,
            'identity': self.window.findChild('identity').text}

class MonteCarloGUI():

  def __init__(self):
    import os
  
    ui_file = pya.QFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "files", "monte_carlo_gui.ui"))
    ui_file.open(pya.QIODevice().ReadOnly)
    self.window = pya.QFormBuilder().load(ui_file, pya.Application.instance().main_window())
    ui_file.close
    
  def show(self):
    pass
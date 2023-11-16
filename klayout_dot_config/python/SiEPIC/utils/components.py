
'''
SiEPIC-Tools/components

Utilities to make SiEPIC-Tools components


'''

import pya

def get_deviceonly_layers(cell, verbose=False):
    '''
    Find the device-only layers, as defined by Verification.xml
        EBeam PDK:
            - Si 1\0
            - SiN 1\5
            - M1_heater 11\0

    '''
    if verbose:
        print(" - SiEPIC.utils.components.get_deviceonly_layers." )
        
    from SiEPIC.utils import load_Verification
    verification = load_Verification()

    if verbose:
        print(" - get_deviceonly_layers, verification: %s" % verification)


    if not verification:
        raise Exception ('Device layers need to be defined in Verification.xml')
        
    # define device-only layers
    try:
        deviceonly_layers = eval(verification['verification']['shapes-inside-components']['deviceonly-layers'])
    except:
        deviceonly_layers = [ [1,0] ]

    ly = cell.layout()
    deviceonly_layers_ids = [ly.find_layer(*l) for l in deviceonly_layers if ly.find_layer(*l) is not None]

    if verbose:
        print(" - Device-only layers: %s" % deviceonly_layers_ids)

    return deviceonly_layers_ids

def get_device_bbox(cell, deviceonly_layers_ids, verbose=False):
    '''Find the bounding box for the device
    '''
    if verbose:
        print('SiEPIC.utils.components.get_device_bbox')

    bbox = pya.Box(0,0)        
    for l in deviceonly_layers_ids:
        bbox += cell.bbox(l)

    if verbose:
        print(" - Bounding box: %s" % bbox )
    
    return bbox

def create_device_DevRec(cell, ports, offset=0.5, verbose=False):
    '''Create a DevRec layer, with DevRec aligned with ports, and extended elsewhere
    ports: L: Left, R: Right, T: Top, B: Bottom
    offset: increase the DevRec by this amount (in dbu)
    '''

    if verbose:
        print('SiEPIC.utils.components.create_device_DevRec')

    from SiEPIC.extend import to_itype

    deviceonly_layers_ids = get_deviceonly_layers(cell, verbose)
    bbox = get_device_bbox(cell, deviceonly_layers_ids, verbose)
    if 'T' not in ports:
        bbox.top = bbox.top + to_itype(offset,cell.layout().dbu)
    if 'B' not in ports:
        bbox.bottom = bbox.bottom - to_itype(offset,cell.layout().dbu)
    if 'L' not in ports:
        bbox.left = bbox.left - to_itype(offset,cell.layout().dbu)
    if 'R' not in ports:
        bbox.right = bbox.right + to_itype(offset,cell.layout().dbu)

    layout = cell.layout()
    cell.shapes(layout.layer(layout.TECHNOLOGY['DevRec'])).insert(bbox)

    return bbox
    
def create_device_PinRec(cell, ports, devrec, verbose=False):
    '''Create the Pins
    ports: L: Left, R: Right, T: Top, B: Bottom
    '''

    if verbose:
        print('SiEPIC.utils.components.create_device_PinRec')

    from SiEPIC.utils.layout import make_pin

    deviceonly_layers_ids = get_deviceonly_layers(cell, verbose)
    LayerPinRecN = cell.layout().layer(cell.layout().TECHNOLOGY['PinRec'])

    # make Region from all device shapes
    region = pya.Region()
    iter1 = pya.RecursiveShapeIterator(cell.layout(), cell, deviceonly_layers_ids )
    while not iter1.at_end():
        if verbose:
            print("   - Shape: %s" % iter1.shape())
        region += pya.Region(iter1)
        iter1.next()        
    region = region.merged()
    
    if verbose:
        print("   - Region: %s" % region)

    # find DevRec edges that are interacting with device shapes, and make optical pins
    pinN = 0
    if 'L' in ports:
        edge = pya.Edge(devrec.left, devrec.top, devrec.left, devrec.bottom)
        port_edges = pya.Edges(edge) & region # find edges that overlap with shapes
        port_edges = sorted(port_edges, key=lambda x: -x.y1) # sort top to bottom
        for e in port_edges:
            pinN += 1
            make_pin(cell, "opt" + str(pinN), [e.x1,int((e.y1+e.y2)/2)], abs(e.y1-e.y2), LayerPinRecN, 180)
        
    if 'R' in ports:
        edge = pya.Edge(devrec.right, devrec.top, devrec.right, devrec.bottom)
        port_edges = pya.Edges(edge) & region
        port_edges = sorted(port_edges, key=lambda x: -x.y1) # sort top to bottom
        for e in port_edges:
            pinN += 1
            make_pin(cell, "opt" + str(pinN), [e.x1,int((e.y1+e.y2)/2)], abs(e.y1-e.y2), LayerPinRecN, 0)

    if 'T' in ports:
        edge = pya.Edge(devrec.left, devrec.top, devrec.right, devrec.top)
        port_edges = pya.Edges(edge) & region # find edges that overlap with shapes
        port_edges = sorted(port_edges, key=lambda x: x.x1) # sort left to right
        for e in port_edges:
            pinN += 1
            make_pin(cell, "opt" + str(pinN), [int((e.x1+e.x2)/2),e.y1], abs(e.x1-e.x2), LayerPinRecN, 90)

    if 'B' in ports:
        edge = pya.Edge(devrec.left, devrec.bottom, devrec.right, devrec.bottom)
        port_edges = pya.Edges(edge) & region # find edges that overlap with shapes
        port_edges = sorted(port_edges, key=lambda x: x.x1) # sort left to right
        for e in port_edges:
            pinN += 1
            make_pin(cell, "opt" + str(pinN), [int((e.x1+e.x2)/2),e.y1], abs(e.x1-e.x2), LayerPinRecN, 270)


def cell_to_component(cell, ports = ['L','R'], verbose=False):
    '''Take a cell with geometries, and turn it into a SiEPIC Component
    as per https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout
    - Add a DevRec box
    - Add PinRec paths and text labels
    '''
    if verbose:
        print('SiEPIC.utils.components.cell_to_component')
        print(' cell: %s' % cell.name)
        print(' ports: %s' % ports)

    # this script can be run inside KLayout's GUI application, or
    # or from the command line: klayout -zz -r H3LoQP.py
    from SiEPIC._globals import Python_Env
    if Python_Env == "KLayout_GUI":
        lv = pya.Application.instance().main_window().current_view()
        lv.transaction("Cell to SiEPIC Component")        
        
    devrec = create_device_DevRec(cell, ports, verbose)
    pinrec = create_device_PinRec(cell, ports, devrec, verbose)

    if Python_Env == "KLayout_GUI":
        lv.commit()


if __name__ == "__main__":
    TECHNOLOGY, lv, layout, topcell = get_layout_variables()  
    layout.TECHNOLOGY = TECHNOLOGY
    # Find the selected objects
    o_selection = lv.object_selection   # returns ObjectInstPath[].
    
    if len(o_selection) != 1:
        v = pya.MessageBox.warning(
            "No selection", "Select one cell you wish to have turned into a SiEPIC Component, with the DevRec and PinRec layers added.\nRead more about component requirements: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout.", pya.MessageBox.Ok)
    else:
        o_selection = o_selection[0]
        if o_selection.is_cell_inst() == False:
            v = pya.MessageBox.warning(
                "No selection", "The selected object must be an instance of a cell (not primitive polygons)", pya.MessageBox.Ok)
        else:
            cell = o_selection.inst().cell
            cell_to_component(cell, ports = ['L','R'], verbose=True)

if 0:
    '''
    Test case
        in EBeam PDK
    '''

    from SiEPIC.utils import get_layout_variables, load_Waveguides_by_Tech
    TECHNOLOGY, lv, layout, topcell = get_layout_variables()  

    waveguide_type='Strip TE 1550 nm, w=500 nm'
    from SiEPIC.scripts import connect_pins_with_waveguide
    
    # create test cell 1
    cell = layout.create_cell('testH')
    box = Box(0, 500, 10000, 1000)
    cell.shapes(layout.layer(TECHNOLOGY['Si'])).insert(box)
    box = Box(0, -500, 10000, -1000)
    cell.shapes(layout.layer(TECHNOLOGY['Si'])).insert(box)
    # instantiate test cell
    t = Trans(Trans.R0, 0, 0)
    inst = topcell.insert(CellInstArray(cell.cell_index(), t))
    
    cell_to_component(cell, ports = ['L','R'], verbose=True)

    connect_pins_with_waveguide(inst, 'opt1', inst, 'opt3', waveguide_type=waveguide_type, turtle_B = [10, 90, 10, 90], turtle_A = [10, -90, 10, -90])
    connect_pins_with_waveguide(inst, 'opt2', inst, 'opt4', waveguide_type=waveguide_type, turtle_A = [10, 90, 10, 90], turtle_B = [10, -90, 10, -90])
    
    # create test cell 2
    cell = layout.create_cell('testV')
    box = Box(1000, 10000, 500, 0)
    cell.shapes(layout.layer(TECHNOLOGY['Si'])).insert(box)
    box = Box(-1000, 10000, -500, 0)
    cell.shapes(layout.layer(TECHNOLOGY['Si'])).insert(box)
    # instantiate test cell
    t = Trans(Trans.R0, 40000, 0)
    inst = topcell.insert(CellInstArray(cell.cell_index(), t))
    
    cell_to_component(cell, ports = ['T','B'], verbose=True)

    connect_pins_with_waveguide(inst, 'opt1', inst, 'opt3', waveguide_type=waveguide_type, turtle_A = [10, 90, 10, 90], turtle_B = [10, -90, 10, -90])
    connect_pins_with_waveguide(inst, 'opt2', inst, 'opt4', waveguide_type=waveguide_type, turtle_B = [10, 90, 10, 90], turtle_A = [10, -90, 10, -90])
    
def find_all_components(cell):
  # function to traverse the entire layout hierarchy and find all the components
  # returns list of components, location, orientation
  # use the DevRec shapes.  One DevRec shape per component.
  
  # Find all the DevRec shapes; identify the component it belongs; record the instance info as an Optical_component 
  # for each component instance, also find the Pins and Fibre ports.
  
  # Use the pin names on layer PinRec to sort the pins in alphabetical order
  #   Requires that a text label be in PinRec layer, co-linear inside the PinRec path.

  _globals.NET.refresh().pins

  # Find all the DevRec shapes
  iter1 = cell.begin_shapes_rec(LayerDevRecN)
  i=0
  while not(iter1.at_end()):
    i+=1
    subcell = iter1.cell()             # cell (component) to which this shape belongs
    component = subcell.basic_name()   # name library component
    instance = subcell.name      
    subcell.name                # name of the cell; for PCells, different from basic_name
    found_component = False
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
      box= iter1.shape().box.transformed(iter1.itrans())
      print("%s: DevRec in cell {%s}, box -- %s; %s"   % (i, subcell.basic_name(), box.p1, box.p2) )
      found_component = True
    if iter1.shape().is_polygon():
      polygon = iter1.shape().polygon.transformed(iter1.itrans())
      print("%s: DevRec in cell {%s}, polygon -- %s"   % (i, subcell.basic_name(), polygon))
      found_component = True

    # A component was found. record the instance info as an Optical_component 
    if found_component:
      # Find text label for DevRec, to get Library name
      library = None
      iter2 = subcell.begin_shapes_rec(LayerDevRecN)
      spice_params = ""
      while not(iter2.at_end()):
        if iter2.shape().is_text():
          text = iter2.shape().text
          print("%s: DevRec label: %s" % (i, text))
          if text.string.find("Lumerical_INTERCONNECT_library="):
            library = text.string[len("Lumerical_INTERCONNECT_library="):]
          if text.string.find("Lumerical_INTERCONNECT_component="):
            component = text.string[len("Lumerical_INTERCONNECT_component="):]
          if text.string.find("Spice_param:"):
            spice_params = text.string[len("Spice_param:"):]
        iter2.next()
      if library == None:
        print("Missing library information for component: %s" % component )
      # get the cell's x,y coordinates
      x = iter1.itrans().disp.x*dbu
      y = iter1.itrans().disp.y*dbu
      flip = iter1.trans().is_mirror()
      rotate = (int(iter1.trans().rot())*90) % 360
      component_idx = len(optical_components)
      optical_components.append ( Optical_component (component_idx, \
                  component, instance, x, y, flip, rotate, library, spice_params) )
      
      # Find the PinRec and record info as Optical_pin
      
      # Use the pin names on layer PinRec to sort the pins in alphabetical order
      # read; sort; save data.
      iter2 = subcell.begin_shapes_rec(LayerPinRecN)
      pin_info1 = []  # array for Pin_info
      path_points=[]
      path_shape=[]   # for optical pins
      box_shape=[]    # for electrical pins
      while not(iter2.at_end()):
        # Find text label for PinRec, to get the port numbers
        if iter2.shape().is_text():
          texto= iter2.shape().text.transformed(iter2.itrans())
          texto= texto.transformed(iter1.itrans())
          x = texto.x
          y = texto.y
#          print( "PinRec label: %s at (%s, %s)" % (iter2.shape().text, x, y) )
          pin_info1.append(Pin_info(iter2.shape().text.string, x, y))
        if iter2.shape().is_path():
          # assume this is an optical pin
          path= iter2.shape().path.transformed(iter2.itrans()).transformed(iter1.itrans())
          points = path_to_points(path)  
#          print( "%s: PinRec in cell {%s}, path -- %s"   % (i, iter2.cell().name, path) )
          path_points.append(points)
          path_shape.append(iter2.shape())
        if iter2.shape().is_box():
          # assume this is an electrical pin
          box = iter2.shape().box.transformed(iter2.itrans()).transformed(iter1.itrans())
          box_shape.append(box)
#          print( "%s: PinRec in cell {%s}, box -- %s"   % (i, iter2.cell().name, box) )
        iter2.next()
      # Sort pin text labels
      pin_info2 = sorted(pin_info1, key=lambda  pin_info0: pin_info0.pin_text)
      
      # find pin labels that are inside the DevRec shape:
      for p1 in range(0,len(pin_info2)):
        for p2 in range(0,len(path_shape)):  # (optical)
          # Check if the pin text label is somewhere along the pin path
#          check_text_in_pin = check_point_in_segment( \
#            pya.Point( *path_points[p2][0] ), \
#            pya.Point( *path_points[p2][1] ), \
#            pya.Point( pin_info2[p1].pin_x, pin_info2[p1].pin_y ) )
          # Check if the pin text label is exactly in the middle of pin path
          points = path_points[p2]  
          x = (points[0][0]+points[1][0])/2   # midpoint of pin path
          y = (points[0][1]+points[1][1])/2
          check_text_in_pin = ( x == pin_info2[p1].pin_x ) &amp; \
                              ( y == pin_info2[p1].pin_y )
          if check_text_in_pin:
            pin_idx = len(optical_pins)
            optical_pins.append (Optical_pin (pin_idx, points, component_idx, x, y, 1, pin_info2[p1].pin_text) )
            optical_components[component_idx].npins += 1
            optical_components[component_idx].pins.append( pin_idx )
            print("%s: PinRec (text=%s) in cell {%s}, component #%s, path -- %s"   \
              % (i, pin_info2[p1].pin_text, iter2.cell().name, component_idx, path_to_points(path_shape[p2].path)) )
        for p2 in range(0,len(box_shape)):
          # Check if the pin text label is inside the DevRec pin box (electrical)
          print('epin x: %s, %s, %s' % (pin_info2[p1].pin_x,box_shape[p2].left,box_shape[p2].right))
          print('epin y: %s, %s, %s' % (pin_info2[p1].pin_y,box_shape[p2].bottom,box_shape[p2].top))
          if (pin_info2[p1].pin_x &gt; box_shape[p2].left) &amp; \
             (pin_info2[p1].pin_x &lt; box_shape[p2].right) &amp; \
             (pin_info2[p1].pin_y &gt; box_shape[p2].bottom) &amp; \
             (pin_info2[p1].pin_y &lt; box_shape[p2].top):
            optical_components[component_idx].nepins += 1
            optical_components[component_idx].epins.append(pin_info2[p1].pin_text  )
            print("%s: PinRec (text=%s) in cell {%s}, component #%s, box -- %s"   \
              % (i, pin_info2[p1].pin_text, iter2.cell().name, component_idx, box_shape[p2]) )
          

      # reserve space for netlist for this component, based on the number of pins.
      optical_components[component_idx].nets = [-1] * (optical_components[component_idx].npins)

      # Find the FbrTgt, and record info as Optical_pin
      iter2 = subcell.begin_shapes_rec(LayerFbrTgtN)
      while not(iter2.at_end()):
        if iter2.shape().is_polygon():
          polygon = iter2.shape().polygon.transformed(iter2.itrans())
          polygon = polygon.transformed(iter1.itrans())
          # What do we want to do with the Fibre ports?
          # create a net (with only 1 member = the optical IO)
          net_idx = len(optical_nets)
          optical_nets.append ( Optical_net (net_idx, 2, component_idx, -1, -1) ) #ok
          # create a pin, optical IO
          bb = polygon.bbox()
          x, y = xy_mean_mult([[bb.p1.x, bb.p1.y], [bb.p2.x, bb.p2.y]],1)
#          x, y = numpy.mean([bb.p1.x, bb.p2.x]), numpy.mean([bb.p1.y, bb.p2.y])
          pin_idx = len(optical_pins)
          optical_pins.append (Optical_pin (pin_idx, polygon_to_points(polygon), component_idx, x, y, 2, "pin0") )
          # register the net with the pin and component
          optical_pins[pin_idx].net = net_idx
          optical_components[component_idx].nets.insert(0, net_idx )
          # register the pin with the component
          optical_components[component_idx].npins += 1
          optical_components[component_idx].pins.insert(0, pin_idx ) 
          print("%s: FbrTgt in cell {%s}, at (%s, %s), net %s, pins %s, component # %s, component nets %s" \
              % (i, iter2.cell().name, x, y, net_idx, optical_components[component_idx].pins, \
              component_idx, optical_components[component_idx].nets ) )
        iter2.next()


    # end if found_component   
     
    iter1.next()
 
  # end while iter1 
# end def find_all_components

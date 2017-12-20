

  # Determine the positions of all the components, in order to determine scaling
  sch_positions = []
  for o in optical_components:
    sch_positions.append ([ o.x, o.y ])
  for o in optical_waveguides:
    x,y = xy_mean_mult(o.points, dbu)
    sch_positions.append ([ x,y ])
  sch_distances = []
  for j in range(len(sch_positions)):
    for k in range(j+1,len(sch_positions)):
      dist = distance_xy ( sch_positions[j], sch_positions[k] )
      sch_distances.append ( dist )

  # find minimum distance between objects, but skip the closest x %, to make the layout more compact
  # e.g., terminators connected to ring resonators are very close.
  # not implemented...

  sch_distances.sort()
  print("Distances between components: %s" % sch_distances)
  
  # remove any 0 distances:
  while 0.0 in sch_distances: sch_distances.remove(0.0)
  
  # scaling based on nearest neighbour:
  Lumerical_schematic_scaling = 0.0006 / min(sch_distances)

  # but if the layout is too big, limit the size
  MAX_size = 0.05
  if max(sch_distances)*Lumerical_schematic_scaling &gt; MAX_size:
    Lumerical_schematic_scaling = MAX_size / max(sch_distances) 
  print ("Scaling for Lumerical INTERCONNECT schematic: %s" % Lumerical_schematic_scaling)

#  Lumerical_schematic_scaling = 5e-2
#  Lumerical_schematic_scaling = 20e-2
#  Lumerical_schematic_scaling = 5e-3

  # convert KLayout GDS rotation/flip to Lumerical INTERCONNECT
  # KLayout defines mirror as an x-axis flip, whereas INTERCONNECT does y-axis flip
  # KLayout defines rotation as counter-clockwise, whereas INTERCONNECT does clockwise
  # input is KLayout Rotation,Flip; output is INTERCONNECT:
  KLayoutInterconnectRotFlip = \
      {(0, False):[0, False], \
       (90, False):[270, False], \
       (180, False):[180, False], \
       (270, False):[90, False], \
       (0, True):[180,True], \
       (90, True):[90, True], \
       (180, True):[0,True], \
       (270, True):[270, False]}

  from time import strftime 
  
  text_main = '* Spice output from KLayout SiEPIC PDK v%s, %s.\n\n' % (SiEPIC_Version, strftime("%Y-%m-%d %H:%M:%S") )
  text_subckt = text_main

#wtext.insertHtml('.subckt %s %s:&lt;br&gt;' % ( topcell.name, find_optical_IO_pins(optical_pins) ))
  opticalIO_pins = find_optical_IO_pins(optical_pins)  

  # find electrical IO pins
  electricalIO_pins = ""
  DCsources = "" # string to create DC sources for each pin
  Vn = 1
  SINGLE_DC_SOURCE = 2
  # (1) attach all electrical pins to the same DC source
  # (2) or to individual DC sources
  # (3) or choose based on number of DC sources, if &gt; 5, use single DC source
  for o in optical_components:
    for p in o.epins:  # electrical pins
      NetName = " N$" + o.component +'_' + str(o.idx) + '_' + p
#      NetName = " C" + str(o.idx) + '_' + p
      electricalIO_pins += NetName
      DCsources += "N" + str(Vn) + NetName + " 0 dcsource amplitude=0 sch_x=%s sch_y=%s\n" % (-2-Vn/10., -2+Vn/8.)
      Vn += 1
  electricalIO_pins_subckt = electricalIO_pins

  if (SINGLE_DC_SOURCE == 1) or ( (SINGLE_DC_SOURCE == 2) &amp; (Vn &gt; 5)):
    electricalIO_pins_subckt = ""
    for o in optical_components:
      for p in o.epins:  # electrical pins
        NetName = " N$"
        electricalIO_pins_subckt += NetName
        DCsources = "N1" + NetName + " 0 dcsource amplitude=0 sch_x=-2 sch_y=0\n"
      
  # create the top subckt:
  text_subckt += '.subckt %s%s%s\n' % (topcell.name, electricalIO_pins, opticalIO_pins)
  text_subckt += '.param MC_uniformity_width=0 \n' # assign MC settings before importing netlist components
  text_subckt += '.param MC_uniformity_thickness=0 \n' 
  text_subckt += '.param MC_resolution_x=100 \n' 
  text_subckt += '.param MC_resolution_y=100 \n' 
  text_subckt += '.param MC_grid=10e-6 \n' 
  text_subckt += '.param MC_non_uniform=99 \n' 

  # Get information about the laser and detectors:
  laser_net, detector_nets, wavelength_start, wavelength_stop, wavelength_points, orthogonal_identifier, ignoreOpticalIOs = \
        get_LumericalINTERCONNECT_analyzers(topcell, optical_pins)
  
  for o in optical_components:
  
    nets_str = ""
    for p in o.epins:  # electrical pins
      nets_str += " N$" + o.component +'_' + str(o.idx) + '_' + p
    for n in o.nets:  # optical nets
      nets_str += " N$" + str(n)

    trans = KLayoutInterconnectRotFlip[(o.rotate, o.flip)]
     
    flip = ' sch_f=true' if trans[1] else ''
    if trans[0] &gt; 0:
      rotate = ' sch_r=%s' % str(trans[0])
    else:
      rotate = ''
#    t = '%s %s %s library="%s" lay_x=%s lay_y=%s sch_x=%s sch_y=%s %s%s'  % \
#         ( "X"+o.component+"_"+str(o.idx), nets_str, o.component, o.library, str (o.x * 1e-6), o.y * 1e-6, o.x, o.y, rotate, flip)
#    t = '  %s %s %s library="%s" %s $X=%s $Y=%s sch_x=%s sch_y=%s %s%s'  % \
    
    # Check to see if this component is an Optical IO type.
    pinIOtype = 0
    for p in o.pins:
      for each_pin in optical_pins:
        if(each_pin.idx == p):
          if each_pin.pin_type == 2:
            pinIOtype = 1
          break  
        
    if ignoreOpticalIOs and pinIOtype:
      # Replace the Grating Coupler or Edge Coupler with a 0-length waveguide.
      component1 = "ebeam_wg_strip_1550"
      params1 = "wg_length=0u wg_width=0.500u"
    else:
      component1 =  o.component 
      params1 = o.params
      
    text_subckt += ' %s %s %s ' % ( component1 +"_"+str(o.idx), nets_str, component1 ) 
    if o.library != None:
      text_subckt += 'library="%s" ' % o.library
    text_subckt += '%s lay_x=%s lay_y=%s sch_x=%s sch_y=%s %s%s\n' % \
       ( params1,
         eng_str(o.x * 1e-6), eng_str(o.y * 1e-6), \
         eng_str(o.x * Lumerical_schematic_scaling), eng_str(o.y * Lumerical_schematic_scaling), \
         rotate, flip)

  # list all Optical_waveguides objects from an array
  # input array, optical_waveguides
  # example output:         
  # X5 9 10 ebeam_wg_strip_1550 library="Design kits/ebeam_v1.0" wg_length=7.86299e-06 wg_width=5.085e-07 sch_x=-1.42 sch_y=-0.265

  for o in optical_waveguides:
    nets_str = "N$%s N$%s" %(o.net1, o.net2)
    x,y = xy_mean_mult(o.points, dbu)
    wg_angle =  ( 360- int(round(angle_segment ( [o.points[0], o.points[len(o.points)-1]] ) / 90) * 90) ) % 360
    
#    t = '%s %s %s library="%s" wg_length=%s wg_width=%s lay_x=%s lay_y=%s sch_x=%5.3f sch_y=%5.3f'  % \
#           ( "Xwg" + str(o.idx), nets_str, o.component, o.library, eng_str(o.length*1e-6), eng_str(o.wg_width*1e-6), \
#             eng_str(x * 1e-6), eng_str(y * 1e-6), x, y)
    t = '  %s %s %s library="%s" wg_length=%s wg_width=%s sch_x=%s sch_y=%s sch_r=%s points="%s"'  % \
           ( "wg" + str(o.idx), nets_str, o.component, o.library, \
             eng_str(o.length*1e-6), eng_str(o.wg_width*1e-6), \
             eng_str(x * Lumerical_schematic_scaling), eng_str(y * Lumerical_schematic_scaling), \
             #str(o.points).replace('[','(').replace(']',')')  )   # format of waveguide section points
             str(wg_angle), \
             str(o.points).replace(', ',',')  )   # change format, delete space    
    text_subckt += '%s\n' %t

  text_subckt += '.ends %s\n\n' % (topcell.name)

  if laser_net &gt; -1:
    text_main += '* Optical Network Analyzer:\n'
    text_main += '.ona input_unit=wavelength input_parameter=start_and_stop\n  + minimum_loss=80\n  + analysis_type=scattering_data\n  + multithreading=user_defined number_of_threads=1\n' 
    text_main += '  + orthogonal_identifier=%s\n' % orthogonal_identifier
    text_main += '  + start=%4.3fe-9\n' % wavelength_start
    text_main += '  + stop=%4.3fe-9\n' % wavelength_stop
    text_main += '  + number_of_points=%s\n' % wavelength_points
    for i in range(0,len(detector_nets)):
      text_main += '  + input(%s)=%s,N$%s\n' % (i+1, topcell.name, detector_nets[i])
    text_main += '  + output=%s,N$%s\n' % (topcell.name, laser_net)

  # main circuit
  text_main += '%s %s %s %s sch_x=-1 sch_y=-1 ' % (topcell.name, electricalIO_pins_subckt, opticalIO_pins, topcell.name)
  if len(DCsources) &gt; 0:
    text_main += 'sch_r=270\n\n'
  else:
    text_main += '\n\n'

  text_main += DCsources
  
  print(text_main)




'''
def check_segments_same_direction( segment1, segment2 ):
  # check that they have the same direction
  is_slope_equal = False
  # check ∆x = 0 first to avoid division by 0
  dx1 = (segment1[0].x-segment1[1].x)
  dx2 = (segment2[0].x-segment2[1].x)
  if dx2 == 0 and dx1 == 0:
    is_slope_equal = True  # both vertical
  elif dx1 != 0 and dx2 != 0:
    # check slopes
    slope1 = (segment1[0].y-segment1[1].y) / (segment1[0].x-segment1[1].x)
    slope2 = (segment2[0].y-segment2[1].y) / (segment2[0].x-segment2[1].x)
    if slope1 == slope2:
      is_slope_equal = True  # both have the same slope
  return is_slope_equal
'''

'''
def check_segments_collinear_overlapping( segment1, segment2 ):
  """ 
  we want to identify ONLY the following situation:
        X  O  X  O
  where XX is a segment, and OO is another segment
  namely, collinear, but also XX overlapping OO
  example usage: 
    a = pya.Point(0,0)
    b = pya.Point(50,0)
    c = pya.Point(50,0)
    d = pya.Point(100,0)
    segment1 = [ a, b ]
    segment2 = [ c, d ]
    print(check_segments_collinear_overlapping( segment1, segment2 ))
  """  

  # check for one of the segment2 points being inside segment1
  check_between1 = pt_intersects_segment( segment1[0], segment1[1], segment2[0] ) | \
                  pt_intersects_segment( segment1[0], segment1[1], segment2[1] )
  # check for one of the segment1 points being inside segment2
  check_between2 = pt_intersects_segment( segment2[0], segment2[1], segment1[0] ) | \
                  pt_intersects_segment( segment2[0], segment2[1], segment1[1] )

  # check that they have the same direction
#  is_slope_equal = check_segments_same_direction( segment1, segment2 )
  is_slope_equal = (inner_angle_b_vectors(segment1, segment2) == 0)
  
#  print( "check_segments_collinear_overlapping: %s, %s, %s, %s: %s, %s, %s" % (segment1[0], segment1[1], segment2[0], segment2[1], check_between1, check_between2, is_slope_equal) )
  
  return (check_between1 or check_between2) and is_slope_equal
'''




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


#################################################################################
#                SiEPIC Tools - scripts                                         #
#################################################################################
'''

connect_pins_with_waveguide
path_to_waveguide
path_to_waveguide2
roundpath_to_waveguide
waveguide_to_path
waveguide_length
waveguide_length_diff
waveguide_heal
auto_route
connect_cell
snap_component
delete_top_cells
compute_area
calibreDRC
auto_coord_extract
find_SEM_labels_gui
calculate_area
trim_netlist
layout_check
open_PDF_files
open_folder
user_select_opt_in
fetch_measurement_data_from_github
measurement_vs_simulation
resize waveguide
replace_cell
svg_from_cell
zoom_out: When running in the GUI, Zoom out and show full hierarchy
export_layout

'''


from . import _globals
if _globals.Python_Env == "KLayout_GUI":
    import pya
import pya

class Turtle:
    '''
    Manhattan definitions for Turtle-style vectors; 
      https://en.wikipedia.org/wiki/Turtle_graphics
    each point is: forward, then turn
    '''
    def __init__(self, forward_turn):
        self.forward = forward_turn[0] # distance to travel forward, in microns
        self.turn = forward_turn[1]    # -90 (right) or 90 (left) turn
        if self.turn not in [-90,90]:
          raise Exception("Either left (90) or right (-90) turn allowed.")
          return
#        rotation = {1:pya.Trans.R270, -1:pya.Trans.R90}[turn]
#        rotation = {1:270, -1:90}[self.turn]
        self.cplxtrans = pya.CplxTrans(1,self.turn,False,self.forward,0)
        self.vector = pya.Vector(self.forward,0)
        # trans(pya.CplxTrans(1,self.turn,False,0,0))
    def display(self):
        print('- turtle: %s, %s' % (self.forward, self.turn))

'''
Convert a list of points from a path into Turtle instructions (forward, turn)
Split it up into half, with one turtle starting at the beginning, 
and the other starting at the end of the list.
'''
def pointlist_to_turtles(pointlist):
    from SiEPIC.scripts import pointlist_to_turtle    
    listA=pointlist[0:round(len(pointlist)/2+1)]
    listB=pointlist[round(len(pointlist)/2-1):len(pointlist)][::-1]
    return pointlist_to_turtle(listA), pointlist_to_turtle(listB)

'''
Convert a list of points from a path into Turtle instructions (forward, turn)
'''
def pointlist_to_turtle(pointlist):
    print(pointlist)
    from SiEPIC.utils import angle_b_vectors
    turtle = []
    pts = [pya.DPoint(*p) for p in pointlist]
    for i in range(1, len(pointlist)-1):
        turtle.append(round(pts[i].distance(pts[i-1]),4))
        turtle.append(int(angle_b_vectors(pts[i]-pts[i-1],pts[i+1]-pts[i])+90)%360-90)
    return turtle
  #  return pointlist
          

def connect_pins_with_waveguide(instanceA, pinA, instanceB, pinB, waveguide = None, waveguide_type = None, turtle_A=None, turtle_B=None, verbose=False, debug_path=False, r=None, error_min_bend_radius=True, relaxed_pinnames=True):
    '''
    Create a Path connecting instanceA:pinA to instanceB:pinB
        where instance = pya.Instance; pin = string, e.g. 'pin1'
    and convert to a Waveguide with waveguide_type = [string] name, from WAVEGUIDES.XML
    using one of the following approaches:
     - fewer than 4 vertices (including endpoints): automatic, no need to specify turtles
     - turtle_A: list of Turtle (forward x microns, turn left -1 or right 1), starting from the pinA end, except for the first and last
         e.g.    [5, -90, 10, 90]
     - turtle_B: list of Turtle (forward x microns, turn left -1 or right 1), starting from the pinA end, except for the first and last
         - both turtle_A and turtle_B: 
                    relative from both pinA and pinB sides
     - the script automatically completes the path as long as the turtles are:
            - going in the same direction, or
            - having their paths crossing
     - doesn't work if they are on diverging paths; in that case add vertices.

    * works for instances that are individual components
    * works for instances that are sub-circuits with many components, but one unique pin name
     
    originally thought about implementing the following, but perhaps not useful: 
     - absolute_vertices: list of Points, except for the first and last
     - relative_vertices_from_A: list of Points, starting from the pinA end, except for the first and last
     - relative_vertices_from_B: list of Points, starting from the pinB end, except for the first and last
         - both relative_vertices_from_A and relative_vertices_from_B: 
                relative from both pinA and pinB sides
     
     
    Uses SiEPIC-Tools find_components to determine the component size, pins, etc.
    
    
    Example code:

    from SiEPIC import scripts    
    from SiEPIC.utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    # clean all cells within the present cell
    top_cell = ly.top_cells()[0]
    ly.prune_subcells(top_cell.cell_index(), 10)

    cell_amf_Terminator_TE_1550 = ly.create_cell('ebeam_crossing4', 'EBeam')
    t = pya.Trans.from_s('r270 230175,190500')
    inst_amf_Terminator_TE_1550_3 = cell.insert(pya.CellInstArray(cell_amf_Terminator_TE_1550.cell_index(), t))

    cell_AMF_IRPH_MRR_0 = ly.create_cell('ebeam_bragg_te1550', 'EBeam',
             {'r': 10.0, 'w': 0.35, 'g': 0.12, 'gmon': 0.5})

    from SiEPIC.scripts import connect_cell
    
    connect_pins_with_waveguide(inst_amf_Terminator_TE_1550_3, 'opt2', cell_AMF_IRPH_MRR_0, 'pin1')

    
    '''

    # layout information
    ly=instanceA.parent_cell.layout()
    dbu=ly.dbu
    
    # check if the two instances share the same cell
    if instanceA.parent_cell != instanceB.parent_cell:

        # check if they share a common parent-parent cell
        # only works if the parent is instantiated once, otherwise we get really confused
                
        if instanceA.parent_cell.parent_cells() > 1 | instanceA.parent_cell.parent_cells() > 1: 
                raise Exception ('connect_pins_with_waveguide function only supports routing where each instance only has only one parent cell.')

#        if instanceA.parent_cell.parent_cell.child_instances() <= 1 & instanceA.parent_cell.parent_cell.child_instances() <= 1: 
        if instanceA.parent_cell.parent_cells() == 1:
                iterA=instanceA.parent_cell.each_parent_cell()
                parentA=ly.cell(next(iterA))
                # check if the parent is only instantiated once
                each_parent_inst = instanceA.parent_cell.each_parent_inst()
                try:
                        next(each_parent_inst).inst()
                        next(each_parent_inst).inst()
                        raise Exception ('connect_pins_with_waveguide function only supports routing where each instance is only instantiated once.')
                except StopIteration:
                        pass                
        else:
                parentA=''
        if instanceB.parent_cell.parent_cells() == 1:
                # find parent parent cell
                iterB=instanceB.parent_cell.each_parent_cell()
                parentB=ly.cell(next(iterB))
                # check if the parent is only instantiated once
                each_parent_inst = instanceB.parent_cell.each_parent_inst()
                try:
                        next(each_parent_inst).inst()
                        next(each_parent_inst).inst()
                        raise Exception ('connect_pins_with_waveguide function only supports routing where each instance is only instantiated once.')
                except StopIteration:
                        pass                
        else:
                parentB=''
        # find the common parent
        parentsA = [instanceA.parent_cell, parentA]
        parentsB = [instanceB.parent_cell, parentB]
        common_cell = list(set(parentsA).intersection(parentsB))
        if verbose:
            print('%s, %s: %s' % (parentsA, parentsB, common_cell))
        if len(common_cell)==0:
                raise Exception ('connect_pins_with_waveguide function could not find a common parent for the two instances.')
        cell=common_cell[0]
                
    else:
        cell=instanceA.parent_cell
    
    # Find the two components:
    from time import time
    t = time()
    # benchmarking: find_components here takes 0.015 s
    componentA = instanceA.parent_cell.find_components(inst=instanceA)
    componentB = instanceB.parent_cell.find_components(inst=instanceB)
#    print('Time elapsed: %s' % (time() - t))    
    if componentA==[]:
        print('InstA: %s, %s' % (instanceA.cell.name, instanceA) )
        print('componentA: %s' % (componentA) )
        print('parent_cell A: %s, cell A: %s' % (instanceA.parent_cell, instanceA.cell) )
        print('all found components A: instance variable: %s' %    ([n.instance for n in instanceA.parent_cell.find_components()]) )
        print('all found components A: component variable: %s' %    ([n.component for n in instanceA.parent_cell.find_components()]) )
        raise Exception("Component '%s' not found. \nCheck that the component is correctly built (DevRec and PinRec layers). \nTry SiEPIC > Layout > Show Selected Component Information for debugging." %instanceA.cell.name)
    if componentB==[]:
        print('InstB: %s, %s' % (instanceB.cell.name, instanceB) )
        print('componentB: %s' % (componentB) )
        print('parent_cell B: %s, cell B: %s' % (instanceB.parent_cell, instanceB.cell) )
        print('all found components B: instance variable: %s' %    ([n.instance for n in instanceB.parent_cell.find_components()]) )
        print('all found components B: component variable: %s' %    ([n.component for n in instanceB.parent_cell.find_components()]) )
        raise Exception("Component '%s' not found. \nCheck that the component is correctly built (DevRec and PinRec layers). \nTry SiEPIC > Layout > Show Selected Component Information for debugging." %instanceB.cell.name)

    # if the instance had sub-cells, then there will be many components. Pick the first one.
    if type(componentA) == type([]):
        componentA = componentA[0]
    if type(componentB) == type([]):
        componentB = componentB[0]

    if verbose:
        print('InstA: %s, InstB: %s' % (instanceA, instanceB) )
        print('componentA: %s, componentB: %s' % (componentA, componentB) )

        componentA.display()
        componentB.display()
        
    # Find pinA and pinB
    cpinA = [p for p in componentA.pins if p.pin_name == pinA]
    cpinB = [p for p in componentB.pins if p.pin_name == pinB]        

    # relaxed_pinnames:  scan for only the number
    if relaxed_pinnames==True:
        import re
        try:
            if cpinA==[]:
                if re.findall(r'\d+', pinA):
                    cpinA = [p for p in componentA.pins if re.findall(r'\d+', pinA)[0] in p.pin_name]
            if cpinB==[]:
                if re.findall(r'\d+', pinB):
                    cpinB = [p for p in componentB.pins if re.findall(r'\d+', pinB)[0] in p.pin_name]
        except:
            print('error in siepic.scripts.connect_cell, relaxed_pinnames')      


    if cpinA==[]:
        try:  
            # this checks if the cell (which could contain multiple components) 
            # contains only one pin matching the name, e.g. unique opt_input in a sub-circuit
            cpinA = [instanceA.find_pin(pinA)]
        except:
              error_message = "SiEPIC-Tools, in function connect_pins_with_waveguide: Pin (%s) not found in componentA (%s). Available pins: %s" % (pinA,componentA.component, [p.pin_name for p in componentA.pins])
              if _globals.Python_Env == "KLayout_GUI":
                question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
                question.setText("SiEPIC-Tools scripted layout, requested pin not found")
                question.setInformativeText(error_message)
                pya.QMessageBox_StandardButton(question.exec_())
                return
              else:          
                raise Exception(error_message)
    if cpinB==[]:
        try:  
            # this checks if the cell (which could contain multiple components) 
            # contains only one pin matching the name, e.g. unique opt_input in a sub-circuit
            cpinB = [instanceB.find_pin(pinB)]
        except:
              error_message = "SiEPIC-Tools, in function connect_pins_with_waveguide: Pin (%s) not found in componentB (%s). Available pins: %s" % (pinB,componentB.component, [p.pin_name for p in componentB.pins])
              if _globals.Python_Env == "KLayout_GUI":
                question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
                question.setText("SiEPIC-Tools scripted layout, requested pin not found")
                question.setInformativeText(error_message)
                pya.QMessageBox_StandardButton(question.exec_())
                return
              else:          
                raise Exception(error_message)

    cpinA=cpinA[0]
    cpinB=cpinB[0]
    if verbose:
        cpinA.display()
        cpinB.display()

    # apply hierarchical transformation on the pins, if necessary
    if cell != instanceA.parent_cell:
        iterA=instanceA.parent_cell.each_parent_inst()
        parentA=next(iterA).inst()
        cpinA.transform(parentA.trans.inverted())
    if cell != instanceB.parent_cell:
        iterB=instanceB.parent_cell.each_parent_inst()
        parentB=next(iterB).inst()
        cpinB.transform(parentB.trans.inverted())
        
    # check if the pins are already connected
    if cpinA.center == cpinB.center:
        print('Pins are already connected; returning')
        return True

    # split this into separate function
    # connect_Pins_with_waveguide
    # input would be two Pins.
    # Instance.find_pins
            
    from .utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name(ly.technology().name)
    technology_name = TECHNOLOGY['technology_name']
    
    # Waveguide type:  
    # Backwards compatible with Waveguides that don't use "waveguide_types"
    if not(waveguide):
        waveguides = ly.load_Waveguide_types()
        if verbose:
            print('Time elapsed, to load waveguide types: %s' % (time() - t))    
            print(waveguides)
        if not(waveguide_type):
            waveguide = waveguides[0]
        if waveguide_type:
            waveguide1 = [w for w in waveguides if w['name']==waveguide_type]
            if verbose:
                print('matching waveguide types: %s' % waveguide1)
            if type(waveguide1) == type([]) and len(waveguide1)>0:
                waveguide = waveguide1[0]
            else:
                waveguide = waveguides[0]
                print('error: waveguide type not found in PDK waveguides')
                raise Exception('error: waveguide type (%s) not found in PDK waveguides' % waveguide_type)
        # check if the waveguide type is compound waveguide
        if 'compound_waveguide' in waveguide:
            waveguide = [w for w in waveguides if w['name']==waveguide['compound_waveguide']['singlemode']]
            waveguide = waveguide[0]
    if verbose:
        print('waveguide type: %s' % waveguide )    
    # Find the 'Waveguide' layer in the waveguide.XML definition, and use that for the width paramater.
    
    waveguide_component = [c for c in waveguide['component'] if c['layer']=='Waveguide']
    if len(waveguide_component) > 0:
        width_um = waveguide_component[0]['width']
    else: # pick the first one:
        width_um = waveguide['component'][0]['width']
    width_um=float(width_um)
    from SiEPIC.extend import to_itype
    width=to_itype(width_um,dbu)
#    layer=waveguide['component'][0]['layer']    # pick the first layer in the waveguide definition, for the path.
        
    # Create the path
    points_fromA = [cpinA.center] # first point A
    points_fromB = [cpinB.center] # last    point B

    # if no turtles are specified, assume a forward movement to be the bend radius
    # if user hasn't specified radius, load from XML
    if r is None:
        radius_um = float(waveguide['radius'])
        radius = to_itype(waveguide['radius'],dbu)
    else:
        radius_um = r
        radius = to_itype(radius_um,dbu)
    
    if turtle_A == None:
        turtle_A = [radius_um]
    if turtle_B == None:
        turtle_B = [radius_um]


    # go through all the turtle instructions and build up the points
    from math import floor
    directionA = cpinA.rotation
    directionB = cpinB.rotation
    if turtle_A != None:
        vector=pya.CplxTrans(1,directionA,False,0,0).trans(pya.Vector(to_itype(turtle_A[0],dbu),0))
        points_fromA.append(points_fromA[-1]+vector)
        for i in range(floor(len(turtle_A)/2)-1):
            directionA = (directionA + turtle_A[i*2+1]) % 360
            vector=pya.CplxTrans(1,directionA,False,0,0).trans(pya.Vector(to_itype(turtle_A[i*2+2],dbu),0))
            points_fromA.append(points_fromA[-1]+vector)
    if turtle_B != None:
        vector=pya.CplxTrans(1,directionB,False,0,0).trans(pya.Vector(to_itype(turtle_B[0],dbu),0))
        points_fromB.append(points_fromB[-1]+vector)
        for i in range(floor(len(turtle_B)/2)-1):
            directionB = (directionB + turtle_B[i*2+1]) % 360
            vector=pya.CplxTrans(1,directionB,False,0,0).trans(pya.Vector(to_itype(turtle_B[i*2+2],dbu),0))
            points_fromB.append(points_fromB[-1]+vector)
    directionA = directionA % 360
    directionB = directionB % 360
    
    # check if the turtles directions are coming together at 90 angle, 
    # then add a vertex
    if (directionB - directionA - 90) % 360 in [0,180]:
        if verbose:
            print('Turtles are coming together at 90 degree angle, adding point')
        if directionA==0:
            if     (directionB==270 and points_fromB[-1].y>points_fromA[-1].y and points_fromB[-1].x>points_fromA[-1].x) \
                or (directionB==90    and points_fromB[-1].y<points_fromA[-1].y and points_fromB[-1].x>points_fromA[-1].x):
                points_fromA.append(pya.Point(points_fromB[-1].x,points_fromA[-1].y))
        if directionA==180:
            if     (directionB==270 and points_fromB[-1].y>points_fromA[-1].y and points_fromB[-1].x<points_fromA[-1].x) \
                or (directionB==90    and points_fromB[-1].y<points_fromA[-1].y and points_fromB[-1].x<points_fromA[-1].x):
                points_fromA.append(pya.Point(points_fromB[-1].x,points_fromA[-1].y))
        if directionA==90:
            if     (directionB==180 and points_fromB[-1].x>points_fromA[-1].x and points_fromB[-1].y>points_fromA[-1].y) \
                or (directionB==0     and points_fromB[-1].x<points_fromA[-1].x and points_fromB[-1].y>points_fromA[-1].y):
                points_fromA.append(pya.Point(points_fromA[-1].x,points_fromB[-1].y))
        if directionA==270:
            if     (directionB==180 and points_fromB[-1].x>points_fromA[-1].x and points_fromB[-1].y<points_fromA[-1].y) \
                or (directionB==0     and points_fromB[-1].x<points_fromA[-1].x and points_fromB[-1].y<points_fromA[-1].y):
                points_fromA.append(pya.Point(points_fromA[-1].x,points_fromB[-1].y))

    
    # check if the turtles going towards each other (180)
    #    - check if the turtles are offset from each other, 
    #        then edit their points to match
    #    - check if they do not have an offset; 
    #        then keep only end points
    if (directionB - directionA - 180) % 360 == 0:
        if verbose:
            print('Turtles are going towards each other ...')
        # horizontal
        if directionA in [0, 180]: 
            # check for y offset
            if points_fromA[-1].y != points_fromB[-1].y:
                # average the x position
                x = (points_fromA[-1].x + points_fromB[-1].x)/2
                points_fromA[-1].x = x
                points_fromB[-1].x = x
                if verbose:
                    print('    Turtles are offset, editing points')
            else:
                points_fromA = [points_fromA[0]]
                points_fromB = [points_fromB[0]]
                if verbose:
                    print('    Turtles are not offset, keeping only endpoints')
                # 
        # vertical
        else:
            # check for x offset
            if points_fromA[-1].x != points_fromB[-1].x:
                # average the y position
                y = (points_fromA[-1].y + points_fromB[-1].y)/2
                points_fromA[-1].y = y
                points_fromB[-1].y = y
                if verbose:
                    print('    Turtles are offset, editing points')
            else:
                points_fromA = [points_fromA[0]]
                points_fromB = [points_fromB[0]]
                if verbose:
                    print('    Turtles are not offset, keeping only endpoints')
                
    # check if the turtles are offset from each other, but going the same way (0)
    # then edit their points to match
    # ensuring that there is enough space for 2 x bend radius (U turn)
    if (directionB - directionA - 0) % 360 == 0:
        if verbose:
            print('Turtles are offset, going the same way, editing points')
        # horizontal:
        if directionA in [0, 180]: 
            # check for y offset
            if points_fromA[-1].y != points_fromB[-1].y:
                # find the x position
                if directionA == 180: 
                    x = min(points_fromA[-1].x, points_fromB[-1].x,
                            points_fromA[-2].x-1*radius, points_fromB[-2].x-1*radius)
                else:
                    x = max(points_fromA[-1].x, points_fromB[-1].x,
                            points_fromA[-2].x+1*radius, points_fromB[-2].x+1*radius)
                points_fromA[-1].x = x
                points_fromB[-1].x = x
        # vertical:
        else:
            # check for x offset
            if points_fromA[-1].x != points_fromB[-1].x:
                # find the y position
                if directionA == 270: 
                    y = min(points_fromA[-1].y, points_fromB[-1].y, 
                            points_fromA[-2].y-1*radius, points_fromB[-2].y-1*radius)
                else:
                    y = max(points_fromA[-1].y, points_fromB[-1].y, 
                            points_fromA[-2].y+1*radius, points_fromB[-2].y+1*radius)
                points_fromA[-1].y = y
                points_fromB[-1].y = y

    # check if the turtle directions are 90, 
    # and going away from facing each other, 
    #    -> can't handle this case
#    if (directionB - directionA + 90) % 360 in [0,180]:
#        print('Turtle directions: %s, %s' % (directionB, directionA))
#        print('Points A, B: %s, %s' % (pya.DPath(points_fromA,0).to_s(), pya.DPath(points_fromB,0).to_s()))
#        raise Exception("Turtles are moving away from each other; can't automatically route the path.")
         
    if verbose:
        print('Time elapsed, make path: %s' % (time() - t))    

    # merge the points from the two ends, A and B
    points = points_fromA + points_fromB[::-1]

    # generate the path
    path = pya.Path(points,width).to_dtype(dbu).remove_colinear_points()

    # Check if the path is Manhattan (it should be)
    if not path.is_manhattan():
        print('Turtle directions: %s, %s' % (directionB, directionA))
        print('Points A, B: %s, %s' % (pya.DPath(points_fromA,0).to_s(), pya.DPath(points_fromB,0).to_s()))
        cell.shapes(ly.layer(TECHNOLOGY['Errors'])).insert(path)
        print("Error. Generated Path is non-Manhattan. \nTurtles are moving away from each other; can't automatically route the path.")
#        raise Exception("Error. Generated Path is non-Manhattan. \nTurtles are moving away from each other; can't automatically route the path.")
        return False

    '''
    The Waveguide PCell's functions now return a Path on layer Error, which highlights the error
    if not path.radius_check(radius_um):
        print('Path: %s' % path)
        # draw a path for debugging purposes
        cell.shapes(ly.layer(TECHNOLOGY['Text'])).insert(path)
        if error_min_bend_radius:
            cell.shapes(ly.layer(TECHNOLOGY['Errors'])).insert(path)
            raise Exception("Error. Generated Path does not meet minimum bend radius requirements.")
        else:
            pass
    '''
    
    # generate the Waveguide PCell, and instantiate
    # Backwards compatible with Waveguides that don't use "waveguide_types"
    wg_pcell = ly.create_cell("Waveguide", technology_name, {"waveguide_type":waveguide_type,
                                                             "path": path,
                                                             "radius": radius_um,
                                                             "width": width_um,
                                                             "wg_width": width_um,
                                                             "adiab": waveguide['adiabatic'],
                                                             "bezier": waveguide['bezier'],
                                                             "layers": [wg['layer'] for wg in waveguide['component']],
                                                             "widths": [wg['width'] for wg in waveguide['component']],
                                                             "offsets": [wg['offset'] for wg in waveguide['component']],
                                                             "CML": waveguide['CML'],
                                                             "model": waveguide['model']})
    
    if wg_pcell==None:
        raise Exception("problem! cannot create Waveguide PCell from library: %s" % technology_name)
    inst = cell.insert(pya.CellInstArray(wg_pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))

    if verbose:
        print('Time elapsed, make waveguide: %s' % (time() - t))    

    if debug_path:
        cell.shapes(1).insert(path)
        
    return inst
    # end of def connect_pins_with_waveguide
    


def path_to_waveguide2(params=None, cell=None, snap=True, lv_commit=True, GUI=False, verbose=False, select_waveguides=False):
#    import time
#    time0 = time.perf_counter()
#    verbose=True

    from . import _globals
    from .utils import select_paths, get_layout_variables
    from .utils.crossings import insert_crossing
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
    if not cell:
        cell = top_cell

    if verbose:
        print("SiEPIC.scripts path_to_waveguide2(); start")
#        print("SiEPIC.scripts path_to_waveguide(); start; time = %s" % (time.perf_counter()-time0))

    if params is None:
        params = _globals.WG_GUI.get_parameters(GUI)
    if params is None:
        if verbose:
            print("SiEPIC.scripts path_to_waveguide2(): No parameters returned (user pressed Cancel); returning")
        return
    if verbose:
        print("SiEPIC.scripts path_to_waveguide2(): params = %s" % params)
    selected_paths = select_paths(TECHNOLOGY['Waveguide'], cell, verbose=verbose)
    if verbose:
        print("SiEPIC.scripts path_to_waveguide2(): selected_paths = %s" % (selected_paths))
#        print("SiEPIC.scripts path_to_waveguide(): selected_paths = %s; time = %s" % (selected_paths, time.perf_counter()-time0))
    selection = []

    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    if not selected_paths:
        # check if the Combine Mode in the tool bar is accidentally set to something that converts
        #  Paths to Polygons
        mw = pya.Application.instance().main_window()
        if mw.instance().get_config("combine-mode") != 'add':
            warning.setText(
                "Error: No 'Waveguide' Paths found - Cannot make waveguides.")
            warning.setInformativeText("Path objects on layer 'Waveguide' are required to create a waveguide pcell. \nAlternatively, object selection can be used to convert selected Paths from any layer.  If nothing is selected,  all 'Waveguide' Paths in the present cell will be converted to waveguides.\nFinally, the Toolbar Combine Mode is not set to Add, so possibly a Path was added and converted to a Polygon; do not use the modes Merge, Erase, etc, to create the Path for the waveguide.")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
        else:
            warning.setText(
                "Error: No 'Waveguide' Paths found - Cannot make waveguides.")
            warning.setInformativeText("Path objects on layer 'Waveguide' are required to create a waveguide pcell. \nAlternatively, object selection can be used to convert selected paths from any layer.  If nothing is selected,  all 'Waveguide' Paths in the present cell will be converted to waveguides.")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
            
    for obj in selected_paths:
        path = obj.shape.path
        path.unique_points()
        if not path.is_manhattan_endsegments():
            warning.setText(
                "Warning: Waveguide segments (first, last) are not Manhattan (vertical, horizontal).")
            warning.setInformativeText("Cannot Proceed")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
        if not path.is_manhattan():
            warning.setText(
                "Error: Waveguide segments are not Manhattan (vertical, horizontal). This is not supported in SiEPIC-Tools.")
            warning.setInformativeText("Cannot Proceed")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
        if not path.radius_check(params['radius'] / TECHNOLOGY['dbu']):
            warning.setText(
                "Warning: One of the waveguide segments has insufficient length to accommodate the desired bend radius.")
            warning.setInformativeText("Do you want to Proceed?")
            if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
                return

    if lv_commit:
        lv.transaction("Path to Waveguide")

    # Snapping first
    for i in range(len(selected_paths)):
        path = selected_paths[i].shape.path
        path.unique_points()

        # Snap waveguides to pins of nearby components
        if snap:
            p,_=cell.find_pins()            
            path.snap(p)

        selected_paths[i].shape.path = path

    # Insert crossings
    # this function unfortunately contains a GUI, so can mess up the undo transaction GUI.
    selected_paths = insert_crossing(selected_paths, params, verbose= False)

    # Make waveguides
    for obj in selected_paths:
        path = obj.shape.path
        path.unique_points()
          
        # get path info
        Dpath = path.to_dtype(TECHNOLOGY['dbu'])
        
        # Get user property #1: the waveguide type        
        prop1 = obj.shape.property(1)
        if prop1 and GUI==False:
            if verbose:
                print(' - user property: waveguide_type - %s' % (prop1) )
            waveguide_type = prop1
        else:
            waveguide_type = params['waveguide_type']
        '''
        instantiate Waveguide using "waveguide_type" approach
        '''
        pcell = 0
        try:
            pcell = ly.create_cell("Waveguide", TECHNOLOGY['technology_name'], {"path": Dpath,
                                                                                "waveguide_type": waveguide_type})
            if 'waveguide_type' not in pcell.pcell_parameters_by_name():
                pcell.delete()
                pcell=0
                print("SiEPIC.scripts.path_to_waveguide2(): legacy waveguide PCell does not have 'waveguide_type' parameter")
            else:
                if verbose:
                    print("SiEPIC.scripts.path_to_waveguide2(): Waveguide from %s, %s" %
                      (TECHNOLOGY['technology_name'], pcell))   
        except:
            pass
        if not pcell:
            try:
                for lib_name in TECHNOLOGY['libraries']:
                    pcell = ly.create_cell("Waveguide", lib_name,{"path": Dpath,
                                                                   "waveguide_type": waveguide_type})
                    if pcell:
                        if 'waveguide_type' not in pcell.pcell_parameters_by_name():
                            pcell.delete()
                            pcell=0
                            print("SiEPIC.scripts.path_to_waveguide2(): legacy waveguide PCell does not have 'waveguide_type' parameter")
                        else:
                            print("SiEPIC.scripts.path_to_waveguide2(): Waveguide from %s, %s" %
                              (lib_name, pcell))   
            except:
                pass

        '''
        instantiate Waveguide using "params" approach
        '''
        if not pcell:
            if ('DevRec' not in [wg['layer'] for wg in params['wgs']]):
                width_devrec = max([wg['width'] for wg in params['wgs']]) + _globals.WG_DEVREC_SPACE * 2
                params['wgs'].append({'width': width_devrec, 'layer': 'DevRec', 'offset': 0.0})
            
            try:
                pcell = ly.create_cell("Waveguide", TECHNOLOGY['technology_name'], {"path": Dpath,
                                                                                "radius": params['radius'],
                                                                                "width": params['width'],
                                                                                "adiab": params['adiabatic'],
                                                                                "bezier": params['bezier'],
                                                                                "layers": [wg['layer'] for wg in params['wgs']],
                                                                                "widths": [wg['width'] for wg in params['wgs']],
                                                                                "offsets": [wg['offset'] for wg in params['wgs']],
                                                                                "CML": params['CML'],
                                                                                "model": params['model']})
                print("SiEPIC.scripts.path_to_waveguide2(): Waveguide from %s, %s" % (TECHNOLOGY['technology_name'], pcell))   
#            print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s; time = %s" %
#                  (TECHNOLOGY['technology_name'], pcell, time.perf_counter()-time0))   
            except:
                pass
        if not pcell:
            try:
                for lib_name in TECHNOLOGY['libraries']:
                    pcell = ly.create_cell("Waveguide", lib_name, { "path": Dpath,
                                                                    "radius": params['radius'],
                                                                    "width": params['width'],
                                                                    "adiab": params['adiabatic'],
                                                                    "bezier": params['bezier'],
                                                                    "layers": [wg['layer'] for wg in params['wgs']],
                                                                    "widths": [wg['width'] for wg in params['wgs']],
                                                                    "offsets": [wg['offset'] for wg in params['wgs']],
                                                                    "CML": params['CML'],
                                                                    "model": params['model']})
                    print("SiEPIC.scripts.path_to_waveguide2(): Waveguide from %s, %s" % (lib_name, pcell))   
                    if pcell:
                        break
#                print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s; time = %s" %
#                  (TECHNOLOGY['technology_name'], pcell, time.perf_counter()-time0))   
            except:
                pass
        if not pcell:
            try:
                pcell = ly.create_cell("Waveguide", "SiEPIC General", {"path": Dpath,
                                                                       "radius": params['radius'],
                                                                       "width": params['width'],
                                                                       "adiab": params['adiabatic'],
                                                                       "bezier": params['bezier'],
                                                                       "layers": [wg['layer'] for wg in params['wgs']],
                                                                       "widths": [wg['width'] for wg in params['wgs']],
                                                                       "offsets": [wg['offset'] for wg in params['wgs']]})
                print("SiEPIC.scripts.path_to_waveguide2(): Waveguide from SiEPIC General, %s" % pcell)
            except:
                pass
        if not pcell:
            # Record a transaction, to enable "undo"
            if lv_commit:
                lv.commit()
            raise Exception(
                "'Waveguide' is not available. Check that the library was loaded successfully.")
        selection.append(pya.ObjectInstPath())
        selection[-1].top = obj.top
        selection[-1].append_path(pya.InstElement.new(cell.insert(
            pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, 0, 0)))))

        obj.shape.delete()

    lv.clear_object_selection()
    if select_waveguides:
        lv.object_selection = selection
    pya.Application.instance().main_window().redraw()  
    ly.cleanup([])  

    # Record a transaction, to enable "undo"
    if lv_commit:
        lv.commit()
    
    if verbose:
        print("SiEPIC.scripts path_to_waveguide2(); done" )


def path_to_waveguide(params=None, cell=None, snap=True, lv_commit=True, GUI=False, verbose=False, select_waveguides=False):
#    import time
#    time0 = time.perf_counter()
#    verbose=True

    from . import _globals
    from .utils import select_paths, get_layout_variables
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
    if not cell:
        cell = top_cell

    if verbose:
        print("SiEPIC.scripts path_to_waveguide(); start")
#        print("SiEPIC.scripts path_to_waveguide(); start; time = %s" % (time.perf_counter()-time0))

    if lv_commit:
        lv.transaction("Path to Waveguide")

    if params is None:
        params = _globals.WG_GUI.get_parameters(GUI)
    if params is None:
        if verbose:
            print("SiEPIC.scripts path_to_waveguide(): No parameters returned (user pressed Cancel); returning")
#        raise Exception("SiEPIC.scripts path_to_waveguide(): no params; returning")
        return
    if verbose:
        print("SiEPIC.scripts path_to_waveguide(): params = %s" % params)
    selected_paths = select_paths(TECHNOLOGY['Waveguide'], cell, verbose=verbose)
    if verbose:
        print("SiEPIC.scripts path_to_waveguide(): selected_paths = %s" % (selected_paths))
#        print("SiEPIC.scripts path_to_waveguide(): selected_paths = %s; time = %s" % (selected_paths, time.perf_counter()-time0))
    selection = []

    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    warning.setDefaultButton(pya.QMessageBox.Yes)
    if not selected_paths:
        # check if the Combine Mode in the tool bar is accidentally set to something that converts
        #  Paths to Polygons
        mw = pya.Application.instance().main_window()
        if mw.instance().get_config("combine-mode") != 'add':
            warning.setText(
                "Error: No 'Waveguide' Paths found - Cannot make waveguides.")
            warning.setInformativeText("Path objects on layer 'Waveguide' are required to create a waveguide pcell. \nAlternatively, object selection can be used to convert selected Paths from any layer.  If nothing is selected,  all 'Waveguide' Paths in the present cell will be converted to waveguides.\nFinally, the Toolbar Combine Mode is not set to Add, so possibly a Path was added and converted to a Polygon; do not use the modes Merge, Erase, etc, to create the Path for the waveguide.")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
        else:
            warning.setText(
                "Error: No 'Waveguide' Paths found - Cannot make waveguides.")
            warning.setInformativeText("Path objects on layer 'Waveguide' are required to create a waveguide pcell. \nAlternatively, object selection can be used to convert selected paths from any layer.  If nothing is selected,  all 'Waveguide' Paths in the present cell will be converted to waveguides.")
            pya.QMessageBox_StandardButton(warning.exec_())
            return
            
    # can this be done once instead of each time?  Moved here, by Lukas C, 2020/05/04
    if snap:
        p,_=cell.find_pins()            

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

        # a slow function - needs fixing:
#        p,_=cell.find_pins()
#        if verbose:
#          print("SiEPIC.scripts path_to_waveguide(); cell.find_pins(); time = %s" % (time.perf_counter()-time0))

        if snap:
            path.snap(p)
#        if verbose:
#          print("SiEPIC.scripts path_to_waveguide(); path.snap(...); time = %s" % (time.perf_counter()-time0))

        Dpath = path.to_dtype(TECHNOLOGY['dbu'])
        if ('DevRec' not in [wg['layer'] for wg in params['wgs']]):
            width_devrec = max([wg['width'] for wg in params['wgs']]) + _globals.WG_DEVREC_SPACE * 2
            params['wgs'].append({'width': width_devrec, 'layer': 'DevRec', 'offset': 0.0})
        
        # added 2 new parameters: CML and model to support multiple WG models
        pcell = 0
        if 'CML' in params.keys() and 'model' in params.keys():
          try:
            pcell = ly.create_cell("Waveguide", TECHNOLOGY['technology_name'], {"path": Dpath,
                                                                                "radius": params['radius'],
                                                                                "width": params['width'],
                                                                                "adiab": params['adiabatic'],
                                                                                "bezier": params['bezier'],
                                                                                "layers": [wg['layer'] for wg in params['wgs']],
                                                                                "widths": [wg['width'] for wg in params['wgs']],
                                                                                "offsets": [wg['offset'] for wg in params['wgs']],
                                                                                "CML": params['CML'],
                                                                                "model": params['model']})
            print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s" %
                  (TECHNOLOGY['technology_name'], pcell))   
#            print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s; time = %s" %
#                  (TECHNOLOGY['technology_name'], pcell, time.perf_counter()-time0))   
          except:
              pass
        if not pcell:
            try:
                pcell = ly.create_cell("Waveguide", TECHNOLOGY['technology_name'], {"path": Dpath,
                                                                                    "radius": params['radius'],
                                                                                    "width": params['width'],
                                                                                    "adiab": params['adiabatic'],
                                                                                    "bezier": params['bezier'],
                                                                                    "layers": [wg['layer'] for wg in params['wgs']],
                                                                                    "widths": [wg['width'] for wg in params['wgs']],
                                                                                    "offsets": [wg['offset'] for wg in params['wgs']]})
                print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s" %
                  (TECHNOLOGY['technology_name'], pcell))   
#                print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s; time = %s" %
#                  (TECHNOLOGY['technology_name'], pcell, time.perf_counter()-time0))   
            except:
                pass
        if not pcell:
            try:
                for lib_name in TECHNOLOGY['libraries']:
                    pcell = ly.create_cell("Waveguide", lib_name, { "path": Dpath,
                                                                    "radius": params['radius'],
                                                                    "width": params['width'],
                                                                    "adiab": params['adiabatic'],
                                                                    "bezier": params['bezier'],
                                                                    "layers": [wg['layer'] for wg in params['wgs']],
                                                                    "widths": [wg['width'] for wg in params['wgs']],
                                                                    "offsets": [wg['offset'] for wg in params['wgs']]})
                    print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s" %
                      (lib_name, pcell))   
                    if pcell:
                        break
#                print("SiEPIC.scripts.path_to_waveguide(): Waveguide from %s, %s; time = %s" %
#                  (TECHNOLOGY['technology_name'], pcell, time.perf_counter()-time0))   
            except:
                pass
        if not pcell:
            try:
                pcell = ly.create_cell("Waveguide", "SiEPIC General", {"path": Dpath,
                                                                       "radius": params['radius'],
                                                                       "width": params['width'],
                                                                       "adiab": params['adiabatic'],
                                                                       "bezier": params['bezier'],
                                                                       "layers": [wg['layer'] for wg in params['wgs']],
                                                                       "widths": [wg['width'] for wg in params['wgs']],
                                                                       "offsets": [wg['offset'] for wg in params['wgs']]})
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
    if select_waveguides:
        lv.object_selection = selection
    if lv_commit:
        lv.commit()
    pya.Application.instance().main_window().redraw()    
    
    if verbose:
        print("SiEPIC.scripts path_to_waveguide(); done" )
#        print("SiEPIC.scripts path_to_waveguide(); done; time = %s" % (time.perf_counter()-time0))

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

'''
Find all Waveguide cells, extract the path and add it to the layout, delete the Waveguide
'''
def waveguide_to_path(cell=None, save_waveguide_type=True):
    from . import _globals
    from .utils import select_waveguides, get_layout_variables

    if cell is None:
        TECHNOLOGY, lv, ly, cell = get_layout_variables()
    else:
        TECHNOLOGY, lv, _, _ = get_layout_variables()
        ly = cell.layout()
    dbu = TECHNOLOGY['dbu']
    
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
            path.width = waveguide.cell.pcell_parameters_by_name()['width'] / dbu
            pts = []
            for pt in [pt1 for pt1 in (path1).each_point()]:
                if type(pt) == pya.Point:
                    # for instantiated PCell
                    pts.append(pya.Point())
                else:
                    # for waveguide from path
                    pts.append(pya.Point().from_dpoint(pt * (1 / dbu )))
            path.points = pts

        selection.append(pya.ObjectInstPath())
        selection[-1].layer = ly.layer(TECHNOLOGY['Waveguide'])
        # DPath.transformed requires DTrans. waveguide.trans is a Trans object
        if KLAYOUT_VERSION > 24:
            selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(
                path.transformed(waveguide.trans.to_dtype(dbu)))
        else:
            selection[-1].shape = cell.shapes(ly.layer(TECHNOLOGY['Waveguide'])).insert(
                path.transformed(pya.Trans(waveguide.trans.disp.x, waveguide.trans.disp.y)))

        # Store the waveguide_type information in the path's user property
        # this method invalidates the shape, https://www.klayout.de/doc/code/class_Shape.html#method152
        # so you can't use the object selection
        if save_waveguide_type:
            if 'waveguide_type' in waveguide.cell.pcell_parameters_by_name():
                selection[-1].shape.set_property(1, waveguide.cell.pcell_parameters_by_name()['waveguide_type'])
        
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
    if not save_waveguide_type:
        lv.object_selection = selection

    pya.Application.instance().main_window().redraw()
    ly.cleanup([])  

    # Record a transaction, to enable "undo"
    lv.commit()

def waveguide_length():

    from .utils import get_layout_variables, select_waveguides
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    import SiEPIC.utils

    selection = select_waveguides(cell)
    if len(selection) == 1:
        cell = selection[0].inst().cell
        length = float(cell.find_components()[0].params.split(' ')[0].split('=')[1])*1e6
        pya.MessageBox.warning("Waveguide Length", "Waveguide length (um): %s" %
                               length, pya.MessageBox.Ok)
    else:
        pya.MessageBox.warning("Selection is not a waveguide",
                               "Select one waveguide you wish to measure.", pya.MessageBox.Ok)

    pya.Application.instance().main_window().redraw()    

def waveguide_length_diff():
    import SiEPIC
    TECHNOLOGY, lv, ly, cell = SiEPIC.utils.get_layout_variables()
    from math import exp, sqrt, pi
    from copy import deepcopy
    import numpy as np
    import fnmatch
    import pathlib
    import pya
    import os

    # calculate length difference
    selection = SiEPIC.utils.select_waveguides(cell)

    if len(selection) == 2:
        cell1 = selection[0].inst().cell
        cell2 = selection[1].inst().cell

        length1 = float(cell1.find_components()[0].params.split(' ')[0].split('=')[1])*1e6
        length2 = float(cell2.find_components()[0].params.split(' ')[0].split('=')[1])*1e6

        # function to find the nearest value in aa 2d array
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        def distance(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def is_between(a, c, b):
            return distance(a, c) + distance(c, b) == distance(a, b)

        # location function to get correlation values
        def get_local_correlation_matrix(corr_length, resolution, temp_comp):

            # correlation function
            def get_corr(x1, y1, x2, y2, l):
                sigma = l / 4
                correlation = exp(-(((x1 - x2)**2 + (y1 - y2)**2) / (l**2 / 2)))
                # correlation = exp((-(x1-x2)**2 - (y1-y2)**2)/((l**2)/2))/(sqrt(pi)*(l/2)) # gaussian function in 2d space
                return correlation

            # function to calculate distance between two points
            def distance(x1, y1, x2, y2):
                return sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # split a line into a number of chunks based on length
            def split_line(line_pts, length):
                # dice waveguide into segments of "length"
                length = abs(length)
                temp_pts = []
                for i in range(len(line_pts)):
                    if(i > 0):
                        pt_a = line_pts[i - 1]
                        pt_b = line_pts[i]
                        diff_x = sqrt((pt_a[0] - pt_b[0])**2)
                        diff_y = sqrt((pt_a[1] - pt_b[1])**2)
                        curr_pt = deepcopy(pt_a)
                        temp_pts.append(pt_a)
                        while distance(curr_pt[0], curr_pt[1], pt_b[0], pt_b[1]) > length:

                            # if diff is along x axis
                            if(diff_x > 0):
                                curr_pt[0] = [curr_pt[0] - length if(distance(curr_pt[0] - length, curr_pt[1], pt_b[0], pt_b[1]) < distance(
                                    temp_pts[-1][0], temp_pts[-1][1], pt_b[0], pt_b[1])) else curr_pt[0] + length][0]
                            # else diff is along y axis
                            else:
                                curr_pt[1] = [curr_pt[1] - length if(distance(curr_pt[0], curr_pt[1] - length, pt_b[0], pt_b[1]) < distance(
                                    temp_pts[-1][0], temp_pts[-1][1], pt_b[0], pt_b[1])) else curr_pt[1] + length][0]

                            temp_pts.append(deepcopy(curr_pt))
                        temp_pts.append(pt_b)
                return temp_pts

            # extract layout information
            TECH, lv, ly, cell = SiEPIC.utils.get_layout_variables()
            net, components_list = cell.identify_nets()
            dbu = ly.dbu

            # comp has all the components
            comp = []
            for each_temp_comp in temp_comp:
                for check_comp in components_list:
                    if(each_temp_comp.cell.cell_index() == check_comp.cell.cell_index()):
                        comp.append(check_comp)

            # initialise correlation matrix
            corr_matrix = np.zeros(shape=(len(comp), len(comp)))
            full_matrix_data = []
            # copy of correlation matrix that can be modified independently and exported
            exported_matrix = np.zeros(shape=(len(comp), len(comp)))

            corr_matrix_name = []
            for i in comp:
                corr_matrix_name.append([])

            for idx1 in range(len(comp)):
                for idx2 in range(len(comp)):
                    # optimisation to remove redundant iterations
                    if(corr_matrix[idx1, idx2] == 0 and corr_matrix[idx2, idx1] == 0):
                        # if you are not comparing the same element with itself
                        # get name of either component for same element comparison(i.e. next condition)
                        if(idx1 == idx2):  # if it is the same element
                            corr_value = 1

                        else:
                            first = comp[idx1]
                            second = comp[idx2]
                            corr_value = 0

                            # if any of the components are waveguides
                            if(first.basic_name == 'Waveguide' or second.basic_name == 'Waveguide'):
                                wgs = [first, second]
                                # wgs = [each.cell.pcell_parameters_by_name()['path'] for each in wgs if each.basic_name == 'Waveguide' else each] #get path obj for each waveguide
                                wg_pts = []
                                for each in wgs:
                                    if(each.basic_name == 'Waveguide'):
                                        each_path = each.cell.pcell_parameters_by_name()[
                                            'path']
                                        each_path = each.trans.to_vtrans(
                                            TECH['dbu']).trans(each_path)
                                        pts_itr = each_path.each_point()
                                        #print([each for each in pts_itr])
                                        wg_pts.append([[each.x, each.y]
                                                       for each in pts_itr])

                                    else:
                                        wg_pts.append(
                                            [[each.center.x * dbu, each.center.y * dbu]])

                                # check if coincidental
                                coincidental = []

                                if(second.basic_name == 'Waveguide'):
                                    for pt_idx1 in range(len(wg_pts[0])):
                                        for pt_idx2 in range(1, len(wg_pts[1])):
                                            if(is_between(wg_pts[1][pt_idx2 - 1], wg_pts[0][pt_idx1], wg_pts[1][pt_idx2])):
                                                coincidental.append(1)
                                            else:
                                                coincidental.append(0)
                                else:
                                    for pt_idx2 in range(len(wg_pts[1])):
                                        for pt_idx1 in range(1, len(wg_pts[0])):
                                            if(is_between(wg_pts[0][pt_idx1 - 1], wg_pts[1][pt_idx2], wg_pts[0][pt_idx1])):
                                                coincidental.append(1)
                                            else:
                                                coincidental.append(0)
                                if(min(coincidental) == 1):
                                    corr_value = (1 - 1e-15)

                                else:
                                    new_wg_pts = []
                                    for each in wg_pts:
                                        if(len(each) > 1):
                                            new_wg_pts.append(
                                                split_line(each, resolution))
                                        else:
                                            new_wg_pts.append(each)
                                    #print('new wg points', len(new_wg_pts))
                                    inner_corr_matrix = np.zeros(
                                        shape=(len(new_wg_pts[0]), len(new_wg_pts[1])))
                                    #print('inner corr matrix', inner_corr_matrix.shape)
                                    for i in range(len(new_wg_pts[0])):
                                        for j in range(len(new_wg_pts[1])):
                                            inner_corr_value = get_corr(
                                                new_wg_pts[0][i][0], new_wg_pts[0][i][1], new_wg_pts[1][j][0], new_wg_pts[1][j][1], corr_length)

                                            inner_corr_matrix[i, j] = inner_corr_value
                                    #np.savetxt('test.txt', inner_corr_matrix)
                                    corr_value = inner_corr_matrix.mean()
                                    full_matrix_data.append(
                                        [idx1, idx2, inner_corr_matrix])

                            else:
                                corr_value = get_corr(
                                    first.center.x * dbu, first.center.y * dbu, second.center.x * dbu, second.center.y * dbu, corr_length)
                                full_matrix_data.append([idx1, idx2, corr_value])

                        corr_matrix[idx1, idx2] = corr_matrix[idx2, idx1] = corr_value
                        s2i = comp[idx1].basic_name + "_" + \
                            str(idx1) + " & " + comp[idx2].basic_name + "_" + str(idx2)

                        corr_matrix_name[idx1].insert(idx2, s2i)
                        corr_matrix_name[idx2].insert(idx1, s2i)

            def find_val(val):
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if(corr_matrix[i][j] == val):
                            print(corr_matrix_name[i][j])

            return(corr_matrix)

        # get MC parameters from the MONTECARLO.xml file
        mc_exists = False

        TECHNOLOGY = SiEPIC.utils.get_technology()
        tech_name = TECHNOLOGY['technology_name']
        paths = []
        
        for root, dirnames, filenames in os.walk(pya.Application.instance().application_data_path(), followlinks=True):
            [paths.append(os.path.join(root, filename))
                for filename in fnmatch.filter(filenames, 'MONTECARLO.xml') if tech_name in root]
        if paths:
            print(paths[0])
            with open(paths[0], 'r') as file:
                montecarlo = SiEPIC.utils.xml_to_dict(file.read())
                montecarlo = montecarlo['technologies']['technology']
                if len(montecarlo) > 1:
                    montecarlo = montecarlo[0]

                mc_exists = True

        nsamples = 10000
        lambda_not = 1.55
        phase_arr = np.ones((nsamples)) * np.nan

        if mc_exists:

            wcl = float(montecarlo['wafer']['width']['corr_length']) * 1e6
            tcl = float(montecarlo['wafer']['height']['corr_length']) * 1e6
            wsigma = float(montecarlo['wafer']['width']['std_dev'])
            tsigma = float(montecarlo['wafer']['height']['std_dev'])

            # generate correlation matrix
            correlation_matrix_w = get_local_correlation_matrix(
                wcl, 5, [selection[0].inst(), selection[1].inst()])
            correlation_matrix_h = get_local_correlation_matrix(
                tcl, 5, [selection[0].inst(), selection[1].inst()])

            # generate width covariance matrix (sigma1 * correlation * sigma2)
            wcovariance_matrix = np.zeros((2, 2))
            tcovariance_matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    wcovariance_matrix[i][j] = wsigma * \
                        correlation_matrix_w[i][j] * wsigma
                    tcovariance_matrix[i][j] = tsigma * \
                        correlation_matrix_h[i][j] * tsigma

            # process cholesky decompositions
            wchol = np.linalg.cholesky(wcovariance_matrix)
            tchol = np.linalg.cholesky(tcovariance_matrix)

            # generate random uncorrelated unit distributions

            dwidth_uncorr = np.random.multivariate_normal(
                [0, 0], [[1, 0], [0, 1]], nsamples).T
            twidth_uncorr = np.random.multivariate_normal(
                [0, 0], [[1, 0], [0, 1]], nsamples).T

            # correlate distributions
            dwidth_corr = np.dot(wchol, dwidth_uncorr)
            dthick_corr = np.dot(tchol, twidth_uncorr)

            # load neff data
            filename = 'geo_vs_neff.npy'
            print(pya.Application.instance().application_data_path())
            print(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))), "Simulation")))
            pathsdata = [each for each in os.walk(
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))), "Simulation")), followlinks=True)]
            match = [each for each in pathsdata if (
                len(each) == 3) and (filename in each[-1])]

            if(len(match)>0):
                filedir = match[0][0]
                import os
                geovsneff_data = np.load(os.path.join(
                    filedir, filename), allow_pickle=True).flatten()[0]
                neff_data = geovsneff_data['neff']
                width_data = geovsneff_data['width']
                thickness_data = geovsneff_data['thickness']

                # create arrays for delta propagation constants
                delta_beta = np.zeros((nsamples))
                phase_arr = np.zeros((nsamples))

                if("nom_width" in geovsneff_data):
                    nom_width = geovsneff_data['nom_width']
                else:
                    nom_width = 500

                if("nom_thickness" in geovsneff_data):
                    nom_thick = geovsneff_data['nom_thickness']
                else:
                    nom_thick = 220

                for each_sample in range(nsamples):
                    # wg1
                    temp_thick1 = nom_thick + dthick_corr[0][each_sample]
                    temp_width1 = nom_width + dwidth_corr[0][each_sample]
                    idxx, idxy = np.where(thickness_data == find_nearest(thickness_data, temp_thick1)), np.where(
                        width_data == find_nearest(width_data, temp_width1))
                    neff1 = neff_data[idxy, idxx].flatten()[0]
                    beta1 = (2 * np.pi * neff1) / lambda_not

                    # wg2
                    temp_thick = nom_thick + dthick_corr[1][each_sample]
                    temp_width = nom_width + dwidth_corr[1][each_sample]
                    idxx, idxy = np.where(thickness_data == find_nearest(thickness_data, temp_thick)), np.where(
                        width_data == find_nearest(width_data, temp_width))
                    neff2 = neff_data[idxy, idxx].flatten()[0]
                    beta2 = (2 * np.pi * neff2) / lambda_not

                    delta_beta[each_sample] = np.abs(beta1 - beta2)
                    phase_arr[each_sample] = ((beta1 * length1) - (beta2 * length2)) / np.pi

        pya.MessageBox.warning("Waveguide Length Difference", "Difference in waveguide lengths (um): %s" % str(
            abs(length1 - length2)) + '\r\n RMS phase error: ' + str(round(np.std(phase_arr), 3)) + ' pi radians', pya.MessageBox.Ok)

    else:
        pya.MessageBox.warning("Selection are not a waveguides",
                               "Select two waveguides you wish to measure.", pya.MessageBox.Ok)

    pya.Application.instance().main_window().redraw()

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
                pins_transient = o_transient.inst().find_pins(verbose=True)[0]
                pins_selection = o_selection.inst().find_pins(verbose=True)[0]
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
                                    if (abs(pin_t.rotation - pin_s.rotation) % 360 == 180) and pin_t.type == _globals.PIN_TYPES.OPTICAL and pin_s.type == _globals.PIN_TYPES.OPTICAL],
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

                pya.Application.instance().main_window().redraw()    
                return
# end def snap_component()
    pya.Application.instance().main_window().redraw()    


def add_and_connect_cell(instanceA, pinA, cellB, pinB, verbose=False):
    return connect_cell(instanceA, pinA, cellB, pinB, verbose)
    
def connect_cell(instanceA, pinA, cellB, pinB, mirror = False, verbose=False, translation=pya.Trans.R0, relaxed_pinnames=False):
  '''
  Instantiate, Move & rotate cellB to connect to instanceA, 
   such that their pins (pinB, pinA) match
   
  Input:
  - instanceA, pinA: the fixed component
  - cellB, pinB: the cell to instantiate and place
  - mirror: optionally mirror cellB
  - translation: translate cellB, e.g., Trans(Trans.R90, 5000, 5000)
     useful for subsequent waveguide routing

  Use SiEPIC-Tools find_components to determine the component size, pins, etc.
  
  
  Example code:

  from SiEPIC import scripts  
  from SiEPIC.utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()

  # clean all cells within the present cell
  top_cell = ly.top_cells()[0]
  ly.prune_subcells(top_cell.cell_index(), 10)

  cell_crossing = ly.create_cell('ebeam_crossing4', 'EBeam')
  t = pya.Trans.from_s('r270 230175,190500')
  inst_crossing = cell.insert(pya.CellInstArray(cell_crossing.cell_index(), t))

  cell_bragg = ly.create_cell('ebeam_bragg_te1550', 'EBeam',
       {'r': 10.0, 'w': 0.35, 'g': 0.12, 'gmon': 0.5})

  from SiEPIC.scripts import connect_cell
  
  instanceB = connect_cell(inst_crossing, 'opt2', cell_bragg, 'pin1')

  
  '''
  # print('SiEPIC-Tools, connect_cell: verbose = %s' %verbose)
  
  from . import _globals

  if not(instanceA):
    raise Exception("instanceA not found")
  if not(cellB):
    raise Exception("cellB not found")

  # check cells
  if type(cellB) != pya.Cell:
      raise Exception("cellB needs to be a cell, not a cell index")
  if type(instanceA) != pya.Instance:
      raise Exception("instanceA needs to be an Instance, not an index")

  # Find the two components:
  componentA = instanceA.parent_cell.find_components(cell_selected=instanceA.cell, inst=instanceA)
  componentB = cellB.find_components()
  if componentA==[]:
    componentA = instanceA.parent_cell.find_components(inst=instanceA)
    if componentA==[]:
      if _globals.Python_Env == "KLayout_GUI":
        question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
        question.setText("SiEPIC-Tools scripted layout, requested component not found")
        question.setInformativeText("Component instanceA not found: %s, %s" % (instanceA, instanceA.cell.name))
        pya.QMessageBox_StandardButton(question.exec_())
        return
      else:          
        raise Exception("Component instanceA not found")
  if componentB==[]:
      if _globals.Python_Env == "KLayout_GUI":
        question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
        question.setText("SiEPIC-Tools scripted layout, requested component not found")
        question.setInformativeText("Component cellB not found: %s, %s" % (cellB, cellB.name))
        pya.QMessageBox_StandardButton(question.exec_())
        return
      else:          
        raise Exception("Component cellB not found")

#  for c in componentA:
#    if c.trans.s_trans() == instanceA.trans:
#      componentA = c

  if type(componentA) == type([]):
    componentA = componentA[0]
  componentB = componentB[0]
  if verbose:
    componentA.display()
    componentB.display()
    
  # Find pinA and pinB
  cpinA = [p for p in componentA.pins if p.pin_name == pinA]
  cpinB = [p for p in componentB.pins if p.pin_name == pinB]    

  # relaxed_pinnames:  scan for only the number
  if relaxed_pinnames==True:
      import re
      try:
          if cpinA==[]:
              cpinA = [p for p in componentA.pins if re.findall(r'\d+', pinA)[0] in p.pin_name]
          if cpinB==[]:
              cpinB = [p for p in componentB.pins if re.findall(r'\d+', pinB)[0] in p.pin_name]
      except:
          print('error in siepic.scripts.connect_cell')      

  def error_pinA(pinA,componentA):
    from inspect import getframeinfo, stack
    error_message = "SiEPIC-Tools, in function connect_cell: PinA (%s) not found in componentA (%s). Available pins: %s.\n%s" % (pinA,componentA.component, [p.pin_name for p in componentA.pins], getframeinfo(stack()[2][0]))
    if _globals.Python_Env == "KLayout_GUI":
        question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
        question.setText("SiEPIC-Tools scripted layout, requested pin not found")
        question.setInformativeText(error_message)
        pya.QMessageBox_StandardButton(question.exec_())
        return
    else:          
        raise Exception(error_message)
  def error_pinB(pinB,componentB):
    from inspect import getframeinfo, stack
    error_message = "SiEPIC-Tools, in function connect_cell: PinB (%s) not found in componentB (%s). Available pins: %s.\n%s" % (pinB,componentB.component, [p.pin_name for p in componentB.pins], getframeinfo(stack()[2][0]))
    if _globals.Python_Env == "KLayout_GUI":
        question = pya.QMessageBox().setStandardButtons(pya.QMessageBox.Ok)
        question.setText("SiEPIC-Tools scripted layout, requested pin not found")
        question.setInformativeText(error_message)
        pya.QMessageBox_StandardButton(question.exec_())
        return
    else:          
        raise Exception(error_message)


  # for cells with hierarchy
  if cpinA==[]:
    try:  
        # this checks if the cell (which could contain multiple components) 
        # contains only one pin matching the name, e.g. unique opt_input in a sub-circuit
        cpinA = [instanceA.find_pin(pinA)]
    except:
        error_pinA(pinA,componentA)
        return
  if cpinB==[]:
    try:  
        # this checks if the cell (which could contain multiple components) 
        # contains only one pin matching the name, e.g. unique opt_input in a sub-circuit
        cpinB = [cellB.find_pin(pinB)]
    except:
        error_pinB(pinB,componentB)
        return


  if cpinA==[]:
    error_pinA(pinA,componentA)
    return
  if cpinB==[]:
    error_pinB(pinB,componentB)
    return
    
  cpinA=cpinA[0]
  cpinB=cpinB[0]
  
  if cpinA == None:
    error_pinA(pinA,componentA)
    return
  if cpinB == None:
    error_pinB(pinB,componentB)
    return

  if verbose:
    cpinA.display()
    cpinB.display()
    
  # Find pin angles, and necessary rotation for componentB
  rotation = (cpinA.rotation - cpinB.rotation - 180) % 360
  if verbose:
    print (' cellB required rotation to connect: %s' % (rotation) )
  
  # Calculate vector to move componentB
  if mirror:
    trans = pya.Trans(pya.Trans.R0.M45, cpinA.center - cpinB.center * pya.Trans(rotation/90,False,0,0)) \
        * pya.Trans(rotation/90,False,0,0)
  else:
    trans = pya.Trans(pya.Trans.R0, cpinA.center - cpinB.center * pya.Trans(rotation/90,False,0,0)) \
        * pya.Trans(rotation/90,False,0,0)

  '''
  # adjust the desired translation so that it is relative to pinB
  # this needs more work as the behaviour is not consistent for different cell orientations
  if translation.rot == 1:  # R90
    translation = translation * pya.Trans(0,False, -cpinB.center.x+cpinB.center.y, -cpinB.center.x-cpinB.center.y) 
  if translation.rot == 2:  # R180
    translation = pya.Trans(translation.rot, False, translation.disp.x, -translation.disp.y)
    translation *= pya.Trans(0,False, -2*cpinB.center.x, -2*cpinB.center.y) 
  if translation.rot == 3:  # R270
    translation = pya.Trans(translation.rot, False, translation.disp.x, -translation.disp.y) * pya.Trans(
        0,False, -cpinB.center.x-cpinB.center.y, cpinB.center.x-cpinB.center.y) 
  if translation.rot == 0:  # R0
    pass
  '''
  
#  vector = cpinA.center - componentA.trans.disp - componentB.trans.disp
  if verbose:
    print (' cellB with translation: %s' % (trans*translation) )

  # Instantiate cellB
  if verbose:
    print(instanceA.parent_cell)
  instanceB = instanceA.parent_cell.insert(pya.CellInstArray(cellB.cell_index(), trans*translation))

  
  return instanceB
  # end of def connect_cell


def delete_extra_topcells(ly, keep_topcell):
    '''
    Delete extra top cells
    Input: 
    ly: pya.Layout
    keep_topcell: a top cell that you want to keep
        can be either a pya.Cell or a string cell name
    '''
    if type(keep_topcell) == str:
        keep_topcell = ly.cell(keep_topcell)    
    if type(keep_topcell) != pya.Cell:
        raise Exception("SiEPIC.scripts.delete_extra_topcells: keep_topcell argument was not a cell")
    if keep_topcell in ly.top_cells():
        ly.delete_cells([tcell for tcell in ly.each_top_cell() if tcell != keep_topcell.cell_index()])
        if len(ly.top_cells()) > 1:
            #print(ly.top_cells())
            delete_extra_topcells(ly, keep_topcell)
        

def delete_top_cells():
    '''
    Delete extra top cells
    Input: 
      a layout open in the GUI with an actively selected top cells
    '''

    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    if cell in ly.top_cells():
        lv.transaction("Delete extra top cells")
        delete_extra_topcells(ly, cell)
        lv.commit()
    else:
        v = pya.MessageBox.warning(
            "No top cell selected", "No top cell selected.\nPlease select a top cell to keep\n(not a sub-cell).", pya.MessageBox.Ok)

    pya.Application.instance().main_window().redraw()    

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
    print(version)
    print(platform.system())
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

    elif (version.find("3.")>-1) & (('Darwin' in platform.system()) | ('Linux' in platform.system())):
        import subprocess
        cmd = subprocess.check_output

        progress.format = "Uploading Layout and Scripts"
        progress.set(2, True)
        pya.Application.instance().main_window().repaint()

        try:
            c = ['ssh', server, 'mkdir', '-p', remote_path]
            print(c)
            out += cmd(c).decode('utf-8')
            c = ['scp', os.path.join(local_path,local_file), '%s:%s' %(server, remote_path)]
            print(c)
            out += cmd(c).decode('utf-8')
            c = ['scp',os.path.join(local_path,'run_calibre'),'%s:%s'%(server, remote_path)]
            print(c)
            out += cmd(c).decode('utf-8')
            c = ['scp',os.path.join(local_path,'drc.cal'),'%s:%s'%(server, remote_path)]
            print(c)
            out += cmd(c).decode('utf-8')

            progress.format = "Checking Layout for Errors"
            progress.set(3, True)
            pya.Application.instance().main_window().repaint()

            c = ['ssh', server, 'cd',remote_path,';source','run_calibre']
            print(c)
            out += cmd(c).decode('utf-8')

            progress.format = "Downloading Results"
            progress.set(4, True)
            pya.Application.instance().main_window().repaint()

            c = ['scp','%s:%s/drc.rve'%(server, remote_path), os.path.join(local_path,results_file)]
            print(c)
            out += cmd(c).decode('utf-8')
        except subprocess.CalledProcessError as e:
            out += '\nError running ssh or scp commands. Please check that these programs are available.\n'
            out += str(e.output)

    elif (version.find("3.")>-1) & ('Win' in platform.system()):
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
        pya.MessageBox.warning(
            "Success", "Calibre DRC run complete. Results downloaded and available in the Marker Browser window.",  pya.MessageBox.Ok)

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
    import os
    import time
    from .utils import get_technology
    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, topcell = get_layout_variables()

    TECHNOLOGY = get_technology()

    def gen_ui():
        global wdg
#        if 'wdg' in globals():
#            if wdg is not None and not wdg.destroyed:
#                wdg.destroy()
        global wtext

        def button_clicked(checked):
            """ Event handler: "OK" button clicked """
            wdg.destroy()

        def download_text(checked):
            mw = pya.Application.instance().main_window()
            layout_filename = mw.current_view().active_cellview().filename()
            if len(layout_filename) == 0:
                raise Exception("Please save your layout before exporting.")
            file_out = os.path.join(os.path.dirname(layout_filename),
                                    "{}.txt".format(os.path.splitext(os.path.basename(layout_filename))[0]))
            f = open(file_out, 'w')

            # Find the automated measurement coordinates:
            from .utils import find_automated_measurement_labels
            cell = pya.Application.instance().main_window().current_view().active_cellview().cell
            text_out, opt_in = find_automated_measurement_labels(cell)

            #text_out doesn't have new line spaces

            f.write(text_out.replace("<br>", "\n"))

            wd = pya.QDialog(pya.Application.instance().main_window())

            #        wdg.setAttribute(pya.Qt.WA_DeleteOnClose)
            wd.setAttribute = pya.Qt.WA_DeleteOnClose


            wd.resize(150, 50)
            wd.move(1, 1)
            grid = pya.QGridLayout(wd)
            windowlabel = pya.QLabel(wd)
            windowlabel.setText("Download Complete. Saved to {}".format(file_out))
            grid.addWidget(windowlabel, 2, 2, 4, 4)
            wd.show()

        wdg = pya.QDialog(pya.Application.instance().main_window())

#        wdg.setAttribute(pya.Qt.WA_DeleteOnClose)
        wdg.setAttribute=pya.Qt.WA_DeleteOnClose
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
        Download = pya.QPushButton("Download", wdg)
        ok.clicked(button_clicked)   # attach the event handler
        Download.clicked(download_text)
#    netlist = pya.QPushButton("Save", wdg) # not implemented

        grid.addWidget(windowlabel1, 0, 0, 1, 3)
        grid.addWidget(wtext, 1, 1, 3, 3)
#    grid.addWidget(netlist, 4, 2)
        grid.addWidget(ok, 4, 3)
        grid.addWidget(Download, 4, 2)

        grid.setRowStretch(3, 1)
        grid.setColumnStretch(1, 1)

        wdg.show()

    # Create a GUI for the output:
    gen_ui()
    #wtext.insertHtml('<br>* Automated measurement coordinates:<br><br>')

    # Find the automated measurement coordinates:
    from .utils import find_automated_measurement_labels
    cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    text_out, opt_in = find_automated_measurement_labels(cell)
    wtext.insertHtml(text_out)
    pya.Application.instance().main_window().redraw()    

def find_SEM_labels_gui(topcell=None, LayerSEMN=None, MarkersInTopCell=False):
    from .utils import get_technology
    TECHNOLOGY = get_technology()

    import string
    if not LayerSEMN:
        from .utils import get_technology, find_paths
        TECHNOLOGY = get_technology()
        dbu = TECHNOLOGY['dbu']
        if 'SEM' in TECHNOLOGY:
            LayerSEMN = TECHNOLOGY['SEM']
        else:
            layers = [d for d in TECHNOLOGY.keys() if 'SEM' in d and '_color' not in d]
            if layers != []:
                LayerSEMN = TECHNOLOGY[layers[0]]
            else:
                v = pya.MessageBox.warning("SEM images", "No 'SEM' layer found in the Technology.", pya.MessageBox.Ok)
                return
    if not topcell:
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
        topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
        if topcell == None:
            raise UserWarning("No cell. Make sure you have an open layout.")


    # Create a Results Database
    rdb_i = lv.create_rdb("SiEPIC-Tools SEM images: %s technology" %
                          TECHNOLOGY['technology_name'])
    rdb = lv.rdb(rdb_i)
    rdb.top_cell_name = topcell.name
    rdb_cell = rdb.create_cell(topcell.name)

    # SEM images
    rdb_cell = next(rdb.each_cell())
    rdb_cat_id_SEM = rdb.create_category("SEM images")
    rdb_cat_id_SEM.description = "SEM image"

    dbu = topcell.layout().dbu
    iter = topcell.begin_shapes_rec(topcell.layout().layer(LayerSEMN))
    i = 0
    while not(iter.at_end()):
        if iter.shape().is_box():
            box = iter.shape().box
            if not(MarkersInTopCell):
                cc = [c for c in rdb.each_cell() if c.name()==iter.shape().cell.name]
                if cc==[]:
                    rdb_cell = rdb.create_cell(iter.shape().cell.name)
                else:
                    rdb_cell = cc[0]
            
            i += 1
            if MarkersInTopCell:
                box2 = iter.shape().box.transformed(iter.itrans()).to_dtype(dbu)  # when added to the top cell
            else:
                box2 = iter.shape().box.to_dtype(dbu)  # coordinates within the cell
            
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_SEM.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(box2))
        iter.next()

    # displays results in Marker Database Browser, using Results Database (rdb)
    if rdb.num_items() > 0:
        v = pya.MessageBox.warning(
            "SEM images", "%s SEM images found.  \nPlease review SEM images using the 'Marker Database Browser'." % rdb.num_items(), pya.MessageBox.Ok)
        lv.show_rdb(rdb_i, cv.cell_index)
    else:
        v = pya.MessageBox.warning("SEM images", "No SEM images found.", pya.MessageBox.Ok)
    pya.Application.instance().main_window().redraw()    


def calculate_area():
    from .utils import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    dbu = TECHNOLOGY['dbu']

    try:
      total = cell.each_shape(ly.layer(TECHNOLOGY['FloorPlan'])).__next__().polygon.area()
    except:
      total = 1e99
    area = 0
    itr = cell.begin_shapes_rec(ly.layer(TECHNOLOGY['Waveguide']))
    while not itr.at_end():
        area += itr.shape().area()
        itr.next()
    print("Waveguide area: %s mm^2, chip area: %s mm^2, percentage: %s %%" % (area/1e6*dbu*dbu,total/1e6*dbu*dbu, area/total*100))

    if total == 1e99:
      v = pya.MessageBox.warning(
        "Waveguide area.", "Waveguide area: %.5g mm^2 \n   (%.5g micron^2)" % (area/1e6*dbu*dbu, area/1e6), pya.MessageBox.Ok)
    else:
      v = pya.MessageBox.warning(
        "Waveguide area.", "Waveguide area: %.5g mm^2 \n   (%.5g micron^2),\nChip Floorplan: %.5g mm^2, \nPercentage: %.3g %%" % (area/1e6*dbu*dbu, area/1e6, total/1e6*dbu*dbu, area/total*100), pya.MessageBox.Ok)

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




def trim_netlist(nets, components, selected_component, verbose=None):
    """Trim Netlist
    by Jaspreet Jhoja (c) 2016-2017
    
    This Python function facilitates trimming of netlist based on a selected component.
    Version history:
    
    Jaspreet Jhoja           2017/12/29
     - Initial version

    Inputs, and example of how to generate them:
        nets, components = topcell.identify_nets()
        selected_component = components[5]   (elsewhere the desired component is selected)
    """

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




def layout_check(cell=None, verbose=False):
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
    rdb_cat_id_comp_shapesoutside = rdb.create_category(rdb_cat_id_comp, "Shapes outside component")
    rdb_cat_id_comp_shapesoutside.description = "Shapes need to be inside a component. Read more about requirements for components: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout"

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

    paths = find_paths(TECHNOLOGY['Waveguide'], cell=cell)
    for p in paths:
        if verbose:
            print("%s, %s" % (type(p), p))
        # Check for paths with > 2 vertices
        Dpath = p.to_dtype(dbu)
        if Dpath.num_points() > 2:
            rdb_item = rdb.create_item(rdb_cell.rdb_id(), rdb_cat_id_wg_path.rdb_id())
            rdb_item.add_value(pya.RdbItemValue(Dpath.polygon()))

    '''
    Shapes need to be inside a component. 
    Read more about requirements for components: https://github.com/SiEPIC/SiEPIC-Tools/wiki/Component-and-PCell-Layout
    Method:
        - find all shapes
        - find all components, and their shapes
        - substract the two, and produce errors        
    rdb_cat_id_comp_shapesoutside
    '''
    for i in range(0, len(components)):
        c = components[i]


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
        if not c.has_model() and 0:
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
            opt_in_labels = [o['opt_in'] for o in opt_in if 'opt_in' in o.keys()]
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
            if 'opt_in' in o.keys():
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

    if not opt_in_selection_text:
        raise Exception ('No opt_in labels were selected. \nCannot perform operation.')
        return

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
    Author: Jaspreet Jhoja(2016 - 2022)

    This Python file implements a waveguide resizing tool.
    Version history:
       Jaspreet Jhoja 2022/02/09
       - Fixed incorrect waveguide length calculations
       - Revised code format to be in-line with flake8 recommendations
       
       Jaspreet Jhoja 2018/02/13
        - Resizes Waveguides with selection
        - Users are required to press Ctrl + Shift + R
"""

def resize_waveguide():
    import pya
    import sys
    from pya import (
        QFont,
        QWidget,
        Qt,
        QVBoxLayout,
        QFrame,
        QLabel,
        QComboBox,
        QLineEdit,
        QPushButton,
        QGridLayout,
        QSplitter,
    )
    from SiEPIC import utils

    TECHNOLOGY, lv, ly, cell = utils.get_layout_variables()

    global points, path_edges, edges_orientations

    # Get the waveguide layer
    LayerSiN = ly.layer(TECHNOLOGY["Waveguide"])

    # get the selected waveguide instance
    selection = utils.select_waveguides(cell)

    # if multiple cells or no cells are selected
    if len(selection) > 1 or len(selection) == 0:
        pya.MessageBox.warning("Message", "No waveguide is selected", pya.MessageBox.Ok)
    else:
        wg_obj = selection[0]
        if wg_obj.is_cell_inst():
            oinst = wg_obj.inst()
            if oinst.is_pcell():
                c = oinst.cell

        path_obj = c.pcell_parameters_by_name()["path"]

        if path_obj.points <= 3:
            pya.MessageBox.warning(
                "Message",
                "Cannot perform this operation on the selected cell/path.\n Hint: Select a cell/path with more than 2 vertices.",
                pya.MessageBox.Ok,
            )

        else:
            # calculate the length of the waveguide using the spice parameters
            path_length = float(c.find_components()[0].params.split(' ')[0].split('=')[1])*1e6

            # get path points
            points_obj = path_obj.get_dpoints()
            points = [[each.x, each.y] for each in points_obj]

            # Separate the path_edges of the waveguide
            path_edges_all = []
            for i in range(len(points)):
                if i > 0:
                    pair = [points[i - 1], points[i]]
                    path_edges_all.append(pair)

            # Ignore the first and the last segment from being edited
            path_edges = path_edges_all[1:-1]

            # Check segment orientation
            edges_orientations = []
            for each in path_edges:
                if each[0][0] == each[1][0]:
                    edges_orientations.append("vertical")
                elif each[0][1] == each[1][1]:
                    edges_orientations.append("horizontal")

            # prop variable which determines the segment propagation, i.e. which directions this segment can move in, +x, -x, +y, -y
            prop_points = points
            edge_propagations = []
            # +x, -x , +y , -y
            for each in path_edges:
                index = prop_points.index(each[0])
                prop = ""

                if index == 0:
                    index = index + 1
                    element_idx = index + 1
                else:
                    element_idx = index - 1

                x1 = prop_points[index][0]
                y1 = prop_points[index][1]
                x2 = prop_points[element_idx][0]
                y2 = prop_points[element_idx][1]

                # the edge propagates along y axis if their x coordinates are equal
                if x1 == x2:
                    if y1 < y2:
                        prop = "+y"
                    elif y1 > y2:
                        prop = "-y"

                elif y1 == y2:
                    if x1 < x2:
                        prop = "-x"
                    elif x1 > x2:
                        prop = "+x"

                edge_propagations.append(prop)

            global wdg, hbox, lframe1, titlefont, lf1title, parameters, lf1label1, lf1label2, lf1label3, lf1title2, lf1text3, lf1form, lframe1, leftsplitter, splitter1, container, ok
            wdg = QWidget()
            wdg.setAttribute = pya.Qt.WA_DeleteOnClose
            wdg.setWindowTitle("Waveguide resizer")

            if sys.platform.startswith("linux"):
                # Linux-specific code here...
                titlefont = QFont("Arial", 11, QFont.Bold, False)

            elif sys.platform.startswith("darwin"):
                # OSX specific
                titlefont = QFont("Arial", 9, QFont.Bold, False)

            elif sys.platform.startswith("win"):
                titlefont = QFont("Arial", 9, QFont.Bold, False)

            hbox = QVBoxLayout(wdg)
            wdg.setFixedSize(650, 250)

            def selection(self):
                global path_edges, edges_orientations, lf1label1, lf1label2

                lf1label1.setText(
                    "     Segment length: %s microns"
                    % str(
                        (
                            abs(
                                path_edges[parameters.currentIndex][0][0]
                                - path_edges[parameters.currentIndex][1][0]
                            )
                            + abs(
                                path_edges[parameters.currentIndex][0][1]
                                - path_edges[parameters.currentIndex][1][1]
                            )
                        )
                    )
                )

                lf1label2.setText(
                    "     Segment orientation: %s"
                    % str(edges_orientations[parameters.currentIndex])
                )

            # Left Frame top section
            lframe1 = QFrame()
            lframe1.setFrameShape(QFrame.StyledPanel)
            lframe1.setStyleSheet("background-color: white;")
            lf1title = QLabel(
                "Current waveguide length (microns): %s" % str(path_length)
            )
            parameters = QComboBox()
            # add vertices as params
            params = []
            for each in range(len(path_edges)):

                params.append(
                    "segment %s, points: (%s, %s) - (%s, %s)"
                    % (
                        str(each),
                        path_edges[each][0][0],
                        path_edges[each][0][1],
                        path_edges[each][1][0],
                        path_edges[each][1][1],
                    )
                )

            parameters.addItems(params)
            parameters.currentIndexChanged(selection)
            parameters.setFixedWidth(400)
            parameters.setStyleSheet("background-color: white;")
            lf1label1 = QLabel("Segment length: ")
            lf1label2 = QLabel("Segment orientation: ")
            lf1label3 = QLabel("New target waveguide length (microns): ")
            lf1title2 =  QLabel("Select the segment you wish to move:")
            lf1text3 = QLineEdit()
            lf1text3.setAccessibleName("lf1text3")
            lf1text3.setText(str(path_length))

            def button(self):
                # Close GUI while changing the layout to avoid view problems.
                wdg.close()

                if lf1text3.text == "":
                    return 0

                # Record the layout state, to enable "undo"
                lv.transaction("Waveguide resizing")

                # get the index of selected segment/edge and the axis of propagation
                index = parameters.currentIndex
                diff = float(lf1text3.text) - path_length

                prop = edge_propagations[index]
                if prop == "+x" or prop == "-x":
                    if prop == "-x":
                        diff = diff * -1

                    path_edges[index][0][0] = path_edges[index][0][0] + diff / 2
                    path_edges[index][1][0] = path_edges[index][1][0] + diff / 2

                elif prop == "+y" or prop == "-y":
                    if prop == "+y":
                        diff = diff * -1

                    path_edges[index][0][1] = path_edges[index][0][1] + diff / 2
                    path_edges[index][1][1] = path_edges[index][1][1] + diff / 2

                dpoints = [pya.DPoint(each[0], each[1]) for each in points]
                dpath = pya.DPath(dpoints, 0.5 * c.layout().dbu) # 0.5 is irrelevant to actual waveguide width

                # replace the old waveguide path points with the new path points
                oinst.change_pcell_parameter("path", dpath)
                lv.commit()
                # destroy GUI
                wdg.destroy()

            ok = QPushButton("OK")
            ok.clicked(button)

            lf1form = QGridLayout()
            lf1form.addWidget(lf1title, 0, 0)
            lf1form.addWidget(lf1label3, 1, 0)
            lf1form.addWidget(lf1text3, 1, 1)
            lf1form.addWidget(lf1title2, 2, 0)
            lf1form.addWidget(parameters, 3, 0)
            lf1form.addWidget(lf1label1, 4, 0)
            lf1form.addWidget(lf1label2, 5, 0)
            lf1form.addWidget(ok, 7, 1)
            lframe1.setLayout(lf1form)
            leftsplitter = QSplitter()
            leftsplitter.setOrientation = Qt.Vertical
            leftsplitter.addWidget(lframe1)
            leftsplitter.setSizes([500, 400, 10])
            splitter1 = QSplitter()
            splitter1.setOrientation = Qt.Horizontal
            splitter1.addWidget(leftsplitter)
            splitter1.setSizes([400, 500])
            container = QWidget()
            hbox.addWidget(splitter1)
            selection(None)
            wdg.show()


def replace_cell(layout, cell_x_name, cell_y_name, cell_y_file=None, cell_y_library=None, Exact = True, debug = False):
    '''
    SiEPIC-Tools: scripts.replace_cell
    Search and replace: cell_x with cell_y
    useful for blackbox IP cell replacement
    - load layout containing cell_y_name from cell_y_file or cell_y_library
    - replace all cell_x_name* instances with cell_y
    
    Black box                   True geometry
    Basename_BB, Basename_BB*   YES: Basename
    Basename, Basename*         NO: Basename_extension
    Basename, Basename*         YES: DifferentName
    '''
    
    import os
    if debug:
        print(" - cell replacement for: %s, with cell %s (%s), "  % (cell_x_name, cell_y_name, os.path.basename(cell_y_file)))
    log = ''
    log += "- cell replacement for: %s, with cell %s (%s)\n"  % (cell_x_name, cell_y_name, os.path.basename(cell_y_file))

    # Find the cells that need replacement (cell_x)
    if Exact:
        # find cell name exactly matching cell_x_name
        cells_x = [layout.cell(cell_x_name)]
    else:
        # replacement for all cells that:
        # 1) cell name exact matching cell_x_name, OR
        # 2) that begin with the cell name, i.e., xxx* is matched
        #    i.e., xxx and xxx* are matched
        cells_x = [cell for cell in layout.each_cell() if cell.name.find(cell_x_name) == 0]

        # replacement for all cells that:
        # 1) cell name exact matching cell_x_name, OR
        # 2) that begin with the cell name and have a $
        #    i.e., xxx and xxx$* are matched  (was used for the Phot1x 2022/06 tapeout)
        #cells_x = [cell for cell in layout.each_cell() if cell.name == cell_x_name or cell.name.find(cell_x_name) == 0 and '$' in cell.name]

    # Load the new cell:   
    if cell_y_file:
        # find cell name CELL_Y
#        print(layout.top_cell())
        cell_y = layout.cell(cell_y_name)
        if debug:
            print(" - checking for cell %s in current layout: %s" % (cell_y_name, cell_y))
        if not cell_y:
            # Load cell_y_name:
            if debug:
                print(" - loading cell %s from file %s" % (cell_y_name, cell_y_file))
            layout.read(cell_y_file)
            # find cell name CELL_Y
            cell_y = layout.cell(cell_y_name)
        if not cell_y:
            raise Exception("No cell '%s' found in layout %s." % (cell_y_name, cell_y_file))
        if debug:
            print("   - replacing with cell: %s, from: %s." % (cell_y.name, os.path.basename(cell_y_file)))
    if cell_y_library:
        cell_y = layout.create_cell(cell_y_name, cell_y_library)
        if not cell_y:
            raise Exception ('Cannot import cell %s from library %s' % (cell_y_name, cell_y_library))        

    if cells_x:
        log += "   - replacing cells: %s\n"  % ([c.name for c in cells_x])
        
    for cell_x in cells_x:
        if debug:
            print(" - replace_cell: found cells to be replaced: %s"  % (cell_x.name))
    
        # find caller cells
        caller_cells = cell_x.caller_cells()
        # loop through all caller cells:
        for c in caller_cells:
            cc = layout.cell(c)

            # find instances of CELL_X in caller cell
            itr = cc.each_inst()
            inst = next(itr)
            while inst:
#                if debug:
#                    print("   - found inst: %s, %s" % (inst, inst.cell.name))
                if inst.cell.name == cell_x.name:
                    if cell_y.destroyed():
                        print('   - Warning: cell_y (%s) destroyed, skipping replacement' % (cell_y_name))
                        print("   - destroyed status: cell_y - %s, cell_x - %s, cc - %s" % (cell_y.destroyed(), cell_x.destroyed(), cc.destroyed()))
                        print('   - looking for cell. %s, %s, %s' % (cell_y_name, cell_y, layout.cell(cell_y_name)))
                        log += '   - Warning: cell destroyed, skipping replacement\n'
#                        continue # skip this inst, continue to next; stays in an infinite loop
                        break  # skip this cell
                    # replace with CELL_Y
                    if inst.is_regular_array():
                        if debug:
                            print("    - replacing %s in %s, with cell array: %s" % (cell_x.name, cc.name, cell_y.name))
                        ci = inst.cell_inst
                        cc.replace(inst, pya.CellInstArray(cell_y.cell_index(),inst.trans, ci.a, ci.b, ci.na, ci.nb))
                    else:
                        if debug:
                            print("    - replacing %s in %s, with cell: %s" % (cell_x.name, cc.name, cell_y.name))
                        cc.replace(inst, pya.CellInstArray(cell_y.cell_index(),inst.trans))
                inst = next(itr, None)


    return log




def svg_from_cell(verbose=True):
  if verbose:
    print('SiEPIC.scripts: svg_from_cell()')

  # Get technology and layout details
  from .utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
  dbum = TECHNOLOGY['dbu']*1e-6 # dbu to m conversion

  # get selected instances; only one
  from .utils import select_instances
  selected_instances = select_instances()
  error = pya.QMessageBox()
  error.setStandardButtons(pya.QMessageBox.Ok )
  if len(selected_instances) != 1:
    error.setText("Error: Need to have one component selected.")
    response = error.exec_()
    return

  # get selected component
  if verbose:
    print(" selected component: %s" % selected_instances[0].inst().cell )
  component = cell.find_components(cell_selected=[selected_instances[0].inst().cell])[0]

  # create an SVG icon for the component, for INTC compact model icon
  from . import _globals
  import os
  from .utils import svg_from_component
  svg_filename = os.path.join(_globals.TEMP_FOLDER, '%s.svg' % component.instance)
  if verbose:
    print(" SVG filename: %s" %svg_filename)
  svg_from_component(component, svg_filename)

  message = pya.QMessageBox()
  message.setStandardButtons(pya.QMessageBox.Ok )
  message.setText("Exported SVG file for selected component. File in location: %s." %svg_filename )
  response = message.exec_()
  

def zoom_out(topcell):
    '''When running in the GUI, Zoom out and show full hierarchy
    '''
    from SiEPIC._globals import Python_Env
    if Python_Env == "KLayout_GUI":   
        # get the Layout View
        mw = pya.Application().instance().main_window()
        lv = mw.current_view()
        if lv:
            # Zoom out
            lv.clear_object_selection()
            lv.zoom_fit()
            # Show all cell hierarchy
            lv.max_hier()
    else:
        return


    
def export_layout(topcell, path, filename, relative_path = '', format='oas', screenshot=False):
    '''Export the layout, as a static file without PCells
    runs in GUI mode or in headless mode
    format = 'oas' for compressed OASIS, or 'gds' for GDSII
    optionally save a screenshot
    by Lukas Chrostowski, 2023, SiEPIC-Tools
    '''

    # Save the layout, without PCell info, for fabrication
    save_options = pya.SaveLayoutOptions()
    save_options.write_context_info=False  
    if format == 'oas':
        save_options.format='OASIS' # smaller file size
        save_options.oasis_compression_level=10
        save_options.oasis_permissive=True
        extension = '.oas'
    else:
        save_options.format='GDS2' 
        extension = '.gds'
    layout = topcell.layout()

    # output file
    import os
    file_out = os.path.join(path, relative_path, filename+extension)
    
    # Save the layout, from the GUI
    success = False
    from SiEPIC._globals import Python_Env
    if Python_Env == "KLayout_GUI":
        mw = pya.Application().instance().main_window()
        lv = mw.current_view()
        if lv:
            cv = mw.current_view().active_cellview().index()
            active_layout = pya.CellView.active().layout()
            if active_layout == layout:
                try:
                    lv.save_as(cv, file_out, save_options)
                    success = True
                except:
                    raise Exception("Problem exporting your layout, %s." % file_out)
            if screenshot:
                png_out = os.path.join(path, relative_path, filename+'.png')
                try:
                    lv.save_screenshot(png_out)
                except:
                    raise Exception("Problem creating screenshot, %s." % png_out)
            
    if not success:
        try:
            layout.write(file_out,save_options)
        except:
            try:
                layout.write(file_out)
            except:
                raise Exception("Problem exporting your layout, %s." % file_out)

def instantiate_all_library_cells(topcell, progress_bar = True):
    '''
    Load all cells (fixed and PCells) and instantiate them on the layout. 
    One column per library, one column for fixed and PCells.
    topcell: is a cell in a pya.Layout that has already configured with Layout.technology_name 
    progress_bar: True displays percentage
    '''
    
    
    from SiEPIC._globals import Python_Env
    if True or Python_Env == "KLayout_GUI":
        # Count all the cells for the progress bar
        count = 0
        ly = topcell.layout()
        for lib in pya.Library().library_ids():
            li = pya.Library().library_by_id(lib)
            if not li.is_for_technology(ly.technology_name) or li.name() == 'Basic':
                continue
            # all the pcells
            count += len(li.layout().pcell_names())
            # all the fixed cells
            for c in li.layout().each_top_cell():
                if not li.layout().cell(c).is_pcell_variant():
                    count += 1
        p = pya.RelativeProgress("Instantiate all libraries' cells", count)

    # all the libraries
    ly = topcell.layout()
    x,y,xmax=0,0,0
    for lib in pya.Library().library_ids():
        li = pya.Library().library_by_id(lib)
        if not li.is_for_technology(ly.technology_name) or li.name() == 'Basic':
            print(' - skipping: %s' % li.name())
            continue

        # all the pcells
        print('All PCells: %s' % li.layout().pcell_names())
        for n in li.layout().pcell_names():
            print(" - PCell: ", li.name(), n)
            pcell = ly.create_cell(n,li.name(), {})
            if pcell:
                t = pya.Trans(pya.Trans.R0, x-pcell.bbox().left, y-pcell.bbox().bottom)
                topcell.insert(pya.CellInstArray(pcell.cell_index(), t))
                y += pcell.bbox().height()+2000
                xmax = max(xmax, x+pcell.bbox().width()+2000)
            else:
                print('Error in: %s' % n)
            p.inc()
        x, y = xmax, 0
        
        # all the fixed cells
        for c in li.layout().each_top_cell():
            # instantiate
            if not li.layout().cell(c).is_pcell_variant():
                print(" - Fixed cell: ", li.name(), li.layout().cell(c).name)
                pcell = ly.create_cell(li.layout().cell(c).name,li.name(), {})
                if not pcell:
                    pcell = ly.create_cell(li.layout().cell(c).name,li.name())
                if pcell:
                    t = pya.Trans(pya.Trans.R0, x-pcell.bbox().left, y-pcell.bbox().bottom)
                    topcell.insert(pya.CellInstArray(pcell.cell_index(), t))
                    y += pcell.bbox().height()+2000
                    xmax = max(xmax, x+pcell.bbox().width()+2000)
                else:
                    print('Error in: %s' % li.layout().cell(c).name)
            p.inc()
        x, y = xmax, 0

    if True or Python_Env == "KLayout_GUI":
        p.destroy


def load_klayout_technology(techname, path_module, path_lyt_file):
    '''
    techname: <string> name of the technology
    path_module: <string> where the Python module is loaded from, e.g., import EBeam
    path_lyt_file: <string> where the KLayout technology (.lyt) is located
    returns: <pya.Technology>
    '''
    import sys
    
    # if running in KLayout Application mode, the technology is loaded
    # automatically via the Technology Manager
    if techname in pya.Technology().technology_names():
        return pya.Technology().technology_by_name(techname)

    # if running in KLayout in PyPI mode, the technology needs to be
    # loaded separately
    if techname not in sys.modules:
        if not path_module in sys.path:
            sys.path.append(path_module)
        tech = pya.Technology().create_technology('EBeam')
        tech = tech.load(path_lyt_file)
        # technology needs to be defined and loaded first, before importing
        import importlib
        importlib.import_module(techname)
        return tech


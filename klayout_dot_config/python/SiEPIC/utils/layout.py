""" Layout helper functions.

Authors: Thomas Ferreira de Lima @thomaslima
         Lukas Chrostowski @lukasc-ubc

The following functions are useful for scripted layout, or making
PDK Pcells.

Functions:

layout_waveguide2
layout_waveguide
layout_waveguide_sbend_bezier
make_pin
y_splitter_tree
floorplan(topcell, x, y)
new_layout(tech, topcell_name, overwrite = False)


TODO: enhance documentation
TODO: make some of the functions in util use these.
"""
from itertools import repeat
import pya
import numpy as np
from numpy import cos, sin, pi, sqrt
import math as m

from functools import reduce
from .sampling import sample_function
from .geometry import rotate90, rotate, bezier_optimal, curve_length

'''
Create a waveguide, in a specific technology
inputs
- cell: into which Cell we add the waveguide
- dpath: DPath type
- waveguide_type: a name from Waveguides.XML
    can be a <compound_waveguide>
    or a primitive waveguide type containing <component> info
output
- compound waveguide, or regular waveguide
by Lukas Chrostowski
acknowledgements: Diedrik Vermeulen for the code to place the taper in the correct orientation
'''


def layout_waveguide4(cell, dpath, waveguide_type, debug=False):

    if debug:
        print('SiEPIC.utils.layout.layout_waveguide4: ')
        print(' - waveguide_type: %s' % (waveguide_type))

    # get the path and clean it up
    layout = cell.layout()
    dbu = layout.dbu
    dpath = dpath.to_itype(dbu)
    dpath.unique_points()
    pts = dpath.get_points()
    dpts = dpath.get_dpoints()

    # Load the technology and all waveguide types
    from SiEPIC.utils import load_Waveguides_by_Tech
    technology_name = layout.technology_name
    waveguide_types = load_Waveguides_by_Tech(technology_name)
    if debug:
        print(' - technology_name: %s' % (technology_name))
        print(' - waveguide_types: %s' % (waveguide_types))

    # Load parameters for the chosen waveguide type
    params = [t for t in waveguide_types if t['name'] == waveguide_type]

    if type(params) == type([]) and len(params) > 0:
        params = params[0]
        if 'width' not in params and 'compound_waveguide' not in params:
            params['width'] = params['wg_width']
        params['waveguide_type'] = waveguide_type
    else:
        print('error: waveguide type not found in PDK waveguides')
        raise Exception('error: waveguide type (%s) not found in PDK waveguides' %
                        waveguide_type)

    # compound waveguide types:
    if 'compound_waveguide' in params:
        # find the singlemode and multimode waveguides:
        if 'singlemode' in params['compound_waveguide']:
            singlemode = params['compound_waveguide']['singlemode']
        else:
            raise Exception(
                'error: waveguide type (%s) does not have singlemode defined' % waveguide_type)
        if 'multimode' in params['compound_waveguide']:
            multimode = params['compound_waveguide']['multimode']
        else:
            raise Exception(
                'error: waveguide type (%s) does not have multimode defined' % waveguide_type)
        params_singlemode = [t for t in waveguide_types if t['name'] == singlemode]
        params_multimode = [t for t in waveguide_types if t['name'] == multimode]
        if type(params_singlemode) == type([]) and len(params_singlemode) > 0:
            params_singlemode = params_singlemode[0]
            params_singlemode['waveguide_type'] = singlemode
        else:
            raise Exception(
                'error: waveguide type (%s) not found in PDK waveguides' % singlemode)
        if type(params_multimode) == type([]) and len(params_multimode) > 0:
            params_multimode = params_multimode[0]
            params_multimode['waveguide_type'] = multimode
        else:
            raise Exception(
                'error: waveguide type (%s) not found in PDK waveguides' % multimode)
        # find the taper
        if 'taper_library' in params['compound_waveguide'] and 'taper_cell' in params['compound_waveguide']:
            taper = layout.create_cell(
                params['compound_waveguide']['taper_cell'], params['compound_waveguide']['taper_library'])
            if not taper:
                raise Exception('Cannot import cell %s : %s' % (
                    params['compound_waveguide']['taper_cell'], params['compound_waveguide']['taper_library']))
        else:
            raise Exception(
                'error: waveguide type (%s) does not have taper cell and library defined' % waveguide_type)
        from pya import Trans, CellInstArray

        '''
        find sections of waveguides that are larger than (2 x radius + 2 x taper_length)
         - insert two tapers
         - insert multimode straight section
         - insert singlemode waveguides (including bends) before
        '''
        import math
        from SiEPIC.extend import to_itype
        from pya import Point
        radius = to_itype(params_singlemode['radius'], dbu)
        pins, _ = taper.find_pins()
        taper_length = pins[0].center.distance(pins[1].center)
        min_length = 2*radius + 2*taper_length
        offset = radius
        wg_sm_segment_pts = []
        wg_last = 0
        waveguide_length = 0
        for ii in range(1, len(dpts)):
            start_point = dpts[ii-1]
            end_point = dpts[ii]
            distance_points = end_point.distance(start_point)
            if distance_points < min_length:
                # single mode segment, keep track
                if ii == 1:
                    wg_sm_segment_pts.append(pts[ii-1])
                wg_sm_segment_pts.append(pts[ii])
                if ii == len(pts)-1:
                    subcell = layout.create_cell("Waveguide_sm_%s" % ii)
                    cell.insert(CellInstArray(subcell.cell_index(), Trans()))
                    waveguide_length += layout_waveguide3(subcell,
                                                          wg_sm_segment_pts, params_singlemode, debug=False)
            else:
                # insert two tapers and multimode waveguide
                angle = math.atan2((end_point.y-start_point.y),
                                   (end_point.x-start_point.x))/math.pi*180
                if ii == 1:
                    wg_first = offset
                else:
                    wg_first = 0
                if ii == len(pts)-1:
                    wg_last = offset
                if round(angle) % 360 == 270.0:
                    t = Trans(Trans.R270, start_point.x, start_point.y-offset+wg_first)
                    t2 = Trans(Trans.R90, end_point.x, end_point.y+offset-wg_last)
                    wg_start_pt = Point(start_point.x, start_point.y -
                                        offset-taper_length+wg_first)
                    wg_end_pt = Point(end_point.x, end_point.y+offset+taper_length-wg_last)
                if round(angle) % 360 == 90.0:
                    t = Trans(Trans.R90, start_point.x, start_point.y+offset-wg_first)
                    t2 = Trans(Trans.R270, end_point.x, end_point.y-offset+wg_last)
                    wg_start_pt = Point(start_point.x, start_point.y +
                                        offset+taper_length-wg_first)
                    wg_end_pt = Point(end_point.x, end_point.y-offset-taper_length+wg_last)
                if round(angle) % 360 == 180.0:
                    t = Trans(Trans.R180, start_point.x-offset+wg_first, start_point.y)
                    t2 = Trans(Trans.R0, end_point.x+offset-wg_last, end_point.y)
                    wg_start_pt = Point(start_point.x-offset -
                                        taper_length+wg_first, start_point.y)
                    wg_end_pt = Point(end_point.x+offset+taper_length-wg_last, end_point.y)
                if round(angle) % 360 == 0.0:
                    t = Trans(Trans.R0, start_point.x+offset-wg_first, start_point.y)
                    t2 = Trans(Trans.R180, end_point.x-offset+wg_last, end_point.y)
                    wg_start_pt = Point(start_point.x+offset +
                                        taper_length-wg_first, start_point.y)
                    wg_end_pt = Point(end_point.x-offset-taper_length+wg_last, end_point.y)
                inst_taper = cell.insert(CellInstArray(taper.cell_index(), t))
                inst_taper = cell.insert(CellInstArray(taper.cell_index(), t2))
                waveguide_length += taper_length*2
                subcell = layout.create_cell("Waveguide_mm_%s" % ii)
                cell.insert(CellInstArray(subcell.cell_index(), Trans()))
                waveguide_length += layout_waveguide3(subcell,
                                                      [wg_start_pt, wg_end_pt], params_multimode, debug=False)
                # compound segment
                if ii > 1:
                    wg_sm_segment_pts.append(t.disp.to_p())
                    subcell = layout.create_cell("Waveguide_sm_%s" % ii)
                    cell.insert(CellInstArray(subcell.cell_index(), Trans()))
                    waveguide_length += layout_waveguide3(subcell,
                                                          wg_sm_segment_pts, params_singlemode, debug=False)
                    wg_sm_segment_pts = [t2.disp.to_p(), pts[ii]]
                else:
                    wg_sm_segment_pts = [t2.disp.to_p(), pts[ii]]

    else:
        # primitive waveguide type
        waveguide_length = layout_waveguide3(cell, pts, params, debug=False)

    return waveguide_length


'''
Create a waveguide, in a specific technology
inputs
- cell: into which Cell we add the waveguide
    from SiEPIC.utils import get_layout_variables
    TECHNOLOGY, lv, layout, cell = get_layout_variables()
- pts
- params, obtained from load_Waveguides_by_Tech and Waveguides.XML
    must be a primitive waveguide type containing <component> info
output:
- waveguide
- DevRec, PinRec
by Lukas Chrostowski
'''


def layout_waveguide3(cell, pts, params, debug=False):

    if debug:
        print('SiEPIC.utils.layout.layout_waveguide3: ')

    layout = cell.layout()
    dbu = layout.dbu
    technology_name = layout.technology_name
    from SiEPIC.utils import get_technology_by_name
    TECHNOLOGY = get_technology_by_name(technology_name)

    from SiEPIC.extend import to_itype
    wg_width = to_itype(params['width'], dbu)
    radius = float(params['radius'])
    model = params['model']
    cellName = 'Waveguide'
    CML = params['CML']
    waveguide_type = params['waveguide_type']

    if debug:
        print(' - waveguide params: %s' % (params))

    if 'compound_waveguide' in params:
        print('error: this function cannot handle compound waveguides')
        raise Exception(
            'error: this function cannot handle compound waveguides (%s)' % waveguide_type)

    # draw the waveguide
    sbends = params['sbends'].lower() in ['true', '1', 't', 'y', 'yes'] if 'sbends' in params.keys() else False
    waveguide_length = layout_waveguide2(TECHNOLOGY, layout, cell, [wg['layer'] for wg in params['component']], [
                                         wg['width'] for wg in params['component']], [wg['offset'] for wg in params['component']], 
                                         pts, radius, params['adiabatic'], params['bezier'], sbends)

    # Draw the marking layers
    from SiEPIC.utils import angle_vector
    LayerPinRecN = layout.layer(TECHNOLOGY['PinRec'])

    make_pin(cell, 'opt1', pts[0], wg_width, LayerPinRecN, angle_vector(pts[0]-pts[1]) % 360)
    make_pin(cell, 'opt2', pts[-1], wg_width, LayerPinRecN,
             angle_vector(pts[-1]-pts[-2]) % 360)

    from pya import Trans, Text, Path, Point

    '''
    t1 = Trans(angle_vector(pts[0]-pts[1])/90, False, pts[0])
    cell.shapes(LayerPinRecN).insert(Path([Point(-10, 0), Point(10, 0)], wg_width).transformed(t1))
    cell.shapes(LayerPinRecN).insert(Text("opt1", t1, 0.3/dbu, -1))
    
    t = Trans(angle_vector(pts[-1]-pts[-2])/90, False, pts[-1])
    cell.shapes(LayerPinRecN).insert(Path([Point(-10, 0), Point(10, 0)], wg_width).transformed(t))
    cell.shapes(LayerPinRecN).insert(Text("opt2", t, 0.3/dbu, -1))
    '''

    LayerDevRecN = layout.layer(TECHNOLOGY['DevRec'])

    # Compact model information
    angle_vec = angle_vector(pts[0]-pts[1])/90
    halign = 0  # left
    angle = 0
    dpt = Point(0, 0)
    if angle_vec == 0:  # horizontal
        halign = 2  # right
        angle = 0
        dpt = Point(0, 0.2*wg_width)
    if angle_vec == 2:  # horizontal
        halign = 0  # left
        angle = 0
        dpt = Point(0, 0.2*wg_width)
    if angle_vec == 1:  # vertical
        halign = 2  # right
        angle = 1
        dpt = Point(0.2*wg_width, 0)
    if angle_vec == -1:  # vertical
        halign = 0  # left
        angle = 1
        dpt = Point(0.2*wg_width, 0)
    pt2 = pts[0] + dpt
    pt3 = pts[0] - dpt
    pt4 = pts[0] - 6*dpt
    pt5 = pts[0] + 2*dpt
    pt6 = pts[0] - 2*dpt

    t = Trans(angle, False, pt3)
    import re
    CML = re.sub('design kits/', '', CML, flags=re.IGNORECASE)
#    CML = CML.lower().replace('design kits/','') # lower: to make it case insensitive, in case WAVEGUIDES.XML contains "Design Kits/" rather than "Design kits/"
    text = Text('Lumerical_INTERCONNECT_library=Design kits/%s' % CML, t, 0.1*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt2)
    text = Text('Component=%s' % model, t, 0.1*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt5)
    text = Text('cellName=%s' % cellName, t, 0.1*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pts[0])
    pts_txt = str([[round(p.to_dtype(dbu).x, 3), round(p.to_dtype(dbu).y, 3)]
                  for p in pts]).replace(', ', ',')
    text = Text(
        'Spice_param:wg_length=%.9f wg_width=%.3g points="%s" radius=%.3g' %
        (waveguide_length*1e-6, wg_width*1e-9, pts_txt, radius*1e-6), t, 0.1*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt4)
    text = Text(
        'Length=%.3f (microns)' % (waveguide_length), t, 0.5*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)
    t = Trans(angle, False, pt6)
    text = Text('waveguide_type=%s' % waveguide_type, t, 0.1*wg_width, -1)
    text.halign = halign
    shape = cell.shapes(LayerDevRecN).insert(text)

    return waveguide_length


'''
Create a waveguide, in a specific technology
inputs
- TECHNOLOGY, layout, cell:
    from SiEPIC.utils import get_layout_variables
    TECHNOLOGY, lv, layout, cell = get_layout_variables()
- layers: list of text names, e.g., ['Waveguide']
- widths: list of floats in units Microns, e.g., [0.50]
- offsets: list of floats in units Microns, e.g., [0]
- pts: a list of pya.Points, e.g. 
    L=15/dbu
    pts = [Point(0,0), Point(L,0), Point(L,L)]
- radius: in Microns, e.g., 5
- adiab: 1 = Bezier curve, 0 = radial bend (arc)
- bezier: the bezier parameter, between 0 and 0.45 (almost a radial bend)
- sbends (optional): sbends (Boolean)
Note: bezier parameters need to be simulated and optimized, and will depend on 
    wavelength, polarization, width, etc.  TM and rib waveguides don't benefit from bezier curves
    most useful for TE 
by Lukas Chrostowski
'''


def layout_waveguide2(TECHNOLOGY, layout, cell, layers, widths, offsets, pts, radius, adiab, bezier, sbends = True):
    from SiEPIC.utils import arc_xy, arc_bezier, angle_vector, angle_b_vectors, inner_angle_b_vectors, translate_from_normal
    from SiEPIC.extend import to_itype
    from SiEPIC.utils.geometry import bezier_parallel
    from pya import Path, Polygon, Trans

    dbu = layout.dbu

    if 'Errors' in TECHNOLOGY:
        error_layer = layout.layer(TECHNOLOGY['Errors'])
    else:
        error_layer = layout.layer('999/0')

    width = widths[0]
    turn = 0
    waveguide_length = 0
    for lr in range(0, len(layers)):
        wg_pts = [pts[0]]
        layer = layout.layer(TECHNOLOGY[layers[lr]])
        width = to_itype(widths[lr], dbu)
        offset = to_itype(offsets[lr], dbu)
        
        it = iter(range(1, len(pts)-1))
        for i in it:
            turn = ((angle_b_vectors(pts[i]-pts[i-1], pts[i+1]-pts[i])+90) % 360-90)/90
            dis1 = pts[i].distance(pts[i-1]) # before the "jog"
            dis2 = pts[i].distance(pts[i+1]) # the "jog"
            angle = angle_vector(pts[i]-pts[i-1])/90
            pt_radius = to_itype(radius, dbu)
            error_seg1 = False
            error_seg2 = False
            
            #determine if waveguide does an S-Shape
            if (sbends) and i < len(pts)-2:
                angle2 = angle_vector(pts[i+2]-pts[i+1])/90
                if angle == angle2 and dis2<2*pt_radius:  # An SBend may be inserted
                    dis3 = pts[i+2].distance(pts[i+1]) # after the "jog"
                    h = pts[i+1].y- pts[i].y if not (angle%2) else pts[i+1].x- pts[i].x
                    theta = m.acos(float(pt_radius-abs(h/2))/pt_radius)*180/pi
                    curved_l = int(2*pt_radius*sin(theta/180.0*pi))  
                    if (i < 3 and dis1 < curved_l/2) or (i > len(pts)-4 and dis3 < curved_l/2): 
                        pass    # Check if there is partial clearance for the bend when there is an end near
                    elif (i >= 3 and (dis1 - pt_radius < curved_l/2)) or (i <= len(pts)-4 and (dis3 - pt_radius < curved_l/2)): 
                        pass    # Check if there is full clearance for the bend
                    else:
                      if not (angle%2):
                        t = pya.Trans(angle, (angle == 2), pts[i].x+(angle-1)*int(curved_l/2), pts[i].y)  
                      else:
                        t = pya.Trans(angle, (angle == 1), pts[i].x, pts[i].y-(angle)*int(curved_l/2))
                      bend_pts = pya.DPath(bezier_parallel(pya.DPoint(0, 0), pya.DPoint(curved_l*dbu, h*dbu), 0),0).to_itype(dbu).transformed(t)
                      wg_pts += bend_pts.each_point()
                      turn = 0
                      
                      # Mark the start of the SBend with an "s"
                      tt = pya.Trans(pya.Trans.R0, 0,0)
                      text = pya.Text ("s", tt).transformed(t)
                      text.halign = pya.HAlign(1)
                      text.valign = pya.VAlign(1)
                      cell.shapes(layout.layer(TECHNOLOGY['Text'])).insert(text).text_size = 3/dbu
                      # Mark the start of the SBend with an "s"
                      tt = pya.Trans(pya.Trans.R0, curved_l, h)
                      text = pya.Text ("s", tt).transformed(t)
                      text.halign = pya.HAlign(1)
                      text.valign = pya.VAlign(1)
                      cell.shapes(layout.layer(TECHNOLOGY['Text'])).insert(text).text_size = 3/dbu
                      
                      i = next(it) # skip the step that was replaced by the SBend
                      continue
                        
            # determine the radius, based on how much space is available
            if len(pts) == 3:
                # simple corner, limit radius by the two edges
                if dis1 < pt_radius:
                    error_seg1 = True
                if dis2 < pt_radius:
                    error_seg2 = True
                pt_radius = min(dis1, dis2, pt_radius)
            else:
                if i == 1:
                    # first corner, limit radius by first edge, or 1/2 of second one
                    if dis1 < pt_radius:
                        error_seg1 = True
                    if dis2/2 < pt_radius:
                        error_seg2 = True
                    pt_radius = min(dis1, dis2/2, pt_radius)
                elif i == len(pts)-2:
                    # last corner, limit radius by second edge, or 1/2 of first one
                    if dis1/2 < pt_radius:
                        error_seg1 = True
                    if dis2 < pt_radius:
                        error_seg2 = True
                    pt_radius = min(dis1/2, dis2, pt_radius)
                else:
                    if dis1/2 < pt_radius:
                        error_seg1 = True
                    if dis2/2 < pt_radius:
                        error_seg2 = True
                    pt_radius = min(dis1/2, dis2/2, pt_radius)
            if error_seg1 or error_seg2:
                if error_layer == None:
                    # we have an error, but no Error layer
                    print('- SiEPIC:layout_waveguide2: missing Error layer')
                # and pt_radius < to_itype(radius,dbu):
                elif layer == layout.layer(TECHNOLOGY['Waveguide']):
                    # add an error polygon to flag the incorrect bend
                    if error_seg1:
                        error_pts = pya.Path([pts[i-1], pts[i]], width)
                        cell.shapes(error_layer).insert(error_pts)
                    if error_seg2:
                        error_pts = pya.Path([pts[i], pts[i+1]], width)
                        cell.shapes(error_layer).insert(error_pts)
    #                error_pts = pya.Path([pts[i-1], pts[i], pts[i+1]], width)
    #                cell.shapes(error_layer).insert(error_pts)
            # waveguide bends:
            if abs(turn) == 1:
                if(adiab):
                    wg_pts += Path(arc_bezier(pt_radius, 270, 270 + inner_angle_b_vectors(pts[i-1]-pts[i], pts[i+1]-pts[i]), float(
                        bezier), DevRec='DevRec' in layers[lr]), 0).transformed(Trans(angle, turn < 0, pts[i])).get_points()
                else:
                    wg_pts += Path(arc_xy(-pt_radius, pt_radius, pt_radius, 270, 270 + inner_angle_b_vectors(
                        pts[i-1]-pts[i], pts[i+1]-pts[i]), DevRec='DevRec' in layers[lr]), 0).transformed(Trans(angle, turn < 0, pts[i])).get_points()

        wg_pts += [pts[-1]]
        wg_pts = pya.Path(wg_pts, 0).unique_points().get_points()
        wg_polygon = Polygon(translate_from_normal(wg_pts, width/2 + (offset if turn > 0 else - offset)) +
                             translate_from_normal(wg_pts, -width/2 + (offset if turn > 0 else - offset))[::-1])
        cell.shapes(layer).insert(wg_polygon)

        if layout.layer(TECHNOLOGY['Waveguide']) == layer:
            waveguide_length = wg_polygon.area() / width * dbu

    return waveguide_length


def layout_waveguide(cell, layer, points_list, width):
    """ Lays out a waveguide (or trace) with a certain width with along given points.

    This is very useful for laying out Bezier curves with or without adiabatic tapers.

    Args:
        cell: cell to place into
        layer: layer to place into. It is done with cell.shapes(layer).insert(pya.Polygon)
        points_list: list of pya.DPoint (at least 2 points)
        width (microns): constant or list. If list, then it has to have the same length as points

    """
    if len(points_list) < 2:
        raise NotImplemented("ERROR: points_list too short")
        return

    if type(width) == type(0.0):
        width_iterator = repeat(width)
        points_iterator = iter(points_list)
    else:
        try:
            if len(width) == len(points_list):
                width_iterator = iter(width)
            else:
                width_iterator = repeat(width[0])
        except TypeError:
            width_iterator = repeat(width)
        finally:
            points_iterator = iter(points_list)

    dbu = cell.layout().dbu

    points_low = list()
    points_high = list()

    def norm(self):
        return sqrt(self.x**2 + self.y**2)

    def cos_angle(point1, point2):
        return point1 * point2 / norm(point1) / norm(point2)

    point_width_list = list(zip(points_iterator, width_iterator))
    N = len(point_width_list)

    first_point, first_width = point_width_list[0]
    next_point, next_width = point_width_list[1]

    delta = next_point - first_point
    theta = np.arctan2(delta.y, delta.x)
    first_high_point = first_point + 0.5 * first_width * \
        pya.DPoint(cos(theta + pi / 2), sin(theta + pi / 2))
    first_low_point = first_point + 0.5 * first_width * \
        pya.DPoint(cos(theta - pi / 2), sin(theta - pi / 2))
    points_high.append(first_high_point)
    points_low.append(first_low_point)

    for i in range(1, N - 1):
        prev_point, prev_width = point_width_list[i - 1]
        point, width = point_width_list[i]
        next_point, next_width = point_width_list[i + 1]

        delta_prev = point - prev_point
        delta_next = next_point - point
        theta_prev = np.arctan2(delta_prev.y, delta_prev.x)
        theta_next = np.arctan2(delta_next.y, delta_next.x)

        next_point_high = (next_point + 0.5 * next_width *
                           pya.DPoint(cos(theta_next + pi / 2), sin(theta_next + pi / 2)))
        next_point_low = (next_point + 0.5 * next_width *
                          pya.DPoint(cos(theta_next - pi / 2), sin(theta_next - pi / 2)))

        forward_point_high = (point + 0.5 * width *
                              pya.DPoint(cos(theta_next + pi / 2), sin(theta_next + pi / 2)))
        forward_point_low = (point + 0.5 * width *
                             pya.DPoint(cos(theta_next - pi / 2), sin(theta_next - pi / 2)))

        prev_point_high = (prev_point + 0.5 * prev_width *
                           pya.DPoint(cos(theta_prev + pi / 2), sin(theta_prev + pi / 2)))
        prev_point_low = (prev_point + 0.5 * prev_width *
                          pya.DPoint(cos(theta_prev - pi / 2), sin(theta_prev - pi / 2)))

        backward_point_high = (point + 0.5 * width *
                               pya.DPoint(cos(theta_prev + pi / 2), sin(theta_prev + pi / 2)))
        backward_point_low = (point + 0.5 * width *
                              pya.DPoint(cos(theta_prev - pi / 2), sin(theta_prev - pi / 2)))

        # High point decision
        next_high_edge = pya.DEdge(forward_point_high, next_point_high)
        prev_high_edge = pya.DEdge(backward_point_high, prev_point_high)

        if next_high_edge.crossed_by(prev_high_edge):
            intersect_point = next_high_edge.crossing_point(prev_high_edge)
            points_high.append(intersect_point)
        else:
            if width * (1 - cos_angle(delta_next, delta_prev)) > dbu:
                points_high.append(backward_point_high)
                points_high.append(forward_point_high)
            else:
                points_high.append((backward_point_high + forward_point_high) * 0.5)

        # Low point decision
        next_low_edge = pya.DEdge(forward_point_low, next_point_low)
        prev_low_edge = pya.DEdge(backward_point_low, prev_point_low)

        if next_low_edge.crossed_by(prev_low_edge):
            intersect_point = next_low_edge.crossing_point(prev_low_edge)
            points_low.append(intersect_point)
        else:
            if width * (1 - cos_angle(delta_next, delta_prev)) > dbu:
                points_low.append(backward_point_low)
                points_low.append(forward_point_low)
            else:
                points_low.append((backward_point_low + forward_point_low) * 0.5)

    last_point, last_width = point_width_list[-1]
    point, width = point_width_list[-2]
    delta = last_point - point
    theta = np.arctan2(delta.y, delta.x)
    final_high_point = last_point + 0.5 * last_width * \
        pya.DPoint(cos(theta + pi / 2), sin(theta + pi / 2))
    final_low_point = last_point + 0.5 * last_width * \
        pya.DPoint(cos(theta - pi / 2), sin(theta - pi / 2))
    if (final_high_point - points_high[-1]) * delta > 0:
        points_high.append(final_high_point)
    if (final_low_point - points_low[-1]) * delta > 0:
        points_low.append(final_low_point)

    # Append point only if change in direction is less than 120 degrees.
    def smooth_append(point_list, point):
#        if point_list is None:
#            print(point)
        if len(point_list) < 1:
            point_list.append(point)
            return point_list
        elif len(point_list) < 2:
            curr_edge = point - point_list[-1]
            if norm(curr_edge) > dbu:
                point_list.append(point)
                return point_list

        curr_edge = point - point_list[-1]
        if norm(curr_edge) > dbu:
            prev_edge = point_list[-1] - point_list[-2]
            if cos_angle(curr_edge, prev_edge) > cos(120 / 180 * pi):
                point_list.append(point)
        return point_list

    polygon_points = points_low + list(reversed(points_high))
    polygon_points = list(reduce(smooth_append, polygon_points, list()))

    poly = pya.DPolygon(polygon_points)
    cell.shapes(layer).insert(poly)


def layout_ring(cell, layer, center, r, w):
    # function to produce the layout of a ring
    # cell: layout cell to place the layout
    # layer: which layer to use
    # center: origin DPoint
    # r: radius
    # w: waveguide width
    # units in microns

    # example usage.  Places the ring layout in the presently selected cell.
    # cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    # layout_ring(cell, cell.layout().layer(LayerInfo(1, 0)), pya.DPoint(0,0), 10, 0.5)

    layout_arc(cell, layer, center, r, w, 0, 2 * np.pi)


def layout_arc(cell, layer, center, r, w, theta_start, theta_end, ex=None):
    # function to produce the layout of an arc
    # cell: layout cell to place the layout
    # layer: which layer to use
    # center: origin DPoint
    # r: radius
    # w: waveguide width
    # theta_start, theta_end: angle in radians
    # units in microns

    # example usage.  Places the ring layout in the presently selected cell.
    # cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    # layout_arc(cell, layer, pya.DPoint(0,0), 10, 0.5, 0, np.pi/2)

    # fetch the database parameters

    if ex is None:
        ex = pya.DPoint(1, 0)

    delta_theta = np.arctan2(ex.y, ex.x)
    theta_start += delta_theta
    theta_end += delta_theta

    # optimal sampling
    def arc_function(t): return np.array([center.x + r * np.cos(t), center.y + r * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [theta_start, theta_end], tol=0.002 / r)

    # # This yields a better polygon
    coords = np.insert(coords, 0, arc_function(theta_start - 0.001),
                       axis=1)  # start the waveguide a little bit before
    coords = np.append(coords, np.atleast_2d(arc_function(theta_end + 0.001)).T,
                       axis=1)  # finish the waveguide a little bit after

    layout_waveguide(cell, layer, [pya.DPoint(x, y) for x, y in zip(*coords)], w)


def layout_arc_drc_exclude(cell, drc_layer, center, r, w, theta_start, theta_end, ex=None):
    corner_points = [center + (r + w / 2) * rotate(ex, theta_start),
                     center + (r - w / 2) * rotate(ex, theta_start),
                     center + (r + w / 2) * rotate(ex, theta_end),
                     center + (r - w / 2) * rotate(ex, theta_end)]
    for corner_point in corner_points:
        layout_square(cell, drc_layer, corner_point, 0.1, ex)


def layout_arc_with_drc_exclude(cell, layer, drc_layer, center, r, w, theta_start, theta_end, ex=None):
    layout_arc(cell, layer, center, r, w, theta_start, theta_end, ex)
    layout_arc_drc_exclude(cell, drc_layer, center, r, w, theta_start, theta_end, ex)


def layout_circle(cell, layer, center, r):
    # function to produce the layout of a filled circle
    # cell: layout cell to place the layout
    # layer: which layer to use
    # center: origin DPoint
    # r: radius
    # w: waveguide width
    # theta_start, theta_end: angle in radians
    # units in microns
    # optimal sampling

    def arc_function(t): return np.array([center.x + r * np.cos(t), center.y + r * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [0, 2 * np.pi - 0.001], tol=0.002 / r)

    dbu = cell.layout().dbu
    dpoly = pya.DPolygon([pya.DPoint(x, y) for x, y in zip(*coords)])
    cell.shapes(layer).insert(dpoly.to_itype(dbu))


def layout_path(cell, layer, point_iterator, w):
    path = pya.DPath(list(point_iterator), w, 0, 0).to_itype(cell.layout().dbu)
    cell.shapes(layer).insert(pya.Path.from_dpath(path))


def layout_path_with_ends(cell, layer, point_iterator, w):
    dpath = pya.DPath(list(point_iterator), w, w / 2, w / 2)
    cell.shapes(layer).insert(dpath)


def box_dpolygon(point1, point3, ex=None):
    # position point2 to the right of point1
    if ex is None:
        ex = pya.DPoint(1, 0)
    ey = rotate90(ex)
    point2 = point1 * ex * ex + point3 * ey * ey
    point4 = point3 * ex * ex + point1 * ey * ey

    return pya.DPolygon([point1, point2, point3, point4])


def square_dpolygon(center, width, ex=None):
    # returns the polygon of a square centered at center,
    # aligned with ex, with width in microns
    if ex is None:
        ex = pya.DPoint(1, 0)
    ey = rotate90(ex)
    quadrant = (width / 2) * (ex + ey)
    point1 = center + quadrant
    quadrant = rotate90(quadrant)
    point2 = center + quadrant
    quadrant = rotate90(quadrant)
    point3 = center + quadrant
    quadrant = rotate90(quadrant)
    point4 = center + quadrant

    return pya.DPolygon([point1, point2, point3, point4])


def layout_square(cell, layer, center, width, ex=None):
    """ Lays out a square in the DRC layer

    Args:
        center: pya.DPoint (um units)
        width: float (um units)
        ex: orientation

    """

    if ex is None:
        ex = pya.DPoint(1, 0)

    square = square_dpolygon(center, width, ex)
    cell.shapes(layer).insert(square)


def layout_taper(cell, layer, trans, w1, w2, length, insert=True):
    """ Lays out a taper

    Args:
        trans: pya.Trans: location and rotation
        w1: width of waveguide, float for DPoint type (microns); int for Point type (nm)
        w2: width of waveguide, float for DPoint type (microns); int for Point type (nm)
        length: length, float
        insert: flag to insert drawn waveguide or return shape, boolean

    """
    import pya
    if type(w1) == type(float()):
        pts = [pya.DPoint(0, -w1/2), pya.DPoint(0, w1/2),
               pya.DPoint(length, w2/2), pya.DPoint(length, -w2/2)]
        shape_taper = pya.DPolygon(pts).transformed(trans)
    else:
        pts = [pya.Point(0, -w1/2), pya.Point(0, w1/2),
               pya.Point(length, w2/2), pya.Point(length, -w2/2)]
        shape_taper = pya.Polygon(pts).transformed(trans)

    if insert == True:
        cell.shapes(layer).insert(shape_taper)
    else:
        return shape_taper


def layout_waveguide_sbend_bezier(cell, layer, trans, w=0.5, wo=None, h=2.0, length=15.0, insert=True, debug=False):
    """ Creates a waveguide s-bend using a bezier curve
    Author: Lukas Chrostowski
    Args:
        trans: pya.Trans: location and rotation
        w: width of input waveguide, float for DPoint type (microns); int for Point type (nm)
        wo (optional): width of output waveguide, float
        h: height
        length: length
        insert: flag to insert drawn waveguide or return shape, boolean
    Usage:
        from SiEPIC.utils import get_layout_variables
        TECHNOLOGY, lv, ly, cell = get_layout_variables()
        layer = cell.layout().layer(TECHNOLOGY['Waveguide'])
        layout_waveguide_sbend_bezier(cell, layer, pya.Trans(), w=0.5, h=2.0, length=15.0, insert = True)
    """

    if wo == None:
        wo = w

    from SiEPIC.utils.geometry import bezier_parallel, translate_from_normal2
    from pya import DPoint, DPolygon, Point, Polygon

    if type(w) == type(int()):
        dbu = cell.layout().dbu
        w = w*dbu
        wo = wo*dbu
        h = h*dbu
        length = length*dbu
        trans = trans.to_dtype(dbu)

    p = bezier_parallel(DPoint(0, 0), DPoint(length, h), 0)

    pt1 = translate_from_normal2(p, w/2, wo/2)
    pt2 = translate_from_normal2(p, -w/2, -wo/2)
    pt = pt1+pt2[::-1]

    poly = pya.DPolygon(pt)
#    print(poly)
    poly_t = poly.transformed(trans)
    if insert == True:
        cell.shapes(layer).insert(poly_t)
        return poly_t.area()/((w+wo)/2)
    else:
        return poly_t


def layout_waveguide_sbend(cell, layer, trans, w=500, r=25000, h=2000, length=15000, insert=True):
    """ Lays out an s-bend

    Args:
        trans: pya.Trans: location and rotation
        w: width of waveguide, int
        r: radius, int
        h: height, int
        length: length, int
        insert: flag to insert drawn waveguide or return shape, boolean

    """

    from math import pi, cos, sin, log, sqrt, acos
    from SiEPIC.utils import points_per_circle
    import pya

    theta = acos(float(r-abs(h/2))/r)*180/pi
    x = int(2*r*sin(theta/180.0*pi))
    straight_l = int((length - x)/2)

    if (straight_l < 0):
        # Problem: target length is too short. increase
        print('SBend, too short: straight_l = %s' % straight_l)
        length = x
        straight_l = 0

    # waveguide_length = (2*pi*r*(2*theta/360.0)+straight_l*2)

    # define the cell origin as the left side of the waveguide sbend

    if (straight_l >= 0):
        circle_fraction = abs(theta) / 360.0
        npoints = int(points_per_circle(r*cell.layout().dbu) * circle_fraction)
        if npoints == 0:
            npoints = 1
        da = 2 * pi / npoints * circle_fraction  # increment, in radians
        x1 = straight_l
        x2 = length-straight_l

        if h > 0:
            y1 = r
            theta_start1 = 270
            y2 = h-r
            theta_start2 = 90
            pts = []
            th1 = theta_start1 / 360.0 * 2 * pi
            th2 = theta_start2 / 360.0 * 2 * pi
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, -w/2)))
            for i in range(0, npoints+1):  # lower left
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x1+(r+w/2)*cos(i*da+th1))/1, (y1+(r+w/2)*sin(i*da+th1))/1)))
            for i in range(npoints, -1, -1):  # lower right
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x2+(r-w/2)*cos(i*da+th2))/1, (y2+(r-w/2)*sin(i*da+th2))/1)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(length, h-w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(length, h+w/2)))
            for i in range(0, npoints+1):  # upper right
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x2+(r+w/2)*cos(i*da+th2))/1, (y2+(r+w/2)*sin(i*da+th2))/1)))
            for i in range(npoints, -1, -1):  # upper left
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x1+(r-w/2)*cos(i*da+th1))/1, (y1+(r-w/2)*sin(i*da+th1))/1)))
        else:
            y1 = -r
            theta_start1 = 90-theta
            y2 = r+h
            theta_start2 = 270-theta
            pts = []
            th1 = theta_start1 / 360.0 * 2 * pi
            th2 = theta_start2 / 360.0 * 2 * pi
            pts.append(pya.Point.from_dpoint(pya.DPoint(length, h-w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(length, h+w/2)))
            for i in range(npoints, -1, -1):  # upper right
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x2+(r-w/2)*cos(i*da+th2))/1, (y2+(r-w/2)*sin(i*da+th2))/1)))
            for i in range(0, npoints+1):  # upper left
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x1+(r+w/2)*cos(i*da+th1))/1, (y1+(r+w/2)*sin(i*da+th1))/1)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, -w/2)))
            for i in range(npoints, -1, -1):  # lower left
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x1+(r-w/2)*cos(i*da+th1))/1, (y1+(r-w/2)*sin(i*da+th1))/1)))
            for i in range(0, npoints+1):  # lower right
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x2+(r+w/2)*cos(i*da+th2))/1, (y2+(r+w/2)*sin(i*da+th2))/1)))

        shape_bend = pya.Polygon(pts).transformed(trans)
        if insert == True:
            cell.shapes(layer).insert(shape_bend)
        else:
            return shape_bend

#    print('SBend: theta %s, x %s, straight_l %s, r %s, h %s, length %s' %
#          (theta, x, straight_l, r, h, length))
    return length


def append_relative(points, *relative_vectors):
    """ Appends to list of points in relative steps """
    try:
        if len(points) > 0:
            origin = points[-1]
    except TypeError:
        raise TypeError("First argument must be a list of points")

    for vector in relative_vectors:
        points.append(origin + vector)
        origin = points[-1]
    return points


def place_cell(parent_cell, cell, placement_origin, params=None, relative_to=None):
    """ Places a cell and return ports
    Args:
        parent_cell: cell to place into
        cell: cell to be placed
        placement_origin: pya.Point object to be used as origin
        relative_to: port name

    Returns:
        ports(dict): key:port.name, value: geometry.Port with positions relative to parent_cell's origin
    """
    layout = parent_cell.layout()
    pcell, ports = cell.pcell(layout, params=params)
    if relative_to is not None:
        offset = next((port.position for port in ports if port.name == relative_to), None)
        placement_origin = placement_origin - offset
    parent_cell.insert(pya.CellInstArray(pcell.cell_index(),
                                         pya.Trans(pya.Trans.R0, placement_origin.to_itype(layout.dbu))))
    for port in ports:
        port.position += placement_origin

    return {port.name: port for port in ports}


def layout_connect_ports(cell, layer, port_from, port_to):

    P0 = port_from.position
    P3 = port_to.position
    angle_from = np.arctan2(port_from.direction.y, port_from.direction.x) * 180 / pi
    angle_to = np.arctan2(-port_to.direction.y, -port_to.direction.x) * 180 / pi

    curve = bezier_optimal(P0, P3, angle_from, angle_to)
    layout_waveguide(cell, layer, curve, [port_from.width, port_to.width])
    return curve_length(curve)


def make_pin(cell, name, center, w, layer, direction, debug=False):
    '''
    Makes a pin that SiEPIC-Tools will recognize
    cell: which cell to draw it in
    name: text label for the pin
    center: location, int [x,y]
    w: pin width
    layer: layout.layer() type
    direction = 
        0: right
        90: up
        180: left
        270: down

    Units: input can be float for microns, or int for nm
    '''


    from SiEPIC.extend import to_itype
    from pya import Point, DPoint
    import numpy
    dbu = cell.layout().dbu

#    if type(w) != type(center[0]):
#        raise Exception('SiEPIC.utils.layout.make_pin: mismatch in input types. center (%s) is %s, width (%s) is %s' % (center[0], type(center[0]), w, type(w)))

    if type(w) == type(float()):
        w = to_itype(w, dbu)
        if debug:
            print('SiEPIC.utils.layout.make_pin: w converted to %s' % w)
    else:
        if debug:
            print('SiEPIC.utils.layout.make_pin: w %s' % w)
#    print(type(center[0]))
    if type(center) == type(Point()) or type(center) == type(DPoint()):
        center = [center.x, center.y]

    if type(center[0]) == type(float()) or type(center[0]) == type(numpy.float64()):
        center[0] = to_itype(center[0], dbu)
        center[1] = to_itype(center[1], dbu)
        if debug:
            print('SiEPIC.utils.layout.make_pin: center converted to %s' % (center))
    else:
        if debug:
            print('SiEPIC.utils.layout.make_pin: center %s' % (center))

    from SiEPIC._globals import PIN_LENGTH as pin_length

    direction = direction % 360
    if direction not in [0, 90, 180, 270]:
        raise Exception('error in make_pin: direction (%s) must be one of [0, 90, 180, 270]' % direction )

    # text label
    t = pya.Trans(pya.Trans.R0, center[0], center[1])
    text = pya.Text(name, t)
    shape = cell.shapes(layer).insert(text)
    shape.text_dsize = float(w*dbu/2)
    shape.text_valign = 1

    if direction == 0:
        p1 = pya.Point(center[0]-pin_length/2, center[1])
        p2 = pya.Point(center[0]+pin_length/2, center[1])
        shape.text_halign = 2
    if direction == 90:
        p1 = pya.Point(center[0], center[1]-pin_length/2)
        p2 = pya.Point(center[0], center[1]+pin_length/2)
        shape.text_halign = 2
        shape.text_rot = 1
    if direction == 180:
        p1 = pya.Point(center[0]+pin_length/2, center[1])
        p2 = pya.Point(center[0]-pin_length/2, center[1])
        shape.text_halign = 3
    if direction == 270:
        p1 = pya.Point(center[0], center[1]+pin_length/2)
        p2 = pya.Point(center[0], center[1]-pin_length/2)
        shape.text_halign = 3
        shape.text_rot = 1

    pin = pya.Path([p1, p2], w)
    cell.shapes(layer).insert(pin)


def make_devrec_label(cell, libname, devname, layer, x=0, y=0, text_size=0.5):
    """
    Generate a SiEPIC-Tools DevRec label for the device and library name.

    Mustafa Hammood, 2022
    TODO: add SPICE params generator

    Parameters
    ----------
    cell : pya cell
        Cell to place the label in.
    libname : string
        Name of the library.
    devname : string
        Name of the device.
    layer : pya lyayer
        devRec layer.
    x : float, optional
        X position of the label, microns. The default is 0.
    y : TYPE, optional
        Y position of the label, microns. The default is 0.
    text_size : float, optional
        Text size of the label. The default is 0.5.

    Returns
    -------
    None.

    """
    dbu = cell.layout().dbu
    shapes = cell.shapes
    # Compact model information
    text_size = 0.1/dbu
    t = pya.Trans(pya.Trans.R0, x, y)
    text = pya.Text('Lumerical_INTERCONNECT_library=Design kits/'+libname, t)
    shape = shapes(layer).insert(text)
    shape.text_size = text_size

    t = pya.Trans(pya.Trans.R0, x, y+text_size)
    text = pya.Text('Component='+devname, t)
    shape = shapes(layer).insert(text)
    shape.text_size = text_size


def y_splitter_tree(cell, tree_depth=4, y_splitter_cell="y_splitter_1310", library="SiEPICfab_Shuksan_PDK", wg_type='Strip TE 1310 nm, w=350 nm', draw_waveguides=True):
    '''
    Create a tree of splitters
    - cell: layout cell to create the structures in
    - tree_depth: Tree depth (2^N outputs)
    - y_splitter_cell: name of the y-splitter cell
    - library: the library containing the y_splitter_cell
    - wg_type: waveguide type from WAVEGUIDES.XML
    - draw_waveguides: True draws the waveguides, False is faster for debugging

    Returns
    - inst_in: instance of the input cell
    - inst_out[]: array of instances of the output cells
    - cell_tree: new cell created
    This is useful for subsequent routing

    Limitations:
    - the design uses regular 90 degree bends, rather than S-bends.
      hence it could be made more compact
    '''

    from SiEPIC.scripts import connect_pins_with_waveguide
    from SiEPIC.extend import to_itype
    from math import floor

    # create a new sub-cell where the tree will go
    ly = cell.layout()
    tech = ly.technology().name
    cell_tree = ly.create_cell("y_splitter_tree")

    # load the y-splitter from the library
    if type(y_splitter_cell)==pya.Cell:
        y_splitter=y_splitter_cell
    else:
        y_splitter = ly.create_cell(y_splitter_cell, library)
    if not y_splitter:
        raise Exception('Cannot import cell %s:%s' % (library, y_splitter_cell))

    # Load waveguide information
    from SiEPIC.utils import load_Waveguides_by_Tech
    waveguides = load_Waveguides_by_Tech(tech)
    wg = [w for w in waveguides if wg_type in w['name']][0]
    if not wg:
        raise Exception("Waveguide type not defined in WAVEGUIDES.XML: %s" % wg_type)
        return
    wg_width = to_itype(float(wg['width']), ly.dbu)
    wg_radius = to_itype(float(wg['radius']), ly.dbu)

    # build the tree, using measurements from the cell and waveguide parameters
    x = 0
    dx = y_splitter.bbox().width() + wg_radius*2
    # calculate the spacing for the y-splitters based on waveguide radius and 90 degree bends
    y_wg_offset = (y_splitter.pinPoint("opt2").y-y_splitter.pinPoint("opt3").y)
    dy = max(y_splitter.bbox().height(), wg_radius*4 + y_wg_offset)
    # intialize loop
    inst_out = []
    y0 = 0
    for i in range(0, tree_depth):
        inst = []
        y = y0
        for j in range(0, int(2**(tree_depth-i-1))):
            t = pya.Trans(pya.Trans.R0, x, y)
            inst.append(cell_tree.insert(pya.CellInstArray(y_splitter.cell_index(), t)))
            # perform waveguide routing
            if (i > 0) and draw_waveguides:
                connect_pins_with_waveguide(
                    inst[j], 'opt2',
                    inst_higher[j*2+1], 'opt1', waveguide_type=wg_type)
                connect_pins_with_waveguide(
                    inst[j], 'opt3',
                    inst_higher[j*2], 'opt1', waveguide_type=wg_type)
            y += dy
        inst_higher = inst
        if i == 0:
            inst_out = inst
        if i == tree_depth-1:
            inst_in = inst[0]
        x += -dx
        y0 = y0 + dy/2
        dy = dy * 2

    return inst_in, inst_out, cell_tree



def floorplan(topcell, x, y):
    '''Create a FloorPlan, from (0,0) to (x,y)
    by Lukas Chrostowski, 2023, SiEPIC-Tools
    '''
    ly = topcell.layout()
    cell = ly.create_cell('FloorPlan')
    topcell.insert(pya.CellInstArray(cell.cell_index(), pya.Vector(0,0)))
    box = pya.Box(0,0,x,y)
    cell.shapes(ly.layer(ly.TECHNOLOGY['FloorPlan'])).insert(box)


def new_layout(tech, topcell_name, GUI=True, overwrite = False):
    '''Create a new layout either in KLayout Application mode or in PyPI mode.
    Create the layout in the Application MainWindow (GUI=True) or in memory only (GUI=False)
    in Application mode, 
      (overwrite = False): creates a new Layout View
      (overwrite = True): clears the existing layout, 
        only if the topcell_name matches the existing one
    by Lukas Chrostowski, 2023, SiEPIC-Tools
    '''
    
    from SiEPIC.utils import get_layout_variables, get_technology_by_name
    from SiEPIC._globals import Python_Env

    # this script can be run inside KLayout's GUI application, or
    # or from the command line: klayout -zz -r H3LoQP.py
    if Python_Env == "KLayout_GUI" and GUI:
        mw = pya.Application().instance().main_window()
        if overwrite and mw.current_view() \
                and mw.current_view().active_cellview().layout().top_cells():
            TECHNOLOGY, lv, ly, cell = get_layout_variables()
            if topcell_name in [n.name for n in ly.top_cells()]:
                # only overwrite if the layout has a matching topcell name
                ly.delete_cells([c.cell_index() for c in ly.cells('*')])
            else:
                ly = mw.create_layout(tech, 1).layout()
        else:
            ly = mw.create_layout(tech, 1).layout()
        topcell = ly.create_cell(topcell_name)
        lv = mw.current_view()
        lv.select_cell(topcell.cell_index(), 0)
    else:
        ly = pya.Layout()
        ly.technology_name = tech
        topcell = ly.create_cell(topcell_name)
    ly.TECHNOLOGY = get_technology_by_name(tech)

    return topcell, ly

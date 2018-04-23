""" Layout helper functions.

Author: Thomas Ferreira de Lima @thomaslima

The following functions are useful for scripted layout, or making
PDK Pcells.

TODO: enhance documentation
TODO: make some of the functions in util use these.
"""

from itertools import repeat
import pya
import numpy as np
from numpy import cos, sin, pi, sqrt
from functools import reduce
from . import sample_function
from .geometry import rotate90, rotate, bezier_optimal, curve_length


def layout_waveguide(cell, layer, points_list, width):
    """ Lays out a waveguide (or trace) with a certain width along given points.

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
        if point_list is None:
            print(point)
        if len(point_list) < 1:
            point_list.append(point)
            return point_list
        elif len(point_list) < 2:
            curr_edge = point - point_list[-1]
            if norm(curr_edge) >= dbu:
                point_list.append(point)
                return point_list

        curr_edge = point - point_list[-1]
        if norm(curr_edge) >= dbu:
            prev_edge = point_list[-1] - point_list[-2]
            if cos_angle(curr_edge, prev_edge) > cos(120 / 180 * pi):
                point_list.append(point)
        return point_list

    polygon_points = points_low + list(reversed(points_high))
    polygon_points = list(reduce(smooth_append, polygon_points, list()))

    poly = pya.DPolygon(polygon_points)
    cell.shapes(layer).insert(poly)


def layout_waveguide_angle(cell, layer, points_list, width, angle):
    """ Lays out a waveguide (or trace) with a certain width along
    given points and with fixed orientation at all points.

    This is very useful for laying out Bezier curves with or without adiabatic tapers.

    Args:
        cell: cell to place into
        layer: layer to place into. It is done with cell.shapes(layer).insert(pya.Polygon)
        points_list: list of pya.DPoint (at least 2 points)
        width (microns): constant or list. If list, then it has to have the same length as points
        angle (degrees)
    """
    if len(points_list) < 2:
        raise NotImplemented("ERROR: points_list too short")
        return

    try:
        if len(width) == len(points_list):
            width_iterator = iter(width)
        else:
            width_iterator = repeat(width[0])
    except TypeError:
        width_iterator = repeat(width)
    finally:
        points_iterator = iter(points_list)

    theta = angle * pi / 180

    points_low = list()
    points_high = list()

    point_width_list = list(zip(points_iterator, width_iterator))
    N = len(point_width_list)

    for i in range(0, N):
        point, width = point_width_list[i]
        point_high = (point + 0.5 * width *
                      pya.DPoint(cos(theta + pi / 2), sin(theta + pi / 2)))
        points_high.append(point_high)
        point_low = (point + 0.5 * width *
                     pya.DPoint(cos(theta - pi / 2), sin(theta - pi / 2)))
        points_low.append(point_low)

    polygon_points = points_low + list(reversed(points_high))

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
    arc_function = lambda t: np.array([center.x + r * np.cos(t), center.y + r * np.sin(t)])
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

    arc_function = lambda t: np.array([center.x + r * np.cos(t), center.y + r * np.sin(t)])
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


def layout_connect_ports(cell, layer, port_from, port_to):

    P0 = port_from.position
    P3 = port_to.position
    angle_from = np.arctan2(port_from.direction.y, port_from.direction.x) * 180 / pi
    angle_to = np.arctan2(-port_to.direction.y, -port_to.direction.x) * 180 / pi

    curve = bezier_optimal(P0, P3, angle_from, angle_to)
    layout_waveguide(cell, layer, curve, [port_from.width, port_to.width])
    return curve_length(curve)

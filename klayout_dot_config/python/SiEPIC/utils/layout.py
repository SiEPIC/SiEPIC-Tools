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

debug = False


def insert_shape(cell, layer, shape):
    if layer is not None:
        cell.shapes(layer).insert(shape)


class DSimplePolygon(pya.DSimplePolygon):

    def transform_and_rotate(self, center, ex=None):
        if ex is None:
            ex = pya.DPoint(1, 0)
        ey = rotate90(ex)

        polygon_dpoints_transformed = [center + p.x *
                                       ex + p.y * ey for p in self.each_point()]
        self.assign(pya.DSimplePolygon(polygon_dpoints_transformed))
        return self

    def clip(self, x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
        # Add points exactly at the boundary, so that the filter below works.
        x_bounds = (np.min(x_bounds), np.max(x_bounds))
        y_bounds = (np.min(y_bounds), np.max(y_bounds))

        check_within_bounds = lambda p: x_bounds[0] <= p.x and x_bounds[1] >= p.x and \
            y_bounds[0] <= p.y and y_bounds[1] >= p.y

        def intersect_left_boundary(p1, p2, x_bounds, y_bounds):
            left_most, right_most = (p1, p2) if p1.x < p2.x else (p2, p1)
            bottom_most, top_most = (p1, p2) if p1.y < p2.y else (p2, p1)
            if left_most.x < x_bounds[0]:
                # intersection only if right_most crosses x_bound[0]
                if right_most.x > x_bounds[0]:
                    # outside the box, on the left
                    y_intersect = np.interp(x_bounds[0], [left_most.x, right_most.x], [
                                            left_most.y, right_most.y])
                    if y_bounds[0] < y_intersect and y_bounds[1] > y_intersect:
                        return pya.DPoint(float(x_bounds[0]), float(y_intersect))
            return None

        def intersect(p1, p2, x_bounds, y_bounds):
            intersect_list = list()
            last_intersect = None

            def rotate_bounds90(x_bounds, y_bounds, i_times):
                for i in range(i_times):
                    x_bounds, y_bounds = (-y_bounds[1], -y_bounds[0]), (x_bounds[0], x_bounds[1])
                return x_bounds, y_bounds

            for i in range(4):
                p1i, p2i = rotate(p1, i * pi / 2), rotate(p2, i * pi / 2)
                x_boundsi, y_boundsi = rotate_bounds90(x_bounds, y_bounds, i)
                p = intersect_left_boundary(p1i, p2i, x_boundsi, y_boundsi)
                if p is not None:
                    last_intersect = i
                    intersect_list.append(rotate(p, -i * pi / 2))
            return intersect_list, last_intersect

        polygon_dpoints_clipped = list()
        polygon_dpoints = list(self.each_point())

        def boundary_vertex(edge_from, edge_to):
            # left edge:0, top edge:1 etc.
            # returns the vertex between two edges
            assert abs(edge_from - edge_to) == 1
            if edge_from % 2 == 0:
                vertical_edge = edge_from
                horizontal_edge = edge_to
            else:
                vertical_edge = edge_to
                horizontal_edge = edge_from
            x = x_bounds[(vertical_edge // 2) % 2]
            y = y_bounds[((horizontal_edge - 1) // 2) % 2]
            return pya.DPoint(x, y)

        # Rotate point list so we can start from a point inside
        # (helps the boundary_vertex algorithm)
        for idx, point in enumerate(polygon_dpoints):
            if check_within_bounds(point):
                break
        else:
            # polygon was never within bounds
            # this can only happen if boundaries are finite
            # return boundary vertices
            boundary_vertices = [boundary_vertex(i, i - 1) for i in range(4, 0, -1)]
            self.assign(pya.DSimplePolygon(boundary_vertices))
            return self

        idx += 1  # make previous_point below already be inside
        polygon_dpoints = polygon_dpoints[idx:] + polygon_dpoints[:idx]

        previous_point = polygon_dpoints[-1]
        previous_intersect = None
        for point in polygon_dpoints:
            # compute new intersecting point and add to list
            intersected_points, last_intersect = intersect(
                previous_point, point, x_bounds, y_bounds)
            if previous_intersect is not None and last_intersect is not None and \
                    last_intersect != previous_intersect:
                if check_within_bounds(point):
                    # this means that we are entering the box at a different edge
                    # need to add the edge points

                    # this assumes a certain polygon orientation
                    # assume points go counterlockwise, which means that
                    # from edge 0 to 2, it goes through 3
                    i = previous_intersect
                    while i % 4 != last_intersect:
                        polygon_dpoints_clipped.append(boundary_vertex(i, i - 1))
                        i = i - 1
            polygon_dpoints_clipped.extend(intersected_points)
            if check_within_bounds(point):
                polygon_dpoints_clipped.append(point)
            previous_point = point
            if last_intersect is not None:
                previous_intersect = last_intersect
        self.assign(pya.DSimplePolygon(polygon_dpoints_clipped))
        return self

    def layout(self, cell, layer):
        return insert_shape(cell, layer, self)

    def layout_drc_exclude(self, cell, drclayer, ex):
        """ Places a drc exclude square at every corner.
        A corner is defined by an angle greater than 30 degrees (conservative)
        """
        if drclayer is not None:
            points = list(self.each_point())
            assert len(points) > 3
            prev_delta = points[-1] - points[-2]
            prev_angle = np.arctan2(prev_delta.y, prev_delta.x)
            for i in range(len(points)):
                delta = points[i] - points[i - 1]
                angle = np.arctan2(delta.y, delta.x)
                if delta.y == 0 or delta.x == 0:
                    thresh_angle = pi / 2
                else:
                    thresh_angle = pi * 85 / 180
                delta_angle = angle - prev_angle
                delta_angle = abs(((delta_angle + pi) % (2 * pi)) - pi)
                if delta_angle > thresh_angle:
                    layout_square(cell, drclayer, points[i - 1], 0.1, ex)
                prev_delta, prev_angle = delta, angle

    def resize(self, dx, dbu, magic_flag=False):
        dpoly = pya.DPolygon(self)
        dpoly.size(dx, 5)
        dpoly = pya.EdgeProcessor().simple_merge_p2p([dpoly.to_itype(dbu)], False, False, 1)
        dpoly = dpoly[0].to_dtype(dbu)  # pya.DPolygon

        def norm(p):
            return sqrt(p.x**2 + p.y**2)

        # Filter edges if they are too small
        points = list(dpoly.each_point_hull())
        new_points = list([points[0]])
        for i in range(0, len(points)):
            delta = points[i] - new_points[-1]
            if norm(delta) > min(10 * dbu, abs(dx)):
                new_points.append(points[i])

        sdpoly = DSimplePolygon(new_points)  # convert to SimplePolygon
        self.assign(sdpoly)
        return self

    def round_corners(self, radius, N):
        """ This only works if the polygon points are sparse."""

        dpoly = super().round_corners(radius, radius, N)
        self.assign(dpoly)
        return self


def waveguide_dpolygon(points_list, width, dbu, smooth=True):
    """ Returns a polygon outlining a waveguide.

    Args:
        cell: cell to place into
        points_list: list of pya.DPoint (at least 2 points)
        width (microns): constant or list. If list, then it has to have the same length as points
    Returns:
        polygon DPoints

    """
    if len(points_list) < 2:
        raise NotImplementedError("ERROR: points_list too short")
        return

    def norm(self):
        return sqrt(self.x**2 + self.y**2)

    try:
        if len(width) == len(points_list):
            width_iterator = iter(width)
        elif len(width) == 2:
            # assume width[0] is initial width and
            # width[1] is final width
            # interpolate with points_list
            L = curve_length(points_list)
            distance = 0
            widths_list = [width[0]]
            widths_func = lambda t: (1 - t) * width[0] + t * width[1]
            old_point = points_list[0]
            for point in points_list[1:]:
                distance += norm(point - old_point)
                old_point = point
                widths_list.append(widths_func(distance / L))
            width_iterator = iter(widths_list)
        else:
            width_iterator = repeat(width[0])
    except TypeError:
        width_iterator = repeat(width)
    finally:
        points_iterator = iter(points_list)

    points_low = list()
    points_high = list()

    def cos_angle(point1, point2):
        cos_angle = point1 * point2 / norm(point1) / norm(point2)

        # ensure it's between -1 and 1 (nontrivial numerically)
        if abs(cos_angle) > 1:
            return cos_angle / abs(cos_angle)
        else:
            return cos_angle

    def sin_angle(point1, point2):
        return sin(np.arccos(cos_angle(point1, point2)))

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

        fix_angle = lambda theta: ((theta + pi) % (2 * pi)) - pi

        # High point decision
        next_high_edge = pya.DEdge(forward_point_high, next_point_high)
        prev_high_edge = pya.DEdge(backward_point_high, prev_point_high)

        if next_high_edge.crossed_by(prev_high_edge):
            intersect_point = next_high_edge.crossing_point(prev_high_edge)
            points_high.append(intersect_point)
        else:
            cos_dd = cos_angle(delta_next, delta_prev)
            if width * (1 - cos_dd) > dbu and fix_angle(theta_next - theta_prev) < 0:
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
            cos_dd = cos_angle(delta_next, delta_prev)
            if width * (1 - cos_dd) > dbu and fix_angle(theta_next - theta_prev) > 0:
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

    # Append point only if change in direction is less than 130 degrees.
    def smooth_append(point_list, point):
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

            if norm(prev_edge) * abs(sin_angle(curr_edge + prev_edge, prev_edge)) > dbu:
                if smooth:
                    # avoid corners when smoothing
                    if cos_angle(curr_edge, prev_edge) > cos(130 / 180 * pi):
                        point_list.append(point)
                    else:
                        # edge case when there is prev_edge is small and
                        # needs to be deleted to get rid of the corner
                        if norm(curr_edge) > norm(prev_edge):
                            point_list[-1] = point
                else:
                    point_list.append(point)
            # avoid unnecessary points
            else:
                point_list[-1] = point
        return point_list

    if debug and False:
        print("Points to be smoothed:")
        for point, width in point_width_list:
            print(point, width)

    smooth_points_high = list(reduce(smooth_append, points_high, list()))
    smooth_points_low = list(reduce(smooth_append, points_low, list()))
    # smooth_points_low = points_low
    # polygon_dpoints = points_high + list(reversed(points_low))
    # polygon_dpoints = list(reduce(smooth_append, polygon_dpoints, list()))
    polygon_dpoints = smooth_points_high + list(reversed(smooth_points_low))
    return DSimplePolygon(polygon_dpoints)


def layout_waveguide(cell, layer, points_list, width, smooth=False):
    """ Lays out a waveguide (or trace) with a certain width along given points.

    This is very useful for laying out Bezier curves with or without adiabatic tapers.

    Args:
        cell: cell to place into
        layer: layer to place into. It is done with cell.shapes(layer).insert(pya.Polygon)
        points_list: list of pya.DPoint (at least 2 points)
        width (microns): constant or list. If list, then it has to have the same length as points

    """

    dbu = cell.layout().dbu

    dpolygon = waveguide_dpolygon(points_list, width, dbu, smooth=smooth)
    dpolygon.compress(True)
    dpolygon.layout(cell, layer)
    return dpolygon


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

    def norm(self):
        return sqrt(self.x**2 + self.y**2)

    try:
        if len(width) == len(points_list):
            width_iterator = iter(width)
        elif len(width) == 2:
            # assume width[0] is initial width and
            # width[1] is final width
            # interpolate with points_list
            L = curve_length(points_list)
            distance = 0
            widths_list = [width[0]]
            widths_func = lambda t: (1 - t) * width[0] + t * width[1]
            old_point = points_list[0]
            for point in points_list[1:]:
                distance += norm(point - old_point)
                old_point = point
                widths_list.append(widths_func(distance / L))
            width_iterator = iter(widths_list)
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

    polygon_points = points_high + list(reversed(points_low))

    poly = pya.DSimplePolygon(polygon_points)
    cell.shapes(layer).insert(poly)


def layout_disk(cell, layer, center, r):
    # function to produce the layout of a disk
    # cell: layout cell to place the layout
    # layer: which layer to use
    # center: origin DPoint
    # r: radius
    # units in microns

    # outer arc
    # optimal sampling
    radius = r
    assert radius > 0
    arc_function = lambda t: np.array([radius * np.cos(t), radius * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [0, 2 * pi], tol=0.002 / radius)

    # create original waveguide poligon prior to clipping and rotation
    points_hull = [center + pya.DPoint(x, y) for x, y in zip(*coords)]
    del points_hull[-1]

    dpoly = pya.DPolygon(points_hull)
    insert_shape(cell, layer, dpoly)
    return dpoly


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

    # outer arc
    # optimal sampling
    assert r - w / 2 > 0
    radius = r + w / 2
    arc_function = lambda t: np.array([radius * np.cos(t), radius * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [0, 2 * pi], tol=0.002 / radius)

    # create original waveguide poligon prior to clipping and rotation
    points_hull = [center + pya.DPoint(x, y) for x, y in zip(*coords)]
    del points_hull[-1]

    radius = r - w / 2
    arc_function = lambda t: np.array([radius * np.cos(t), radius * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [0, 2 * pi], tol=0.002 / radius)

    # create original waveguide poligon prior to clipping and rotation
    points_hole = [center + pya.DPoint(x, y) for x, y in zip(*coords)]
    del points_hole[-1]

    dpoly = pya.DPolygon(list(reversed(points_hull)))
    dpoly.insert_hole(points_hole)
    dpoly.compress(True)
    insert_shape(cell, layer, dpoly)
    return dpoly


def layout_arc(cell, layer, center, r, w, theta_start, theta_end, ex=None,
               x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
    """ function to produce the layout of an arc"""
    # cell: layout cell to place the layout
    # layer: which layer to use
    # center: origin DPoint (not affected by ex)
    # r: radius
    # w: waveguide width
    # theta_start, theta_end: angle in radians
    # x_bounds and y_bounds relative to the center, before rotation by ex.
    # units in microns
    # returns a dpolygon

    # example usage.  Places the ring layout in the presently selected cell.
    # cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    # layout_arc(cell, layer, pya.DPoint(0,0), 10, 0.5, 0, np.pi/2)

    # fetch the database parameters

    if r <= 0:
        raise RuntimeError(f"Please give me a positive radius. Bad r={r}")

    # optimal sampling
    if theta_end < theta_start:
        theta_start, theta_end = theta_end, theta_start

    arc_function = lambda t: np.array([r * np.cos(t), r * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [theta_start, theta_end], tol=0.002 / r)

    # # This yields a better polygon
    coords = np.insert(coords, 0, arc_function(theta_start - 0.001),
                       axis=1)  # start the waveguide a little bit before
    coords = np.append(coords, np.atleast_2d(arc_function(theta_end + 0.001)).T,
                       axis=1)  # finish the waveguide a little bit after

    # create original waveguide poligon prior to clipping and rotation
    dpoints_list = [pya.DPoint(x, y) for x, y in zip(*coords)]
    dpolygon = waveguide_dpolygon(dpoints_list, w, cell.layout().dbu)

    # clip dpolygon to bounds
    dpolygon.clip(x_bounds=x_bounds, y_bounds=y_bounds)
    # Transform points (translation + rotation)
    dpolygon.transform_and_rotate(center, ex)
    dpolygon.compress(True)
    dpolygon.layout(cell, layer)
    return dpolygon


def layout_arc2(cell, layer, center, r1, r2, theta_start, theta_end, ex=None,
                x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
    r1, r2 = min(r1, r2), max(r1, r2)

    r = (r1 + r2) / 2
    w = (r2 - r1)
    return layout_arc(cell, layer, center, r, w, theta_start, theta_end,
                      ex=ex, x_bounds=x_bounds, y_bounds=y_bounds)


def layout_section(cell, layer, center, r2, theta_start, theta_end, ex=None,
                   x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):

    assert r2 > 0

    # optimal sampling
    arc_function = lambda t: np.array([r2 * np.cos(t), r2 * np.sin(t)])
    t, coords = sample_function(arc_function,
                                [theta_start, theta_end], tol=0.002 / r2)

    # # This yields a better polygon
    if theta_end < theta_start:
        theta_start, theta_end = theta_end, theta_start

    coords = np.insert(coords, 0, arc_function(theta_start - 0.001),
                       axis=1)  # start the waveguide a little bit before
    coords = np.append(coords, np.atleast_2d(arc_function(theta_end + 0.001)).T,
                       axis=1)  # finish the waveguide a little bit after

    # create original waveguide poligon prior to clipping and rotation
    dpoints_list = [pya.DPoint(x, y) for x, y in zip(*coords)]
    dpolygon = DSimplePolygon(dpoints_list + [pya.DPoint(0, 0)])

    # clip dpolygon to bounds
    dpolygon.clip(x_bounds=x_bounds, y_bounds=y_bounds)
    # Transform points (translation + rotation)
    dpolygon.transform_and_rotate(center, ex)
    dpolygon.compress(True)
    dpolygon.layout(cell, layer)
    return dpolygon


def layout_arc_drc_exclude(cell, drc_layer, center, r, w, theta_start, theta_end, ex=None):
    corner_points = [center + (r + w / 2) * rotate(ex, theta_start),
                     center + (r - w / 2) * rotate(ex, theta_start),
                     center + (r + w / 2) * rotate(ex, theta_end),
                     center + (r - w / 2) * rotate(ex, theta_end)]
    for corner_point in corner_points:
        layout_square(cell, drc_layer, corner_point, 0.1, ex)


def layout_arc_with_drc_exclude(cell, layer, drc_layer, center, r, w, theta_start, theta_end, ex=None, **kwargs):
    dpoly = layout_arc(cell, layer, center, r, w, theta_start, theta_end, ex, **kwargs)
    dpoly.layout_drc_exclude(cell, drc_layer, ex)
    return dpoly


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
    dpoly = pya.DSimplePolygon([pya.DPoint(x, y) for x, y in zip(*coords)])
    cell.shapes(layer).insert(dpoly.to_itype(dbu))


def layout_path(cell, layer, point_iterator, w):
    path = pya.DPath(list(point_iterator), w, 0, 0).to_itype(cell.layout().dbu)
    cell.shapes(layer).insert(pya.Path.from_dpath(path))


def layout_path_with_ends(cell, layer, point_iterator, w):
    dpath = pya.DPath(list(point_iterator), w, w / 2, w / 2)
    cell.shapes(layer).insert(dpath)


def box_dpolygon(point1, point3, ex):
    # position point2 to the right of point1
    ey = rotate90(ex)
    point2 = point1 * ex * ex + point3 * ey * ey
    point4 = point3 * ex * ex + point1 * ey * ey

    return DSimplePolygon([point1, point2, point3, point4])


def layout_box(cell, layer, point1, point3, ex):
    """ Lays out a box

    Args:
        point1: bottom-left point
        point3: top-right point

    """

    box = box_dpolygon(point1, point3, ex)
    insert_shape(cell, layer, box)
    return box


def rectangle_dpolygon(center, width, height, ex):
    # returns the polygon of a rectangle centered at center,
    # aligned with ex, with width and height in microns

    ey = rotate90(ex)

    point1 = center - width / 2 * ex - height / 2 * ey
    point3 = center + width / 2 * ex + height / 2 * ey

    return box_dpolygon(point1, point3, ex=ex)


def square_dpolygon(center, width, ex=None):
    # returns the polygon of a square centered at center,
    # aligned with ex, with width in microns

    return rectangle_dpolygon(center, width, width, ex=ex)


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
    insert_shape(cell, layer, square)
    return square


def layout_rectangle(cell, layer, center, width, height, ex):
    """ Lays out a rectangle

    Args:
        center: pya.DPoint (um units)
        width: float (um units)
        height: float (um unit)
        ex: orientation

    """

    rectangle = rectangle_dpolygon(center, width, height, ex=ex)
    insert_shape(cell, layer, rectangle)
    return rectangle


def append_relative(points, *relative_vectors):
    """ Appends to list of points in relative steps:
        It takes a list of points, and adds new points to it in relative coordinates.
        For example, if you call append_relative([A, B], C, D), the result will be [A, B, B+C, B+C+D].
    """
    try:
        if len(points) > 0:
            origin = points[-1]
    except TypeError:
        raise TypeError("First argument must be a list of points")

    for vector in relative_vectors:
        points.append(origin + vector)
        origin = points[-1]
    return points


def layout_connect_ports(cell, layer, port_from, port_to, smooth=True):

    if port_from.name.startswith("el"):
        assert port_to.name.startswith("el")
        P0 = port_from.position + port_from.direction * port_from.width / 2
        P3 = port_to.position + port_to.direction * port_to.width / 2
        smooth = smooth and True
    else:
        dbu = cell.layout().dbu
        P0 = port_from.position - dbu * port_from.direction
        P3 = port_to.position - dbu * port_to.direction
        smooth = smooth or True
    angle_from = np.arctan2(port_from.direction.y, port_from.direction.x) * 180 / pi
    angle_to = np.arctan2(-port_to.direction.y, -port_to.direction.x) * 180 / pi

    curve = bezier_optimal(P0, P3, angle_from, angle_to)
    if debug:
        for point in curve:
            print(point)
        print(f"bezier_optimal({P0}, {P3}, {angle_from}, {angle_to})")
    return layout_waveguide(cell, layer, curve, [port_from.width, port_to.width], smooth=smooth)


def layout_connect_ports_angle(cell, layer, port_from, port_to, angle):

    if port_from.name.startswith("el"):
        assert port_to.name.startswith("el")
        P0 = port_from.position + port_from.direction * port_from.width / 2
        P3 = port_to.position + port_to.direction * port_to.width / 2

        # straight lines for electrical connectors
        curve = [P0, P3]
    else:
        P0 = port_from.position
        P3 = port_to.position
        curve = bezier_optimal(P0, P3, angle, angle)

    return layout_waveguide_angle(cell, layer, curve, [port_from.width, port_to.width], angle)


def layout_text(cell, layer_text, position, text_string, size):
    dtext = pya.DText(str(text_string), pya.DTrans(
        pya.DTrans.R0, position.x, position.y))
    dtext.size = size
    cell.shapes(layer_text).insert(dtext)

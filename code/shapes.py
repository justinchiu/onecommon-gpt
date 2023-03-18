import shapely
from shapely import Point, MultiPoint
import numpy as np
import math

def is_contiguous(dots, all_dots):
    dots = np.stack(dots)
    xy = dots[:,:2]
    all_xy = all_dots[:,:2]
    hull = MultiPoint(xy).convex_hull
    contiguous = not any([
        shapely.contains_xy(hull, x, y)
        for x,y in all_xy
    ])
    return contiguous

def is_line(dots, all_dots):
    if len(dots) < 2: return False
    if len(dots) == 2: return True

    dots = np.stack(dots)
    xy = dots[:,:2]
    rect = MultiPoint(xy).minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x, y = rect.exterior.coords.xy

    # get length of bounding box edges
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )

    # get length of polygon as the longest edge of the bounding box
    length = max(edge_length)

    # get width of polygon as the shortest edge of the bounding box
    width = min(edge_length)

    # check angle later
    return math.fabs(math.atan(length / width)) < math.pi / 15

def is_triangle(dots, all_dots):
    line = is_line(dots, all_dots)
    return not line and len(dots) == 3 and is_contiguous(dots, all_dots)


def is_square(dots, all_dots):
    return len(dots) == 4 and is_contiguous(dots, all_dots)

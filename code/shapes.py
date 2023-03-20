import shapely
from shapely import Point, MultiPoint
import numpy as np
import math

def is_nearest(x, ctx):
    raise NotImplementedError

def is_contiguous(x, ctx):
    xy = ctx[x,:2]
    all_xy = ctx[:,:2]
    hull = MultiPoint(xy).convex_hull
    contiguous = not any([
        shapely.contains_xy(hull, x, y)
        for x,y in all_xy
    ])
    return contiguous

def is_line(x, ctx):
    if len(x) < 2: return False
    if len(x) == 2: return True

    xy = ctx[x,:2]
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

def is_triangle(x, ctx):
    line = is_line(x, ctx)
    return not line and len(x) == 3 and is_contiguous(x, ctx)


def is_square(x, ctx):
    return len(x) == 4 and is_contiguous(x, ctx)

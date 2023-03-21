import shapely
from shapely import MultiPoint, Point
import numpy as np
import math

# validate
def all_close(idxs, ctx):
    xy = ctx[idxs,:2]

    rect = MultiPoint(xy).minimum_rotated_rectangle

    minx, miny, maxx, maxy = rect.bounds

    width = maxx - minx
    height = maxy - miny
    diagonal = math.hypot(width, height)

    # HARD CODE, try to be a bit generous
    return diagonal < 0.3

def are_close(x, y, ctx):
    return np.linalg.norm(ctx[x,:2]-ctx[y,:2]) < 0.3

def are_direction(x, y, direction, ctx):
    # Relative to view: always True
    if y is None:
        return True
    deltas = ctx[x,None,:2] - ctx[y,:2]
    return ((deltas * direction) >= 0).all()

def are_above(x, y, ctx):
    return are_direction(x, y, np.array([0,1]), ctx)

def are_below(x, y, ctx):
    return are_direction(x, y, np.array([0,-1]), ctx)

def are_right(x, y, ctx):
    return are_direction(x, y, np.array([1,0]), ctx)

def are_left(x, y, ctx):
    return are_direction(x, y, np.array([-1,0]), ctx)

def are_above_right(x, y, ctx):
    return are_direction(x, y, np.array([1,1]), ctx)

def are_above_left(x, y, ctx):
    return are_direction(x, y, np.array([-1,1]), ctx)

def are_below_right(x, y, ctx):
    return are_direction(x, y, np.array([1,-1]), ctx)

def are_below_left(x, y, ctx):
    return are_direction(x, y, np.array([-1,-1]), ctx)

def are_middle(x, y, ctx):
    # Relative to view: always True
    if y is None:
        return True

    # x in convex hull of y
    xy_a = ctx[x,:2]
    xy_b = ctx[y,:2]
    hull = MultiPoint(xy_b).convex_hull
    # very generous
    return all([
        shapely.contains_xy(hull, x, y)
        or shapely.contains_xy(hull, x+0.01, y)
        or shapely.contains_xy(hull, x-0.01, y)
        or shapely.contains_xy(hull, x, y+0.01)
        or shapely.contains_xy(hull, x, y-0.01)
        for x,y in xy_a
    ])

# getters
def get_magnitude(x, direction, ctx):
    xy = ctx[x,:2]
    delta = xy[:,None] - xy[None]
    # magnitude: target x source
    return (delta @ direction).sum(-1)

def get_direction(x, direction, ctx):
    magnitude = get_magnitude(x, direction, ctx)
    return x[magnitude.argmax()]

def get_top(x, ctx):
    return get_direction(x, np.array([0,1]), ctx)

def get_bottom(x, ctx):
    return get_direction(x, np.array([0,-1]), ctx)

def get_right(x, ctx):
    return get_direction(x, np.array([1,0]), ctx)

def get_left(x, ctx):
    return get_direction(x, np.array([-1,0]), ctx)

def get_top_right(x, ctx):
    return get_direction(x, np.array([1,1]), ctx)

def get_top_left(x, ctx):
    return get_direction(x, np.array([-1,1]), ctx)

def get_bottom_right(x, ctx):
    return get_direction(x, np.array([1,-1]), ctx)

def get_bottom_left(x, ctx):
    return get_direction(x, np.array([-1,-1]), ctx)

def get_middle(x, ctx):
    # x in convex hull of y
    xy = ctx[x,:2]
    centroid = MultiPoint(xy).centroid
    return x[
        np.linalg.norm(
            xy - np.array([centroid.x, centroid.y]),
            axis=1,
        ).argmin()
    ]

# global functions
def global_top(x, ctx):
    return True
def global_bottom(x, ctx):
    return True
def global_left(x, ctx):
    return True
def global_right(x, ctx):
    return True
def global_top_left(x, ctx):
    return True
def global_top_right(x, ctx):
    return True
def global_bottom_left(x, ctx):
    return True
def global_bottom_right(x, ctx):
    return True
def global_middle(x, ctx):
    return True

if __name__ == "__main__":
    ctx = np.array([[1,1], [1,0], [0,0], [0,0]])
    x = [0,1]
    y = [2,3]
    assert are_above(x, y, ctx)
    assert are_below(y, x, ctx)
    assert are_right(x, y, ctx)
    assert are_left(y, x, ctx)

    x = [0]
    y = [2]
    assert are_above_right(x, y, ctx)
    assert are_below_left(y, x, ctx)

    ctx = np.array([[0,0], [0,1], [1,-1], [-1,-1]])
    x = [0]
    y = [1,2,3]
    assert are_middle(x, y, ctx)

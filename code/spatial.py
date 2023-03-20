import shapely
from shapely import MultiPoint, Point
import numpy as np
import math

# validate
def all_close(dots):
    dots = np.array(dots)
    xy = dots[:,:2]

    rect = MultiPoint(xy).minimum_rotated_rectangle

    minx, miny, maxx, maxy = rect.bounds

    width = maxx - minx
    height = maxy - miny
    diagonal = math.hypot(width, height)

    # HARD CODE, try to be a bit generous
    return diagonal < 0.3

def is_close(x, y):
    return np.linalg.norm(x[:2]-y[:2]) < 0.3

def is_direction(dots_a, dots_b, direction):
    deltas = dots_a[:,None] - dots_b
    return ((deltas * direction) >= 0).all()

def is_above(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([0,1]))

def is_below(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([0,-1]))

def is_right(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([1,0]))

def is_left(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([-1,0]))

def is_above_right(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([1,1]))

def is_above_left(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([-1,1]))

def is_below_right(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([1,-1]))

def is_below_left(dots_a, dots_b):
    return is_direction(dots_a, dots_b, np.array([-1,-1]))

def is_middle(dots_a, dots_b):
    # dots_a in convex hull of dots_b
    xy_a = dots_a[:,:2]
    xy_b = dots_b[:,:2]
    hull = MultiPoint(xy_b).convex_hull
    return all([
        shapely.contains_xy(hull, x, y)
        for x,y in xy_a
    ])

# getters
def get_magnitude(dots, direction):
    xy = dots[:,:2]
    delta = xy[:,None] - xy[None]
    # magnitude: target x source
    return (delta @ direction).sum(-1)

def get_direction(dots, direction):
    magnitude = get_magnitude(dots, direction)
    return dots[magnitude.argmax()]

def get_top(dots):
    return get_direction(dots, np.array([0,1]))

def get_bottom(dots):
    return get_direction(dots, np.array([0,-1]))

def get_right(dots):
    return get_direction(dots, np.array([1,0]))

def get_left(dots):
    return get_direction(dots, np.array([-1,0]))

def get_top_right(dots):
    return get_direction(dots, np.array([1,1]))

def get_top_left(dots):
    return get_direction(dots, np.array([-1,1]))

def get_bottom_right(dots):
    return get_direction(dots, np.array([1,-1]))

def get_bottom_left(dots):
    return get_direction(dots, np.array([-1,-1]))

if __name__ == "__main__":
    dots_a = np.array([[1,1], [1,0]])
    dots_b = np.array([[0,0], [0,0]])
    assert is_above(dots_a, dots_b)
    assert is_below(dots_b, dots_a)

    dots_a = np.array([[1,1], [1,0]])
    dots_b = np.array([[0,0], [0,0]])
    assert is_right(dots_a, dots_b)
    assert is_left(dots_b, dots_a)

    dots_a = np.array([[1,1]])
    dots_b = np.array([[0,0]])
    assert is_above_right(dots_a, dots_b)
    assert is_below_left(dots_b, dots_a)

    dots_a = np.array([[0,0]])
    dots_b = np.array([[0,1], [1,-1], [-1,-1]])
    assert is_middle(dots_a, dots_b)

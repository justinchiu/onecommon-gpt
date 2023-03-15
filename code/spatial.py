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

def is_above(dot, dots):
    raise NotImplementedError

def is_below(dot, dots):
    raise NotImplementedError

def is_right(dot, dots):
    raise NotImplementedError

def is_left(dot, dots):
    raise NotImplementedError

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


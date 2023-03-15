from shapely import MultiPoint, Point
import numpy as np
import math

# validate
def is_close(dots):
    dots = np.array(dots)
    xy = dots[:,:2]

    rect = MultiPoint(xy).minimum_rotated_rectangle

    minx, miny, maxx, maxy = rect.bounds

    width = maxx - minx
    height = maxy - miny
    diagonal = math.hypot(width, height)

    # HARD CODE
    return diagonal < 0.3

def is_above(dots):
    pass

def is_below(dots):
    pass

def is_right(dots):
    pass

def is_left(dots):
    pass

# getters
def get_top(dots):
    pass

def get_bottom(dots):
    pass

def get_right(dots):
    pass

def get_left(dots):
    pass

def get_top_right(dots):
    pass

def get_top_left(dots):
    pass

def get_bottom_right(dots):
    pass

def get_bottom_left(dots):
    pass


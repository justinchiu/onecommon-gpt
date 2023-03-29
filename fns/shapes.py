import shapely
from shapely import Point, MultiPoint
import numpy as np
import math

from iterators import getcombs

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

def unit_vector(vector, axis=None):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angles(xys):
    num_dots = xys.shape[0]
    pairs = [
        [tgt for tgt in range(num_dots) if src != tgt]
        for src in range(num_dots)
    ]
    xy_pairs = xys[np.array(pairs)]
    diffs = xys[:,None] - xy_pairs
    diffs = unit_vector(diffs, 1)
    return np.arccos(np.clip(
        (diffs[:,0] * diffs[:,1]).sum(-1),
        -1, 1
    ))

def is_line(x, ctx):
    if len(x) < 2: return False
    if len(x) == 2: return True

    xy = ctx[x,:2]

    angles = get_angles(xy)
    return max(angles) * 180 / math.pi > 135


def is_triangle(x, ctx):
    if len(x) != 3: return False

    # only take most compact triangles
    radii = []
    dots = []
    #for idxs in get3idxs(list(range(7))):
    for idxs in getcombs(list(range(7)), 3):
        if not is_line(idxs, ctx) and is_contiguous(idxs, ctx):
            mp = MultiPoint(ctx[idxs,:2])
            radius = shapely.minimum_bounding_radius(mp)
            radii.append(radius)
            dots.append((idxs))
    # 3 smallest triangles. 6 permutations of 3 dots
    dotset = np.array(dots)[np.argsort(radii)[:18]]
    return (dotset == x).all(-1).any()


def is_square(x, ctx):
    return len(x) == 4 and is_contiguous(x, ctx)

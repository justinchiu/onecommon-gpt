
import sys
sys.path.append("fns")

from oc.fns.context import get_ctx
from oc.fns.shapes import is_triangle, is_line, is_square
from oc.fns.spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from oc.fns.spatial import get_top, get_bottom, get_right, get_left
from oc.fns.spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from oc.fns.spatial import get_middle
from oc.fns.spatial import get_distance, get_minimum_radius
from oc.fns.color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from oc.fns.size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from oc.fns.iterators import get1idxs, get2idxs, get3idxs, getsets
from oc.fns.lists import add
from oc.fns.lists import sort_state
import numpy as np
from functools import partial
from itertools import permutations


def get_ctx():
    ctx = np.array([[0.125, -0.815, -1.0, -0.8933333333333333], [-0.21, 0.585, 0.3333333333333333, -0.9733333333333334], [0.645, 0.185, -1.0, -0.96], [0.305, 0.645, -1.0, -0.9733333333333334], [-0.705, 0.015, 0.0, 0.84], [0.345, -0.545, 0.6666666666666666, -0.9066666666666666], [-0.315, 0.165, 0.6666666666666666, 0.8]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = []



print([tuple(x) for x in state])

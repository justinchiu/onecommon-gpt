

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
    ctx = np.array([[-0.46, 0.44, 1.0, -0.09333333333333334], [0.21, -0.53, 0.3333333333333333, 0.32], [0.63, 0.255, -0.3333333333333333, -0.5466666666666666], [-0.095, -0.635, 0.3333333333333333, -0.32], [-0.03, -0.9, 0.3333333333333333, 0.5733333333333334], [0.475, 0.62, -0.6666666666666666, -0.5333333333333333], [0.565, 0.0, 0.3333333333333333, -0.76]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = [(2, 6), (2, 5)]


# Turn 0
# Them: Is there a small size and dark color dot to the left and above those?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b, in permutations(config):
            for c, in get1idxs(idxs, exclude=[a,b,]):
                check_c_small = is_small(c, ctx)
                check_c_dark = is_dark(c, ctx)
                check_c_left_a = is_left(c, a, ctx)
                
                if (
                    True 
                    and check_c_small
                    and check_c_dark
                    and check_c_left_a
                    
                ):
                    dots = frozenset([a,b,c,])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)


if state is not None:
    print([tuple(x) for x in state])
else:
    print("None")


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
    ctx = np.array([[0.125, 0.615, 0.3333333333333333, 0.28], [0.45, 0.18, -0.3333333333333333, -0.41333333333333333], [-0.605, 0.625, -0.3333333333333333, -0.4], [-0.01, -0.32, 0.6666666666666666, 0.7066666666666667], [-0.38, -0.02, -0.6666666666666666, 0.16], [0.125, 0.895, 0.6666666666666666, 0.6533333333333333], [-0.22, -0.325, 0.6666666666666666, 0.7733333333333333]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = None


# Turn 0
# Them: I have a black medium dot and small light dot to its left.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 2):
        for a,b, in permutations(config):
            for _ in [0]:
                check_a_medium = is_medium_size(a, ctx)
                check_a_dark = is_dark(a, ctx)
                check_b_small = is_small(b, ctx)
                check_b_light = is_light(b, ctx)
                check_b_left = b == get_left([a,b], ctx)
                
                if (
                    True 
                    and check_a_medium
                    and check_a_dark
                    and check_b_small
                    and check_b_light
                    and check_b_left
                    
                ):
                    dots = frozenset([a,b,])
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


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
    ctx = np.array([[0.23, -0.95, 0.0, 0.22666666666666666], [-0.96, 0.255, -0.6666666666666666, -0.18666666666666668], [-0.47, 0.02, -0.3333333333333333, -1.0], [-0.01, -0.295, 0.0, -0.41333333333333333], [-0.51, 0.215, 0.6666666666666666, 0.28], [-0.375, 0.815, 0.6666666666666666, 0.10666666666666667], [0.535, 0.69, 0.3333333333333333, 0.48]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = None


# Turn 0
# Them: Pair with large grey on top and black small on bottom.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 2):
        for a,b, in permutations(config):
            for _ in [0]:
                check_ab_pair = all_close([a,b], ctx)
                check_a_large = is_large(a, ctx)
                check_a_grey = is_grey(a, ctx)
                check_a_top = a == get_top([a,b], ctx)
                check_b_small = is_small(b, ctx)
                check_b_dark = is_dark(b, ctx)
                check_b_bottom = b == get_bottom([a,b], ctx)
                
                if (
                    True 
                    and check_ab_pair
                    and check_a_large
                    and check_a_grey
                    and check_a_top
                    and check_b_small
                    and check_b_dark
                    and check_b_bottom
                    
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
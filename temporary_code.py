

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
    ctx = np.array([[-0.635, 0.685, -0.6666666666666666, 0.3466666666666667], [0.035, 0.225, -0.6666666666666666, 0.7733333333333333], [0.02, -0.085, 0.3333333333333333, 0.12], [0.81, 0.565, -0.6666666666666666, 0.6133333333333333], [0.685, -0.39, -1.0, 0.64], [0.48, -0.41, -0.6666666666666666, 0.02666666666666667], [0.015, -0.68, 0.3333333333333333, 0.25333333333333335]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = None


# Turn 0
# Them: Do you see a pair of dots, where the top left dot is small-sized and light and the bottom right dot is small-sized and light?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 2):
        for a,b, in permutations(config):
            for _ in [0]:
                check_ab_pair = all_close([a,b], ctx)
                check_a_top_left = a == get_top_left([a,b], ctx)
                check_a_small = is_small(a, ctx)
                check_a_light = is_light(a, ctx)
                check_b_bottom_right = b == get_bottom_right([a,b], ctx)
                check_b_small = is_small(b, ctx)
                check_b_light = is_light(b, ctx)
                
                if (
                    True 
                    and check_ab_pair
                    and check_a_top_left
                    and check_a_small
                    and check_a_light
                    and check_b_bottom_right
                    and check_b_small
                    and check_b_light
                    
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
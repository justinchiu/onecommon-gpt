

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
    ctx = np.array([[0.42, -0.605, -0.3333333333333333, 0.49333333333333335], [-0.55, 0.03, 1.0, 0.9866666666666667], [-0.22, 0.47, 0.3333333333333333, 0.84], [0.53, -0.19, -0.6666666666666666, -0.7066666666666667], [-0.525, -0.8, -0.3333333333333333, 0.06666666666666667], [0.61, 0.755, -0.6666666666666666, 0.08], [0.7, -0.105, 0.6666666666666666, 0.8133333333333334]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = None


# Turn 0
# Them: Do you see a lone small grey dot?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 1):
        for a, in permutations(config):
            for _ in [0]:
                check_a_small = is_small(a, ctx)
                check_a_grey = is_grey(a, ctx)
                
                if (
                    True 
                    and check_a_small
                    and check_a_grey
                    
                ):
                    dots = frozenset([a,])
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
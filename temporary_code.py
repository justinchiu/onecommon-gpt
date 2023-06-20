

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
    ctx = np.array([[-0.735, -0.46, -1.0, -0.8533333333333334], [0.535, 0.275, -1.0, 0.8533333333333334], [-0.005, 0.455, 1.0, 0.7333333333333333], [0.72, 0.095, -0.3333333333333333, 0.7066666666666667], [0.205, 0.775, 0.0, -0.25333333333333335], [0.72, -0.5, 0.0, -0.18666666666666668], [-0.32, -0.825, -0.3333333333333333, -0.49333333333333335]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = [(2, 3, 5)]


# Turn 0
# Them: Let's select the medium size and grey color one. <selection>.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c, in permutations(config):
            for _ in [0]:
                check_a_medium = is_medium_size(a, ctx)
                check_a_grey = is_grey(a, ctx)
                
                if (
                    True 
                    and check_a_medium
                    and check_a_grey
                    
                ):
                    dots = frozenset([a])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=True)
state = turn(state)


if state is not None:
    print([tuple(x) for x in state])
else:
    print("None")
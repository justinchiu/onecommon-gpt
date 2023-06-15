

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
    ctx = np.array([[0.675, 0.125, -0.3333333333333333, 0.7466666666666667], [0.83, 0.03, -1.0, 0.37333333333333335], [0.485, 0.65, 0.6666666666666666, -0.72], [-0.105, 0.04, 0.6666666666666666, 0.9733333333333334], [0.215, -0.345, 0.0, 0.9333333333333333], [-0.42, -0.57, 0.6666666666666666, 0.17333333333333334], [-0.68, 0.11, 0.6666666666666666, -0.84]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = [(0, 2, 4)]


# Turn 0
# Them: Let's select the large size and dark color one. <selection>
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c, in permutations(config):
            for _ in [0]:
                check_a_large = is_large(a, ctx)
                check_a_dark = is_dark(a, ctx)
                
                if (
                    True 
                    and check_a_large
                    and check_a_dark
                    
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
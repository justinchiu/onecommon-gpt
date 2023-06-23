

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
    ctx = np.array([[-0.57, -0.55, 1.0, -0.76], [-0.88, -0.47, -0.3333333333333333, -0.17333333333333334], [0.54, -0.19, -1.0, -0.5733333333333334], [-0.195, 0.735, -1.0, -0.6666666666666666], [0.22, -0.745, 0.3333333333333333, -0.52], [-0.005, -0.435, 1.0, 0.10666666666666667], [0.795, 0.355, 0.6666666666666666, 0.49333333333333335]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
state = [(0, 2, 5)]


# Turn 0
# Them: Let's select the large size and grey color one. <selection>.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c, in permutations(config):
            for _ in [0]:
                check_a_large = is_large(a, ctx)
                check_a_grey = is_grey(a, ctx)
                
                if (
                    True 
                    and check_a_large
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
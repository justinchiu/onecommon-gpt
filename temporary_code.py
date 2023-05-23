

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
    ctx = np.array([[0.645, 0.33, 0.3333333333333333, -0.88], [0.5, -0.505, 0.6666666666666666, -0.9733333333333334], [-0.275, 0.505, 0.3333333333333333, -0.6133333333333333], [-0.24, 0.105, -0.6666666666666666, 0.10666666666666667], [-0.63, 0.585, -1.0, -0.3466666666666667], [-0.59, 0.04, 0.0, -0.013333333333333334], [-0.245, -0.855, -0.6666666666666666, -0.37333333333333335]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
states = []


# Turn 0
# You: Do you see a pair of dots, where the right dot is medium-sized and dark and the left dot is small-sized and dark?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_xy_pair = all_close([x, y], ctx)
                check_x_left = is_left(x, y, ctx)
                check_x_small = is_small(x, ctx)
                check_x_dark = is_dark(x, ctx)
                check_y_right = is_right(y, x, ctx)
                check_y_medium = is_medium_size(y, ctx)
                check_y_dark = is_dark(y, ctx)
                if (
                    True 
                    and check_xy_pair
                    and check_x_left
                    and check_x_small
                    and check_x_dark
                    and check_y_right
                    and check_y_medium
                    and check_y_dark
                    
                ):
                    dots = frozenset([x,y])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 1
# Them: Yes.
def turn(state): return None
state = None if len(states) > 0 else None
states.append(turn(state))


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
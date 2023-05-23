

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
states = []


# Turn 0
# You: Do you see a pair of dots, where the top dot is small-sized and light and the bottom dot is medium-sized and grey?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_xy_pair = all_close([x,y], ctx)
                check_x_top = x == get_top([x, y], ctx)
                check_x_small = is_small(x, ctx)
                check_x_light = is_light(x, ctx)
                check_y_bottom = y == get_bottom([x, y], ctx)
                check_y_medium = is_medium_size(y, ctx)
                check_y_grey = is_grey(y, ctx)
                if (
                    True 
                    and check_xy_pair
                    and check_x_top
                    and check_x_small
                    and check_x_light
                    and check_y_bottom
                    and check_y_medium
                    and check_y_grey
                    
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

# Turn 2
# You: Is there a small size and light color dot to the right of those?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Follow up question, new dot.
    for config in state:
        for a, b in permutations(config):
            for x, in get1idxs(idxs, exclude=[a, b]):
                check_x_right_ab = x == get_right([a, b, x], ctx)
                check_x_small = is_small(x, ctx)
                check_x_light = is_light(x, ctx)
                if (
                    True 
                    and check_x_right_ab
                    and check_x_small
                    and check_x_light
                    
                ):
                    dots = frozenset([a,b,x,])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = states[0] if len(states) > 0 else None
states.append(turn(state))

# Turn 3
# Them: Yes.
def turn(state): return None
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 4
# You: Let's select the medium size and grey color one. <selection>.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Select a dot.
    for config in state:
        for a, b, x in permutations(config):
            for _ in [0]:
                check_b_medium = is_medium_size(b, ctx)
                check_b_grey = is_grey(b, ctx)
                if (
                    True 
                    and check_b_medium
                    and check_b_grey
                    
                ):
                    dots = frozenset([b])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=True)
state = states[2] if len(states) > 0 else None
states.append(turn(state))


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
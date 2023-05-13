

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
    ctx = np.array([[-0.635, -0.685, -0.6666666666666666, 0.3466666666666667], [0.035, -0.225, -0.6666666666666666, 0.7733333333333333], [0.02, 0.085, 0.3333333333333333, 0.12], [0.81, -0.565, -0.6666666666666666, 0.6133333333333333], [0.685, 0.39, -1.0, 0.64], [0.48, 0.41, -0.6666666666666666, 0.02666666666666667], [0.015, 0.68, 0.3333333333333333, 0.25333333333333335]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
states = []


# Turn 0
# You: Do you see a pair of dots, where the bottom left dot is small-sized and light, and the top right dot is small-sized and light?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_xy_pair = all_close([x,y], ctx)
                check_x_bottom_left = x == get_bottom_left([x, y], ctx)
                check_x_small = is_small(x, ctx)
                check_x_light = is_light(x, ctx)
                check_y_top_right = y == get_top_right([x, y], ctx)
                check_y_small = is_small(y, ctx)
                check_y_light = is_light(y, ctx)
                if (
                    True 
                    and check_xy_pair
                    and check_x_bottom_left
                    and check_x_small
                    and check_x_light
                    and check_y_top_right
                    and check_y_small
                    and check_y_light
                    
                ):
                    dots = frozenset([x, y])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 1
# Them: Yes.
def turn(state): return [None]
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 2
# You: Is there a medium-sized and grey-colored dot above those?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Follow up question, new dot.
    for config in state:
        for a, b in permutations(config):
            for x, in get1idxs(idxs, exclude=[a, b]):
                check_x_medium = is_medium_size(x, ctx)
                check_x_grey = is_grey(x, ctx)
                check_x_above_ab = is_above(x, [a, b], ctx)
                if (
                    True 
                    and check_x_medium
                    and check_x_grey
                    and check_x_above_ab
                    
                ):
                    dots = frozenset([a, b,x,])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = states[0] if len(states) > 0 else None
states.append(turn(state))

# Turn 3
# Them: Yes.
def turn(state): return [None]
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 4
# Them: Let's select the medium size and grey color one above.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Select a dot.
    for config in state:
        for a, b, x in permutations(config):
            for _ in [0]:
                check_x_medium = is_medium_size(x, ctx)
                check_x_grey = is_grey(x, ctx)
                check_x_above_ab = is_above(x, [a, b], ctx)
                if (
                    True 
                    and check_x_medium
                    and check_x_grey
                    and check_x_above_ab
                    
                ):
                    dots = frozenset([x])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=True)
state = states[2] if len(states) > 0 else None
states.append(turn(state))


print([tuple(x) for x in states[-1]])
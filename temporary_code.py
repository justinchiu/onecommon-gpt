

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
    ctx = np.array([[0.565, -0.205, 1.0, -0.9866666666666667], [0.385, 0.225, 1.0, 0.96], [-0.09, 0.19, 0.0, -0.9333333333333333], [0.855, -0.45, -0.3333333333333333, -0.36], [-0.42, 0.46, 0.0, 0.6933333333333334], [0.195, 0.635, -0.6666666666666666, 0.05333333333333334], [0.2, -0.235, -0.3333333333333333, 0.7866666666666666]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
states = []


# Turn 0
# You: Do you see a pair of dots, where the right dot is large-sized and dark, and the left dot is small-sized and light?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_xy_pair = all_close([x,y], ctx)
                check_x_right = is_right(x, y, ctx)
                check_x_large = is_large(x, ctx)
                check_x_dark = is_dark(x, ctx)
                check_y_left = is_left(y, x, ctx)
                check_y_small = is_small(y, ctx)
                check_y_light = is_light(y, ctx)
                if (
                    True 
                    and check_xy_pair
                    and check_x_right
                    and check_x_large
                    and check_x_dark
                    and check_y_left
                    and check_y_small
                    and check_y_light
                    
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
# You: Is there a large size and light color dot above those?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Follow up question, new dot.
    for config in state:
        for a, b in permutations(config):
            for x, in get1idxs(idxs, exclude=[a, b]):
                check_x_large = is_large(x, ctx)
                check_x_light = is_light(x, ctx)
                check_x_above_ab = is_above(x, [a, b], ctx)
                if (
                    True 
                    and check_x_large
                    and check_x_light
                    and check_x_above_ab
                    
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
# You: Let's select the one that is large in size and dark in color. <selection>.
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Select a dot.
    for config in state:
        for a, b in permutations(config):
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
state = states[2] if len(states) > 0 else None
states.append(turn(state))


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
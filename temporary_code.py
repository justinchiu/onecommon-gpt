

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
    ctx = np.array([[0.125, -0.815, -1.0, -0.8933333333333333], [-0.21, 0.585, 0.3333333333333333, -0.9733333333333334], [0.645, 0.185, -1.0, -0.96], [0.305, 0.645, -1.0, -0.9733333333333334], [-0.705, 0.015, 0.0, 0.84], [0.345, -0.545, 0.6666666666666666, -0.9066666666666666], [-0.315, 0.165, 0.6666666666666666, 0.8]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
states = []


# Turn 0
# You: Do you see a pair of dots, where the top dot is medium-sized and dark and the bottom dot is large-sized and light?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_x_top = x == get_top([x, y], ctx)
                check_x_medium = is_medium_size(x, ctx)
                check_x_dark = is_dark(x, ctx)
                check_y_bottom = y == get_bottom([x, y], ctx)
                check_y_large = is_large(y, ctx)
                check_y_light = is_light(y, ctx)
                if (
                    True 
                    and check_x_top
                    and check_x_medium
                    and check_x_dark
                    and check_y_bottom
                    and check_y_large
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
# You: To the right and above those, is there a small, dark-colored dot?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Follow up question, new dot.
    for config in state:
        for a, b in permutations(config):
            for x, in get1idxs(idxs, exclude=[a, b]):
                check_x_small = is_small(x, ctx)
                check_x_dark = is_dark(x, ctx)
                check_x_right_ab = is_right(x, [a, b], ctx)
                check_x_above_ab = is_above(x, [a, b], ctx)
                if (
                    True 
                    and check_x_small
                    and check_x_dark
                    and check_x_right_ab
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


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
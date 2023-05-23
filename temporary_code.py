

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
    ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])
    return ctx

idxs = list(range(7))

# New.
ctx = get_ctx()
states = []


# Turn 0
# Them: Do you see a pair of dots, where the left dot is large-sized and grey and the right dot is small-sized and light?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_x_left = is_left(x, y, ctx)
                check_x_large = is_large(x, ctx)
                check_x_grey = is_grey(x, ctx)
                check_y_right = is_right(y, x, ctx)
                check_y_small = is_small(y, ctx)
                check_y_light = is_light(y, ctx)
                if (
                    True 
                    and check_x_left
                    and check_x_large
                    and check_x_grey
                    and check_y_right
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
# Them: Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            for _ in [0]:
                check_x_bottom_left = is_below(x, y, ctx) and is_left(x, y, ctx)
                check_x_large = is_large(x, ctx)
                check_x_grey = is_grey(x, ctx)
                check_y_top_right = is_above(y, x, ctx) and is_right(y, x, ctx)
                check_y_large = is_large(y, ctx)
                check_y_grey = is_grey(y, ctx)
                if (
                    True 
                    and check_x_bottom_left
                    and check_x_large
                    and check_x_grey
                    and check_y_top_right
                    and check_y_large
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


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
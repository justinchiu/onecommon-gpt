

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
# Them: Do you see a pair of dots, where the bottom left dot is large-sized and grey, and the top right dot is large-sized and grey?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # New question.
    for config in getsets(idxs, 2):
        for a,b in permutations(config):
            for _ in [0]:
                check_ab_pair = all_close([a,b], ctx)
                check_a_bottom_left = a == get_bottom_left([a,b], ctx)
                check_a_large = is_large(a, ctx)
                check_a_grey = is_grey(a, ctx)
                check_b_top_right = b == get_top_right([a,b], ctx)
                check_b_large = is_large(b, ctx)
                check_b_grey = is_grey(b, ctx)
                if (
                    True 
                    and check_ab_pair
                    and check_a_bottom_left
                    and check_a_large
                    and check_a_grey
                    and check_b_top_right
                    and check_b_large
                    and check_b_grey
                    
                ):
                    dots = frozenset([a,b])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = None if len(states) > 0 else None
states.append(turn(state))

# Turn 1
# You: Yes, do you see a pair of dots where the right dot is large-sized and grey and the left dot is large-sized and dark?
def turn(state):
    results = set()
    orderedresults = []
    parents = []
    # Follow up question, new dots.
    for config in state:
        for a,b in permutations(config):
            for c,d in get2idxs(idxs, exclude=[a,b]):
                check_cd_pair = all_close([c,d], ctx)
                check_c_left = c == get_left([c,d], ctx)
                check_c_large = is_large(c, ctx)
                check_c_dark = is_dark(c, ctx)
                check_d_right = d == get_right([c,d], ctx)
                check_d_large = is_large(d, ctx)
                check_d_grey = is_grey(d, ctx)
                if (
                    True 
                    and check_cd_pair
                    and check_c_left
                    and check_c_large
                    and check_c_dark
                    and check_d_right
                    and check_d_large
                    and check_d_grey
                    
                ):
                    dots = frozenset([a,b,c,d])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = states[0] if len(states) > 0 else None
states.append(turn(state))


if states[-1] is not None:
    print([tuple(x) for x in states[-1]])
else:
    print("None")
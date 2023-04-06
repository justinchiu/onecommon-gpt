
# ('S_5sEpQ2A9SzLfOIfk', 'C_1ee5b7a282f14ba59e0067c64573eca0')

import sys
sys.path.append("fns")

from context import get_ctx
from shapes import is_triangle, is_line, is_square
from spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from spatial import get_top, get_bottom, get_right, get_left
from spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from spatial import get_middle
from spatial import get_distance, get_minimum_radius
from color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from iterators import get1idxs, get2idxs, get3idxs, getsets
from lists import add
import numpy as np
from functools import partial
from itertools import permutations


def get_ctx():
    ctx = np.array([[-0.155, 0.295, 0.0, -0.5333333333333333], [0.08, 0.67, -0.6666666666666666, 0.88], [0.975, -0.11, 0.3333333333333333, -0.36], [-0.485, 0.325, 1.0, 0.09333333333333334], [0.64, -0.005, 1.0, -0.88], [-0.13, -0.295, -1.0, -0.04], [-0.335, -0.275, -0.6666666666666666, -0.14666666666666667]])
    return ctx



idxs = list(range(7))

# New.
ctx = get_ctx()
state = set()

# Them: Got a triangle of 3 light grey dots by itself.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_triangle = is_triangle([x,y,z], ctx)
            check_xyz_light = all([is_light(dot, ctx) for dot in [x,y,z]])
            check_xyz_alone = all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])
            if (
                check_xyz_triangle
                and check_xyz_light
                and check_xyz_alone
            ):
                results.add(frozenset([x,y,z]))
    return results
state = turn(state)
# End.

# You: Could be. One on right is largest with a tiny gray on top??
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_a_right = a == get_right([a,b,c], ctx)
            check_a_largest = a == largest([a,b,c], ctx)
            check_b_tiny = is_small(b, ctx)
            check_b_grey = is_grey(b, ctx)
            check_b_top = b == get_top([a,b,c], ctx)
            if (
                check_a_right
                and check_a_largest
                and check_b_tiny
                and check_b_grey
                and check_b_top
            ):
                results.add(forzenset([a,b,c]))
    return results
state = turn(state)
# End.

# Them: Nevermind. Do you see a pair of dark dots? One with another above and to the right of it? Same size as well.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            check_xy_pair = all_close([x,y], ctx)
            check_xy_dark = is_dark(x, ctx) and is_dark(y, ctx)
            check_y_right_x = is_right(y, x, ctx)
            check_y_above_x = is_above(y, x, ctx)
            check_xy_same_size = same_size([x,y], ctx)
            if (
                check_xy_pair
                and check_xy_dark
                and check_y_right_x
                and check_y_above_x
                and check_xy_same_size
            ):
                results.append(frozenset([x,y]))
    return results
state = turn(state)
# End.

# You: No.
def turn(state):
    # New question.
    results = set()
    return results
state = turn(state)
# End.

# Them: What about a large medium grey dot near the center?
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 1):
        for x, in permutations(config):
            check_x_large = is_large(x, ctx)
            check_x_grey = is_grey(x, ctx)
            check_x_center = is_middle(x, None, ctx)
            if (
                check_x_large
                and check_x_grey
                and check_x_center
            ):
                results.add(frozenset([x]))
    return results
state = turn(state)
# End.

# You: Is there a smaller black one next to it?
def turn(state):
    # Follow up question, new dot.
    results = set()
    for config in state:
        for a, in permutations(config):
            for x, in get1idxs(idxs):
                check_x_smaller_a = is_smaller(x, a, ctx)
                check_x_dark = is_dark(x, ctx)
                check_x_next_to_a = all_close([a,x], ctx)
                if(
                    check_x_smaller_a
                    and check_x_dark
                    and check_x_next_to_a
                ):
                    results.add(frozenset([a, x]))
    return results
state = turn(state)
# End.

# Them: No. Do you see three dots in a diagonal line, where the top left dot is light, middle dot is grey, and bottom right dot is dark?
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_line = is_line([x,y,z], ctx)
            check_x_top_left = x == get_top_left([x, y, z], ctx)
            check_x_light = is_light(x, ctx)
            check_y_middle = is_middle(y, [x,y,z], ctx)
            check_y_grey = is_grey(y, ctx)
            check_z_bottom_right = z == get_bottom_right([x, y, z], ctx)
            check_z_dark = is_dark(z, ctx)
            if (
                check_xyz_line
                and check_x_top_left
                and check_x_light
                and check_y_middle
                and check_y_grey
                and check_z_bottom_right
                and check_z_dark
            ):
                results.add(frozenset([x,y,z]))
    return results
state = turn(state)
# End.

# You: Yes. Is the top one close to the middle darker one?
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_a_top = a == get_top([a,b,c], ctx)
            check_b_middle = b == get_middle([a,b,c], ctx)
            check_ab_close = all_close([a, b], ctx)
            check_b_darker_a = is_darker(b, a, ctx)
            if (
                check_a_top
                and check_b_middle
                and check_ab_close
                and check_b_darker_a
            ):
                results.add(frozenset([a,b,c]))
    return results
state = turn(state)
# End.

# Them: Yes. And the smallest is on the bottom right.
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_a_smallest = a == smallest([a,b,c], ctx)
            check_a_bottom_right = a == get_bottom_right([a,b,c], ctx)
            if (
                check_a_smallest
                and check_a_bottom_right
            ):
                results.add(frozenset([a,b,c]))
    return results
state = turn(state)
# End.

# You: Yes, let's select the large one. <selection>.
def select(state):
    # Select a dot.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_a_large = is_large(a, ctx)
            check_b_not_large = not is_large(b, ctx)
            check_c_not_large = not is_large(c, ctx)
            if (
                check_a_large
                and check_b_not_large
                and check_c_not_large
            ):
                results.append(frozenset([a]))
    return results
state = select(state)
# End.

# New.
ctx = get_ctx()
state = []

# You: Do you see a large black dot on the bottom left?
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 1):
        for x, in permutations(config):
            check_x_large = is_large(x, ctx)
            check_x_dark = is_dark(x, ctx)
            check_x_below_left = is_below(x, None, ctx) and is_left(x, None, ctx)
            if (
                check_x_large
                and check_x_dark
                and check_x_below_left
            ):
                results.add(frozenset([x]))
    return results
state = turn(state)
# End.
 
# Them: I see a large black dot next to two smaller lighter dots. The two smaller ones are the same size and color. We have different views though.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_close = all_close([x,y,z], ctx)
            check_x_large = is_large(x, ctx)
            check_z_dark = is_dark(z, ctx)
            check_y_smaller_x = is_smaller(y, x, ctx)
            check_z_smaller_x = is_smaller(z, x, ctx)
            check_y_lighter_x = is_lighter(y, x, ctx)
            check_z_lighter_x = is_lighter(z, x, ctx)
            check_yz_same_size = same_size([y,z], ctx)
            check_yz_same_color = same_color([y,z], ctx)
            if (
                check_xyz_close
                and check_x_large
                and check_z_dark
                and check_y_smaller_x
                and check_z_smaller_x
                and check_y_lighter_x
                and check_z_lighter_x
                and check_yz_same_size
                and check_yz_same_color
            ):
                results.add(frozenset([x,y,z]))
    return results
state = turn(state)
# End.

# You: Select the largest one.
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_a_largest = a == get_largest([a,b,c], ctx)
            if check_a_largest:
                results.add(frozenset([a]))
    return results
state = turn(state)
# End.
 
# Them: Okay.
def turn(state):
    # No op.
    return state
state = turn(state)
# End.
 
# You: Okay. <selection>.
def select(state):
    # Select a dot.
    return state
state = select(state)
# End.


# New.
ctx = get_ctx()
state = set()

# Them: I have a pair of tiny grays.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            check_xy_pair = all_close([x, y], ctx)
            check_xy_tiny = is_small(x, ctx) and is_small(y, ctx)
            check_xy_grey = is_grey(x, ctx) and is_grey(y, ctx)
            if (
                check_xy_pair
                and check_xy_tiny
                and check_xy_grey
            ):
                results.add(frozenset([x, y]))
    return results
state = turn(state)
# End.

# You: I have 3 pairs of darkish dots and one little lighter dot at the top.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 7):
        for a, b, c, d, e, f, g in permutations(config):
            check_abcdef_pairs = (
                all_close([a, b], ctx)
                and all_close([c, d], ctx)
                and all_close([e, f], ctx)
            )
            check_abcdef_darkish = (
                is_dark(a, ctx)
                and is_dark(b, ctx)
                and is_dark(c, ctx)
                and is_dark(d, ctx)
                and is_dark(e, ctx)
                and is_dark(f, ctx)
            )
            check_g_top = g == get_top([a, b, c, d, e, f, g], ctx)
            check_g_lighter = (
                is_lighter(g, a, ctx)
                and is_lighter(g, b, ctx)
                and is_lighter(g, c, ctx)
                and is_lighter(g, d, ctx)
                and is_lighter(g, e, ctx)
                and is_lighter(g, f, ctx)
            )
            check_g_small = is_small(g, ctx)
            if (
                check_abcdef_pairs
                and check_abcdef_darkish
                and check_g_top
                and check_g_lighter
                and check_g_small
            ):
                results.add(frozenset([a, b, c, d, e, f, g]))
    return results
state = turn(state)
# End.

# Them: I only have two pairs. In each pair, the dot on the right is slightly smaller.
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 4):
        for a, b, c, d in permutations(config):
            check_ab_pair = all_close([a, b], ctx)
            check_cd_pair = all_close([c, d], ctx)
            check_b_right_a = is_right(b, a, ctx)
            check_d_right_c = is_right(d, c, ctx)
            check_b_smaller_a = is_smaller(b, a, ctx)
            check_d_smaller_c = is_smaller(d, c, ctx)
            if (
                check_ab_pair
                and check_cd_pair
                and check_b_right_a
                and check_d_right_c
                and check_b_smaller_a
                and check_d_smaller_c
            ):
                results.add(frozenset([a, b, c, d]))
    return results
state = turn(state)
# End.

# You: Where are your pairs positioned? One of mine is all the way to the edge at 3 o'clock, one is toward the center at 7 o'clock, and one is halfway to the edge at about 10 o'clock.
def turn(state):
    # New question.
    results = set()
    for config in state:
        for a, b, c, d in permutations(config):
            check_ab_right = a == get_right([a, b, c, d], ctx) or b == get_right([a, b, c, d], ctx)
            check_ab_3_oclock = is_right(a, None, ctx) and is_right(b, None, ctx)
            check_cd_center = c == get_middle([a, b, c, d], ctx) or d == get_middle([a, b, c, d], ctx)
            check_cd_7_oclock = is_below(c, None, ctx) and is_left(c, None, ctx) and is_below(d, None, ctx) and is_left(d, None, ctx)
            check_ab_10_oclock = (
                (is_above(a, None, ctx) and is_left(a, None, ctx) and is_above(b, None, ctx) and is_left(b, None, ctx))
                or (is_above(c, None, ctx) and is_left(c, None, ctx) and is_above(d, None, ctx) and is_left(d, None, ctx))
            )
            if (
                check_ab_right
                and check_ab_3_oclock
                and check_cd_center
                and check_cd_7_oclock
                and check_ab_10_oclock
            ):
                results.add(frozenset([a, b, c, d]))
    return results
state = turn(state)
# End.

# Them: My layout doesn't look like that. For your pairs, is there a larger pair above a smaller pair?
def turn(state):
    # New question.
    results = set()
    for config in state:
        for a, b, c, d in permutations(config):
            check_ab_pair = all_close([a, b], ctx)
            check_cd_pair = all_close([c, d], ctx)
            check_ab_larger_cd = is_larger(a, c, ctx) and is_larger(b, d, ctx)
            check_ab_above_cd = is_above(a, c, ctx) and is_above(b, d, ctx)
            if (
                check_ab_pair
                and check_cd_pair
                and check_ab_larger_cd
                and check_ab_above_cd
            ):
                results.add(frozenset([a, b, c, d]))
    return results
state = turn(state)
# End.

# You: I have the smaller pair, yes. Click on the smaller right-side dot of the smaller pair.
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a, b, c, d in permutations(config):
            check_ab_pair = all_close([a, b], ctx)
            check_cd_pair = all_close([c, d], ctx)
            check_ab_smaller_cd = is_smaller(a, c, ctx) and is_smaller(b, d, ctx)
            check_b_right_a = is_right(b, a, ctx)
            check_d_right_c = is_right(d, c, ctx)
            if (
                check_ab_pair
                and check_cd_pair
                and check_ab_smaller_cd
                and check_b_right_a
                and check_d_right_c
            ):
                results.add(frozenset([b]))
    return results
state = turn(state)
# End.

# Them: OK. <selection>.
def select(state):
    # Select a dot.
    return state
state = select(state)


print(sorted(
    [tuple(x) for x in state],
    key = lambda x: get_minimum_radius(x, ctx),
))
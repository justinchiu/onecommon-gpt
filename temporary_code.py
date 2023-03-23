
# ('S_zesr8xr8W5lZo159', 'C_f875d0ff0e014d7d9d7426b40dc542f9')

import sys
sys.path.append("fns")

from context import get_ctx
from shapes import is_triangle, is_line, is_square
from spatial import all_close, are_above, are_below, are_right, are_left
from spatial import are_above_left, are_above_right, are_below_right, are_below_left
from spatial import are_middle
from spatial import get_top, get_bottom, get_right, get_left
from spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from spatial import get_middle
from color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, are_darker, are_lighter
from size import is_large, is_small, is_medium, largest, smallest, same_size, different_size, are_larger, are_smaller
from iterators import get1idxs, get2idxs, get3idxs
from lists import add
import numpy as np
from functools import partial


def get_ctx():
    ctx = np.array([[0.275, 0.665, 0.3333333333333333, 0.04], [-0.295, 0.08, 0.0, 1.0], [0.26, -0.835, 0.3333333333333333, -0.8], [-0.865, -0.23, -1.0, 0.32], [-0.74, -0.595, 0.3333333333333333, 0.5733333333333334], [0.005, -0.77, -1.0, -0.3333333333333333], [-0.15, -0.4, -0.3333333333333333, -0.6533333333333333]])
    return ctx



idxs = list(range(7))

# New.
ctx = get_ctx()
state = []

# Them: Got a triangle of 3 light grey dots by itself.
def turn(state):
    # New question.
    results = []
    for x,y,z in get3idxs(idxs):
        check_xyz_triangle = is_triangle([x,y,z], ctx)
        check_xyz_light = all([is_light(dot, ctx) for dot in [x,y,z]])
        check_xyz_alone = all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])
        if (
            check_xyz_triangle
            and check_xyz_light
            and check_xyz_alone
        ):
            results.append([x,y,z])
    return results
state = turn(state)
# End.

# You: Could be. One on right is largest with a tiny gray on top??
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        check_largest_right = largest([a,b,c], ctx) == get_right([a,b,c], ctx)
        check_small_top = is_small(get_top([a,b,c], ctx), ctx)
        check_grey_top = is_grey(get_top([a,b,c], ctx), ctx)
        if (
            check_largest_right
            and check_small_top
            and check_grey_top
        ):
            results.append([a,b,c])
    return results
state = turn(state)
# End.

# Them: Nevermind. Do you see a pair of dark dots? One with another above and to the right of it? Same size as well.
def turn(state):
    # New question.
    results = []
    for x, y in get2idxs(idxs):
        check_pair = all_close([x,y], ctx)
        check_all_dark = all([is_dark(dot, ctx) for dot in [x,y]])
        check_right = are_right([y], [x], ctx)
        check_above = are_above([y], [x], ctx)
        check_size = same_size([x,y], ctx)
        if (
            check_pair
            and check_all_dark
            and check_right
            and check_above
            and check_size
        ):
            results.append([x,y])
    return results
state = turn(state)
# End.

# You: No.
def turn(state):
    # New question.
    results = []
    return results
state = turn(state)
# End.

# Them: What about a large medium grey dot?
def turn(state):
    # New question.
    results = []
    for x, in get1idxs(idxs):
        check_x_large = is_large(x, ctx)
        check_x_grey = is_grey(x, ctx)
        if (
            check_x_large
            and check_x_grey
        ):
            results.append([x])
    return results
state = turn(state)
# End.

# You: Is there a smaller black one next to it?
def turn(state):
    # Follow up question, new dot.
    results = []
    for a, in state:
        for x, in get1idxs(idxs):
            check_x_smaller_a = are_smaller([x], [a], ctx)
            check_x_dark = is_dark(x, ctx)
            check_x_next_to_a = all_close([a,x], ctx) and not are_middle([x], [a], ctx)
            if(
                check_x_smaller_a
                and check_x_dark
                and check_x_next_to_a
            ):
                results.append([a, x])
    return results
state = turn(state)
# End.

# Them: No. Do you see three dots in a diagonal line, where the top left dot is light, middle dot is grey, and bottom right dot is dark?
def turn(state):
    # New question.
    results = []
    for x, y, z in get3idxs(idxs):
        check_xyz_line = is_line([x,y,z], ctx)
        check_x_top_left = x == get_top_left([x, y, z], ctx)
        check_x_light = is_light(x, ctx)
        check_y_middle = are_middle([y], [x,y,z], ctx)
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
            results.append([x,y,z])
    return results
state = turn(state)
# End.

# You: Yes. Is the top one close to the middle one?
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        top_one = get_top([a,b,c], ctx)
        middle_one = get_middle([a,b,c], ctx)
        check_close = all_close([top_one, middle_one], ctx)
        if (
            check_close
        ):
            results.append([a,b,c])
    return results
state = turn(state)
# End.

# Them: Yes, let's select the large one. <selection>.
def select(state):
    # Select a dot.
    results = []
    for a,b,c in state:
        check_a_large = is_large(a, ctx)
        check_b_not_large = not is_large(b, ctx)
        check_c_not_large = not is_large(c, ctx)
        if (
            check_a_large
            and check_b_not_large
            and check_c_not_large
        ):
            results.append([a])
    return results
state = select(state)
# End.

# New.
ctx = get_ctx()
state = []

# You: Do you see a large black dot on the bottom left?
def turn(state):
    # New question.
    results = []
    for x, in get1idxs(idxs):
        check_x_large = is_large(x, ctx)
        check_x_dark = is_dark(x, ctx)
        check_x_below_left = are_below_left([x], None, ctx)
        if (
            check_x_large
            and check_x_dark
            and check_x_below_left
        ):
            results.append(x)
    return results
state = turn(state)
# End.
 
# Them: I see a large black dot next to two smaller lighter dots. We have different views though.
def turn(state):
    # New question.
    results = []
    for x,y,z in get3idxs(idxs):
        check_xyz_close = all_close([x,y,z], ctx)
        check_x_large = is_large(x, ctx)
        check_z_dark = is_dark(z, ctx)
        check_y_smaller_x = are_smaller([y], [x], ctx)
        check_z_smaller_x = are_smaller([z], [x], ctx)
        check_y_lighter_x = are_lighter([y], [x], ctx)
        check_z_lighter_x = are_lighter([z], [x], ctx)
        if (
            check_xyz_close
            and check_x_large
            and check_z_dark
            and check_y_smaller_x
            and check_z_smaller_x
            and check_y_lighter_x
            and check_z_lighter_x
        ):
            results.append([x,y,z])
    return results
state = turn(state)
# End.

# You: Select the largest one.
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        results.append(get_largest(result, ctx))
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
state = []

# You: Do you see a large dark dot with a tiny lighter grey dot above and to the left?
def turn(state):
    # New question.
    results = []
    for x, y in get2idxs(idxs):
        check_x_large = is_large(x, ctx)
        check_x_dark = is_dark(x, ctx)
        check_y_tiny = is_small(y, ctx)
        check_y_lighter_grey = is_light(y, ctx) and is_grey(y, ctx)
        check_y_above_left = are_above_left([y], [x], ctx)
        if (
            check_x_large
            and check_x_dark
            and check_y_tiny
            and check_y_lighter_grey
            and check_y_above_left
        ):
            results.append([x, y])
    return results
state = turn(state)
# End.

# Them: I see a medium black dot at the middle between a light grey dot and a smaller dot, making a line going down.
def turn(state):
    # New question.
    results = []
    for x, y, z in get3idxs(idxs):
        check_xyz_line = is_line([x, y, z], ctx)
        check_x_medium = is_medium(x, ctx)
        check_x_dark = is_dark(x, ctx)
        check_y_light_grey = is_light(y, ctx) and is_grey(y, ctx)
        check_z_smaller = is_small(z, ctx)
        check_line_down = get_top([x, y, z], ctx) == y and get_bottom([x, y, z], ctx) == z
        if (
            check_xyz_line
            and check_x_medium
            and check_x_dark
            and check_y_light_grey
            and check_z_smaller
            and check_line_down
        ):
            results.append([x, y, z])
    return results
state = turn(state)
# End.

# You: OK, I see that line.
def turn(state):
    # No op.
    return state
state = turn(state)
# End.

# Them: Do you see the black dot in the middle?
def turn(state):
    # Follow up question.
    results = []
    for a, b, c in state:
        check_middle_black = is_dark(get_middle([a, b, c], ctx), ctx)
        if (
            check_middle_black
        ):
            results.append([a, b, c])
    return results
state = turn(state)
# End.

# You: Yes, choose it <selection>.
def select(state):
    # Select a dot.
    results = []
    for a, b, c in state:
        middle_dot = get_middle([a, b, c], ctx)
        results.append([middle_dot])
    return results
state = select(state)


print(state)
# state: num_candidates x size x feats=4
# dots: 7 x feats=4
# heuristic: take first candidate state[0]
"""
if state:
    print(state[0].tolist())
else:
    print("None")
"""
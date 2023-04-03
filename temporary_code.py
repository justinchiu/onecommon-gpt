
# ('S_zesr8xr8W5lZo159', 'C_f875d0ff0e014d7d9d7426b40dc542f9')

import sys
sys.path.append("fns")

from context import get_ctx
from shapes import is_triangle, is_line, is_square
from spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from spatial import get_top, get_bottom, get_right, get_left
from spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from spatial import get_middle
from spatial import get_distance
from color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from iterators import get1idxs, get2idxs, get3idxs
from lists import add
import numpy as np
from functools import partial


def get_ctx():
    ctx = np.array([[-0.405, -0.025, -1.0, 0.64], [0.36, 0.47, 0.0, 1.0], [-0.21, 0.16, -1.0, 0.32], [-0.09, -0.205, 0.3333333333333333, 0.5733333333333334], [-0.23, -0.43, 0.0, 0.38666666666666666], [0.655, -0.375, -1.0, -0.3333333333333333], [0.5, -0.01, -0.3333333333333333, -0.6533333333333333]])
    return ctx



idxs = list(range(7))

# New.
ctx = get_ctx()
state = []

"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* New dots A B C
* A light
* B light
* C light
* A B C triangle
* A B C alone
"""
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

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A largest of A B C
* A on right of A B C
* B tiny and grey
* B top of A B C
"""
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        check_a_largest = a == largest([a,b,c], ctx)
        check_a_right = a == get_right([a,b,c], ctx)
        check_b_tiny_grey = is_small(b, ctx) and is_grey(b, ctx)
        check_b_top = b == get_top([a,b,c], ctx)
        if (
            check_largest_right
            and check_tiny_top
            and check_grey_top
        ):
            results.append([a,b,c])
    return results
state = turn(state)
# End.

"""
Confirmation: Deny.
Give names to the dots and list the properties described.
* New dots A B
* A pair B
* A dark
* B dark
* B above right A
* A same size B
"""
def turn(state):
    # New question.
    results = []
    for x, y in get2idxs(idxs):
        check_xy_pair = all_close([x,y], ctx)
        check_xy_dark = is_dark(x, ctx) and is_dark(y, ctx)
        check_y_above_right_x = is_above(y, x, ctx) and is_right(y, x, ctx)
        check_xy_same_size = same_size([x,y], ctx)
        if (
            check_xy_pair
            and check_xy_dark
            and check_y_above_right_x
            and check_xy_same_size
        ):
            results.append([x,y])
    return results
state = turn(state)
# End.

"""
Confirmation: Deny.
"""
def turn(state):
    # No op.
    results = []
    return results
state = turn(state)
# End.

"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* New dots A
* A large and grey
* A near center
"""
def turn(state):
    # New question.
    results = []
    for x, in get1idxs(idxs):
        check_x_large = is_large(x, ctx)
        check_x_grey = is_grey(x, ctx)
        check_x_center = is_middle(x, None, ctx)
        if (
            check_x_large
            and check_x_grey
            and check_x_center
        ):
            results.append([x])
    return results
state = turn(state)
# End.

"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* Previous dots A
* New dots B
* B black
* B smaller than A 
* A next to B
"""
def turn(state):
    # Follow up question, new dot.
    results = []
    for a, in state:
        for x, in get1idxs(idxs):
            check_x_smaller_a = is_smaller(x, a, ctx)
            check_x_dark = is_dark(x, ctx)
            check_x_next_to_a = all_close([a,x], ctx)
            if(
                check_x_smaller_a
                and check_x_dark
                and check_x_next_to_a
            ):
                results.append([a, x])
    return results
state = turn(state)
# End.

"""
Confirmation: Deny.
Give names to the dots and list the properties described.
* New dots A B C
* A light
* B grey
* C dark
* A B C diagonal line
* A is top left A B C
* B is middle A B C
* C is bottom right A B C
"""
def turn(state):
    # New question.
    results = []
    for x, y, z in get3idxs(idxs):
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
            results.append([x,y,z])
    return results
state = turn(state)
# End.

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A is top of A B C
* B is middle of A B C
* A darker than B
* A close B
"""
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        check_a_top = a == get_top([a,b,c], ctx)
        check_b_middle = b == get_middle([a,b,c], ctx)
        check_darker = is_darker(a, b, ctx)
        check_close = all_close([a, b], ctx)
        if (
            check_close
            and check_darker
        ):
            results.append([a,b,c])
    return results
state = turn(state)
# End.

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A is smallest in A B C
* A is bottom right of A B C
"""
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        check_a_smallest = a == smallest([a,b,c], ctx)
        check_a_bottom_right = a == get_bottom_right([a,b,c], ctx)
        if (
            check_a_smallest
            and check_a_bottom_right
        ):
            results.append([a,b,c])
    return results
state = turn(state)
# End.

"""
Confirmation: Confirm.
Selection.
"""
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

"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* New dots A
* A large and black
* A is bottom left
"""
def turn(state):
    # New question.
    results = []
    for x, in get1idxs(idxs):
        check_x_large = is_large(x, ctx)
        check_x_dark = is_dark(x, ctx)
        check_x_below_left = is_below(x, None, ctx) and is_left(x, None, ctx)
        if (
            check_x_large
            and check_x_dark
            and check_x_below_left
        ):
            results.append([x])
    return results
state = turn(state)
# End.
 
"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* New dots A B C
* A large and black
* B smaller and lighter than A
* C smaller and lighter than A
* B C same size and color
"""
def turn(state):
    # New question.
    results = []
    for x,y,z in get3idxs(idxs):
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
            results.append([x,y,z])
    return results
state = turn(state)
# End.

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A largest
"""
def turn(state):
    # Follow up question.
    results = []
    for a,b,c in state:
        largest_one = get_largest([a,b,c], ctx)
        results.append(largest_one)
    return results
state = turn(state)
# End.
 
"""
Confirmation: Confirm.
"""
def turn(state):
    # No op.
    return state
state = turn(state)
# End.
 
"""
Confirmation: Confirm.
Selection.
"""
def select(state):
    # Select a dot.
    return state
state = select(state)
# End.


# New.
ctx = get_ctx()
state = []

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* New dots A B
* A large and dark
* B tiny and lighter grey
* B above and to the left of A
"""
def turn(state):
    # New question.
    results = []
    for x, y in get2idxs(idxs):
        check_xy_large_dark = is_large(x, ctx) and is_dark(x, ctx)
        check_y_tiny_light_grey = is_small(y, ctx) and is_lighter(y, ctx) and is_grey(y, ctx)
        check_y_above_left_x = is_above(y, x, ctx) and is_left(y, x, ctx)
        if (
            check_xy_large_dark
            and check_y_tiny_light_grey
            and check_y_above_left_x
        ):
            results.append([x,y])
    return results
state = turn(state)
# End.

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* New dots A B C
* A medium and black
* B light grey
* C smaller than A and B
* A B C line going down
* A in middle of A B C
"""
def turn(state):
    # New question.
    results = []
    for x,y,z in get3idxs(idxs):
        check_xyz_close = all_close([x,y,z], ctx)
        check_x_medium_black = is_medium_size(x, ctx) and is_dark(x, ctx)
        check_y_light_grey = is_lighter(y, ctx) and is_grey(y, ctx)
        check_z_smaller_ab = is_smaller(z, x, ctx) and is_smaller(z, y, ctx)
        check_xyz_line_down = is_line([x,y,z], ctx) and is_below(x, y, ctx) and is_below(y, z, ctx)
        check_x_middle_abc = is_middle(x, [x,y,z], ctx)
        if (
            check_xyz_close
            and check_x_medium_black
            and check_y_light_grey
            and check_z_smaller_ab
            and check_xyz_line_down
            and check_x_middle_abc
        ):
            results.append([x,y,z])
    return results
state = turn(state)


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
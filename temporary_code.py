
# ('S_CEQqwHXvbR5j5fGz', 'C_888a97927a5b4014b39e92645313ae06')

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
    ctx = np.array([[-0.97, -0.12, 0.0, -0.3333333333333333], [-0.12, 0.98, 0.0, 0.9066666666666666], [0.175, 0.615, -0.6666666666666666, 0.6533333333333333], [0.03, -0.93, -0.3333333333333333, -0.3333333333333333], [-0.095, 0.045, -0.3333333333333333, 0.6266666666666667], [-0.69, 0.115, -0.3333333333333333, -0.04], [0.635, 0.315, 1.0, 0.17333333333333334]])
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
                results.add(frozenset([a,b,c]))
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

"""
Confirmation: Deny.
"""
def turn(state):
    # No op.
    results = set()
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

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A is smallest in A B C
* A is bottom right of A B C
"""
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

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A large
Selection.
"""
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
state = set()

"""
Confirmation: Neither.
Give names to the dots and list the properties described.
* New dots A
* A large and black
* A is bottom left
"""
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

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B C
* A largest
"""
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
state = set()

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* New dots A B C D E F G
* B largest of A B C D E F G
* B middle right of A B C D E F G
* B medium gray
"""
def turn(state):
    # New question.
    results = set()
    for config in getsets(idxs, 7):
        for a,b,c,d,e,f,g in permutations(config):
            check_b_largest = b == get_largest([a,b,c,d,e,f,g], ctx)
            check_b_middle_right = b == get_middle([a,b,c,d,e,f,g], [None, get_right([a,b,c,d,e,f,g], ctx)], ctx)
            check_b_medium_gray = is_grey(b, ctx) and not is_light(b, ctx) and not is_dark(b, ctx)
            if (
                check_b_largest
                and check_b_middle_right
                and check_b_medium_gray
            ):
                results.add(frozenset([a,b,c,d,e,f,g]))
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
Give names to the dots and list the properties described.
* Previous dots A B C
* B smaller and lighter than A
* C smaller and lighter than A
* B C to the left of A
"""
def turn(state):
    # Follow up question.
    results = set()
    for config in state:
        for a,b,c in permutations(config):
            check_b_smaller_a = is_smaller(b, a, ctx)
            check_c_smaller_a = is_smaller(c, a, ctx)
            check_bc_left_a = is_left(b, a, ctx) and is_left(c, a, ctx)
            if (
                check_b_smaller_a
                and check_c_smaller_a
                and check_bc_left_a
            ):
                results.add(frozenset([a,b,c]))
    return results
state = turn(state)
# End.

"""
Confirmation: Neither.
"""
def 
# End.

"""
Confirmation: Confirm.
Give names to the dots and list the properties described.
* Previous dots A B
* A larger than B
* A grey
Selection.
"""
def 
# End.

"""
Confirmation: Confirm.
Selection.
"""
def 

print(sorted(
    [tuple(x) for x in state],
    key = lambda x: get_minimum_radius(x, ctx),
))
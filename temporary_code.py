
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
from iterators import get1dots, get2dots, get3dots
from lists import add
import numpy as np
from functools import partial


def get_ctx():
    ctx = np.array([[0.275, 0.665, 0.3333333333333333, 0.04], [-0.295, 0.08, 0.0, 1.0], [0.26, -0.835, 0.3333333333333333, -0.8], [-0.865, -0.23, -1.0, 0.32], [-0.74, -0.595, 0.3333333333333333, 0.5733333333333334], [0.005, -0.77, -1.0, -0.3333333333333333], [-0.15, -0.4, -0.3333333333333333, -0.6533333333333333]])
    return ctx



all_dots = np.arange(7)

# New.
ctx = get_ctx()
state = []

# Them: got a triangle of 3 light grey dots.
def turn(state):
    # New question.
    results = []
    for x,y,z in get3dots(all_dots):
        if is_triangle([x,y,z], ctx) and all(map(partial(is_light, ctx=ctx), [x,y,z])):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)
# End.

# You: Could be. One on right is largest with a small gray on top??
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        if (largest(result, ctx) == get_right(result, ctx)).all() and is_small(get_top(result, ctx), ctx):
            results.append(result)
    return results
state = turn(state)
# End.

# Them: Nevermind. Do you see a pair of dark dots? One with another above and to the right of it? Same size as well.
def turn(state):
    # New question.
    results = []
    for result in get2dots(all_dots):
        if (
            all_close(result, ctx)
            and all(map(partial(is_dark, ctx=ctx), result))
            and are_right(result[1], result[0], ctx)
            and are_above(result[1], result[0], ctx)
            and same_size(result, ctx)
        ):
            results.append(result)
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
    for dot in get1dots(all_dots):
        if is_large(dot, ctx):
            results.append(dot)
    return results
state = turn(state)
# End.

# You: Is there a small black one next to it?
def turn(state):
    # Follow up question, new dot.
    results = []
    for result in state:
        for dot in get1dots(all_dots):
            if is_small(dot, ctx) and is_dark(dot, ctx) and all_close(add(result, dot), ctx) and not are_middle(dot, result, ctx):
                results.append(add(result, dot))
    return results
state = turn(state)
# End.

# Them: No. Do you see three dots in a diagonal line, where the top left dot is light, middle dot is grey, and bottom right dot is dark?
def turn(state):
    # New question.
    results = []
    for x, y, z in get3dots(all_dots):
        if (
            is_line([x,y,z], ctx)
            and x == get_top_left([x, y, z], ctx)
            and is_light(x, ctx)
            and are_middle([y], [x,y,z], ctx)
            and is_grey(y, ctx)
            and z == get_bottom_right([x, y, z], ctx)
            and is_dark(z, ctx)
        ):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)
# End.

# You: Yes. Is the top one close to the middle one?
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        if all_close([get_top(result, ctx), get_middle(result, ctx)], ctx):
            results.append(result)
    return results
state = turn(state)
# End.

# Them: Yes, let's select the large one. <selection>.
def select(state):
    # Select a dot.
    results = []
    for result in state:
        for dot in result:
            if is_large([dot], ctx):
                results.append(np.array([dot]))
state = select(state)
# End.

# New.
ctx = get_ctx()
state = []

# You: Do you see a large black dot on the bottom left?
def turn(state):
    # New question.
    results = []
    for dot in get1dots(all_dots):
        if is_large(dot, ctx) and is_dark(dot, ctx) and are_below_left(dot, None, ctx):
            results.append(dot)
    return results
state = turn(state)
# End.
 
# Them: I see a large black dot next to two small dots. We have different views though.
def turn(state):
    # New question.
    results = []
    for x,y,z in get3dots(all_dots):
        if all_close(np.array([x,y,z]), ctx) and is_large(x, ctx) and is_dark(z, ctx) and is_small(y, ctx) and is_small(z, ctx):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)
# End.

# You: Select the large black one.
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        for dot in result:
            if is_large([dot], ctx) and is_dark([dot], ctx):
                results.append(np.array([dot]))
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
dots = get_ctx()
state = []

# You: Do you see a large, dark dot with a tiny, lighter-grey dot above and to the left?
def turn(state):
    # New question.
    results = []
    for x,y in get2dots(all_dots):
        if (
            is_large(x, ctx)
            and is_dark(x, ctx)
            and is_small(y, ctx)
            and are_above_left(y, x, ctx)
            and are_lighter(y, x, ctx)
        ):
            results.append(np.array([x,y]))
    return results
state = turn(state)
# End.

# Them: I see a medium-sized black dot at the middle, between a light-grey dot and a smaller dot, making a line going down.
def turn(state):
    # New question.
    results = []
    for x,y,z in get3dots(all_dots):
        if (
            is_line([x,y,z], ctx)
            and is_medium(y, ctx)
            and is_dark(y, ctx)
            and are_middle([y], [x,y,z], ctx)
            and is_light(x, ctx)
            and is_small(z, ctx)
        ):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)
# End.

# You: Ok, I see that line.
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        if is_line(result, ctx):
            results.append(result)
    return results
state = turn(state)
# End.

# Them: Do you see the dot at the middle, black?
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        for dot in result:
            if is_dark([dot], ctx) and are_middle([dot], result, ctx):
                results.append(np.array([dot]))
    return results
state = turn(state)
# End.

# You: Yes, choose it.
def turn(state):
    # Follow up question.
    results = []
    for result in state:
        for dot in result:
            if is_dark([dot], ctx) and are_middle([dot], result, ctx):
                results.append(np.array([dot]))
    return results
state = turn(state)


print([x.tolist() for x in state])
# state: num_candidates x size x feats=4
# dots: 7 x feats=4
# heuristic: take first candidate state[0]
"""
if state:
    print(state[0].tolist())
else:
    print("None")
"""
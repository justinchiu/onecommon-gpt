
# ('S_N3atbPCA1hsEIsRn', 'C_5e57c484d8d24b788d3e13577b8617ef')

import sys
sys.path.append("code")

from dot import get_dots
from shapes import is_triangle, is_line, is_square
from spatial import all_close, are_close, are_above, are_below, are_right, are_left
from spatial import are_above_left, are_above_right, are_below_right, are_below_left
from spatial import are_middle
from spatial import get_top, get_bottom, get_right, get_left, get_top_right, get_top_left, get_bottom_right, get_bottom_left
from color import is_dark, is_grey, is_light
from size import is_large, is_small, largest, smallest, is_medium
from iterators import get1dots, get2dots, get3dots
from lists import add
import numpy as np
from functools import partial


def get_dots():
    dots = np.array([[-0.765, 0.33, 0.6666666666666666, 0.9066666666666666], [-0.575, 0.76, 0.0, -0.24], [0.565, -0.085, -1.0, 0.9866666666666667], [-0.83, -0.405, 0.0, -0.6], [-0.365, -0.035, 0.3333333333333333, -0.88], [0.785, 0.025, 0.0, 0.30666666666666664], [0.59, -0.5, -0.6666666666666666, -0.22666666666666666]])
    return dots



all_dots = np.arange(7)

# NEW
ctx = get_dots()
state = []

# Them: got a triangle of 3 light grey dots.
def turn(state):
    results = []
    for x,y,z in get3dots(all_dots):
        if is_triangle([x,y,z], ctx) and all(map(partial(is_light, ctx=ctx), [x,y,z])):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)

# You: Could be. One on right is largest?
def turn(state):
    # Follow up question.
    results = state
    for result in results:
        if (largest(result, ctx) == get_right(result, ctx)).all():
            results.append(result)
    return results
state = turn(state)

# Them: Nevermind. Do you see a pair of dark dots?
def turn(state):
    # New question.
    results = []
    for result in get2dots(all_dots):
        if all_close(result, ctx) and all(map(partial(is_dark, ctx=ctx), result)):
            results.append(result)
    return results
state = turn(state)

# You: No.
def turn(state):
    # New question.
    results = []
    return results
state = turn(state)

# Them: What about a large medium grey dot?
def turn(state):
    # New question.
    results = []
    for dot in get1dots(all_dots):
        if is_large(dot, ctx):
            results.append(dot)
    return results
state = turn(state)

# You: Is there a small black one next to it?
def turn(state):
    # Follow up question, new dot.
    results = []
    for prev_dots in state:
        for dot in get1dots(all_dots):
            if is_small(dot, ctx) and is_dark(dot, ctx) and all_close(add(prev_dots, dot), ctx) and not are_middle(dot, prev_dots, ctx):
                results.append(add(prev_dots, dot))
    return results
state = turn(state)

# Them: No. Do you see three dots in a line, where the top left dot is light, middle dot is grey, and bottom right dot is dark?
def turn(state):
    # New question.
    results = []
    for x, y, z in get3dots(all_dots):
        if (
            is_line([x,y,z], ctx)
            and x == get_top_left([x, y, z], ctx)
            and is_light(x, ctx)
            and are_middle(y, [x,y,z], ctx)
            and is_grey(y, ctx)
            and z == get_bottom_right([x, y, z], ctx)
            and is_dark(z, ctx)
        ):
            results.append(np.array([x,y,z]))
    return results
state = turn(state)

# You: Yes, let's select the large one.
def select(state):
    results = [dot for dots in state for dot in dots]
    for dot in results:
        if is_large(dot):
            return [dot]
state = select(state)

# NEW
dots = get_dots()
state = []

# Them: i have a light grey small dot next to a medium grey medium dot.
def turn(state):
    results = []
    for x,y in get2dots(all_dots):
        if (
            is_light(x, dots)
            and is_small(x, dots)
            and is_grey(y, dots)
            and is_medium(y, dots)
            and are_close(x, y, dots)
        ):
            results.append(np.array([x,y]))
    return results
state = turn(state)



#print(state)
# state: num_candidates x size x feats=4
# dots: 7 x feats=4
# heuristic: take first candidate state[0]
if state:
    print(state[0].tolist())
else:
    print([0])
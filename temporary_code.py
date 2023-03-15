import sys
sys.path.append("code")

from dot import get_dots
from shapes import is_triangle, is_line, is_square
from spatial import is_close, is_above, is_below, is_right, is_left
from spatial import get_top, get_bottom, get_right, get_left, get_top_right, get_top_left, get_bottom_right, get_bottom_left
from color import is_dark, is_grey, is_light
from size import is_large, is_small, largest, smallest, is_medium
from iterators import get2dots, get3dots
import numpy as np


def get_dots():
    dots = np.array([[-0.765, 0.33, 0.6666666666666666, 0.9066666666666666], [-0.575, 0.76, 0.0, -0.24], [0.565, -0.085, -1.0, 0.9866666666666667], [-0.83, -0.405, 0.0, -0.6], [-0.365, -0.035, 0.3333333333333333, -0.88], [0.785, 0.025, 0.0, 0.30666666666666664], [0.59, -0.5, -0.6666666666666666, -0.22666666666666666]])
    return dots

None

dots = get_dots()
context = []

# Them: got a triangle of 3 light grey dots.
def turn(dots, context):
    results = context
    for x,y,z in get3dots(dots):
        if is_triangle([x,y,z], dots) and all(map(is_light, [x,y,z])):
            results.append([x,y,z])
    return results
context = turn(dots, context)

# You: Could be. One on right is largest?
def turn(dots, context):
    results = context
    for result in results:
        if largest(result) == get_right(result):
            results.append(result)
    return results
context = turn(dots, context)

# Them: Nevermind. Do you see a pair of dark dots?
def turn(dots, context):
    results = []
    for result in get2dots(dots):
        if is_close(result) and all(map(is_dark, result)):
            results.append(result)
    return results
context = turn(dots, context)

# You: No.
def turn(dots, context):
    results = []
    return results
context = turn(dots, context)

# Them: What about a large medium grey dot?
def turn(dots, context):
    results = []
    for dot in dots:
        if is_large(dot):
            results.append(dot)
    return results
context = turn(dots, context)

# You: Don't see that either.
def turn(dots, context):
    results = []
    return results
context = turn(dots, context)


dots = get_dots()
context = []


# Them: i have a light grey small dot next to a medium grey medium dot
def turn(dots, context):
    results = []
    for x,y in get2dots(dots):
        if is_close([x,y]) and is_light(x) and is_small(x) and is_medium(y) and is_grey(y):
            results.append([x,y])
    return results
context = turn(dots, context)



print(context)
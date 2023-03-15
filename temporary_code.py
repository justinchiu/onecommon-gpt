
# ('S_N3atbPCA1hsEIsRn', 'C_5e57c484d8d24b788d3e13577b8617ef')

import sys
sys.path.append("code")

from dot import get_dots
from shapes import is_triangle, is_line, is_square
from spatial import all_close, is_close, is_above, is_below, is_right, is_left
from spatial import get_top, get_bottom, get_right, get_left, get_top_right, get_top_left, get_bottom_right, get_bottom_left
from color import is_dark, is_grey, is_light
from size import is_large, is_small, largest, smallest, is_medium
from iterators import get2dots, get3dots
import numpy as np


def get_dots():
    dots = np.array([[-0.765, 0.33, 0.6666666666666666, 0.9066666666666666], [-0.575, 0.76, 0.0, -0.24], [0.565, -0.085, -1.0, 0.9866666666666667], [-0.83, -0.405, 0.0, -0.6], [-0.365, -0.035, 0.3333333333333333, -0.88], [0.785, 0.025, 0.0, 0.30666666666666664], [0.59, -0.5, -0.6666666666666666, -0.22666666666666666]])
    return dots



dots = get_dots()
context = []

# Them: got a triangle of 3 light grey dots.
def turn(dots, context):
    results = []
    for x,y,z in get3dots(dots):
        if is_triangle([x,y,z], dots) and all(map(is_light, [x,y,z])):
            results.append(np.array([x,y,z]))
    return results
context = turn(dots, context)

# You: Could be. One on right is largest?
def turn(dots, context):
    results = context
    for result in results:
        if (largest(result) == get_right(result)).all():
            results.append(result)
    return results
context = turn(dots, context)

# Them: Nevermind. Do you see a pair of dark dots?
def turn(dots, context):
    results = []
    for result in get2dots(dots):
        if all_close(result) and all(map(is_dark, result)):
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

# You: Is there a small black one next to it?
def turn(dots, context):
    results = []
    for prev_dots in context:
        for dot in dots:
            if is_small(dot) and is_dark(dot) and all_close(prev_dots + [dot]) and not_in(dot, dots):
                results.append(dots + [dot])
    return results
context = turn(dots, context)

# Them: Yes, let's select the large one.
def select(dots, context):
    results = [dot for dots in context for dot in dots]
    for dot in results:
        if is_large(dot):
            return dot[None,None]
context = select(dots, context)


dots = get_dots()
context = []


# Them: i have a light grey small dot next to a medium grey medium dot.
def turn(dots, context):
    results = []
    for x,y in get2dots(dots):
        if is_close(x,y) and is_small(x) and is_light(x) and is_medium(y) and is_grey(y):
            results.append(np.array([x,y]))
    return results
context = turn(dots, context)


# You: yes i see that pair choose the small light grey dot <selection>.
def select(dots, context):
    results = [dot for dots in context for dot in dots]
    for dot in results:
        if is_small(dot) and is_light(dot) and is_grey(dot):
            return dot[None,None]
context = select(dots, context)



#print(context)
# context: num_candidates x size x feats=4
# dots: 7 x feats=4
# heuristic: take first candidate res[0]
res = (np.array(context)[:,None] == dots[:,None]).all(-1)
print(res[0].nonzero()[0].tolist())
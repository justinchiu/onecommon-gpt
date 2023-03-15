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
    dots = np.array([[0.83, -0.245, -0.3333333333333333, -0.44], [0.445, -0.72, 0.3333333333333333, -0.5466666666666666], [0.575, 0.39, -1.0, -0.8933333333333333], [-0.865, 0.32, -1.0, 0.9066666666666666], [0.215, -0.37, -0.3333333333333333, 0.84], [0.675, -0.39, 1.0, 0.6], [-0.57, 0.485, 0.3333333333333333, -0.6533333333333333]])
    return dots



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

# Them: i have a larger black dot all by itself down and to the left.
def turn(dots, context):
    results = []
    for dot in dots:
        if is_large(dot) and is_dark(dot) and is_left(dot) and is_below(dot):
            results.append(dot)
    return results
context = turn(dots, context)


#print(context)
# context: num_candidates x size x feats=4
# dots: 7 x feats=4
# heuristic: take first candidate res[0]
res = (np.array(context)[:,None] == dots[:,None]).all(-1)
print(res[0].nonzero()[0].tolist())

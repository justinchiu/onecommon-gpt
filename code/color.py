import numpy as np
# dots are x, y, size, color

# -1 is darkest
def is_dark(x, ctx):
    #return dot[-1] < -0.4
    return ctx[x,-1] < -0.3

# 1 is lightest
def is_light(x, ctx):
    # colors are in [-1,1]
    #return dot[-1] > 0.4
    return ctx[x,-1] > 0.3

def is_grey(x, ctx):
    return True
    return not is_dark(dot) and not is_light(dot)

def all_color(x, ctx):
    colors = ctx[x,-1]
    return np.abs(colors[None] - colors).max() < 0.1

def are_darker(x, y, ctx):
    return (ctx[x,None,-1] < ctx[y,-1]).all()

def are_lighter(x, y, ctx):
    return (ctx[x,None,-1] > ctx[y,-1]).all()

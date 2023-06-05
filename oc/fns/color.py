import numpy as np
# dots are x, y, size, color

# -1 is darkest
def is_dark(x, ctx):
    #return dot[-1] < -0.4
    #return ctx[x,-1] < -0.3
    return ctx[x,-1] < -0.2

# 1 is lightest
def is_light(x, ctx):
    # colors are in [-1,1]
    #return dot[-1] > 0.4
    #return ctx[x,-1] > 0.3
    return ctx[x,-1] > 0.2

def is_grey(x, ctx):
    #return True
    return ctx[x,-1] < 0.4 and ctx[x,-1] > -0.4
    return not is_dark(dot, ctx) and not is_light(dot, ctx)


def darkest(x, ctx):
    return x[np.argmin(ctx[x,-1])]

def lightest(x, ctx):
    return x[np.argmax(ctx[x,-1])]


def same_color(x, ctx):
    colors = ctx[x,-1]
    return np.abs(colors[:,None] - colors).max() < 0.1

def different_color(x, ctx):
    colors = ctx[x,-1]
    return np.abs(colors[:,None] - colors + np.eye(len(x))).min() > 0.3


def are_darker(x, y, ctx):
    return (ctx[x,None,-1] < ctx[y,-1]).all()

def are_lighter(x, y, ctx):
    return (ctx[x,None,-1] > ctx[y,-1]).all()

# simplify
def is_darker(x, y, ctx):
    if y is None:
        return is_dark(x, ctx)
    elif isinstance(y, int) or isinstance(y, np.int64):
        return ctx[x,-1] < ctx[y,-1]
    elif isinstance(y, list):
        return are_darker([x], y, ctx)
    else:
        ValueError
def is_lighter(x, y, ctx):
    if y is None:
        return is_light(x, ctx)
    elif isinstance(y, int) or isinstance(y, np.int64):
        return ctx[x,-1] > ctx[y,-1]
    elif isinstance(y, list):
        return are_lighter([x], y, ctx)
    else:
        ValueError

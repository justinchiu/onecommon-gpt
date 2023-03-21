import numpy as np


def is_large(x, ctx):
    return ctx[x, -2] > 0.3

def is_small(x, ctx):
    return ctx[x, -2] < -0.3

def is_medium(x, ctx):
    return True


def largest(x, ctx):
    return x[np.argmax(ctx[x,-2])]

def smallest(x, ctx):
    return x[np.argmin(ctx[x,-2])]


def same_size(x, ctx):
    sizes = ctx[x,-2] 
    return np.abs(sizes[:,None] - sizes).max() < 0.1

def different_size(x, ctx):
    sizes = ctx[x,-2]
    return np.abs(sizes[:,None] - sizes + np.eye(len(x))).min() > 0.3


def are_larger(x, y, ctx):
    return (ctx[x,None,-1] > ctx[y,-1]).all()

def are_smaller(x, y, ctx):
    return (ctx[x,None,-1] < ctx[y,-1]).all()

import numpy as np


def is_large(x, ctx):
    return ctx[x, -2] > 0.4

def is_small(x, ctx):
    return ctx[x, -2] < -0.4

def is_medium(x, ctx):
    return True


def largest(x, ctx):
    return np.argmax(ctx[x,-2])

def smallest(x, ctx):
    return np.argmin(ctx[x,-2])


def all_size(dots, ctx):
    sizes = ctx[x,-2] 
    return np.abs(sizes[None] - sizes).max() < 0.1

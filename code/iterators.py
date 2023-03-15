import numpy as np
import itertools


def get2dots(dots):
    N = dots.shape[0]
    idx = np.stack(list(itertools.combinations(np.arange(N), r=2)))
    return dots[idx]

def get3dots(dots):
    N = dots.shape[0]
    idx = np.stack(list(itertools.combinations(np.arange(N), r=3)))
    return dots[idx]

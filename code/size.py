import numpy as np


def is_large(dot):
    return dot[-2] > 0.4

def is_small(dot):
    return dot[-2] < -0.4

def is_medium(dot):
    return True


def largest(dots):
    return dots[np.argmax(dots[:,-2])]

def smallest(dots):
    return dots[np.argmin(dots[:,-2])]

import numpy as np

def add(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.concatenate([x,y], -1)
    elif isinstance(x, list) and isinstance(y, list):
        return np.array(x + y)

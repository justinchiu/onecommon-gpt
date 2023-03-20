import numpy as np
import itertools

def getnr(x, r):
    n = len(x)
    return x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]

def get2dots(x):
    return getnr(x, 2)

def get3dots(x):
    return getnr(x, 3)

if __name__ == "__main__":
    print(get2dots(np.arange(7))) 

import numpy as np
import itertools

def getnr(x, r):
    n = len(x)
    #return x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]
    x = np.array(x)
    out = x[np.stack(list(itertools.permutations(np.arange(n), r=r)))]
    return out.tolist()

def get1idxs(x):
    return [[x] for x in x]

def get2idxs(x):
    return getnr(x, 2)

def get3idxs(x):
    return getnr(x, 3)

if __name__ == "__main__":
    print(get2idxs(np.arange(7))) 

import numpy as np
import itertools



def getcombs(x, r, d=None):
    if d is not None:
        x = set(x).difference(set(d))
    n = len(x)
    x = np.array(x)
    out = x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]
    #out = x[np.stack(list(itertools.permutations(np.arange(n), r=r)))]
    return out.tolist()

def getsets(x, r, d=None):
    combs = getcombs(x,r, d)
    return [set(c) for c in combs]

def getnr(x, r, d=None):
    if d is not None:
        x = set(x).difference(set(d))
    n = len(x)
    #return x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]
    x = np.array(x)
    out = x[np.stack(list(itertools.permutations(np.arange(n), r=r)))]
    return out.tolist()

def get1idxs(x, d=None):
    if d is not None:
        return [[x] for x in x if x not in d]
    else:
        return [[x] for x in x]

def get2idxs(x, d=None):
    return getnr(x, 2, d=d)

def get3idxs(x, d=None):
    return getnr(x, 3, d=d)

if __name__ == "__main__":
    print(get2idxs(np.arange(7))) 

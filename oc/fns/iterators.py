import numpy as np
import itertools



def getcombs(x, r, exclude=None):
    if exclude is not None:
        x = set(x).difference(set(exclude))
    n = len(x)
    x = np.array(x)
    out = x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]
    #out = x[np.stack(list(itertools.permutations(np.arange(n), r=r)))]
    return out.tolist()

def getsets(x, r, exclude=None):
    combs = getcombs(x,r, exclude)
    return [set(c) for c in combs]

def getnr(x, r, exclude=None):
    if exclude is not None:
        x = set(x).difference(set(exclude))
    n = len(x)
    #return x[np.stack(list(itertools.combinations(np.arange(n), r=r)))]
    x = np.array(x)
    out = x[np.stack(list(itertools.permutations(np.arange(n), r=r)))]
    return out.tolist()

def get1idxs(x, exclude=None):
    if exclude is not None:
        return [[x] for x in x if x not in exclude]
    else:
        return [[x] for x in x]

def get2idxs(x, exclude=None):
    return getnr(x, 2, exclude=exclude)

def get3idxs(x, exclude=None):
    return getnr(x, 3, exclude=exclude)

if __name__ == "__main__":
    print(get2idxs(np.arange(7))) 

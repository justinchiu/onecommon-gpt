import numpy as np
from oc.fns.spatial import get_minimum_radius

def add(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.concatenate([x,y], -1)
    elif isinstance(x, list) and isinstance(y, list):
        return np.array(x + y)

def sort_state(configs, parents, ctx, select):
    if len(configs) == 0:
        return configs

    if not select:
        return list(sorted(configs, key=lambda x: get_minimum_radius(x, ctx)))
    else:
        radii = [get_minimum_radius(x, ctx) for x in parents]
        idxs = np.argsort(radii)
        return [configs[x] for x in idxs]

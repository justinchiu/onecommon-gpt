
import math

from enum import Enum

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from scipy.spatial import ConvexHull, Delaunay

def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(
        chain.from_iterable(combinations(range(n), k)), 
        int,
        count=count*k,
    )
    return index.reshape(-1, k)

def safe_log(x, eps=1e-10):
    result = np.where(x > eps, x, 0)
    np.log(result, out=result, where=result > 0)
    return result

# discrete entropy
def entropy(px):
    Hx = px * safe_log(px)
    return -(Hx).sum(-1)

# entropy for computing dot marginals
# px: num_dots
def marginal_entropy(px):
    px = np.stack((1-px, px), -1)
    return entropy(px)

def is_contiguous(x, xy, num_dots=7):
    if x.sum() <= 1:
        return True

    rg = np.arange(num_dots)
    pairs = np.array(list(itertools.product(rg, rg)))
    xy_pairs = xy[pairs].reshape((num_dots, num_dots, 2, 2))
    dist_pairs = np.linalg.norm(xy_pairs[:,:,0] - xy_pairs[:,:,1], axis=-1)
    idxs = dist_pairs.argsort()
    ranks = idxs.argsort()

    dots = x.nonzero()[0]

    def score_rec(dots, remaining_dots, score):
        if len(remaining_dots) == 0:
            return score
        remainder = np.delete(rg, dots)
        trunc_dists = dist_pairs[
            np.array(dots)[:,None],
            remainder,
        ]
        trunc_idxs = trunc_dists.argsort()
        trunc_ranks = trunc_idxs.argsort()

        dot_dists = dist_pairs[
            np.array(dots)[:,None],
            remaining_dots,
        ]
        closest_dots = remaining_dots[dot_dists.argmin(-1)]
        #best_ranks = ranks[np.array(dots), closest_dots]
        col, row = np.where(remainder[:,None] == closest_dots)
        best_ranks = trunc_ranks[row, col]
        best_rank = best_ranks.min()

        best_dot = closest_dots[best_ranks.argmin()]
        idx = np.where(remaining_dots == best_dot)[0].item()
        return score_rec(dots + [best_dot], np.delete(remaining_dots, idx), score + best_rank)

    scores = []
    for i, dot in enumerate(dots):
        remaining_dots = np.delete(dots, i)
        score = score_rec([dot], remaining_dots, 0)
        scores.append(score)

    return min(scores) == 0



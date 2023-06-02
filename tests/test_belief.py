
import math

from enum import Enum

import matplotlib.pyplot as plt

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from oc.belief.belief_utils import comb_index, entropy, marginal_entropy, get_config_idx

from oc.belief.belief import process_ctx, CostBelief, PriorType


#np.seterr(all="raise")

def get_last(belief, belief_dist, history, responses, n):
    confirmed_dots = [
        (
            dots,
            #get_config_idx(dots, belief.configs)
            belief.p_response(belief_dist, dots)[1]
        )
        for dots, r in zip(history, responses) if r == 1
    ]
    if len(confirmed_dots) == 0: return None
    print(confirmed_dots)
    #if len(confirmed_dots) == 2: import pdb; pdb.set_trace()
    for dots, prob in reversed(confirmed_dots):
        return dots
        if prob > 0.5: return dots
    return None


def rollout(ctx, ids, belief, responses, strategy):
    belief.history = []
    sc = belief.sc
    xy = belief.xy

    N = len(responses)
    fig, ax = plt.subplots(1, N, figsize=(4*N, 4))

    prior = belief.prior
    configs = belief.configs
    for n in range(N):
        print("n", n)
        repeat_mask = np.ones(128)
        if belief.history:
            history = np.stack(belief.history)
            repeat_mask = ~(belief.configs[:,None] == history).all(-1).any(-1)
        if strategy == "ig":
            EdHs = belief.compute_EdHs(prior)
            idx = (EdHs * repeat_mask).argmax()
        elif strategy == "igc":
            EdHs = belief.compute_EdHs(prior)
            last = get_last(belief, prior, belief.history, responses, n)
            plan_mask = belief.configs.sum(-1) == 2
            if last is not None:
                last_mask = ((belief.configs & last) == last).all(-1)
                size_mask = belief.configs.sum(-1) == last.sum() + 1
                plan_mask = last_mask * size_mask
            idx = (EdHs * repeat_mask * plan_mask).argmax()
        elif strategy == "ml":
            last = get_last(belief, prior, belief.history, responses, n)
            plan_mask = belief.configs.sum(-1) == 2
            if last is not None:
                last_mask = ((belief.configs & last) == last).all(-1)
                size_mask = belief.configs.sum(-1) == last.sum() + 1
                plan_mask = last_mask * size_mask
            confirms = np.array([
                # confirmation prob
                belief.p_response(prior, config)[1]
                for config in belief.configs
            ])
            idx = (confirms * repeat_mask * plan_mask).argmax()

        utt = belief.configs[idx]
        #print("utt", belief.configs[EdHs.argmax()])
        #print("marg utt", belief.configs[mEdHs.argmax()])

        uttb = utt.astype(bool)
        ax[n].scatter(
            xy[:,0], xy[:,1],
            marker='o',
            s=50*(1+sc[:,0]),
            #c=-80*(sc[:,1]),
            #s = 50*(ctx[:,2] + ctx[:,2].min() + 1),
            c = -ctx[:,3],
            cmap="binary",
            edgecolor="black",
            linewidth=1,
        )
        ax[n].scatter(xy[uttb,0], xy[uttb,1], marker="x", s=100, c="r")
        for i, id in enumerate(ids):
            ax[n].annotate(id, (xy[i,0]+.025, xy[i,1]+.025))

        print("utt", utt)
        print("prior", belief.marginals(prior))
        print(responses[n])
        new_prior = belief.posterior(prior, utt, responses[n])
        print("posterior", belief.marginals(new_prior))

        belief.history.append(utt)
        prior = new_prior

    plt.savefig(f"plan_plots/{strategy}.png")


def main():
    num_dots = 7

    # scenario S_pGlR0nKz9pQ4ZWsw
    # streamlit run main.py
    ctx = np.array([
        0.635, -0.4,   2/3, -1/6,  # 8
        0.395, -0.7,   0.0,  3/4,  # 11
        -0.74,  0.09,  2/3, -2/3,  # 13
        -0.24, -0.63, -1/3, -1/6,  # 15
        0.15,  -0.58,  0.0,  0.24, # 40
        -0.295, 0.685, 0.0, -8/9,  # 50
        0.035, -0.79, -2/3,  0.56, # 77
    ], dtype=float).reshape(-1, 4)
    ids = np.array(['8', '11', '13', '15', '40', '50', '77'], dtype=int)

    # reflect across y axis
    ctx[:,1] = -ctx[:,1]

    beliefs = [
        #CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5, prior_type=PriorType.UNIFORM),
        CostBelief(num_dots, ctx, num_size_buckets=3, num_color_buckets=3, prior_type=PriorType.ISING),
        #CostBelief(num_dots, ctx, num_size_buckets=3, num_color_buckets=3, prior_type=PriorType.MST),
    ]
    responses = [
        [1,0,1,0,0,0,0],
        #[0,1,0,1,1],
    ]
    strategies = [
        "ig",
        "igc",
        "ml",
    ]
    for belief in beliefs:
        belief.ids = ids
        for response in responses:
            for strategy in strategies:
                print(strategy)
                rollout(ctx, ids, belief, response, strategy)


if __name__ == "__main__":
    main()

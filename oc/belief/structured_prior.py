import matplotlib.pyplot as plt
import math

import numpy as np
from scipy.special import logsumexp as lse


def visualize(configs, prior, name=None):
    fig, ax = plt.subplots(1, 2**num_dots, figsize=(3*(2**num_dots), 2))
    for i, (config, prob) in enumerate(zip(configs, prior)):
        bconfig = config.astype(bool)
        ax[i].scatter(
            xy[bconfig,0], xy[bconfig,1], marker="o", s=100,
        )
        ax[i].set_title(f"Prob {math.exp(prob):.2f}")
        ax[i].set_xlim(-1, 1)
        ax[i].set_ylim(-1, 1)
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()
    plt.clf()


# ising model prior based on distance
def ising_prior(configs, dists, tau=16):
    # convert to rademacher
    rad_configs = np.where(configs == 0, -1, configs)
    log_unnormalized_prior = np.einsum("di,ij,dj->d", rad_configs, -dists, rad_configs) / tau
    Z = lse(log_unnormalized_prior)
    log_prior = log_unnormalized_prior - Z
    return log_prior
    prior = np.exp(log_prior)
    return prior


def mst_rec(num_dots, dist_pairs, dots, remaining_dots, score, edges):
    if len(remaining_dots) == 0:
        return score, edges
    trunc_dists = dist_pairs[
        dots[:,None],
        remaining_dots,
    ]
    src, tgt = np.unravel_index(np.argmin(trunc_dists), trunc_dists.shape)
    distance = trunc_dists[src, tgt]

    best_dot = remaining_dots[tgt]
    idx = np.where(remaining_dots == best_dot)[0].item()
    return mst_rec(
        num_dots,
        dist_pairs,
        np.append(dots, best_dot),
        np.delete(remaining_dots, idx),
        score + distance,
        edges + [(dots[src], best_dot)],
    )


def mst_prior(configs, dists, tau=3):
    num_dots = configs.shape[1]
    log_unnormalized_prior = np.array([
        -mst_rec(num_dots, dists, x.nonzero()[0][:1], x.nonzero()[0][1:], 0, [])[0] / tau
        for x in configs
    ])
    Z = lse(log_unnormalized_prior)
    log_prior = log_unnormalized_prior - Z
    return log_prior
    prior = np.exp(log_prior)
    return prior


if __name__ == "__main__":
    from rich.progress import track

    num_dots = 4

    configs = np.array([
        np.unpackbits(np.array([x], dtype=np.ubyte))[8-num_dots:]
        for x in range(2 ** num_dots)
    ])

    xy = np.random.rand(num_dots,2) * 2 - 1
    dists = ((xy[:,None] - xy[None]) ** 2).sum(-1)
    visualize(configs, ising_prior(configs, dists))
    visualize(configs, mst_prior(configs, dists))

    score, edges = mst_rec(
        num_dots, dists,
        np.array([0]),
        np.array([1,2,3]),
        0,
        [],
    )

    # check likelihood of configurations
    from hfdata import corpus
    num_dots = 7
    configs = np.array([
        np.unpackbits(np.array([x], dtype=np.ubyte))[8-num_dots:]
        for x in range(2 ** num_dots)
    ])
    tau_mst_losses = {
        t: 0 for t in [2,3,4]
        # best was 3, -39939 train -4998 valid
    }
    tau_ising_losses = {
        t: 0 for t in [12,16,20]
        # best was 16, -38544 train -4848 valid
    }
    num_examples = 0
    # takes 3 minutes
    for example in track(corpus.train):
        context = np.array(example.input_vals, dtype=float).reshape(-1, 4)
        ids = [int(x) for x in example.real_ids]
        partner_ids = set([int(x) for x in example.partner_real_ids])
        intersect = np.array([x in partner_ids for x in ids])

        config_idx = (configs == intersect).all(-1).nonzero()[0]

        xy = context[:,:2]
        dists = ((xy[:,None] - xy[None]) ** 2).sum(-1)
        for tau in tau_mst_losses.keys():
            log_p_intersect = mst_prior(configs, dists, tau)
            tau_mst_losses[tau] += log_p_intersect[config_idx]
        for tau in tau_ising_losses.keys():
            log_p_intersect = ising_prior(configs, dists, tau)
            tau_ising_losses[tau] += log_p_intersect[config_idx]
        num_examples += 1
    print(tau_mst_losses)
    print(tau_ising_losses)


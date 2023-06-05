
import math

from enum import Enum

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb
from scipy.special import logsumexp as lse

from scipy.spatial import ConvexHull, Delaunay

from oc.belief.belief_utils import comb_index, entropy, marginal_entropy
from oc.belief.structured_prior import ising_prior, mst_prior


class Label(Enum):
    ERROR = 0
    COARSE = 1
    UNRESOLVABLE = 2
    SPECIFIC = 3


class PriorType(Enum):
    UNIFORM = 1
    ISING = 2
    MST = 3


def label_config_sets(writer_configs, reader_configs):
    """
    The configs passed in give the dot ids of the plans.
    The writer_configs give which dots the writer intended to refer to,
    while the reader_configs give the set of resolved reader dots.
    """
    # writer_configs: num_writer_configs x utt_size
    # reader_configs: num_reader_configs x utt_size
    num_writer_configs = writer_configs.shape[0]
    num_reader_configs = reader_configs.shape[0]

    if (
        # reader did not resolve to any dots
        num_writer_configs > 0 and num_reader_configs == 0
    ):
        return Label.UNRESOLVABLE
    elif (
        # configs are exactly identical
        (num_writer_configs == 0
            and num_reader_configs == 0)
        or (
            writer_configs.shape == reader_configs.shape
            and (writer_configs == reader_configs).all()
        )
    ):
        return Label.SPECIFIC
    elif (
        # all resolved dots were intended
        num_writer_configs > 0
        and num_reader_configs > 0
        and (writer_configs[:,None] == reader_configs[None]).all(-1).any()
    ):
        return Label.COARSE
    elif (
        # not all resolved dots are intended
        num_writer_configs > 0
        and num_reader_configs > 0
        and not (writer_configs[:,None] == reader_configs[None]).all(-1).any()
    ):
        return Label.ERROR
    else:
        raise ValueError("Unhandled config label case")

def process_ctx(
    ctx,
    absolute=True,
    num_size_buckets = 5,
    num_color_buckets = 5,
):
    # ctx: [x, y, size, color]
    eps = 1e-3
    min_ = ctx.min(0)
    max_ = ctx.max(0)

    if absolute:
        # ABSOLUTE BUCKET
        min_size, max_size = -1, 1
        min_color, max_color = -1, 1
    else:
        # RELATIVE BUCKET
        min_size, max_size = min_[2], max_[2]
        min_color, max_color = min_[3], max_[3]
    

    size_buckets = np.linspace(min_size, max_size + eps, num_size_buckets+1)
    color_buckets = np.linspace(min_color, max_color + eps, num_color_buckets+1)
    sizes = ctx[:,2]
    colors = ctx[:,3]

    size_idxs = (size_buckets[:-1,None] <= sizes) & (sizes < size_buckets[1:,None])
    color_idxs = (color_buckets[:-1,None] <= colors) & (colors < color_buckets[1:,None])
    return np.stack((size_idxs.T.nonzero()[1], color_idxs.T.nonzero()[1]), 1)


# convert plan to sequence of mentions
def expand_plan(plan, unroll=True):
    # plan: {0,1}^7
    num_dots = plan.sum().item()
    if num_dots == 0:
        print("EMPTY PLAN")
        mentions = plan
    elif num_dots <= 1 or not unroll:
        mentions = plan[None, None]
    else:
        mentions = np.zeros((num_dots + 1, 1, 7))
        mentions[0,0] = plan
        mentions[np.arange(1,num_dots+1), 0, plan.nonzero()[0]] = 1
    return mentions

class Dot:
    def __init__(self, item):
        for k,v in item.items():
            setattr(self, k, v)

    def html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size
        f = self.color
        label = f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id}</text>'
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" /> {label}'

    def select_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 2
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="3" stroke-dasharray="3,3"  />'

    def intersect_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 4
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="blue" stroke-width="3" stroke-dasharray="3,3"  />'

    def __repr__(self):
        return f"Dot {self.id}: ({self.x}, {self.y}) r={self.size} f={self.color}"


class Belief:
    def __init__(self, num_dots, overlap_size=None):
        self.num_dots = num_dots
        self.overlap_size = overlap_size
        self.configs = np.array([
            np.unpackbits(np.array([x], dtype=np.ubyte))[8-num_dots:]
            for x in range(2 ** num_dots)
        ])
        self.num_configs = 2 ** num_dots
        self.history = []

    def joint(self, prior, utt):
        raise NotImplementedError

    def p_response(self, prior, utt):
        raise NotImplementedError

    def posterior(self, prior, utt, response):
        raise NotImplementedError

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return Hs - Hs_r

    def marginal_info_gain(self, prior, utt, response):
        marginal_prior = self.marginals(prior)
        marginal_prior = np.stack((1-marginal_prior, marginal_prior), -1)
        Hs = entropy(marginal_prior)
        marginal_post = self.marginals(self.posterior(prior, utt, response))
        marginal_post = np.stack((1-marginal_post, marginal_post), -1)
        Hs_r = entropy(marginal_post)
        return Hs - Hs_r

    def expected_info_gain(self, prior, utt):
        raise NotImplementedError

    def expected_marginal_info_gain(self, prior, utt):
        raise NotImplementedError

    def expected_marginal_posterior(self, prior, utt):
        raise NotImplementedError

    def compute_EdHs(self, prior):
        EdHs = []
        for utt in self.configs:
            EdH = self.expected_info_gain(prior, utt)
            EdHs.append(EdH)
        return np.array(EdHs)

    def compute_marginal_EdHs(self, prior):
        EdHs = []
        for utt in self.configs:
            EdH = self.expected_marginal_info_gain(prior, utt)
            EdHs.append(EdH)
        return np.stack(EdHs)

    def compute_marginal_posteriors(self, prior):
        posteriors = []
        for utt in self.configs:
            posterior = self.expected_marginal_posterior(prior, utt)
            posteriors.append(posterior)
        return np.stack(posteriors)

    def compute_lengths(self):
        return self.configs.sum(-1)

    def compute_diameters(self):
        diameters = []
        for utt in self.configs:
            utt_size = utt.sum().item()
            if utt_size == 0:
                diameters.append(10)
                continue
            elif utt_size == 1:
                diameters.append(0)
                continue
            xy = self.ctx[utt.astype(bool), :2]
            pairs = comb_index(utt_size, 2)
            # num_combs x num_pairs x dots=2 x position
            pairwise_positions = xy[pairs]
            pairwise_dists = np.linalg.norm(
                pairwise_positions[:,:,0] - pairwise_positions[:,:,1],
                axis=-1,
            )
            diameter = pairwise_dists.max(-1)
            diameters.append(diameter)
        return np.array(diameters)

    def compute_contiguity(self):
        contiguous = []
        xy = self.ctx[:,:2]
        for utt in self.configs:
            utt_size = utt.sum().item()
            # only compute contiguity for > 2?
            # delaunay fails for line segments
            if utt_size <= 2:
                contiguous.append(1)
                continue
            uttb = utt.astype(bool)
            tri = Delaunay(xy[uttb])
            # check if other dots are in hull
            outside = True
            for i in range(7):
                if utt[i] == 0:
                    outside &= tri.find_simplex(xy[i]) == -1
                """
                if not outside:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.plot(xy[:,0], xy[:,1], 'o')
                    for j in range(7):
                        ax.annotate(str(j), (xy[j,0], xy[j,1]))
                    ax.triplot(xy[uttb,0], xy[uttb,1], tri.simplices)
                    print(i)
                    plt.show()
                """
            contiguous.append(1 if outside else 0)
        return np.array(contiguous)

    def compute_processing_costs(self, new_weight, distance_weight):
        MAX = self.num_dots * new_weight
        if len(self.history) == 0:
            scores = self.configs.sum(-1) * new_weight
            scores[0] = MAX
            logits = -scores.astype(float)
            return logits - lse(logits) 

        scores = []
        history = np.stack(self.history)[::-1]
        past = history.sum(0)
        for utt in self.configs:
            dots = utt.nonzero()[0]

            is_new = past[dots] == 0
            distance = 1 + history[
                np.arange(history.shape[0])[:,None],
                utt.astype(bool),
            ].argmax(0)

            # linear distance
            score = is_new * new_weight + distance * distance_weight * ~is_new

            # no repeats
            is_repeat = (history == utt).all(-1).any()

            scores.append(score.sum() if not is_repeat else MAX)
        logits = -np.array(scores, dtype=float)
        logits[0] = -MAX


        return logits - lse(logits) 

    def compute_utilities(
        self,
        prior,
        length_coef = 0,
        diameter_coef = 0,
        contiguity_coef = 0,
        processing_coef = 0,
    ):
        # we want to MAXIMIZE utility
        EdHs = self.compute_EdHs(prior)
        utility = EdHs
        if length_coef > 0:
            lengths = self.compute_lengths()
            # penalize longer plans
            utility -= length_coef * lengths
        if diameter_coef > 0:
            # penalize distant plans
            utility -= diameter_coef * self.diameters
        if contiguity_coef > 0:
            # reward contiguous plans
            utility += contiguity_coef * self.contiguous
        if processing_coef > 0:
            utility += processing_coef * self.compute_processing_costs()
        return utility

    def viz_belief(self, p, n=5):
        # decreasing order
        idxs = (-p).argsort()[:n]
        cs = self.configs[idxs]
        ps = p[idxs]
        return cs, ps

    def marginals(self, p):
        return (self.configs * p[:,None]).sum(0)

    def marginal_size(self, p, size=3):
        size_mask = self.configs.sum(-1) >= size
        #N = int(size_mask.sum())

        idxs = np.arange(self.num_configs)

        config_prob = np.zeros(self.num_configs)
        for i, utt in enumerate(self.configs):
            utt = utt.astype(bool)
            config_mask = self.configs[:,utt].all(-1)
            config_prob[i] = p.dot(config_mask)

        config_scores = size_mask * config_prob
        #config_scores = config_prob
        config_scores[0] = 0
        config_idxs = (-config_scores).argsort()

        for idx in config_idxs:
            config = self.configs[idx]
            feats = self.get_feats(config)
            idxs = self.resolve_utt(*feats)
            if config_scores[idx] == 0:
                print("TRIED TO SELECT CONFIG WITH 0 PROB")
            #if len(idxs) > 1:
                #import pdb; pdb.set_trace()
            if len(idxs) == 1:
                return config
                #return self.configs[(size_mask * config_prob).argmax()]


class IndependentBelief(Belief):
    """
    Fully independent partner model
    * response r: num_dots
    * utterance u: num_dots
    * state s: num_dots
    p(r|u,s) = prod_i p(r_i|u_i,s_i)
    Underestimates failures from large configurations due to
    independence assumption.
    """
    def __init__(self, num_dots, correct=0.9):
        super().__init__(num_dots)
        # VERY PESSIMISTIC PRIOR
        # actually, this is incorrect.
        # if we know 6/7 overlap, the marginal dist should be 1/7 not included
        # given K overlap, marginal dist is 1 - 6Ck / 7Ck = k/7
        # guess we should not assume K overlap though, be even dumber?
        state_prior = np.ones((num_dots,)) / 2
        self.prior= np.stack((state_prior, 1-state_prior), 1)

        # initialize basic likelihood
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # utt about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont utt about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    # RESPONSE IS FOR ALL DOTS INDEPENDENTLY
    def p_response(self, prior, utt):
        return (self.likelihood[:,utt] * prior).sum(-1).T

    def posterior(self, prior, utt, response):
        f = self.likelihood[response, utt] * prior
        return f / f.sum(-1, keepdims=True)

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return (Hs - Hs_r)[utt.astype(bool)].sum()

    def expected_info_gain(self, prior, utt):
        p_response = self.p_response(prior, utt)
        Hs = entropy(prior)
        r0 = np.zeros((self.num_dots,), dtype=int)
        r1 = np.ones((self.num_dots,), dtype=int)
        Hs_r0 = entropy(self.posterior(prior, utt, r0))
        Hs_r1 = entropy(self.posterior(prior, utt, r1))
        EHs_r = (p_response * np.stack((Hs_r0, Hs_r1), 1)).sum(-1)
        return (Hs - EHs_r)[utt.astype(bool)].sum()



class AndBelief(Belief):
    """
    Noisy-and model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    p(r=1|u,s) = prod_i p(r=1|u_i=1,s_i)
    Accurately estimates failure of large configurations,
    under-estimates failure of small configurations due to ignoring partial observability.
    """

    def __init__(
        self,
        num_dots,
        ctx = None,
        overlap_size = None,
        correct = 0.95,
        num_size_buckets = 5,
        num_color_buckets = 5,
        absolute = True,
        prior_type = PriorType.UNIFORM,
    ):
        super().__init__(num_dots, overlap_size)

        self.ctx = np.array(ctx, dtype=float).reshape(num_dots, 4)
        self.size_color = process_ctx(
            self.ctx,
            absolute = absolute,
            num_size_buckets = num_size_buckets,
            num_color_buckets = num_color_buckets,
        )
        self.sc = self.size_color
        self.xy = self.ctx[:,:2]

        self.prior_type = prior_type
        if prior_type == PriorType.UNIFORM:
            self.prior = np.ones((2 ** num_dots,))
            if overlap_size is not None:
                self.prior[self.configs.sum(-1) < overlap_size] = 0
                #self.prior[self.configs.sum(-1) != overlap_size] = 0
                self.prior[-1] = 0
            self.prior = self.prior / self.prior.sum()
        elif prior_type == PriorType.ISING:
            dists = ((self.xy[:,None] - self.xy[None]) ** 2).sum(-1)
            prior = np.exp(ising_prior(self.configs, dists))
            self.prior = prior
        elif prior_type == PriorType.MST:
            dists = ((self.xy[:,None] - self.xy[None]) ** 2).sum(-1)
            prior = np.exp(mst_prior(self.configs, dists))
            self.prior = prior
        else:
            raise ValueError(f"Invalid prior_type {prior_type}")
        self.prior[0] = 0
        self.prior = self.prior / self.prior.sum()


        # initialize basic likelihood
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # utt about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont utt about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    def p_response(self, prior, utt):
        # prior: num_configs * 7
        # \sum_s p(r=1 | u, s)p(s) = \sum_s \prod_i p(r=1 | ui, si)p(s)
        p_r1 = 0
        p_r0 = 0
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(utt):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1 += likelihood * p
            p_r0 += (1-likelihood) * p
        #p_r0 = 1 - p_r1 # this is equivalent
        return np.array((p_r0, p_r1))

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(utt):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        Z1 = p_r1s_u.sum(-1, keepdims=True)
        p_s_ur1 = p_r1s_u / Z1 if Z1 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        Z2 = p_r0s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / Z2 if Z2 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        return p_s_ur1 if response == 1 else p_s_ur0

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return Hs - Hs_r

    def expected_info_gain(self, prior, utt):
        p_response = self.p_response(prior, utt)
        Hs = entropy(prior)
        Hs_r0 = entropy(self.posterior(prior, utt, 0))
        Hs_r1 = entropy(self.posterior(prior, utt, 1))
        EHs_r = (p_response * np.array((Hs_r0, Hs_r1))).sum()
        return Hs - EHs_r

    def expected_marginal_info_gain(self, prior, utt):
        p_response = self.p_response(prior, utt)
        Hs = marginal_entropy(self.marginals(prior))
        Hs_r0 = marginal_entropy(self.marginals(self.posterior(prior, utt, 0)))
        Hs_r1 = marginal_entropy(self.marginals(self.posterior(prior, utt, 1)))
        EHs_r = (p_response[:,None] * np.stack((Hs_r0, Hs_r1))).sum(0)
        return Hs - EHs_r

    def expected_marginal_posterior(self, prior, utt):
        p_response = self.p_response(prior, utt)
        s_r0 = self.posterior(prior, utt, 0)
        s_r1 = self.posterior(prior, utt, 1)
        return (
            p_response[:,None] * np.log(np.stack((
                self.marginals(s_r0), self.marginals(s_r1)
            )))
        ).sum(0)

        posterior = (p_response[:,None] * np.stack((s_r0, s_r1))).sum(0)
        E_log_posterior = (p_response[:,None] * np.log(np.stack((s_r0, s_r1)))).sum(0)
        return self.marginals(posterior)

class AndOrBelief(AndBelief):
    """
    And-or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the dot-level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|
    p(r=1|u,s) = prod_i p(r=1|u_i,s_i,z) = prod_i 1 - p(r=0|ui,si)p(r=0|ui,z)
    Accurately estimates failure of small and large configurations.
    As the OR happens at the dot level, does not prefer large configurations.

    Note on p(r=0|ui,z) = (8/9)^|z|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def p_response(self, prior, utt):
        # prior: num_configs * 7
        # \sum_s p(r=1 | u, s)p(s)
        # = \sum_s,z p(s)p(z|s) \prod_i 1-p(r=0|ui,si)p(r=0|ui,z)
        # = \sum_s p(s) \prod_i 1-p(r=0|ui,si)(8/9)^{n-|s|}
        p_r1 = 0
        p_r0 = 0
        for s,ps in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            for i,d in enumerate(utt):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1 += likelihood * ps
            p_r0 += (1-likelihood) * ps
        #p_r0 = 1 - p_r1 # this is equivalent
        return np.array((p_r0, p_r1))

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            for i,d in enumerate(utt):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        Z1 = p_r1s_u.sum(-1, keepdims=True)
        p_s_ur1 = p_r1s_u / Z1 if Z1 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        Z0 = p_r0s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / Z0 if Z0 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        return p_s_ur1 if response == 1 else p_s_ur0

class OrAndBelief(AndBelief):
    """
    Or-and model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    Noisy-AND for dots and state
        p(r=1|u,s) = prod_i p(r=1|u_i,s_i)
        p(r=0|u,s) = 1-p(r=1|u,s)
    Noisy-OR
        p(r=0|u,s,z) = 1-p(r=0|u,s)p(r=0|u,z)
    Dot distractors
        p(r=0|u,z) = 1 - |z|C|u| 9^-|u|

    Accurately estimates failure of small and large configurations.

    Note on p(r=0|u,z) = 1-|z|C|u|9^-|u|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """

    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1-\prod_i p(r=1|ui,si)) \sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1-\prod_i p(r=1|ui,si)) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            likelihood = 1
            for i,d in enumerate(utt):
                if d == 1:
                    likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r0.append((1 - likelihood)*distractor_prob * ps)
            p_r1.append((1- (1-likelihood)*distractor_prob) * ps)
        return np.array((p_r0, p_r1))

    def p_response(self, prior, utt):
        return self.joint(prior, utt).sum(1)

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_rs_u = self.joint(prior, utt)
        Z = p_rs_u.sum(1, keepdims=True)
        unif = np.ones((2, 2 ** self.num_dots)) / 2 ** self.num_dots
        p_s_ur = np.divide(p_rs_u, Z, out=unif, where=Z>0)
        return p_s_ur[response]

class OrBelief(OrAndBelief):
    """
    Or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    Normal model for dots and state
        p(r=1|u,s) = initialization
        p(r=0|u,s) = 1-p(r=1|u,s)
    Noisy-OR
        p(r=1|u,s,z) = 1-p(r=0|u,s)p(r=0|u,z)
    Dot distractors
        p(r=0|u,z) = 1 - |z|C|u| 9^-|u|

    Accurately estimates failure of small and large configurations.

    Note on p(r=0|u,z) = 1-|z|C|u|9^-|u|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def __init__(
        self,
        num_dots,
        ctx,
        correct = 0.95,
        overlap_size = None,
        absolute = True,
        use_diameter = False,
        use_contiguity = False,
        num_size_buckets = 5,
        num_color_buckets = 5,
        prior_type = PriorType.MST,
    ):
        super().__init__(
            num_dots,
            ctx = ctx,
            overlap_size = overlap_size,
            correct = correct,
            absolute = absolute,
            num_size_buckets = num_size_buckets,
            num_color_buckets = num_color_buckets,
            prior_type = prior_type,
        )
        # initialize config_likelihood based on configuration resolution
        self.config_likelihood = np.zeros(
            (self.num_configs, self.num_configs), dtype=float)
        self.resolvable= np.zeros(
            (self.num_configs, self.num_configs), dtype=bool)
        for u, utt in enumerate(self.configs):
            for s, config in enumerate(self.configs):
                self.resolvable[u,s] = self.can_resolve_utt(utt, config)
                self.config_likelihood[u,s] = (
                    correct if self.resolvable[u,s] else 1 - correct
                )
        # for computing utility
        self.use_diameter = use_diameter
        self.use_contiguity = use_contiguity
        if use_diameter:
            self.diameters = self.compute_diameters()
        if use_contiguity:
            self.contiguous = self.compute_contiguity()

    def can_resolve_utt(self, utt, config):
        size_color = self.size_color
        xy = self.xy

        utt_size = utt.sum().item()
        config_size = config.sum().item()
        if utt_size == 0 or config_size == 0:
            return False

        utt_sc = size_color[utt.astype(bool)]
        utt_xy = xy[utt.astype(bool)]
        if utt_size == 1:
            return (utt_sc == size_color[config.astype(bool)]).all(-1).any()

        arange = np.arange(self.num_dots)
        perms = np.array(list(itertools.permutations(np.arange(utt_size))))
        num_perms = perms.shape[0]
        pairwise_combs = np.array(list(itertools.combinations(np.arange(utt_size), 2)))

        utt_pairwise_xy = utt_xy[pairwise_combs]
        utt_rel_xy = utt_pairwise_xy[:,0] > utt_pairwise_xy[:,1]

        # get all dot combinations of size utt_size
        config_ids = arange[config.astype(bool)]
        combs = np.array(list(itertools.combinations(config_ids, utt_size)))
        # will check permutations within combinations below (batched)
        for comb in combs:
            comb_sc = size_color[comb]
            comb_xy = xy[comb]

            # filters if all dots have a match with unary features
            sc_match = (utt_sc[:,None] == comb_sc[None,:]).all(-1).any(1).all()

            if sc_match:
                # check if there is a permutation that matches
                sc_perms = comb_sc[perms]
                xy_perms = comb_xy[perms]

                sc_matches = (utt_sc == sc_perms).reshape(num_perms, -1).all(-1)
                xy_potential_matches = xy_perms[sc_matches]
                for xy_config in xy_potential_matches:
                    config_pairwise_xy = xy_config[pairwise_combs]
                    config_rel_xy = config_pairwise_xy[:,0] > config_pairwise_xy[:,1]

                    xy_match = (utt_rel_xy == config_rel_xy).all()
                    if xy_match:
                        return True
        return False

    def get_feats(self, utt):
        utt_size = utt.sum().item()
        utt_sc = self.size_color[utt.astype(bool)]
        utt_xy = self.xy[utt.astype(bool)]
        return utt_size, utt_sc, utt_xy

    def resolve_utt(self, utt_size, utt_sc, utt_xy):
        # helper function for resolving utterances to our context
        # based on shape / color and relative xy locations
        # somewhat repeated code
         
        size_color = self.size_color
        xy = self.xy

        #utt_size = utt.sum().item()
        #config_size = config.sum().item()
        config_size = self.num_dots
        if utt_size == 0 or config_size == 0:
            return []

        #utt_sc = size_color[utt.astype(bool)]
        #utt_xy = xy[utt.astype(bool)]
        if utt_size == 1:
            return (utt_sc == size_color).all(-1).nonzero()[0][:,None]

        arange = np.arange(self.num_dots)
        perms = np.array(list(itertools.permutations(np.arange(utt_size))))
        num_perms = perms.shape[0]
        pairwise_combs = np.array(list(itertools.combinations(np.arange(utt_size), 2)))

        utt_pairwise_xy = utt_xy[pairwise_combs]
        utt_rel_xy = utt_pairwise_xy[:,0] > utt_pairwise_xy[:,1]

        # get all dot combinations of size utt_size
        config_ids = arange
        combs = np.array(list(itertools.combinations(config_ids, utt_size)))
        matches = []
        # will check permutations within combinations below (batched)
        for comb in combs:
            comb_sc = size_color[comb]
            comb_xy = xy[comb]

            # filters if all dots have a match with unary features
            sc_match = (utt_sc[:,None] == comb_sc[None,:]).all(-1).any(1).all()

            if sc_match:
                # check if there is a permutation that matches
                sc_perms = comb_sc[perms]
                xy_perms = comb_xy[perms]

                sc_matches = (utt_sc == sc_perms).reshape(num_perms, -1).all(-1)
                xy_potential_matches = xy_perms[sc_matches]
                for xy_config in xy_potential_matches:
                    config_pairwise_xy = xy_config[pairwise_combs]
                    config_rel_xy = config_pairwise_xy[:,0] > config_pairwise_xy[:,1]

                    xy_match = (utt_rel_xy == config_rel_xy).all()
                    if xy_match:
                        matches.append(comb)
        return np.stack(matches) if matches else np.array([], dtype=int)

    def joint(self, prior, utt):
        # p(r,s | u)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1 - p(r=1|u,s)) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots)
            likelihood = self.config_likelihood[utt_idx,s].item()
            #for i,d in enumerate(utt):
                #if d == 1:
                    #likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r0.append((1 - likelihood)*distractor_prob * ps)
            p_r1.append((1- (1-likelihood)*distractor_prob) * ps)
        return np.array((p_r0, p_r1))

    """
    def p_response(self, prior, utt):
        joint = self.joint(prior, utt) 
        std = joint.sum(1)
        log = np.exp(lse(np.log(joint), 1))
        if (np.abs(std - log) > 0.001).any():
            import pdb; pdb.set_trace()
        return log
    """


class CostBelief(OrBelief):
    """
    Or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    Normal model for dots and state
        p(r=1|u,s) = initialization
        p(r=0|u,s) = 1-p(r=1|u,s)
    Noisy-OR
        p(r=1|u,s,z) = 1-p(r=0|u,s)p(r=0|u,z)
    Dot distractors
        p(r=0|u,z) = 1 - |z|C|u| 9^-|u|

    Accurately estimates failure of small and large configurations.

    Note on p(r=0|u,z) = 1-|z|C|u|9^-|u|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def __init__(
        self,
        num_dots,
        ctx,
        correct = 0.95,
        overlap_size = None,
        absolute = True,
        use_diameter = False,
        use_contiguity = False,
        num_size_buckets = 5,
        num_color_buckets = 5,
        prior_type = PriorType.MST,
    ):
        super().__init__(
            num_dots,
            ctx,
            overlap_size = overlap_size,
            absolute = absolute,
            correct = correct,
            use_diameter = use_diameter,
            use_contiguity = use_contiguity,
            num_size_buckets = num_size_buckets,
            num_color_buckets = num_color_buckets,
            prior_type = prior_type,
        )

        # redo the dot likelihood
        self.spatial_resolvable = np.zeros((self.num_configs,), dtype=bool)
        for u, utt in enumerate(self.configs):
            self.spatial_resolvable[u] = self.is_contiguous(utt)
            for s, config in enumerate(self.configs):
                self.config_likelihood[u,s] = (
                    correct
                    if self.resolvable[u,s] and self.spatial_resolvable[u]
                    else 1 - correct
                )

    def is_contiguous(self, x):
        if x.sum() <= 1:
            return True

        rg = np.arange(self.num_dots)
        xy = self.xy
        pairs = np.array(list(itertools.product(rg, rg)))
        xy_pairs = xy[pairs].reshape((self.num_dots, self.num_dots, 2, 2))
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

        return min(scores) == 0 if x.sum() == 2 else min(scores) <= 2


class EgoCostBelief(CostBelief):
    # ABLATED version of CostBelief
    # Does not consider unshared dots
    # same method as belief.py:ConfigBelief
    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1 - p(r=1|u,s)) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots)
            likelihood = self.config_likelihood[utt_idx,s].item()
            #for i,d in enumerate(utt):
                #if d == 1:
                    #likelihood *= self.likelihood[1,d,state_config[i]]
            #distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r1.append(likelihood * ps)
            p_r0.append((1 - likelihood) * ps)
        return np.array((p_r0, p_r1))


import numpy as np
import itertools

from oc.agent.utils import Plan

from oc.fns.spatial import get_minimum_radius

def new_and_old_dots(plan, history):
    if len(history) > 0:
        plans = [x.dots for x in history if x is not None]
        if len(plans) == 0:
            return plan, None
        plans = np.any(plans, axis=0)
        return plan & ~plans, plans
    else:
        return plan, None

def idxs_to_dots(plan_idxs, plan_idxs2, view):
    dotsets = [set(x) for x in plan_idxs]
    dotsets2 = [set(x) for x in plan_idxs2]
    setpairs = list(itertools.product(dotsets, dotsets2))
    smalldiffs = [(x,y) for x,y in setpairs if len(y.difference(x)) == 1]
    radii = [get_minimum_radius(list(y), view) for x,y in smalldiffs]
    smallest_idx = np.argmin(radii)
    olddotset = smalldiffs[smallest_idx][0]
    newdotset = smalldiffs[smallest_idx][1]
    newdot = list(newdotset.difference(olddotset))
    olddots = list(olddotset)
    newdots = list(newdotset)
    return newdots, olddots

class PlannerMixin:

    def update_belief(self, dots, response):
        #plan = self.plans[-1].dots
        #plan = self.preds[-1][0]
        self.belief_dist = self.belief.posterior(
            self.belief_dist,
            dots.astype(int),
            response,
        )

    def plan(self):
        if self.should_select():
            plan = self.plan_select(self.belief_dist, self.plans)
        elif len(self.preds) > 0 and self.preds[-1].sum() > 0:
            # first check if we see partner's plan
            dots = self.preds[-1][0]
            plan = self.plan_followup(self.belief_dist, dots)
        elif len(self.confirmations) > 0 and self.confirmations[-1]:
            dots = self.plans[-1].dots
            plan = self.plan_followup(self.belief_dist, dots)
        else:
            plan = self.plan_start(self.belief_dist, self.plans)
        self.plans.append(plan)
        return plan

    def plan_start(self, belief_dist, plans):
        EdHs = self.belief.compute_EdHs(belief_dist)
        planbool = self.belief.configs[EdHs.argmax()].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        # forget about this.
        #new, old = new_and_old_dots(planbool, plans)

        if len(self.plans) == 0:
            confirmation = None
        else:
            confirmation = self.preds[-1].sum() > 0

        plan = Plan(
            dots = planbool,
            newdots = planbool,
            olddots = None,
            plan_idxs = plan_idxs,
            should_select = False,
            confirmation = confirmation,
        )
        return plan

    def plan_followup(self, belief_dist, plans):
        EdHs = self.belief.compute_EdHs(belief_dist)

        # TODO:assume a 1st order chain for now
        #dots = plans[-1].dots
        dots = self.preds[-1][0]
        # mask out plans that don't have the desired configs
        EdHs_mask = (self.belief.configs & dots).sum(-1) == dots.sum()
        newdot_mask = self.belief.configs.sum(-1) == dots.sum() + 1
        EdHs *= EdHs_mask * newdot_mask
        planbool = self.belief.configs[EdHs.argmax()].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        prev_feats = self.belief.get_feats(self.preds[-1][0])
        prev_plan_idxs = self.belief.resolve_utt(*prev_feats)

        new_idxs, old_idxs = idxs_to_dots(prev_plan_idxs, plan_idxs, self.ctx)

        # disambiguated
        planbool2 = np.zeros(7, dtype=bool)
        planbool2[new_idxs] = 1

        old = np.zeros(7, dtype=bool)
        old[old_idxs] = 1

        new = planbool2 & ~old

        if len(self.plans) == 0:
            confirmation = None
        else:
            confirmation = self.preds[-1].sum() > 0

        plan = Plan(
            dots = planbool2,
            newdots = new,
            olddots = old,
            plan_idxs = plan_idxs,
            should_select = False,
            confirmation = confirmation,
        )
        return plan

    def plan_select(self, belief_dist, plans):
        """
        if True in pred_successes:
            # find the last partner plan we see
            ridx = list(reversed(pred_successes)).index(True)
            idx = len(self.confirmations) - ridx - 1
            import pdb; pdb.set_trace()
        else:
            # otherwise find the last confirmed plan
            ridx = list(reversed(self.confirmations)).index(True)
            idx = len(self.confirmations) - ridx - 1
            lastplan = self.plans[idx-1]
            dots = lastplan.dots
            olddots = lastplan.olddots
        """

        pred_successes = [x.sum() > 0 for x in self.preds]
        revsuc = list(reversed(pred_successes)) 
        ridx1 = revsuc.index(True)
        ridx2 = revsuc[ridx1+1:].index(True) + ridx1 + 1

        idx1 = len(self.preds) - ridx1 - 1
        idx2 = len(self.preds) - ridx2 - 1
        dots = self.preds[idx1][0]
        olddots = self.preds[idx2][0]

        feats = self.belief.get_feats(dots)
        plan_idxs = self.belief.resolve_utt(*feats)

        # choose the second last confirmed dot
        prev_feats = self.belief.get_feats(olddots)
        prev_plan_idxs = self.belief.resolve_utt(*prev_feats)

        new_idxs, old_idxs = idxs_to_dots(prev_plan_idxs, plan_idxs, self.ctx)

        # disambiguated
        planbool = np.zeros(7, dtype=bool)
        planbool[new_idxs] = 1

        old = np.zeros(7, dtype=bool)
        old[old_idxs] = 1

        new = planbool & ~old

        if len(self.plans) == 0:
            confirmation = None
        else:
            confirmation = self.preds[-1].sum() > 0

        plan = Plan(
            dots = planbool,
            newdots = new,
            olddots = old,
            plan_idxs = new_idxs,
            should_select = True,
            confirmation = confirmation,
        )
        return plan

    def choose(self):
        # TODO: REMOVE HACK
        if self.preds[-1].sum() == 0:
            return 0
        return self.preds[-1][0].nonzero()[0].item()

    def should_select(self):
        max_belief = self.belief.marginals(self.belief_dist).max()
        select = max_belief > self.belief_threshold
        return select

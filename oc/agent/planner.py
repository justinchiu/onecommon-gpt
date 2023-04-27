import numpy as np
import itertools

from oc.agent.utils import Plan

from oc.fns.spatial import get_minimum_radius

def new_and_old_dots(plan, history):
    if len(history) > 0:
        plans = [x.plan for x in history]
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

    def update_belief(self, response):
        plan = self.plans[-1].dots
        self.belief_dist = self.belief.posterior(
            self.belief_dist,
            plan.astype(int),
            response,
        )

    def plan(self):
        # was the previous plan confirmed or denied?
        if len(self.confirmations) == 0:
            previous_plan_confirmed = False
        else:
            previous_plan_confirmed = self.confirmations[-1]

        plan = (
            self.plan_followup(self.belief_dist, self.plans)
            if previous_plan_confirmed is True or previous_plan_confirmed is None
            else self.plan_start(self.belief_dist, self.plans)
        )
        self.plans.append(plan)
        return plan

    def plan_start(self, belief_dist, plans):
        EdHs = self.belief.compute_EdHs(belief_dist)
        planbool = self.belief.configs[EdHs.argmax()].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        new, old = new_and_old_dots(planbool, plans)

        if len(self.plans) == 0:
            confirmation = None
        else:
            confirmation = self.plans[-1].sum() > 0

        plan = Plan(
            dots = planbool,
            newdots = new,
            olddots = old,
            plan_idxs = plan_idxs,
            should_select = self.should_select(),
            confirmation = confirmation,
        )
        return plan

    def plan_followup(self, belief_dist, plans):
        EdHs = self.belief.compute_EdHs(belief_dist)

        # TODO:assume a 1st order chain for now
        dots = plans[-1].dots
        # mask out plans that don't have the desired configs
        EdHs_mask = (self.belief.configs & dots).sum(-1) == dots.sum()
        EdHs *= EdHs_mask
        planbool = self.belief.configs[EdHs.argmax()].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        new_idxs, old_idxs = idxs_to_dots(self.plans[-1].plan_idxs, plan_idxs, self.ctx)

        # disambiguated
        planbool2 = np.zeros(7, dtype=bool)
        planbool2[new_idxs] = 1

        old = np.zeros(7, dtype=bool)
        old[old_idxs] = 1

        new = planbool2 & ~old

        if len(self.plans) == 0:
            confirmation = None
        else:
            confirmation = self.plans[-1].sum() > 0

        plan = Plan(
            dots = planbool2,
            newdots = new,
            olddots = old,
            plan_idxs = plan_idxs,
            should_select = self.should_select(),
            confirmation = confirmation,
        )
        return plan

    def choose(self):
        import pdb; pdb.set_trace()
        return self.preds[-1]

    def should_select(self):
        max_belief = self.belief.marginals(self.belief_dist).max()
        select = max_belief > self.belief_threshold
        return select
